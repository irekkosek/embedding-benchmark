package main

import (
	"context"
	"flag"
	"fmt"
	"github.com/marosiak/embedding-benchmark/embedding"
	"math"
	"net/http"
	"os"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

type job struct {
	id int
}

func main() {
	workers := flag.Int("workers", envInt("WORKERS", 2), "number of concurrent workers")
	duration := flag.Duration("duration", envDuration("DURATION", 2*time.Minute), "total run duration (default 2 min)")
	model := flag.String("model", "hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q8_0", "ollama model to use, default: hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q8_0")
	input := flag.String("input", "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer convallis.", "input to embed (will be appended with number in order to skip possible caching")
	flag.Parse()

	httpClient := http.Client{Timeout: 20 * time.Second}
	emb := embedding.NewOllamaEmbedding(*model, &httpClient)

	ctx, cancel := context.WithTimeout(context.Background(), *duration)
	defer cancel()

	jobs := make(chan job, *workers*2)
	durations := make([]float64, 0, 1000)
	var durationsMu sync.Mutex

	var processed atomic.Int64
	var wg sync.WaitGroup

	for w := 0; w < *workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-ctx.Done():
					return
				case j, ok := <-jobs:
					if !ok {
						return
					}
					text := fmt.Sprintf("%s %d", input, j.id)
					start := time.Now()
					_, err := emb.GenerateEmbeddings(ctx, []embedding.EmbeddingPayload{{Data: text, ID: int64(j.id)}})
					if err == nil {
						fmt.Printf("GenerateEmbeddings error: %v\n", err)
						processed.Add(1)
						tookMs := float64(time.Since(start).Microseconds()) / 1000.0
						durationsMu.Lock()
						durations = append(durations, tookMs)
						durationsMu.Unlock()
					} else {
						if ctx.Err() != nil {
							return
						}
						return
					}
				}
			}
		}()
	}

	startAll := time.Now()
	go func() {
		i := 0
		for {
			select {
			case <-ctx.Done():
				close(jobs)
				return
			default:
				select {
				case jobs <- job{id: i}:
					i++
				case <-ctx.Done():
					close(jobs)
					return
				}
			}
		}
	}()

	wg.Wait()
	totalDur := time.Since(startAll)
	fmt.Println("processed all", "took", totalDur, "amount", processed.Load())

	durationsMu.Lock()
	stats := computeStats(durations)
	durationsMu.Unlock()

	if stats.count > 0 {
		fmt.Printf("min %.2f ms | avg %.2f ms | max %.2f ms\n", stats.min, stats.mean, stats.max)
		fmt.Printf("p50 %.2f ms | p90 %.2f ms | p95 %.2f ms | p99 %.2f ms\n", stats.p50, stats.p90, stats.p95, stats.p99)
		fmt.Printf("stddev %.2f ms\n", stats.stddev)
		rps := float64(processed.Load()) / totalDur.Seconds()
		fmt.Printf("throughput %.2f req/s\n", rps)
	} else {
		fmt.Println("no successful samples collected")
	}
}

type summary struct {
	count  int
	min    float64
	mean   float64
	max    float64
	p50    float64
	p90    float64
	p95    float64
	p99    float64
	stddev float64
}

func computeStats(samples []float64) summary {
	n := len(samples)
	if n == 0 {
		return summary{}
	}
	cp := make([]float64, n)
	copy(cp, samples)
	sort.Float64s(cp)

	vMin := cp[0]
	vMax := cp[n-1]
	sum := 0.0
	for _, v := range cp {
		sum += v
	}
	mean := sum / float64(n)

	var variance float64
	if n > 1 {
		m := mean
		ss := 0.0
		for _, v := range cp {
			d := v - m
			ss += d * d
		}
		variance = ss / float64(n-1)
	}
	stddev := math.Sqrt(variance)

	p50 := percentile(cp, 50)
	p90 := percentile(cp, 90)
	p95 := percentile(cp, 95)
	p99 := percentile(cp, 99)

	return summary{
		count:  n,
		min:    vMin,
		mean:   mean,
		max:    vMax,
		p50:    p50,
		p90:    p90,
		p95:    p95,
		p99:    p99,
		stddev: stddev,
	}
}

func percentile(sorted []float64, p int) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 100 {
		return sorted[len(sorted)-1]
	}
	rank := (float64(p) / 100.0) * float64(len(sorted)-1)
	l := int(math.Floor(rank))
	u := int(math.Ceil(rank))
	if l == u {
		return sorted[l]
	}
	w := rank - float64(l)
	return sorted[l]*(1.0-w) + sorted[u]*w
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envDuration(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}
