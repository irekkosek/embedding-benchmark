# What's this?
Simple embedding benchmark made with Go - it works only with Ollama, and most likely won't be developed further as it was created for one single purpose.


## How to?

### Install Ollama  
For Linux:
`curl -fsSL https://ollama.com/install.sh | sh`

Other platforms: https://ollama.com/download

### Fetch Ollama model
```bash
ollama pull hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q8_0
```
### Run
```bash
git clone git@github.com:marosiak/embedding-benchmark.git
cd embedding-benchmark
go mod tidy
go run main.go --duration 5m
```

## CLI flags
```go run main.go --help```