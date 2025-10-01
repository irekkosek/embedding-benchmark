package embedding

import (
	"context"
	"fmt"
	"github.com/openai/openai-go"
)

type OpenAIEmbeddingProvider struct {
	openai openai.Client
	model  openai.EmbeddingModel
}

func NewOpenAIEmbedding(client openai.Client, model openai.EmbeddingModel) *OpenAIEmbeddingProvider {
	return &OpenAIEmbeddingProvider{
		openai: client,
		model:  model,
	}
}

func (p *OpenAIEmbeddingProvider) GenerateEmbeddings(ctx context.Context, payloads []EmbeddingPayload) ([]GenerationResponse, error) {
	var payloadsOfStrings []string
	for _, payload := range payloads {
		if payload.Data == "" {
			return nil, fmt.Errorf("payload data cannot be empty")
		}
		payloadsOfStrings = append(payloadsOfStrings, payload.Data)
	}

	input := openai.EmbeddingNewParamsInputUnion{
		OfArrayOfStrings: payloadsOfStrings,
	}

	payload := openai.EmbeddingNewParams{
		Input:          input,
		Model:          p.model,
		EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
	}

	res, err := p.openai.Embeddings.New(ctx, payload)
	if err != nil {
		panic(err)
	}

	if len(res.Data) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	var embeddings []GenerationResponse
	for i, data := range res.Data {
		if len(data.Embedding) == 0 {
			return nil, fmt.Errorf("data has no embeddings")
		}

		embeddings = append(embeddings, GenerationResponse{
			Data:      payloads[i].Data,
			ID:        payloads[i].ID,
			Embedding: Float64ListToFloat32List(data.Embedding),
		})
	}

	return embeddings, nil
}
func Float64ListToFloat32List(fs []float64) []float32 {
	out := make([]float32, len(fs))
	for i, f := range fs {
		out[i] = float32(f)
	}
	return out
}
