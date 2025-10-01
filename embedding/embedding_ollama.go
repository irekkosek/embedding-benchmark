package embedding

import (
	"context"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"net/http"
)

var _ Provider = (*OllamaEmbedding)(nil)

type OllamaEmbedding struct {
	model  string
	client *api.Client
}

func NewOllamaEmbedding(model string, httpClient *http.Client) *OllamaEmbedding {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	client := api.NewClient(envconfig.Host(), httpClient)

	return &OllamaEmbedding{
		model:  model,
		client: client,
	}
}

func (o OllamaEmbedding) GenerateEmbeddings(ctx context.Context, payload []EmbeddingPayload) ([]GenerationResponse, error) {
	var rawInputs []string
	for _, v := range payload {
		rawInputs = append(rawInputs, v.Data)
	}

	req := &api.EmbedRequest{
		Model: o.model,
		Input: rawInputs,
	}

	resp, err := o.client.Embed(ctx, req)
	if err != nil {
		return nil, err
	}

	var output []GenerationResponse
	for i, v := range resp.Embeddings {
		output = append(output, GenerationResponse{
			Data:      rawInputs[i],
			ID:        payload[i].ID,
			Embedding: v,
		})
	}

	return output, nil
}
