package embedding

import (
	"context"
)

type Provider interface {
	GenerateEmbeddings(ctx context.Context, payload []EmbeddingPayload) ([]GenerationResponse, error)
}

type EmbeddingPayload struct {
	Data string `json:"data"`
	// TODO: Instead of ID there could be map like "Metadata" for tracking
	ID int64 `json:"id"` // ID is optional, it can be used to track element and later add metadata for example for qdrant or other vector databases
}

type GenerationResponse struct {
	Data      string
	Embedding []float32
	ID        int64 // ID is optional, it can be used to track element and later add metadata for example for qdrant or other vector databases
}
