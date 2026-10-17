package processors

import (
	"context"
	"nu-hackathon/backend/internal/models"
)

// Analyze must honor ctx, run locally, and return genuine evidence frames.
// Implementations must not use cloud inference, audio, or face identification.
type Vision interface {
	Available() bool
	Analyze(ctx context.Context, videoPath string) (models.Analysis, error)
}
