package processors

import (
	"context"
	"nu-hackathon/backend/internal/models"
)

// DisabledVision keeps the API honest until a local CV implementation is selected.
// Production must never substitute an invented successful verdict.
type DisabledVision struct{}

func NewVision() *DisabledVision        { return &DisabledVision{} }
func (*DisabledVision) Available() bool { return false }
func (*DisabledVision) Analyze(ctx context.Context, _ string) (models.Analysis, error) {
	if err := ctx.Err(); err != nil {
		return models.Analysis{}, err
	}
	return models.Analysis{}, models.ErrAnalysisUnavailable
}
