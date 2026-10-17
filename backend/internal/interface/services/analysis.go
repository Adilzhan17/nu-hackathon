package services

import (
	"context"
	"io"
	"nu-hackathon/backend/internal/models"
	"time"
)

type Analysis interface {
	Submit(context.Context, string, io.Reader) (models.Job, error)
	Available() bool
}
type Scheduler interface {
	Enqueue(string) error
}

type submissionStartKey struct{}

// WithSubmissionStart accounts for multipart parsing in the total analysis budget.
func WithSubmissionStart(ctx context.Context, at time.Time) context.Context {
	return context.WithValue(ctx, submissionStartKey{}, at)
}

func SubmissionStart(ctx context.Context) time.Time {
	now := time.Now().UTC()
	if at, ok := ctx.Value(submissionStartKey{}).(time.Time); ok && !at.IsZero() && !at.After(now) {
		return at.UTC()
	}
	return now
}
