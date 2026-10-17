package processors

import (
	"context"
	"io"
	"time"
)

type Storage interface {
	Save(ctx context.Context, id, filename string, src io.Reader, maxBytes int64) error
	Path(id string) (string, error)
	Remove(ctx context.Context, id string) error
	// Cleanup removes owned files older than cutoff, except IDs in keep.
	Cleanup(ctx context.Context, cutoff time.Time, keep map[string]bool) error
}
