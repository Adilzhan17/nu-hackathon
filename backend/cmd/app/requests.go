package app

import (
	"context"
	"net/http"
	"nu-hackathon/backend/internal/controller/http/v1/response"
	"sync"
)

// Stop prevents new dependency users before Wait observes the final active count.
type requestTracker struct {
	next     http.Handler
	mu       sync.Mutex
	active   sync.WaitGroup
	stopping bool
	done     chan struct{}
}

func newRequestTracker(next http.Handler) *requestTracker {
	return &requestTracker{next: next, done: make(chan struct{})}
}
func (t *requestTracker) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	t.mu.Lock()
	if t.stopping {
		t.mu.Unlock()
		response.Error(w, 503, "service_busy", "The service is shutting down.")
		return
	}
	t.active.Add(1)
	t.mu.Unlock()
	defer t.active.Done()
	t.next.ServeHTTP(w, r)
}
func (t *requestTracker) Stop() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.stopping {
		return
	}
	t.stopping = true
	go func() { t.active.Wait(); close(t.done) }()
}
func (t *requestTracker) Wait(ctx context.Context) error {
	select {
	case <-t.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
