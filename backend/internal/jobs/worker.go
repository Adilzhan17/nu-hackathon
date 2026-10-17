package jobs

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"nu-hackathon/backend/internal/enums"
	"nu-hackathon/backend/internal/interface/processors"
	"nu-hackathon/backend/internal/interface/repositories"
	"nu-hackathon/backend/internal/models"
)

type WorkerOptions struct {
	Workers         int
	QueueSize       int
	Retention       time.Duration
	CleanupInterval time.Duration
}

type Worker struct {
	repo     repositories.Jobs
	storage  processors.Storage
	vision   processors.Vision
	opts     WorkerOptions
	log      *slog.Logger
	queue    chan string
	mu       sync.Mutex
	started  bool
	stopping bool
	pending  map[string]bool
	ctx      context.Context
	cancel   context.CancelFunc
	done     chan struct{}
	stopErr  error
}

func NewWorker(repo repositories.Jobs, storage processors.Storage, vision processors.Vision, opts WorkerOptions, log *slog.Logger) *Worker {
	if opts.Workers <= 0 {
		opts.Workers = 1
	}
	if opts.QueueSize <= 0 {
		opts.QueueSize = 8
	}
	if opts.Retention <= 0 {
		opts.Retention = 24 * time.Hour
	}
	if opts.CleanupInterval <= 0 {
		opts.CleanupInterval = 15 * time.Minute
	}
	if log == nil {
		log = slog.Default()
	}
	return &Worker{repo: repo, storage: storage, vision: vision, opts: opts, log: log, queue: make(chan string, opts.QueueSize), pending: make(map[string]bool), done: make(chan struct{})}
}

func (w *Worker) Start(ctx context.Context) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.stopping {
		return models.ErrShuttingDown
	}
	if w.started {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := w.interruptProcessing(ctx); err != nil {
		return fmt.Errorf("recover jobs: %w", err)
	}
	if err := w.cleanup(ctx, false); err != nil {
		return fmt.Errorf("clean job storage: %w", err)
	}
	w.ctx, w.cancel = context.WithCancel(ctx)
	w.started = true
	var wg sync.WaitGroup
	wg.Add(w.opts.Workers + 1)
	for i := 0; i < w.opts.Workers; i++ {
		go func() { defer wg.Done(); w.run() }()
	}
	go func() { defer wg.Done(); w.maintain() }()
	go func() {
		wg.Wait()
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		err := w.interruptProcessing(cleanupCtx)
		w.mu.Lock()
		w.stopErr = err
		w.stopping = true
		w.mu.Unlock()
		close(w.done)
	}()
	return nil
}

func (w *Worker) Enqueue(id string) error {
	if !models.ValidJobID(id) {
		return models.ErrNotFound
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if !w.started || w.stopping || w.ctx.Err() != nil {
		return models.ErrShuttingDown
	}
	if w.pending[id] {
		return models.ErrConflict
	}
	select {
	case w.queue <- id:
		w.pending[id] = true
		return nil
	default:
		return models.ErrQueueFull
	}
}

// Stop waits for the analyzer to honor cancellation before the database can close.
// If its context expires, callers must leave resources open while workers finish.
func (w *Worker) Stop(ctx context.Context) error {
	w.mu.Lock()
	if !w.started {
		w.stopping = true
		w.mu.Unlock()
		return nil
	}
	w.stopping = true
	w.cancel()
	w.mu.Unlock()
	select {
	case <-w.done:
		w.mu.Lock()
		defer w.mu.Unlock()
		return w.stopErr
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Stopped reports whether no worker or cleanup operation can use the repository.
// A cleanup failure returned by Stop does not imply that workers are still live.
func (w *Worker) Stopped() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	if !w.started {
		return w.stopping
	}
	select {
	case <-w.done:
		return true
	default:
		return false
	}
}

func (w *Worker) run() {
	for {
		select {
		case <-w.ctx.Done():
			return
		case id := <-w.queue:
			if w.ctx.Err() != nil {
				return
			}
			w.execute(id)
			w.mu.Lock()
			delete(w.pending, id)
			w.mu.Unlock()
		}
	}
}

func (w *Worker) execute(id string) {
	job, err := w.repo.Get(w.ctx, id)
	if err != nil {
		w.log.Error("load queued job", "job_id", id, "error", err)
		return
	}
	if job.Status != enums.JobProcessing {
		return
	}
	ctx, cancel := context.WithDeadline(w.ctx, job.DeadlineAt)
	defer cancel()
	var result models.Analysis
	if err = ctx.Err(); err == nil {
		if !w.vision.Available() {
			err = models.ErrAnalysisUnavailable
		} else {
			var path string
			path, err = w.storage.Path(id)
			if err == nil {
				result, err = analyze(ctx, w.vision, path)
			}
		}
	}
	if ctx.Err() != nil {
		err = ctx.Err()
	}
	failure := "Analysis failed. Please try again."
	if err == nil {
		if err = result.Validate(); err != nil {
			failure = "The analyzer returned an invalid result. Please try again."
		}
	}
	if ctx.Err() != nil {
		err = ctx.Err()
	}
	job.UpdatedAt = time.Now().UTC()
	if err != nil {
		switch {
		case errors.Is(err, context.DeadlineExceeded):
			failure = "The analysis time limit was exceeded."
		case errors.Is(err, context.Canceled):
			failure = "Analysis was interrupted. Please submit the video again."
		case errors.Is(err, models.ErrAnalysisUnavailable):
			failure = "The local analyzer is unavailable."
		}
		job.Status, job.Result, job.Failure = enums.JobFailed, nil, failure
		// Analyzer errors may contain paths or model input; never log their text.
		w.log.Warn("analysis failed", "job_id", id, "reason", failure)
	} else {
		if result.Violations == nil {
			result.Violations = []models.Violation{}
		}
		job.Status, job.Result, job.Failure = enums.JobDone, &result, ""
	}
	persistCtx, persistCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer persistCancel()
	if err := w.repo.Update(persistCtx, job); err != nil {
		w.log.Error("persist analysis result", "job_id", id, "error", err)
		// Keep the input until a terminal status is durable. Periodic maintenance
		// reconciles this abandoned processing row once its deadline has elapsed.
		return
	}
	if err := w.storage.Remove(persistCtx, id); err != nil {
		w.log.Error("remove analyzed video", "job_id", id, "error", err)
	}
}

func analyze(ctx context.Context, vision processors.Vision, path string) (result models.Analysis, err error) {
	defer func() {
		if recover() != nil {
			result = models.Analysis{}
			err = errors.New("analyzer panicked")
		}
	}()
	return vision.Analyze(ctx, path)
}

func (w *Worker) interruptProcessing(ctx context.Context) error {
	jobs, err := w.repo.List(ctx)
	if err != nil {
		return err
	}
	for _, job := range jobs {
		if job.Status != enums.JobProcessing {
			continue
		}
		job.Status = enums.JobFailed
		job.Result = nil
		job.Failure = "Analysis was interrupted. Please submit the video again."
		job.UpdatedAt = time.Now().UTC()
		if err := w.repo.Update(ctx, job); err != nil {
			return err
		}
		if err := w.storage.Remove(ctx, job.ID); err != nil {
			w.log.Error("remove interrupted video", "job_id", job.ID, "error", err)
		}
	}
	return nil
}

func (w *Worker) maintain() {
	ticker := time.NewTicker(w.opts.CleanupInterval)
	defer ticker.Stop()
	for {
		select {
		case <-w.ctx.Done():
			return
		case <-ticker.C:
			if err := w.cleanup(w.ctx, true); err != nil && w.ctx.Err() == nil {
				w.log.Error("clean job storage", "error", err)
			}
		}
	}
}

func (w *Worker) cleanup(ctx context.Context, reconcile bool) error {
	jobs, err := w.repo.List(ctx)
	if err != nil {
		return err
	}
	now := time.Now().UTC()
	keep := make(map[string]bool)
	for _, job := range jobs {
		if job.Status == enums.JobProcessing {
			if reconcile && !job.DeadlineAt.After(now) {
				terminal, err := w.reconcileExpired(ctx, job.ID, now)
				if err != nil {
					return err
				}
				if terminal {
					if err := w.storage.Remove(ctx, job.ID); err != nil {
						return err
					}
					continue
				}
			}
			keep[job.ID] = true
			continue
		}
		if err := w.storage.Remove(ctx, job.ID); err != nil {
			return err
		}
		if job.UpdatedAt.Before(now.Add(-w.opts.Retention)) {
			if err := w.repo.Delete(ctx, job.ID); err != nil {
				return err
			}
		}
	}
	// A save may not yet have a DB row. Preserve recent files, including uploads
	// currently being streamed; configured request uploads expire in < 1 hour.
	return w.storage.Cleanup(ctx, now.Add(-time.Hour), keep)
}

func (w *Worker) reconcileExpired(ctx context.Context, id string, now time.Time) (bool, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.pending[id] {
		return false, nil
	}
	// Re-read under the scheduling lock: the snapshot may precede a successful
	// completion, which must never be overwritten with a failure.
	job, err := w.repo.Get(ctx, id)
	if err != nil {
		return false, err
	}
	if job.Status != enums.JobProcessing {
		return true, nil
	}
	if job.DeadlineAt.After(now) {
		return false, nil
	}
	job.Status, job.Result = enums.JobFailed, nil
	job.Failure = "The analysis time limit was exceeded."
	job.UpdatedAt = now
	if err := w.repo.Update(ctx, job); err != nil {
		return false, err
	}
	return true, nil
}
