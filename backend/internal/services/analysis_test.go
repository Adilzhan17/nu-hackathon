package services_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"log/slog"
	"path/filepath"
	"sync"
	"testing"
	"time"

	ports "nu-hackathon/backend/internal/interface/processors"
	serviceports "nu-hackathon/backend/internal/interface/services"
	"nu-hackathon/backend/internal/models"
	"nu-hackathon/backend/internal/processors"
	"nu-hackathon/backend/internal/repositories"
	"nu-hackathon/backend/internal/services"
)

type testScheduler struct{ called bool }

func (s *testScheduler) Enqueue(string) error { s.called = true; return nil }

type enabledVision struct{}

func (enabledVision) Available() bool { return true }
func (enabledVision) Analyze(context.Context, string) (models.Analysis, error) {
	panic("service must not run inference")
}

type forbiddenReader struct{ t *testing.T }

func (r forbiddenReader) Read([]byte) (int, error) {
	r.t.Fatal("unavailable service read request body")
	return 0, io.EOF
}

type cancelAfterSave struct {
	ports.Storage
	cancel  context.CancelFunc
	savedID *string
}

func (s cancelAfterSave) Save(ctx context.Context, id, filename string, src io.Reader, limit int64) error {
	if err := s.Storage.Save(ctx, id, filename, src, limit); err != nil {
		return err
	}
	*s.savedID = id
	s.cancel()
	return nil
}

type blockedStorage struct {
	ports.Storage
	started, release chan struct{}
}

func (s blockedStorage) Save(ctx context.Context, id, filename string, src io.Reader, limit int64) error {
	close(s.started)
	select {
	case <-s.release:
	case <-ctx.Done():
		return ctx.Err()
	}
	return s.Storage.Save(ctx, id, filename, src, limit)
}

func adapters(t *testing.T) (*repositories.JobsRepository, *processors.LocalStorage, *slog.Logger) {
	t.Helper()
	root := t.TempDir()
	repo, err := repositories.NewJobs(filepath.Join(root, "jobs.db"))
	if err != nil {
		t.Fatal(err)
	}
	storage, err := processors.NewStorage(filepath.Join(root, "uploads"))
	if err != nil {
		repo.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() { storage.Close(); repo.Close() })
	return repo, storage, slog.New(slog.NewTextHandler(io.Discard, nil))
}

func movFixture() []byte {
	box := func(kind string) []byte {
		out := make([]byte, 9)
		binary.BigEndian.PutUint32(out, 9)
		copy(out[4:8], kind)
		return out
	}
	return append(box("moov"), box("mdat")...)
}

func TestUnavailableAnalysisDoesNotConsumeOrPersist(t *testing.T) {
	repo, storage, log := adapters(t)
	scheduler := &testScheduler{}
	service := services.NewAnalysis(repo, storage, processors.NewVision(), scheduler, services.AnalysisOptions{}, log)
	if _, err := service.Submit(context.Background(), "clip.mov", forbiddenReader{t}); !errors.Is(err, models.ErrAnalysisUnavailable) {
		t.Fatalf("unavailable error: %v", err)
	}
	rows, err := repo.List(context.Background())
	if err != nil || len(rows) != 0 || scheduler.called {
		t.Fatalf("unavailable request was persisted or scheduled: rows=%d scheduled=%v err=%v", len(rows), scheduler.called, err)
	}
}

func TestCancellationAfterUploadRollsBackWithFreshContext(t *testing.T) {
	repo, storage, log := adapters(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	scheduler := &testScheduler{}
	var savedID string
	wrappedStorage := cancelAfterSave{Storage: storage, cancel: cancel, savedID: &savedID}
	service := services.NewAnalysis(repo, wrappedStorage, enabledVision{}, scheduler, services.AnalysisOptions{}, log)
	if _, err := service.Submit(ctx, "clip.mov", bytes.NewReader(movFixture())); !errors.Is(err, context.Canceled) {
		t.Fatalf("want cancellation, got %v", err)
	}
	rows, err := repo.List(context.Background())
	if err != nil || len(rows) != 0 || scheduler.called {
		t.Fatalf("canceled upload retained job: rows=%d scheduled=%v err=%v", len(rows), scheduler.called, err)
	}
	if _, err := storage.Path(savedID); !errors.Is(err, models.ErrNotFound) {
		t.Fatalf("canceled upload retained media: %v", err)
	}
}

func TestSimultaneousUploadLimitRejectsBeforeReading(t *testing.T) {
	repo, storage, log := adapters(t)
	started, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	service := services.NewAnalysis(repo, blockedStorage{Storage: storage, started: started, release: release}, enabledVision{}, &testScheduler{}, services.AnalysisOptions{MaxUploads: 1}, log)
	completed := make(chan error, 1)
	go func() {
		_, err := service.Submit(context.Background(), "first.mov", bytes.NewReader(movFixture()))
		completed <- err
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("upload did not begin")
	}
	if _, err := service.Submit(context.Background(), "second.mov", forbiddenReader{t}); !errors.Is(err, models.ErrQueueFull) {
		t.Fatalf("want upload capacity error, got %v", err)
	}
	once.Do(func() { close(release) })
	select {
	case err := <-completed:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("first upload did not complete")
	}
}

func TestJobsRejectsNoncanonicalIDs(t *testing.T) {
	repo, _, _ := adapters(t)
	service := services.NewJobs(repo)
	for _, id := range []string{"", "../jobs.db", "ABCDEF0123456789ABCDEF0123456789", "unknown"} {
		if _, err := service.Get(context.Background(), id); !errors.Is(err, models.ErrNotFound) {
			t.Fatalf("%q: %v", id, err)
		}
	}
}

func TestAnalysisBudgetIncludesTimeBeforeServiceSubmission(t *testing.T) {
	repo, storage, log := adapters(t)
	scheduler := &testScheduler{}
	service := services.NewAnalysis(repo, storage, enabledVision{}, scheduler, services.AnalysisOptions{Timeout: time.Minute}, log)
	started := time.Now().UTC().Add(-10 * time.Second)
	ctx := serviceports.WithSubmissionStart(context.Background(), started)
	job, err := service.Submit(ctx, "clip.mov", bytes.NewReader(movFixture()))
	if err != nil {
		t.Fatal(err)
	}
	if !job.CreatedAt.Equal(started) || !job.DeadlineAt.Equal(started.Add(time.Minute)) {
		t.Fatalf("multipart parsing time was excluded from budget: %+v", job)
	}

	expired := serviceports.WithSubmissionStart(context.Background(), time.Now().Add(-2*time.Minute))
	if _, err := service.Submit(expired, "late.mov", bytes.NewReader(movFixture())); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expired total budget accepted: %v", err)
	}
	rows, err := repo.List(context.Background())
	if err != nil || len(rows) != 1 {
		t.Fatalf("expired upload persisted: rows=%d err=%v", len(rows), err)
	}
}
