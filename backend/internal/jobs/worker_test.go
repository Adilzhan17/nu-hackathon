package jobs_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nu-hackathon/backend/internal/enums"
	repositoryports "nu-hackathon/backend/internal/interface/repositories"
	"nu-hackathon/backend/internal/jobs"
	"nu-hackathon/backend/internal/models"
	"nu-hackathon/backend/internal/processors"
	"nu-hackathon/backend/internal/repositories"
	"nu-hackathon/backend/internal/services"
)

type testVision struct {
	run func(context.Context, string) (models.Analysis, error)
}

type failingRepository struct {
	repositoryports.Jobs
	failUpdates  atomic.Bool
	updateFailed chan struct{}
	failStopList bool
	listCalls    atomic.Int32
}

func (r *failingRepository) Update(ctx context.Context, job models.Job) error {
	if r.failUpdates.Load() {
		select {
		case r.updateFailed <- struct{}{}:
		default:
		}
		return errors.New("test database write unavailable")
	}
	return r.Jobs.Update(ctx, job)
}

func (r *failingRepository) List(ctx context.Context) ([]models.Job, error) {
	if r.listCalls.Add(1) >= 3 && r.failStopList {
		return nil, errors.New("test shutdown cleanup failed")
	}
	return r.Jobs.List(ctx)
}

func (testVision) Available() bool { return true }
func (v testVision) Analyze(ctx context.Context, path string) (models.Analysis, error) {
	return v.run(ctx, path)
}

func fixture(t *testing.T) (*repositories.JobsRepository, *processors.LocalStorage, string, *slog.Logger) {
	t.Helper()
	root := t.TempDir()
	dbPath := filepath.Join(root, "jobs.db")
	repo, err := repositories.NewJobs(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	storage, err := processors.NewStorage(filepath.Join(root, "uploads"))
	if err != nil {
		repo.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() { storage.Close(); repo.Close() })
	return repo, storage, dbPath, slog.New(slog.NewTextHandler(io.Discard, nil))
}

func startWorker(t *testing.T, worker *jobs.Worker) {
	t.Helper()
	if err := worker.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		if err := worker.Stop(ctx); err != nil {
			t.Errorf("stop worker: %v", err)
		}
	})
}

// This minimal container exercises upload boundaries; testVision supplies the
// result directly. It is deliberately not advertised as decodable footage.
func movie() []byte {
	box := func(kind string, body []byte) []byte {
		b := make([]byte, 8+len(body))
		binary.BigEndian.PutUint32(b, uint32(len(b)))
		copy(b[4:8], kind)
		copy(b[8:], body)
		return b
	}
	return append(box("moov", box("mvhd", []byte{0})), box("mdat", []byte{0})...)
}

func waitJob(t *testing.T, repo *repositories.JobsRepository, id string, want enums.JobStatus) models.Job {
	t.Helper()
	until := time.Now().Add(3 * time.Second)
	for time.Now().Before(until) {
		job, err := repo.Get(context.Background(), id)
		if err != nil {
			t.Fatal(err)
		}
		if job.Status == want {
			return job
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("job %s did not reach %s", id, want)
	return models.Job{}
}

func TestPipelinePersistsCompletedJobAndRemovesVideo(t *testing.T) {
	repo, storage, dbPath, log := fixture(t)
	vision := testVision{run: func(_ context.Context, path string) (models.Analysis, error) {
		if _, err := os.Stat(path); err != nil {
			return models.Analysis{}, err
		}
		return models.Analysis{Verdict: enums.VerdictNoViolation}, nil
	}}
	worker := jobs.NewWorker(repo, storage, vision, jobs.WorkerOptions{}, log)
	startWorker(t, worker)
	service := services.NewAnalysis(repo, storage, vision, worker, services.AnalysisOptions{Timeout: time.Minute}, log)
	job, err := service.Submit(context.Background(), "camera.mov", bytes.NewReader(movie()))
	if err != nil {
		t.Fatal(err)
	}
	if !models.ValidJobID(job.ID) || job.DeadlineAt.Sub(job.CreatedAt) != time.Minute {
		t.Fatalf("invalid submitted job: %+v", job)
	}
	got := waitJob(t, repo, job.ID, enums.JobDone)
	if got.Result == nil || got.Result.Verdict != enums.VerdictNoViolation || got.Result.Violations == nil {
		t.Fatalf("invalid completed result: %+v", got)
	}
	if err := worker.Stop(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := storage.Path(job.ID); !errors.Is(err, models.ErrNotFound) {
		t.Fatalf("raw video retained: %v", err)
	}
	if err := repo.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := repositories.NewJobs(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	got, err = reopened.Get(context.Background(), job.ID)
	if err != nil || got.Status != enums.JobDone || got.Result == nil {
		t.Fatalf("result lost on restart: %+v %v", got, err)
	}
}

func TestTimeoutAndInvalidOutputFailSafely(t *testing.T) {
	cases := []struct {
		name    string
		run     func(context.Context, string) (models.Analysis, error)
		message string
	}{
		{"deadline", func(ctx context.Context, _ string) (models.Analysis, error) {
			<-ctx.Done()
			return models.Analysis{}, ctx.Err()
		}, "time limit"},
		{"late success", func(ctx context.Context, _ string) (models.Analysis, error) {
			<-ctx.Done()
			return models.Analysis{Verdict: enums.VerdictNoViolation}, nil
		}, "time limit"},
		{"invalid output", func(context.Context, string) (models.Analysis, error) {
			return models.Analysis{Verdict: enums.VerdictViolation}, nil
		}, "invalid result"},
		{"error privacy", func(context.Context, string) (models.Analysis, error) {
			return models.Analysis{}, errors.New("secret model/path/key")
		}, "Analysis failed"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			repo, storage, _, log := fixture(t)
			vision := testVision{run: tc.run}
			worker := jobs.NewWorker(repo, storage, vision, jobs.WorkerOptions{}, log)
			startWorker(t, worker)
			service := services.NewAnalysis(repo, storage, vision, worker, services.AnalysisOptions{Timeout: 80 * time.Millisecond}, log)
			job, err := service.Submit(context.Background(), "clip.mov", bytes.NewReader(movie()))
			if err != nil {
				t.Fatal(err)
			}
			got := waitJob(t, repo, job.ID, enums.JobFailed)
			if got.Result != nil || !strings.Contains(got.Failure, tc.message) || strings.Contains(got.Failure, "secret") {
				t.Fatalf("unsafe or incorrect failed job: %+v", got)
			}
		})
	}
}

func TestAnalyzerPanicDoesNotLoseWorker(t *testing.T) {
	repo, storage, _, log := fixture(t)
	var calls atomic.Int32
	vision := testVision{run: func(context.Context, string) (models.Analysis, error) {
		if calls.Add(1) == 1 {
			panic("secret panic payload")
		}
		return models.Analysis{Verdict: enums.VerdictNoViolation}, nil
	}}
	worker := jobs.NewWorker(repo, storage, vision, jobs.WorkerOptions{}, log)
	startWorker(t, worker)
	service := services.NewAnalysis(repo, storage, vision, worker, services.AnalysisOptions{}, log)
	first, err := service.Submit(context.Background(), "one.mov", bytes.NewReader(movie()))
	if err != nil {
		t.Fatal(err)
	}
	failed := waitJob(t, repo, first.ID, enums.JobFailed)
	if strings.Contains(failed.Failure, "secret") {
		t.Fatal("panic leaked into job")
	}
	next, err := service.Submit(context.Background(), "two.mov", bytes.NewReader(movie()))
	if err != nil {
		t.Fatal(err)
	}
	waitJob(t, repo, next.ID, enums.JobDone)
}

func TestQueueBackpressureRollsBackAndStopFinalizesPending(t *testing.T) {
	repo, storage, _, log := fixture(t)
	entered := make(chan struct{}, 1)
	vision := testVision{run: func(ctx context.Context, _ string) (models.Analysis, error) {
		entered <- struct{}{}
		<-ctx.Done()
		return models.Analysis{}, ctx.Err()
	}}
	worker := jobs.NewWorker(repo, storage, vision, jobs.WorkerOptions{QueueSize: 1, Workers: 1}, log)
	startWorker(t, worker)
	service := services.NewAnalysis(repo, storage, vision, worker, services.AnalysisOptions{}, log)
	first, err := service.Submit(context.Background(), "one.mov", bytes.NewReader(movie()))
	if err != nil {
		t.Fatal(err)
	}
	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("worker never started")
	}
	second, err := service.Submit(context.Background(), "two.mov", bytes.NewReader(movie()))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := service.Submit(context.Background(), "three.mov", bytes.NewReader(movie())); !errors.Is(err, models.ErrQueueFull) {
		t.Fatalf("want queue full, got %v", err)
	}
	records, err := repo.List(context.Background())
	if err != nil || len(records) != 2 {
		t.Fatalf("unqueued row leaked: %d %v", len(records), err)
	}
	firstPath, err := storage.Path(first.ID)
	if err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(filepath.Dir(firstPath))
	if err != nil || len(entries) != 2 {
		t.Fatalf("unqueued upload leaked: %d %v", len(entries), err)
	}
	if err := worker.Stop(context.Background()); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{first.ID, second.ID} {
		waitJob(t, repo, id, enums.JobFailed)
		if _, err := storage.Path(id); !errors.Is(err, models.ErrNotFound) {
			t.Fatalf("shutdown retained video: %v", err)
		}
	}
	if err := worker.Enqueue(first.ID); !errors.Is(err, models.ErrShuttingDown) {
		t.Fatalf("enqueue after stop: %v", err)
	}
}

func TestRestartRecoveryAndRetention(t *testing.T) {
	repo, storage, _, log := fixture(t)
	now := time.Now().UTC()
	interrupted := models.Job{ID: strings.Repeat("a", 32), Status: enums.JobProcessing, CreatedAt: now, UpdatedAt: now, DeadlineAt: now.Add(time.Minute)}
	if err := storage.Save(context.Background(), interrupted.ID, "clip.mov", bytes.NewReader(movie()), 1024); err != nil {
		t.Fatal(err)
	}
	path, err := storage.Path(interrupted.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err := repo.Create(context.Background(), interrupted); err != nil {
		t.Fatal(err)
	}
	expired := models.Job{ID: strings.Repeat("b", 32), Status: enums.JobDone, CreatedAt: now.Add(-2 * time.Hour), UpdatedAt: now.Add(-2 * time.Hour), Result: &models.Analysis{Verdict: enums.VerdictNoViolation}}
	if err := repo.Create(context.Background(), expired); err != nil {
		t.Fatal(err)
	}
	orphan := filepath.Join(filepath.Dir(path), strings.Repeat("c", 32)+".part")
	foreign := filepath.Join(filepath.Dir(path), "user-file.mov")
	for _, filename := range []string{orphan, foreign} {
		if err := os.WriteFile(filename, []byte("partial upload"), 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(filename, now.Add(-2*time.Hour), now.Add(-2*time.Hour)); err != nil {
			t.Fatal(err)
		}
	}
	worker := jobs.NewWorker(repo, storage, processors.NewVision(), jobs.WorkerOptions{Retention: time.Hour}, log)
	startWorker(t, worker)
	waitJob(t, repo, interrupted.ID, enums.JobFailed)
	if _, err := storage.Path(interrupted.ID); !errors.Is(err, models.ErrNotFound) {
		t.Fatalf("interrupted video retained: %v", err)
	}
	if _, err := repo.Get(context.Background(), expired.ID); !errors.Is(err, models.ErrNotFound) {
		t.Fatalf("expired record retained: %v", err)
	}
	if _, err := os.Stat(orphan); !os.IsNotExist(err) {
		t.Fatalf("orphan retained: %v", err)
	}
	if _, err := os.Stat(foreign); err != nil {
		t.Fatalf("foreign file touched: %v", err)
	}
}

func TestStoppedDistinguishesCleanupFailureFromRunningAnalyzer(t *testing.T) {
	t.Run("cleanup failure still stopped", func(t *testing.T) {
		repo, storage, _, log := fixture(t)
		wrapped := &failingRepository{Jobs: repo, failStopList: true}
		worker := jobs.NewWorker(wrapped, storage, processors.NewVision(), jobs.WorkerOptions{}, log)
		if err := worker.Start(context.Background()); err != nil {
			t.Fatal(err)
		}
		if worker.Stopped() {
			t.Fatal("running worker reported stopped")
		}
		if err := worker.Stop(context.Background()); err == nil {
			t.Fatal("cleanup error lost")
		}
		if !worker.Stopped() {
			t.Fatal("completed shutdown must permit resource closure despite cleanup error")
		}
	})
	t.Run("timed out stop leaves analyzer running", func(t *testing.T) {
		repo, storage, _, log := fixture(t)
		entered, release := make(chan struct{}), make(chan struct{})
		vision := testVision{run: func(context.Context, string) (models.Analysis, error) {
			close(entered)
			<-release
			return models.Analysis{Verdict: enums.VerdictNoViolation}, nil
		}}
		worker := jobs.NewWorker(repo, storage, vision, jobs.WorkerOptions{}, log)
		startWorker(t, worker)
		service := services.NewAnalysis(repo, storage, vision, worker, services.AnalysisOptions{}, log)
		if _, err := service.Submit(context.Background(), "clip.mov", bytes.NewReader(movie())); err != nil {
			close(release)
			t.Fatal(err)
		}
		select {
		case <-entered:
		case <-time.After(time.Second):
			close(release)
			t.Fatal("analyzer did not start")
		}
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
		defer cancel()
		err := worker.Stop(ctx)
		stopped := worker.Stopped()
		close(release)
		if !errors.Is(err, context.DeadlineExceeded) || stopped {
			t.Fatalf("live analyzer must retain resources: stopped=%v err=%v", stopped, err)
		}
		if err := worker.Stop(context.Background()); err != nil {
			t.Fatal(err)
		}
		if !worker.Stopped() {
			t.Fatal("completed analyzer still reported running")
		}
	})
}

func TestFailedPersistenceRetainsVideoUntilReconciliation(t *testing.T) {
	repo, storage, _, log := fixture(t)
	wrapped := &failingRepository{Jobs: repo, updateFailed: make(chan struct{}, 1)}
	wrapped.failUpdates.Store(true)
	vision := testVision{run: func(context.Context, string) (models.Analysis, error) {
		return models.Analysis{Verdict: enums.VerdictNoViolation}, nil
	}}
	worker := jobs.NewWorker(wrapped, storage, vision, jobs.WorkerOptions{CleanupInterval: 10 * time.Millisecond}, log)
	startWorker(t, worker)
	service := services.NewAnalysis(wrapped, storage, vision, worker, services.AnalysisOptions{Timeout: 300 * time.Millisecond}, log)
	job, err := service.Submit(context.Background(), "clip.mov", bytes.NewReader(movie()))
	if err != nil {
		t.Fatal(err)
	}
	select {
	case <-wrapped.updateFailed:
	case <-time.After(2 * time.Second):
		t.Fatal("result persistence never attempted")
	}
	got, err := repo.Get(context.Background(), job.ID)
	if err != nil || got.Status != enums.JobProcessing {
		t.Fatalf("unexpected persistent row: %+v %v", got, err)
	}
	if _, err := storage.Path(job.ID); err != nil {
		t.Fatalf("raw video deleted before durable status: %v", err)
	}
	wrapped.failUpdates.Store(false)
	got = waitJob(t, repo, job.ID, enums.JobFailed)
	if !strings.Contains(got.Failure, "time limit") {
		t.Fatalf("unexpected reconciliation failure: %q", got.Failure)
	}
	if err := worker.Stop(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := storage.Path(job.ID); !errors.Is(err, models.ErrNotFound) {
		t.Fatalf("reconciled video retained: %v", err)
	}
}
