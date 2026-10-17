package repositories

import (
	"context"
	"errors"
	"path/filepath"
	"sync"
	"testing"
	"time"

	bolt "go.etcd.io/bbolt"
	boltErrors "go.etcd.io/bbolt/errors"

	"nu-hackathon/backend/internal/enums"
	"nu-hackathon/backend/internal/models"
)

const repositoryJobID = "0123456789abcdef0123456789abcdef"

func sampleJob() models.Job {
	now := time.Now().UTC().Truncate(time.Second)
	return models.Job{ID: repositoryJobID, Status: enums.JobProcessing, CreatedAt: now, UpdatedAt: now, DeadlineAt: now.Add(time.Minute)}
}

func TestJobsDurableLifecycle(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "nested", "jobs.db")
	repo, err := NewJobs(path)
	if err != nil {
		t.Fatal(err)
	}
	job := sampleJob()
	if err := repo.Create(ctx, job); err != nil {
		t.Fatal(err)
	}
	if err := repo.Create(ctx, job); !errors.Is(err, models.ErrConflict) {
		t.Fatalf("duplicate create: %v", err)
	}
	job.Status = enums.JobDone
	job.Result = &models.Analysis{Violations: []models.Violation{{Type: "sample", EvidenceFrames: []models.Evidence{{TimestampSec: 1.5, JPEG: []byte{0xff, 0xd8, 0xff, 0xd9}}}}}}
	if err := repo.Update(ctx, job); err != nil {
		t.Fatal(err)
	}
	if err := repo.Close(); err != nil {
		t.Fatal(err)
	}
	repo, err = NewJobs(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = repo.Close() })
	got, err := repo.Get(ctx, job.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got.Status != enums.JobDone || !got.CreatedAt.Equal(job.CreatedAt) || got.Result == nil || string(got.Result.Violations[0].EvidenceFrames[0].JPEG) != string(job.Result.Violations[0].EvidenceFrames[0].JPEG) {
		t.Fatalf("reopened job lost persisted fields: %#v", got)
	}
	metadata, err := repo.List(ctx)
	if err != nil || len(metadata) != 1 || metadata[0].Result != nil {
		t.Fatalf("List must return metadata only: %#v, %v", metadata, err)
	}
	got.Result.Violations[0].EvidenceFrames[0].JPEG[0] = 0
	again, err := repo.Get(ctx, job.ID)
	if err != nil || again.Result.Violations[0].EvidenceFrames[0].JPEG[0] != 0xff {
		t.Fatalf("caller mutated persisted result: %#v, %v", again, err)
	}
	job.Result = nil
	job.Status = enums.JobFailed
	job.Failure = "worker interrupted"
	if err := repo.Update(ctx, job); err != nil {
		t.Fatal(err)
	}
	got, err = repo.Get(ctx, job.ID)
	if err != nil || got.Result != nil || got.Failure != job.Failure {
		t.Fatalf("update did not remove previous result: %#v, %v", got, err)
	}
	if err := repo.Delete(ctx, job.ID); err != nil {
		t.Fatal(err)
	}
	if err := repo.Delete(ctx, job.ID); err != nil {
		t.Fatalf("delete must be retryable: %v", err)
	}
	if _, err := repo.Get(ctx, job.ID); !errors.Is(err, models.ErrNotFound) {
		t.Fatalf("deleted job: %v", err)
	}
	if err := repo.Update(ctx, job); !errors.Is(err, models.ErrNotFound) {
		t.Fatalf("update missing job: %v", err)
	}
}

func TestJobsListDoesNotReadResultPayload(t *testing.T) {
	repo, err := NewJobs(filepath.Join(t.TempDir(), "jobs.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = repo.Close() })
	ctx := context.Background()
	job := sampleJob()
	if err := repo.Create(ctx, job); err != nil {
		t.Fatal(err)
	}
	// A malformed result must not prevent the recovery worker listing metadata.
	if err := repo.db.Update(func(tx *bolt.Tx) error {
		return tx.Bucket(jobResultsBucket).Put([]byte(job.ID), []byte("broken result"))
	}); err != nil {
		t.Fatal(err)
	}
	if jobs, err := repo.List(ctx); err != nil || len(jobs) != 1 || jobs[0].Result != nil {
		t.Fatalf("metadata listing loaded the result: %#v, %v", jobs, err)
	}
	if _, err := repo.Get(ctx, job.ID); err == nil {
		t.Fatal("Get hid corrupt result data")
	}
}

func TestJobsContextAndExclusiveLock(t *testing.T) {
	path := filepath.Join(t.TempDir(), "jobs.db")
	repo, err := NewJobs(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = repo.Close() })
	second, err := NewJobs(path)
	if second != nil {
		_ = second.Close()
	}
	if !errors.Is(err, boltErrors.ErrTimeout) {
		t.Fatalf("second writer should time out on exclusive database lock: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := repo.Create(ctx, sampleJob()); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled create: %v", err)
	}
	if err := repo.Ping(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled ping: %v", err)
	}
	if jobs, err := repo.List(context.Background()); err != nil || len(jobs) != 0 {
		t.Fatalf("canceled write changed database: %#v, %v", jobs, err)
	}
	// Context cancellation also interrupts waiting for another operation.
	repo.gate <- struct{}{}
	waitCtx, waitCancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	err = repo.Create(waitCtx, sampleJob())
	waitCancel()
	<-repo.gate
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("waiting request ignored deadline: %v", err)
	}
	if err := repo.Close(); err != nil {
		t.Fatal(err)
	}
	if err := repo.Ping(context.Background()); !errors.Is(err, boltErrors.ErrDatabaseNotOpen) {
		t.Fatalf("closed repository still ready: %v", err)
	}
}

func TestJobsConcurrentCreateHasOneWinner(t *testing.T) {
	repo, err := NewJobs(filepath.Join(t.TempDir(), "jobs.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = repo.Close() })
	var group sync.WaitGroup
	results := make(chan error, 8)
	for range cap(results) {
		group.Add(1)
		go func() {
			defer group.Done()
			results <- repo.Create(context.Background(), sampleJob())
		}()
	}
	group.Wait()
	close(results)
	winners := 0
	for err := range results {
		if err == nil {
			winners++
		} else if !errors.Is(err, models.ErrConflict) {
			t.Fatalf("unexpected create failure: %v", err)
		}
	}
	if winners != 1 {
		t.Fatalf("got %d successful creates, expected 1", winners)
	}
}
