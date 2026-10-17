package repositories

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	bolt "go.etcd.io/bbolt"

	repository "nu-hackathon/backend/internal/interface/repositories"
	"nu-hackathon/backend/internal/models"
)

var (
	jobMetadataBucket = []byte("jobs.v1")
	jobResultsBucket  = []byte("job-results.v1")
)

// JobsRepository persists each job and its result in one bbolt transaction.
// The separate result bucket keeps large evidence payloads out of List.
type JobsRepository struct {
	db   *bolt.DB
	gate chan struct{}
}

var _ repository.Jobs = (*JobsRepository)(nil)

func NewJobs(path string) (*JobsRepository, error) {
	if path == "" {
		return nil, fmt.Errorf("job database path is empty")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, fmt.Errorf("create database directory: %w", err)
	}
	db, err := bolt.Open(path, 0o600, &bolt.Options{Timeout: time.Second})
	if err != nil {
		return nil, fmt.Errorf("open job database: %w", err)
	}
	if err := db.Update(func(tx *bolt.Tx) error {
		for _, name := range [][]byte{jobMetadataBucket, jobResultsBucket} {
			if _, err := tx.CreateBucketIfNotExists(name); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("initialize job database: %w", err)
	}
	return &JobsRepository{db: db, gate: make(chan struct{}, 1)}, nil
}

// lock allows canceled requests to stop waiting for another database operation.
func (r *JobsRepository) lock(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case r.gate <- struct{}{}:
		if err := ctx.Err(); err != nil {
			<-r.gate
			return err
		}
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (r *JobsRepository) Create(ctx context.Context, job models.Job) error {
	return r.write(ctx, job, true)
}

func (r *JobsRepository) Update(ctx context.Context, job models.Job) error {
	return r.write(ctx, job, false)
}

func (r *JobsRepository) write(ctx context.Context, job models.Job, create bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if !models.ValidJobID(job.ID) {
		return fmt.Errorf("%w: invalid job ID", models.ErrInvalidUpload)
	}
	metadata := job
	metadata.Result = nil
	encoded, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("encode job metadata: %w", err)
	}
	var result []byte
	if job.Result != nil {
		result, err = json.Marshal(job.Result)
		if err != nil {
			return fmt.Errorf("encode job result: %w", err)
		}
	}
	if err := r.lock(ctx); err != nil {
		return err
	}
	defer func() { <-r.gate }()
	return r.db.Update(func(tx *bolt.Tx) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		metadataBucket, resultsBucket := tx.Bucket(jobMetadataBucket), tx.Bucket(jobResultsBucket)
		if metadataBucket == nil || resultsBucket == nil {
			return fmt.Errorf("job database schema is missing")
		}
		key := []byte(job.ID)
		exists := metadataBucket.Get(key) != nil
		if create && exists {
			return models.ErrConflict
		}
		if !create && !exists {
			return models.ErrNotFound
		}
		if err := metadataBucket.Put(key, encoded); err != nil {
			return err
		}
		if result == nil {
			if err := resultsBucket.Delete(key); err != nil {
				return err
			}
		} else if err := resultsBucket.Put(key, result); err != nil {
			return err
		}
		return ctx.Err()
	})
}

func (r *JobsRepository) Get(ctx context.Context, id string) (models.Job, error) {
	var job models.Job
	if err := ctx.Err(); err != nil {
		return job, err
	}
	if !models.ValidJobID(id) {
		return job, models.ErrNotFound
	}
	if err := r.lock(ctx); err != nil {
		return job, err
	}
	defer func() { <-r.gate }()
	err := r.db.View(func(tx *bolt.Tx) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		metadataBucket, resultsBucket := tx.Bucket(jobMetadataBucket), tx.Bucket(jobResultsBucket)
		if metadataBucket == nil || resultsBucket == nil {
			return fmt.Errorf("job database schema is missing")
		}
		key := []byte(id)
		data := metadataBucket.Get(key)
		if data == nil {
			return models.ErrNotFound
		}
		if err := json.Unmarshal(data, &job); err != nil {
			return fmt.Errorf("decode job metadata: %w", err)
		}
		if result := resultsBucket.Get(key); result != nil {
			job.Result = &models.Analysis{}
			if err := json.Unmarshal(result, job.Result); err != nil {
				return fmt.Errorf("decode job result: %w", err)
			}
		}
		return ctx.Err()
	})
	if err != nil {
		return models.Job{}, err
	}
	return job, nil
}

func (r *JobsRepository) List(ctx context.Context) ([]models.Job, error) {
	if err := r.lock(ctx); err != nil {
		return nil, err
	}
	defer func() { <-r.gate }()
	jobs := make([]models.Job, 0)
	err := r.db.View(func(tx *bolt.Tx) error {
		bucket := tx.Bucket(jobMetadataBucket)
		if bucket == nil {
			return fmt.Errorf("job database schema is missing")
		}
		return bucket.ForEach(func(_, data []byte) error {
			if err := ctx.Err(); err != nil {
				return err
			}
			var job models.Job
			if err := json.Unmarshal(data, &job); err != nil {
				return fmt.Errorf("decode job metadata: %w", err)
			}
			job.Result = nil
			jobs = append(jobs, job)
			return nil
		})
	})
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	sort.Slice(jobs, func(i, j int) bool {
		if jobs[i].CreatedAt.Equal(jobs[j].CreatedAt) {
			return jobs[i].ID < jobs[j].ID
		}
		return jobs[i].CreatedAt.After(jobs[j].CreatedAt)
	})
	return jobs, nil
}

// Delete is idempotent, so expired-job cleanup can safely be retried.
func (r *JobsRepository) Delete(ctx context.Context, id string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if !models.ValidJobID(id) {
		return models.ErrNotFound
	}
	if err := r.lock(ctx); err != nil {
		return err
	}
	defer func() { <-r.gate }()
	return r.db.Update(func(tx *bolt.Tx) error {
		for _, name := range [][]byte{jobMetadataBucket, jobResultsBucket} {
			bucket := tx.Bucket(name)
			if bucket == nil {
				return fmt.Errorf("job database schema is missing")
			}
			if err := bucket.Delete([]byte(id)); err != nil {
				return err
			}
		}
		return ctx.Err()
	})
}

func (r *JobsRepository) Ping(ctx context.Context) error {
	if err := r.lock(ctx); err != nil {
		return err
	}
	defer func() { <-r.gate }()
	return r.db.View(func(tx *bolt.Tx) error {
		if tx.Bucket(jobMetadataBucket) == nil || tx.Bucket(jobResultsBucket) == nil {
			return fmt.Errorf("job database schema is missing")
		}
		return ctx.Err()
	})
}

func (r *JobsRepository) Close() error {
	r.gate <- struct{}{}
	defer func() { <-r.gate }()
	return r.db.Close()
}
