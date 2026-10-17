package services

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"io"
	"log/slog"
	"time"

	"nu-hackathon/backend/internal/enums"
	"nu-hackathon/backend/internal/interface/processors"
	"nu-hackathon/backend/internal/interface/repositories"
	ports "nu-hackathon/backend/internal/interface/services"
	"nu-hackathon/backend/internal/models"
)

type AnalysisOptions struct {
	MaxBytes   int64
	Timeout    time.Duration
	MaxUploads int
}

type AnalysisService struct {
	repo      repositories.Jobs
	storage   processors.Storage
	vision    processors.Vision
	scheduler ports.Scheduler
	opts      AnalysisOptions
	uploads   chan struct{}
	log       *slog.Logger
}

func NewAnalysis(repo repositories.Jobs, storage processors.Storage, vision processors.Vision, scheduler ports.Scheduler, opts AnalysisOptions, log *slog.Logger) *AnalysisService {
	if opts.MaxBytes <= 0 {
		opts.MaxBytes = 500 << 20
	}
	if opts.Timeout <= 0 {
		opts.Timeout = 270 * time.Second
	}
	if opts.MaxUploads <= 0 {
		opts.MaxUploads = 2
	}
	if log == nil {
		log = slog.Default()
	}
	return &AnalysisService{repo: repo, storage: storage, vision: vision, scheduler: scheduler, opts: opts, uploads: make(chan struct{}, opts.MaxUploads), log: log}
}

func (s *AnalysisService) Available() bool { return s.vision.Available() }

func (s *AnalysisService) Submit(ctx context.Context, filename string, src io.Reader) (models.Job, error) {
	if !s.Available() {
		return models.Job{}, models.ErrAnalysisUnavailable
	}
	if err := ctx.Err(); err != nil {
		return models.Job{}, err
	}
	select {
	case s.uploads <- struct{}{}:
		defer func() { <-s.uploads }()
	default:
		return models.Job{}, models.ErrQueueFull
	}
	created := ports.SubmissionStart(ctx)
	deadline := created.Add(s.opts.Timeout)
	ctx, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()
	var bytes [16]byte
	if _, err := rand.Read(bytes[:]); err != nil {
		return models.Job{}, fmt.Errorf("generate job ID: %w", err)
	}
	id := hex.EncodeToString(bytes[:])
	if err := s.storage.Save(ctx, id, filename, src, s.opts.MaxBytes); err != nil {
		return models.Job{}, err
	}
	rollback := func(removeRecord bool) {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cleanupCancel()
		if removeRecord {
			if err := s.repo.Delete(cleanupCtx, id); err != nil {
				s.log.Error("remove unqueued job", "job_id", id, "error", err)
			}
		}
		if err := s.storage.Remove(cleanupCtx, id); err != nil {
			s.log.Error("remove unqueued video", "job_id", id, "error", err)
		}
	}
	if err := ctx.Err(); err != nil {
		rollback(false)
		return models.Job{}, err
	}
	job := models.Job{ID: id, Status: enums.JobProcessing, CreatedAt: created, UpdatedAt: created, DeadlineAt: deadline}
	if err := s.repo.Create(ctx, job); err != nil {
		rollback(false)
		return models.Job{}, fmt.Errorf("create job: %w", err)
	}
	if err := ctx.Err(); err != nil {
		rollback(true)
		return models.Job{}, err
	}
	if err := s.scheduler.Enqueue(id); err != nil {
		rollback(true)
		return models.Job{}, err
	}
	return job, nil
}
