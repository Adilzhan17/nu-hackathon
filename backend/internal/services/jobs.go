package services

import (
	"context"
	"nu-hackathon/backend/internal/interface/repositories"
	"nu-hackathon/backend/internal/models"
)

type JobsService struct{ repo repositories.Jobs }

func NewJobs(repo repositories.Jobs) *JobsService { return &JobsService{repo: repo} }

func (s *JobsService) Get(ctx context.Context, id string) (models.Job, error) {
	if !models.ValidJobID(id) {
		return models.Job{}, models.ErrNotFound
	}
	return s.repo.Get(ctx, id)
}
