package services

import (
	"context"
	"nu-hackathon/backend/internal/models"
)

type Jobs interface {
	Get(context.Context, string) (models.Job, error)
}
