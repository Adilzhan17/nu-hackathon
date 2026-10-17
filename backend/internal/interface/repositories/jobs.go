package repositories

import (
	"context"
	"nu-hackathon/backend/internal/models"
)

type Jobs interface {
	Create(context.Context, models.Job) error
	Get(context.Context, string) (models.Job, error)
	Update(context.Context, models.Job) error
	// List returns metadata only; evidence is loaded through Get.
	List(context.Context) ([]models.Job, error)
	Delete(context.Context, string) error
	Ping(context.Context) error
	Close() error
}
