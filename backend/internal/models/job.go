package models

import (
	"errors"
	"nu-hackathon/backend/internal/enums"
	"regexp"
	"time"
)

var (
	ErrNotFound            = errors.New("job not found")
	ErrConflict            = errors.New("job already exists")
	ErrInvalidUpload       = errors.New("invalid upload")
	ErrInvalidVideo        = errors.New("unsupported or invalid MOV container")
	ErrTooLarge            = errors.New("upload exceeds limit")
	ErrQueueFull           = errors.New("analysis capacity exhausted")
	ErrAnalysisUnavailable = errors.New("vision analyzer is not configured")
	ErrShuttingDown        = errors.New("service is shutting down")
)
var jobIDPattern = regexp.MustCompile(`^[a-f0-9]{32}$`)

func ValidJobID(id string) bool { return jobIDPattern.MatchString(id) }

type Job struct {
	ID         string          `json:"id"`
	Status     enums.JobStatus `json:"status"`
	CreatedAt  time.Time       `json:"created_at"`
	UpdatedAt  time.Time       `json:"updated_at"`
	DeadlineAt time.Time       `json:"deadline_at"`
	Result     *Analysis       `json:"result,omitempty"`
	Failure    string          `json:"failure,omitempty"`
}
