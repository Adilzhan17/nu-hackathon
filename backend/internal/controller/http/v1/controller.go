package v1

import (
	"log/slog"
	"nu-hackathon/backend/internal/controller/http/v1/controllers"
	"nu-hackathon/backend/internal/interface/services"
	"time"
)

type Container struct {
	Analysis *controllers.AnalysisController
	Jobs     *controllers.JobsController
}
type Options struct {
	MaxBytes      int64
	UploadTimeout time.Duration
	MaxUploads    int
}

func New(analysis services.Analysis, jobs services.Jobs, opts Options, log *slog.Logger) *Container {
	return &Container{Analysis: controllers.NewAnalysis(analysis, opts.MaxBytes, opts.UploadTimeout, opts.MaxUploads, log), Jobs: controllers.NewJobs(jobs)}
}
