package controllers

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"mime/multipart"
	"net"
	"net/http"
	"nu-hackathon/backend/internal/controller/http/v1/response"
	"nu-hackathon/backend/internal/interface/services"
	"nu-hackathon/backend/internal/models"
	"nu-hackathon/backend/internal/models/dto"
	"time"
)

type AnalysisController struct {
	service       services.Analysis
	maxBytes      int64
	uploadTimeout time.Duration
	log           *slog.Logger
	uploads       chan struct{}
}

func NewAnalysis(service services.Analysis, maxBytes int64, uploadTimeout time.Duration, maxUploads int, log *slog.Logger) *AnalysisController {
	if maxUploads <= 0 {
		maxUploads = 2
	}
	return &AnalysisController{service: service, maxBytes: maxBytes, uploadTimeout: uploadTimeout, log: log, uploads: make(chan struct{}, maxUploads)}
}
func (c *AnalysisController) Submit(w http.ResponseWriter, r *http.Request) {
	r = r.WithContext(services.WithSubmissionStart(r.Context(), time.Now()))
	mediaType, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || mediaType != "multipart/form-data" {
		response.FromError(w, models.ErrInvalidUpload)
		return
	}
	if !c.service.Available() {
		response.FromError(w, models.ErrAnalysisUnavailable)
		return
	}
	select {
	case c.uploads <- struct{}{}:
		defer func() { <-c.uploads }()
	default:
		response.FromError(w, models.ErrQueueFull)
		return
	}
	if r.ContentLength > c.maxBytes+(1<<20) {
		response.FromError(w, models.ErrTooLarge)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), c.uploadTimeout)
	defer cancel()
	_ = http.NewResponseController(w).SetReadDeadline(time.Now().Add(c.uploadTimeout))
	r.Body = http.MaxBytesReader(w, r.Body, c.maxBytes+(1<<20))
	stop := context.AfterFunc(ctx, func() { _ = r.Body.Close() })
	defer stop()
	defer r.Body.Close()
	reader, err := r.MultipartReader()
	if err != nil {
		response.FromError(w, models.ErrInvalidUpload)
		return
	}
	part, err := reader.NextPart()
	if err != nil {
		response.FromError(w, uploadError(ctx, err))
		return
	}
	if part.FormName() != "file" || part.FileName() == "" {
		response.FromError(w, models.ErrInvalidUpload)
		return
	}
	// The reader checks the final multipart boundary before the service commits a job.
	body := &singleFileReader{part: part, multipart: reader}
	job, err := c.service.Submit(ctx, part.FileName(), body)
	if err != nil {
		if ctx.Err() != nil {
			err = ctx.Err()
		}
		c.log.Debug("upload rejected", "error_type", fmt.Sprintf("%T", err))
		response.FromError(w, err)
		return
	}
	response.JSON(w, http.StatusAccepted, dto.AnalysisAccepted{JobID: job.ID})
}
func uploadError(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	var tooLarge *http.MaxBytesError
	if errors.As(err, &tooLarge) {
		return models.ErrTooLarge
	}
	var network net.Error
	if errors.As(err, &network) && network.Timeout() {
		return context.DeadlineExceeded
	}
	if errors.Is(err, io.EOF) {
		return models.ErrInvalidUpload
	}
	return fmt.Errorf("%w: %w", models.ErrInvalidUpload, err)
}

type singleFileReader struct {
	part      *multipart.Part
	multipart *multipart.Reader
	ended     bool
}

func (r *singleFileReader) Read(p []byte) (int, error) {
	if r.ended {
		return 0, io.EOF
	}
	n, err := r.part.Read(p)
	if err == io.EOF {
		if n > 0 {
			return n, nil
		}
		_, nextErr := r.multipart.NextPart()
		if nextErr != io.EOF {
			if nextErr == nil {
				return 0, models.ErrInvalidUpload
			}
			return 0, uploadError(context.Background(), nextErr)
		}
		r.ended = true
		return 0, io.EOF
	}
	if err != nil {
		return n, uploadError(context.Background(), err)
	}
	return n, nil
}
