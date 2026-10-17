package response

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"nu-hackathon/backend/internal/models"
)

type errorDetail struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}
type errorResponse struct {
	Error errorDetail `json:"error"`
}

func JSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
func Error(w http.ResponseWriter, status int, code, message string) {
	JSON(w, status, errorResponse{errorDetail{Code: code, Message: message}})
}
func FromError(w http.ResponseWriter, err error) {
	var bodyLimit *http.MaxBytesError
	var network net.Error
	switch {
	case errors.Is(err, models.ErrNotFound):
		Error(w, 404, "job_not_found", "Job not found.")
	case errors.Is(err, models.ErrTooLarge), errors.As(err, &bodyLimit):
		Error(w, 413, "upload_too_large", "Video exceeds the configured upload limit.")
	case errors.Is(err, models.ErrInvalidVideo):
		Error(w, 415, "unsupported_video", "A structurally valid MOV video is required.")
	case errors.Is(err, models.ErrInvalidUpload):
		Error(w, 400, "invalid_upload", "Send exactly one video in multipart field file.")
	case errors.Is(err, models.ErrAnalysisUnavailable):
		Error(w, 503, "analysis_unavailable", "The local vision analyzer has not been configured yet.")
	case errors.Is(err, models.ErrQueueFull), errors.Is(err, models.ErrShuttingDown):
		w.Header().Set("Retry-After", "5")
		Error(w, 503, "service_busy", "The service cannot accept another analysis right now.")
	case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled), errors.As(err, &network) && network.Timeout():
		Error(w, 408, "upload_timeout", "The upload did not finish in time.")
	default:
		Error(w, 500, "internal_error", "The request could not be completed.")
	}
}
