package middleware

import (
	"log/slog"
	"net/http"
	"time"
)

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }
func (w *statusWriter) WriteHeader(status int) {
	if w.status != 0 {
		return
	}
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}
func (w *statusWriter) Write(p []byte) (int, error) {
	if w.status == 0 {
		w.WriteHeader(200)
	}
	return w.ResponseWriter.Write(p)
}
func Logger(log *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		out := &statusWriter{ResponseWriter: w}
		next.ServeHTTP(out, r)
		status := out.status
		if status == 0 {
			status = 200
		}
		// Matched route only: never log raw URLs, query strings, headers, or payloads.
		log.Info("http request", "method", r.Method, "route", r.Pattern, "status", status, "duration_ms", time.Since(start).Milliseconds())
	})
}
