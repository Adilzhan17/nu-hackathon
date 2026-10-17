package http

import (
	"context"
	"log/slog"
	"net/http"
	"nu-hackathon/backend/docs"
	"nu-hackathon/backend/internal/controller/http/middleware"
	v1 "nu-hackathon/backend/internal/controller/http/v1"
	"nu-hackathon/backend/internal/controller/http/v1/response"
)

func NewRouter(controllers *v1.Container, apiKey string, ready func(context.Context) bool, log *slog.Logger) http.Handler {
	mux := http.NewServeMux()
	controllers.Register(mux, apiKey)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) { response.JSON(w, 200, map[string]string{"status": "ok"}) })
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, r *http.Request) {
		if !ready(r.Context()) {
			response.JSON(w, 503, map[string]string{"status": "not_ready"})
			return
		}
		response.JSON(w, 200, map[string]string{"status": "ready"})
	})
	mux.HandleFunc("GET /openapi.yaml", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/yaml; charset=utf-8")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		_, _ = w.Write(docs.OpenAPI)
	})
	return middleware.Logger(log, middleware.Recovery(log, mux))
}
