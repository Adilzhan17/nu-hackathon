package middleware

import (
	"log/slog"
	"net/http"
	"nu-hackathon/backend/internal/controller/http/v1/response"
)

func Recovery(log *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if recover() != nil {
				log.Error("http handler panic")
				response.Error(w, 500, "internal_error", "The request could not be completed.")
			}
		}()
		next.ServeHTTP(w, r)
	})
}
