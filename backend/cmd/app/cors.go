package app

import (
	"net/http"
	"nu-hackathon/backend/internal/controller/http/v1/response"
	"strings"
)

func CORS(origins []string, next http.Handler) http.Handler {
	allowed := make(map[string]bool, len(origins))
	for _, origin := range origins {
		allowed[origin] = true
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Add("Vary", "Origin")
		origin := r.Header.Get("Origin")
		if origin != "" {
			if !allowed[origin] {
				response.Error(w, 403, "origin_not_allowed", "Origin is not allowed.")
				return
			}
			w.Header().Set("Access-Control-Allow-Origin", origin)
		}
		if r.Method == http.MethodOptions && origin != "" {
			w.Header().Add("Vary", "Access-Control-Request-Method")
			w.Header().Add("Vary", "Access-Control-Request-Headers")
			method := r.Header.Get("Access-Control-Request-Method")
			if method != "POST" && method != "GET" {
				response.Error(w, 403, "method_not_allowed", "CORS method is not allowed.")
				return
			}
			for _, header := range strings.Split(r.Header.Get("Access-Control-Request-Headers"), ",") {
				h := strings.ToLower(strings.TrimSpace(header))
				if h != "" && h != "content-type" && h != "x-api-key" {
					response.Error(w, 403, "header_not_allowed", "CORS header is not allowed.")
					return
				}
			}
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
			w.Header().Set("Access-Control-Max-Age", "600")
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}
