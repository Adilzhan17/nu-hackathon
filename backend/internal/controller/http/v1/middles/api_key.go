package middles

import (
	"crypto/sha256"
	"crypto/subtle"
	"net/http"
	"nu-hackathon/backend/internal/controller/http/v1/response"
)

func APIKey(key string, next http.Handler) http.Handler {
	expected := sha256.Sum256([]byte(key))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		provided := sha256.Sum256([]byte(r.Header.Get("X-API-Key")))
		if key == "" || len(r.Header.Values("X-API-Key")) != 1 || subtle.ConstantTimeCompare(expected[:], provided[:]) != 1 {
			response.Error(w, 401, "unauthorized", "A valid X-API-Key is required.")
			return
		}
		next.ServeHTTP(w, r)
	})
}
