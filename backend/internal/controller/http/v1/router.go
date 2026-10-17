package v1

import (
	"net/http"
	"nu-hackathon/backend/internal/controller/http/v1/middles"
)

func (c *Container) Register(mux *http.ServeMux, apiKey string) {
	mux.Handle("POST /api/v1/analyze", middles.APIKey(apiKey, http.HandlerFunc(c.Analysis.Submit)))
	mux.Handle("GET /api/v1/jobs/{job_id}", middles.APIKey(apiKey, http.HandlerFunc(c.Jobs.Get)))
}
