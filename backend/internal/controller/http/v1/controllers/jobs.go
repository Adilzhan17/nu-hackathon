package controllers

import (
	"net/http"
	"nu-hackathon/backend/internal/controller/http/v1/response"
	"nu-hackathon/backend/internal/interface/services"
	"nu-hackathon/backend/internal/models/dto"
)

type JobsController struct{ service services.Jobs }

func NewJobs(service services.Jobs) *JobsController { return &JobsController{service: service} }
func (c *JobsController) Get(w http.ResponseWriter, r *http.Request) {
	job, err := c.service.Get(r.Context(), r.PathValue("job_id"))
	if err != nil {
		response.FromError(w, err)
		return
	}
	response.JSON(w, http.StatusOK, dto.FromJob(job))
}
