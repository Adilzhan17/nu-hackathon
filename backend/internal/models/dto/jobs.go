package dto

import (
	"nu-hackathon/backend/internal/enums"
	"nu-hackathon/backend/internal/models"
)

type JobResponse struct {
	JobID      string              `json:"job_id"`
	Status     enums.JobStatus     `json:"status"`
	Verdict    enums.Verdict       `json:"verdict,omitempty"`
	Violations *[]models.Violation `json:"violations,omitempty"`
	Error      string              `json:"error,omitempty"`
}

func FromJob(job models.Job) JobResponse {
	response := JobResponse{JobID: job.ID, Status: job.Status}
	if job.Status == enums.JobDone && job.Result != nil {
		response.Verdict = job.Result.Verdict
		violations := job.Result.Violations
		if violations == nil {
			violations = []models.Violation{}
		}
		response.Violations = &violations
	}
	if job.Status == enums.JobFailed {
		response.Error = job.Failure
	}
	return response
}
