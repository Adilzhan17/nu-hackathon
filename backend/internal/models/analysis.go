package models

import "nu-hackathon/backend/internal/enums"

type Analysis struct {
	Verdict    enums.Verdict `json:"verdict"`
	Violations []Violation   `json:"violations"`
}
