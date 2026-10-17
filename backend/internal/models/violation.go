package models

type Violation struct {
	Type             string     `json:"type"`
	TimecodeSec      float64    `json:"timecode_sec"`
	Description      string     `json:"description"`
	EvidenceFrames   []Evidence `json:"evidence_frames"`
	RegulatorySource *Rule      `json:"regulatory_source,omitempty"`
}
