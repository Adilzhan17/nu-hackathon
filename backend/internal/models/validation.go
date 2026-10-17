package models

import (
	"bytes"
	"fmt"
	"image/jpeg"
	"math"
	"net/url"
	"strings"

	"nu-hackathon/backend/internal/enums"
)

// Validate bounds persisted results and rejects malformed evidence. Provenance and
// correctness of the supplied rule and frames remain the local analyzer's duty.
func (a Analysis) Validate() error {
	switch a.Verdict {
	case enums.VerdictNoViolation:
		if len(a.Violations) != 0 {
			return fmt.Errorf("no_violation must have an empty violations list")
		}
		return nil
	case enums.VerdictViolation:
		if len(a.Violations) < 1 || len(a.Violations) > 32 {
			return fmt.Errorf("violation requires 1 to 32 events")
		}
	default:
		return fmt.Errorf("unknown verdict")
	}
	totalBytes := 0
	for i, violation := range a.Violations {
		if !boundedText(violation.Type, 256) || !boundedText(violation.Description, 8192) {
			return fmt.Errorf("event %d has missing or oversized text", i)
		}
		if !validSeconds(violation.TimecodeSec) {
			return fmt.Errorf("event %d has invalid timestamp", i)
		}
		rule := violation.RegulatorySource
		if rule == nil || !boundedText(rule.Title, 1024) || len(rule.Section) > 1024 || len(rule.URL) > 2048 {
			return fmt.Errorf("event %d requires a bounded regulatory source", i)
		}
		if rule.URL != "" {
			u, err := url.ParseRequestURI(rule.URL)
			if err != nil || u.Scheme != "https" || u.Hostname() == "" || u.User != nil || u.Opaque != "" {
				return fmt.Errorf("event %d has invalid source URL", i)
			}
		}
		if len(violation.EvidenceFrames) < 1 || len(violation.EvidenceFrames) > 3 {
			return fmt.Errorf("event %d requires 1 to 3 evidence frames", i)
		}
		for j, frame := range violation.EvidenceFrames {
			if !validSeconds(frame.TimestampSec) {
				return fmt.Errorf("event %d frame %d has invalid timestamp", i, j)
			}
			if len(frame.JPEG) == 0 || len(frame.JPEG) > 1<<20 {
				return fmt.Errorf("event %d frame %d has invalid JPEG size", i, j)
			}
			totalBytes += len(frame.JPEG)
			if totalBytes > 16<<20 {
				return fmt.Errorf("evidence exceeds aggregate limit")
			}
			config, err := jpeg.DecodeConfig(bytes.NewReader(frame.JPEG))
			if err != nil || config.Width < 1 || config.Height < 1 || config.Width > 4096 || config.Height > 4096 {
				return fmt.Errorf("event %d frame %d has invalid JPEG dimensions", i, j)
			}
			if _, err := jpeg.Decode(bytes.NewReader(frame.JPEG)); err != nil {
				return fmt.Errorf("event %d frame %d is not a complete JPEG", i, j)
			}
		}
	}
	return nil
}

func boundedText(value string, max int) bool {
	return strings.TrimSpace(value) != "" && len(value) <= max
}
func validSeconds(seconds float64) bool {
	return !math.IsNaN(seconds) && !math.IsInf(seconds, 0) && seconds >= 0
}
