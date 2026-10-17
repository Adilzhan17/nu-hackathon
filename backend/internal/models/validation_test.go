package models

import (
	"bytes"
	"image"
	"image/jpeg"
	"math"
	"strings"
	"testing"

	"nu-hackathon/backend/internal/enums"
)

func validAnalysis(t *testing.T) Analysis {
	t.Helper()
	var imageBytes bytes.Buffer
	if err := jpeg.Encode(&imageBytes, image.NewRGBA(image.Rect(0, 0, 4, 4)), nil); err != nil {
		t.Fatal(err)
	}
	return Analysis{
		Verdict: enums.VerdictViolation,
		Violations: []Violation{{
			Type: "test_event", TimecodeSec: 1.25, Description: "Synthetic test description",
			RegulatorySource: &Rule{Title: "Test fixture, not an actual regulation", URL: "https://example.com/test-rule"},
			EvidenceFrames:   []Evidence{{TimestampSec: 1.25, JPEG: imageBytes.Bytes()}},
		}},
	}
}

func TestAnalysisValidationAcceptsBoundedJPEGAndEmptyVerdict(t *testing.T) {
	if err := validAnalysis(t).Validate(); err != nil {
		t.Fatal(err)
	}
	if err := (Analysis{Verdict: enums.VerdictNoViolation}).Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestAnalysisValidationRejectsInvalidResults(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Analysis)
	}{
		{"unknown verdict", func(a *Analysis) { a.Verdict = "uncertain" }},
		{"no violation with event", func(a *Analysis) { a.Verdict = enums.VerdictNoViolation }},
		{"empty violation", func(a *Analysis) { a.Violations = nil }},
		{"too many events", func(a *Analysis) { a.Violations = make([]Violation, 33) }},
		{"blank description", func(a *Analysis) { a.Violations[0].Description = "  " }},
		{"oversized text", func(a *Analysis) { a.Violations[0].Description = strings.Repeat("x", 8193) }},
		{"negative timestamp", func(a *Analysis) { a.Violations[0].TimecodeSec = -1 }},
		{"nan timestamp", func(a *Analysis) { a.Violations[0].TimecodeSec = math.NaN() }},
		{"infinite evidence time", func(a *Analysis) { a.Violations[0].EvidenceFrames[0].TimestampSec = math.Inf(1) }},
		{"missing source", func(a *Analysis) { a.Violations[0].RegulatorySource = nil }},
		{"empty source", func(a *Analysis) { a.Violations[0].RegulatorySource.Title = "" }},
		{"nonhttps source", func(a *Analysis) { a.Violations[0].RegulatorySource.URL = "http://example.com" }},
		{"relative source", func(a *Analysis) { a.Violations[0].RegulatorySource.URL = "/rule" }},
		{"source credentials", func(a *Analysis) { a.Violations[0].RegulatorySource.URL = "https://user:password@example.com" }},
		{"no evidence", func(a *Analysis) { a.Violations[0].EvidenceFrames = nil }},
		{"excess evidence", func(a *Analysis) { a.Violations[0].EvidenceFrames = make([]Evidence, 4) }},
		{"fake jpeg", func(a *Analysis) { a.Violations[0].EvidenceFrames[0].JPEG = []byte("not an image") }},
		{"truncated jpeg", func(a *Analysis) {
			frame := &a.Violations[0].EvidenceFrames[0]
			frame.JPEG = frame.JPEG[:len(frame.JPEG)-10]
		}},
		{"oversized jpeg", func(a *Analysis) { a.Violations[0].EvidenceFrames[0].JPEG = make([]byte, (1<<20)+1) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result := validAnalysis(t)
			tc.mutate(&result)
			if err := result.Validate(); err == nil {
				t.Fatal("invalid analysis accepted")
			}
		})
	}
}
