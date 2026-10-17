package models

type Rule struct {
	Title   string `json:"title"`
	Section string `json:"section,omitempty"`
	URL     string `json:"url,omitempty"`
}
