package models

// JPEG holds actual JPEG bytes. JSON encodes them as raw base64, without a data URL.
type Evidence struct {
	TimestampSec float64 `json:"timestamp_sec"`
	JPEG         []byte  `json:"image_base64"`
}
