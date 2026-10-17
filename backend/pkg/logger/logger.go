package logger

import (
	"io"
	"log/slog"
)

func New(level string, output io.Writer) *slog.Logger {
	var l slog.Level
	_ = l.UnmarshalText([]byte(level))
	return slog.New(slog.NewJSONHandler(output, &slog.HandlerOptions{Level: l}))
}
