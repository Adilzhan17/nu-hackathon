package config

import (
	"bufio"
	"errors"
	"fmt"
	"net"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	HTTPAddr        string
	APIKey          string
	LogLevel        string
	StorageDir      string
	UploadMaxBytes  int64
	UploadTimeout   time.Duration
	AnalysisTimeout time.Duration
	ShutdownTimeout time.Duration
	Workers         int
	QueueSize       int
	MaxUploads      int
	JobRetention    time.Duration
	CleanupInterval time.Duration
	CORSOrigins     []string
}

// Load reads an optional .env without modifying the process environment.
// Existing environment variables take precedence, including empty values.
func Load(envFile string) (Config, error) {
	values := map[string]string{}
	if envFile != "" {
		f, err := os.Open(envFile)
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return Config{}, fmt.Errorf("read env file: %w", err)
		}
		if err == nil {
			defer f.Close()
			scanner := bufio.NewScanner(f)
			for line := 1; scanner.Scan(); line++ {
				s := strings.TrimSpace(strings.TrimPrefix(scanner.Text(), "\ufeff"))
				if s == "" || strings.HasPrefix(s, "#") {
					continue
				}
				key, value, ok := strings.Cut(s, "=")
				key, value = strings.TrimSpace(key), strings.TrimSpace(value)
				if !ok || key == "" || strings.ContainsAny(key, " \t") {
					return Config{}, fmt.Errorf("invalid env assignment at line %d", line)
				}
				if strings.HasPrefix(value, "\"") || strings.HasPrefix(value, "'") {
					if len(value) < 2 || value[len(value)-1] != value[0] {
						return Config{}, fmt.Errorf("unclosed env quote at line %d", line)
					}
					value = value[1 : len(value)-1]
				}
				values[key] = value
			}
			if err := scanner.Err(); err != nil {
				return Config{}, fmt.Errorf("read env file: %w", err)
			}
		}
	}
	get := func(key, fallback string) string {
		if value, ok := os.LookupEnv(key); ok {
			return value
		}
		if value, ok := values[key]; ok {
			return value
		}
		return fallback
	}
	cfg := Config{
		HTTPAddr:   get("HTTP_ADDR", "127.0.0.1:8080"),
		APIKey:     get("API_KEY", ""),
		LogLevel:   strings.ToLower(get("LOG_LEVEL", "info")),
		StorageDir: get("STORAGE_DIR", "./storage"),
	}
	if len(cfg.APIKey) < 16 || strings.TrimSpace(cfg.APIKey) != cfg.APIKey {
		return Config{}, errors.New("API_KEY must contain at least 16 bytes and no surrounding whitespace")
	}
	_, port, err := net.SplitHostPort(cfg.HTTPAddr)
	if err != nil {
		return Config{}, errors.New("HTTP_ADDR must be host:port")
	}
	p, err := strconv.Atoi(port)
	if err != nil || p < 1 || p > 65535 {
		return Config{}, errors.New("HTTP_ADDR requires a port between 1 and 65535")
	}
	if cfg.StorageDir == "" {
		return Config{}, errors.New("STORAGE_DIR must not be empty")
	}
	switch cfg.LogLevel {
	case "debug", "info", "warn", "error":
	default:
		return Config{}, errors.New("LOG_LEVEL must be debug, info, warn or error")
	}
	integer := func(key, fallback string, min, max int64) (int64, error) {
		n, err := strconv.ParseInt(get(key, fallback), 10, 64)
		if err != nil || n < min || n > max {
			return 0, fmt.Errorf("%s must be between %d and %d", key, min, max)
		}
		return n, nil
	}
	if cfg.UploadMaxBytes, err = integer("UPLOAD_MAX_BYTES", "524288000", 1, 1<<30); err != nil {
		return Config{}, err
	}
	for _, field := range []struct {
		key, fallback string
		target        *int
		max           int64
	}{
		{"WORKERS", "1", &cfg.Workers, 16}, {"QUEUE_SIZE", "8", &cfg.QueueSize, 256}, {"MAX_UPLOADS", "2", &cfg.MaxUploads, 16},
	} {
		n, err := integer(field.key, field.fallback, 1, field.max)
		if err != nil {
			return Config{}, err
		}
		*field.target = int(n)
	}
	for _, field := range []struct {
		key, fallback string
		target        *time.Duration
		min, max      time.Duration
	}{
		{"UPLOAD_TIMEOUT", "110s", &cfg.UploadTimeout, time.Second, 110 * time.Second},
		{"ANALYSIS_TIMEOUT", "270s", &cfg.AnalysisTimeout, time.Second, 270 * time.Second},
		{"SHUTDOWN_TIMEOUT", "15s", &cfg.ShutdownTimeout, time.Second, time.Minute},
		{"JOB_RETENTION", "24h", &cfg.JobRetention, time.Hour, 30 * 24 * time.Hour},
		{"CLEANUP_INTERVAL", "10m", &cfg.CleanupInterval, time.Minute, 24 * time.Hour},
	} {
		d, err := time.ParseDuration(get(field.key, field.fallback))
		if err != nil || d < field.min || d > field.max {
			return Config{}, fmt.Errorf("%s must be a duration between %s and %s", field.key, field.min, field.max)
		}
		*field.target = d
	}
	if cfg.UploadTimeout >= cfg.AnalysisTimeout {
		return Config{}, errors.New("UPLOAD_TIMEOUT must be shorter than ANALYSIS_TIMEOUT")
	}
	if cfg.CleanupInterval > cfg.JobRetention {
		return Config{}, errors.New("CLEANUP_INTERVAL must not exceed JOB_RETENTION")
	}
	for _, origin := range strings.Split(get("CORS_ORIGINS", "http://localhost:8081,http://127.0.0.1:8081"), ",") {
		origin = strings.TrimSpace(origin)
		if origin == "" {
			continue
		}
		u, err := url.Parse(origin)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" || u.Path != "" || u.RawQuery != "" || u.Fragment != "" || u.User != nil {
			return Config{}, errors.New("CORS_ORIGINS must contain comma-separated http(s) origins without paths")
		}
		cfg.CORSOrigins = append(cfg.CORSOrigins, origin)
	}
	return cfg, nil
}
