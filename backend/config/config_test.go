package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func isolateConfig(t *testing.T) {
	t.Helper()
	for _, key := range []string{"HTTP_ADDR", "API_KEY", "LOG_LEVEL", "STORAGE_DIR", "UPLOAD_MAX_BYTES", "UPLOAD_TIMEOUT", "ANALYSIS_TIMEOUT", "SHUTDOWN_TIMEOUT", "WORKERS", "QUEUE_SIZE", "MAX_UPLOADS", "JOB_RETENTION", "CLEANUP_INTERVAL", "CORS_ORIGINS"} {
		value, exists := os.LookupEnv(key)
		t.Cleanup(func() {
			if exists {
				_ = os.Setenv(key, value)
			} else {
				_ = os.Unsetenv(key)
			}
		})
		_ = os.Unsetenv(key)
	}
}
func TestLoadFileAndEnvironmentPrecedence(t *testing.T) {
	isolateConfig(t)
	path := filepath.Join(t.TempDir(), ".env")
	if err := os.WriteFile(path, []byte("API_KEY='test-key-from-file-123'\nHTTP_ADDR=127.0.0.1:9090\n"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HTTP_ADDR", "127.0.0.1:8088")
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.HTTPAddr != "127.0.0.1:8088" || cfg.APIKey != "test-key-from-file-123" || cfg.UploadMaxBytes != 500<<20 || cfg.AnalysisTimeout != 270*time.Second {
		t.Fatalf("unexpected config: address=%s limit=%d timeout=%s", cfg.HTTPAddr, cfg.UploadMaxBytes, cfg.AnalysisTimeout)
	}
	if _, exists := os.LookupEnv("API_KEY"); exists {
		t.Fatal("Load mutated the environment")
	}
	if len(cfg.CORSOrigins) != 0 {
		t.Fatal("browser origins must be explicitly configured")
	}
	t.Setenv("API_KEY", "")
	if _, err := Load(path); err == nil {
		t.Fatal("empty environment key must override dotenv")
	}
}
func TestRejectInvalidConfigWithoutLeakingValues(t *testing.T) {
	for _, tc := range []struct{ key, value string }{
		{"API_KEY", "private-short"}, {"UPLOAD_MAX_BYTES", "0"}, {"UPLOAD_TIMEOUT", "111s"}, {"ANALYSIS_TIMEOUT", "10s"},
		{"WORKERS", "0"}, {"QUEUE_SIZE", "-1"}, {"CORS_ORIGINS", "*"}, {"CORS_ORIGINS", "https://example.com/path"},
		{"JOB_RETENTION", "1m"}, {"CLEANUP_INTERVAL", "25h"}, {"LOG_LEVEL", "secret-unknown-level"}, {"HTTP_ADDR", "localhost:0"},
	} {
		t.Run(tc.key+"="+tc.value, func(t *testing.T) {
			isolateConfig(t)
			t.Setenv("API_KEY", "test-valid-secret-123456")
			t.Setenv(tc.key, tc.value)
			_, err := Load("")
			if err == nil {
				t.Fatal("invalid configuration accepted")
			}
			if tc.key == "API_KEY" && strings.Contains(err.Error(), tc.value) {
				t.Fatal("secret leaked")
			}
		})
	}
}
