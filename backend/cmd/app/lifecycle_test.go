package app

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"nu-hackathon/backend/config"
	"nu-hackathon/backend/internal/enums"
	"nu-hackathon/backend/internal/models"
	"nu-hackathon/backend/internal/repositories"
)

type blockedBody struct {
	started, done        chan struct{}
	startOnce, closeOnce sync.Once
}

func (b *blockedBody) Read([]byte) (int, error) {
	b.startOnce.Do(func() { close(b.started) })
	<-b.done
	return 0, io.ErrClosedPipe
}
func (b *blockedBody) Close() error { b.closeOnce.Do(func() { close(b.done) }); return nil }
func TestUploadSlotsCoverMultipartHeadersAndCancelReader(t *testing.T) {
	f := newFixture(t, testVision{result: models.Analysis{Verdict: enums.VerdictNoViolation}})
	body := &blockedBody{started: make(chan struct{}), done: make(chan struct{})}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r := httptest.NewRequestWithContext(ctx, "POST", "/api/v1/analyze", body)
	r.Header.Set("Content-Type", "multipart/form-data; boundary=blocked")
	r.Header.Set("X-API-Key", testKey)
	response := httptest.NewRecorder()
	finished := make(chan struct{})
	go func() { f.handler.ServeHTTP(response, r); close(finished) }()
	select {
	case <-body.started:
	case <-time.After(time.Second):
		t.Fatal("upload did not start")
	}
	second := &unreadBody{}
	rejected := request(f.handler, "POST", "/api/v1/analyze", testKey, "multipart/form-data; boundary=second", second)
	if rejected.Code != 503 || second.read {
		t.Fatalf("header admission failed: %d read=%v", rejected.Code, second.read)
	}
	cancel()
	select {
	case <-finished:
	case <-time.After(2 * time.Second):
		t.Fatal("canceled request reader did not stop")
	}
	if response.Code != 408 {
		t.Fatalf("cancel status=%d %s", response.Code, response.Body.String())
	}
}
func TestRequestDrainWaitsForHandlerAndRejectsNewWork(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	tracker := newRequestTracker(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { close(started); <-release; w.WriteHeader(204) }))
	finished := make(chan struct{})
	go func() {
		tracker.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/", nil))
		close(finished)
	}()
	<-started
	tracker.Stop()
	rejected := request(tracker, "GET", "/", "", "", nil)
	if rejected.Code != 503 {
		t.Fatal("request accepted during shutdown")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	if tracker.Wait(ctx) == nil {
		t.Fatal("drain returned while handler still uses dependencies")
	}
	close(release)
	waitCtx, waitCancel := context.WithTimeout(context.Background(), time.Second)
	defer waitCancel()
	if err := tracker.Wait(waitCtx); err != nil {
		t.Fatal(err)
	}
	<-finished
}

type addressLogWriter struct{ address chan string }

func (w addressLogWriter) Write(p []byte) (int, error) {
	var entry struct {
		Message string `json:"msg"`
		Address string `json:"address"`
	}
	if json.Unmarshal(p, &entry) == nil && entry.Message == "backend listening" {
		w.address <- entry.Address
	}
	return len(p), nil
}
func TestLiveServerStartsRespondsAndShutsDown(t *testing.T) {
	dir := t.TempDir()
	cfg := config.Config{HTTPAddr: "127.0.0.1:0", APIKey: testKey, StorageDir: dir, UploadTimeout: time.Second, ShutdownTimeout: 2 * time.Second,
		UploadMaxBytes: 128, AnalysisTimeout: time.Second, Workers: 1, QueueSize: 2, MaxUploads: 1}
	address := make(chan string, 1)
	log := slog.New(slog.NewJSONHandler(addressLogWriter{address}, nil))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- Run(ctx, cfg, log) }()
	var addr string
	select {
	case addr = <-address:
	case err := <-done:
		t.Fatalf("startup failed: %v", err)
	case <-time.After(3 * time.Second):
		t.Fatal("startup timeout")
	}
	client := &http.Client{Timeout: time.Second}
	for _, tc := range []struct {
		path   string
		status int
	}{{"/healthz", 200}, {"/readyz", 503}, {"/openapi.yaml", 200}, {"/api/v1/jobs/missing", 401}} {
		resp, err := client.Get("http://" + addr + tc.path)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		if resp.StatusCode != tc.status {
			t.Fatalf("%s: %d", tc.path, resp.StatusCode)
		}
	}
	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("shutdown timeout")
	}
	repo, err := repositories.NewJobs(dir + "/jobs.db")
	if err != nil {
		t.Fatalf("database lock held after shutdown: %v", err)
	}
	_ = repo.Close()
}
