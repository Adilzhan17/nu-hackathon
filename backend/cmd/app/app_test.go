package app

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"image"
	"image/jpeg"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"nu-hackathon/backend/config"
	api "nu-hackathon/backend/internal/controller/http"
	v1 "nu-hackathon/backend/internal/controller/http/v1"
	"nu-hackathon/backend/internal/enums"
	ip "nu-hackathon/backend/internal/interface/processors"
	"nu-hackathon/backend/internal/managers"
	"nu-hackathon/backend/internal/models"
	"nu-hackathon/backend/internal/models/dto"
)

const testKey = "local-test-key-not-a-real-secret"

type testVision struct {
	result models.Analysis
	err    error
}

func (testVision) Available() bool { return true }
func (v testVision) Analyze(ctx context.Context, path string) (models.Analysis, error) {
	if _, err := os.Stat(path); err != nil {
		return models.Analysis{}, err
	}
	return v.result, v.err
}

type fixture struct {
	main    *managers.MainManager
	handler http.Handler
	dir     string
}

func newFixture(t *testing.T, vision ip.Vision) fixture {
	t.Helper()
	dir := t.TempDir()
	cfg := config.Config{StorageDir: dir, Workers: 1, QueueSize: 2, MaxUploads: 1, UploadMaxBytes: 128, UploadTimeout: time.Second, AnalysisTimeout: time.Second, JobRetention: time.Hour, CleanupInterval: time.Hour}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	main, err := managers.RegisterMainManager(cfg, log, vision)
	if err != nil {
		t.Fatal(err)
	}
	if err := main.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		if err := main.Close(ctx); err != nil {
			t.Error(err)
		}
	})
	service := main.GetServicesManager()
	controller := v1.New(service.Analysis(), service.Jobs(), v1.Options{MaxBytes: cfg.UploadMaxBytes, UploadTimeout: cfg.UploadTimeout, MaxUploads: cfg.MaxUploads}, log)
	handler := CORS([]string{"http://localhost:3000"}, api.NewRouter(controller, testKey, main.Ready, log))
	return fixture{main, handler, dir}
}
func movData() []byte {
	var buf bytes.Buffer
	for _, box := range []struct {
		name string
		data []byte
	}{{"moov", []byte{0}}, {"mdat", []byte{1, 2, 3}}} {
		_ = binary.Write(&buf, binary.BigEndian, uint32(8+len(box.data)))
		buf.WriteString(box.name)
		buf.Write(box.data)
	}
	return buf.Bytes()
}
func multipartBody(t *testing.T, payload []byte, secondPart bool) ([]byte, string) {
	t.Helper()
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	part, err := writer.CreateFormFile("file", "cashier.mov")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = part.Write(payload)
	if secondPart {
		_ = writer.WriteField("unexpected", "reject")
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes(), writer.FormDataContentType()
}
func request(handler http.Handler, method, path, key, contentType string, body io.Reader) *httptest.ResponseRecorder {
	r := httptest.NewRequest(method, path, body)
	if key != "" {
		r.Header.Set("X-API-Key", key)
	}
	if contentType != "" {
		r.Header.Set("Content-Type", contentType)
	}
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, r)
	return w
}

type unreadBody struct{ read bool }

func (b *unreadBody) Read([]byte) (int, error) {
	b.read = true
	return 0, errors.New("body must not be read")
}
func (*unreadBody) Close() error { return nil }

func TestUnavailableAnalyzerAndAuthenticationDoNotReadUpload(t *testing.T) {
	f := newFixture(t, nil)
	for _, tc := range []struct {
		key    string
		status int
	}{{"", 401}, {"incorrect", 401}, {testKey, 503}} {
		body := &unreadBody{}
		w := request(f.handler, "POST", "/api/v1/analyze", tc.key, "multipart/form-data; boundary=test", body)
		if w.Code != tc.status || body.read {
			t.Fatalf("status=%d read=%v body=%s", w.Code, body.read, w.Body.String())
		}
	}
	r := httptest.NewRequest("GET", "/api/v1/jobs/not-an-id", nil)
	r.Header.Add("X-API-Key", testKey)
	r.Header.Add("X-API-Key", testKey)
	w := httptest.NewRecorder()
	f.handler.ServeHTTP(w, r)
	if w.Code != 401 {
		t.Fatalf("duplicate key accepted: %d", w.Code)
	}
	for _, tc := range []struct {
		path   string
		status int
	}{{"/healthz", 200}, {"/readyz", 503}, {"/openapi.yaml", 200}, {"/api/v1/jobs/not-an-id", 404}} {
		w := request(f.handler, "GET", tc.path, testKey, "", nil)
		if w.Code != tc.status {
			t.Fatalf("%s: %d", tc.path, w.Code)
		}
	}
	jobs, err := f.main.GetRepositoriesManager().Jobs().List(context.Background())
	if err != nil || len(jobs) != 0 {
		t.Fatalf("unexpected jobs: %v %v", jobs, err)
	}
}

func TestMalformedUploadsDoNotLeaveFilesOrJobs(t *testing.T) {
	f := newFixture(t, testVision{result: models.Analysis{Verdict: enums.VerdictNoViolation}})
	valid, contentType := multipartBody(t, movData(), false)
	// Each generated body has its own boundary.
	extra, extraType := multipartBody(t, movData(), true)
	large, largeType := multipartBody(t, bytes.Repeat([]byte("x"), 129), false)
	invalid, invalidType := multipartBody(t, []byte("not a video"), false)
	for _, tc := range []struct {
		name        string
		data        []byte
		contentType string
		status      int
	}{
		{"truncated", valid[:len(valid)-8], contentType, 400},
		{"extra part", extra, extraType, 400},
		{"oversize", large, largeType, 413},
		{"invalid MOV", invalid, invalidType, 415},
		{"not multipart", nil, "application/json", 400},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := request(f.handler, "POST", "/api/v1/analyze", testKey, tc.contentType, bytes.NewReader(tc.data))
			if w.Code != tc.status {
				t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
			}
			records, err := f.main.GetRepositoriesManager().Jobs().List(context.Background())
			if err != nil || len(records) != 0 {
				t.Fatalf("residual records: %v %v", records, err)
			}
			files, err := os.ReadDir(filepath.Join(f.dir, "videos"))
			if err != nil || len(files) != 0 {
				t.Fatalf("residual uploads: %v %v", files, err)
			}
		})
	}
}
func TestHTTPPipelineProducesFrontendCompatibleResults(t *testing.T) {
	var jpegData bytes.Buffer
	if err := jpeg.Encode(&jpegData, image.NewRGBA(image.Rect(0, 0, 2, 2)), nil); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name   string
		vision testVision
		status enums.JobStatus
	}{
		{"clear", testVision{result: models.Analysis{Verdict: enums.VerdictNoViolation}}, enums.JobDone},
		{"violation", testVision{result: models.Analysis{Verdict: enums.VerdictViolation, Violations: []models.Violation{{
			Type: "test_event", TimecodeSec: 1, Description: "Synthetic test only",
			RegulatorySource: &models.Rule{Title: "Test fixture; not a legal rule"},
			EvidenceFrames:   []models.Evidence{{TimestampSec: 1, JPEG: jpegData.Bytes()}},
		}}}}, enums.JobDone},
		{"failed", testVision{err: errors.New("private path must not appear in API")}, enums.JobFailed},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newFixture(t, tc.vision)
			if w := request(f.handler, "GET", "/readyz", "", "", nil); w.Code != 200 {
				t.Fatal(w.Body.String())
			}
			body, contentType := multipartBody(t, movData(), false)
			w := request(f.handler, "POST", "/api/v1/analyze", testKey, contentType, bytes.NewReader(body))
			if w.Code != 202 {
				t.Fatalf("submit: %d %s", w.Code, w.Body.String())
			}
			var accepted dto.AnalysisAccepted
			if err := json.Unmarshal(w.Body.Bytes(), &accepted); err != nil || !models.ValidJobID(accepted.JobID) {
				t.Fatalf("invalid acceptance: %s", w.Body.String())
			}
			var result map[string]json.RawMessage
			deadline := time.Now().Add(3 * time.Second)
			for {
				w = request(f.handler, "GET", "/api/v1/jobs/"+accepted.JobID, testKey, "", nil)
				if w.Code != 200 {
					t.Fatalf("get: %d %s", w.Code, w.Body.String())
				}
				if err := json.Unmarshal(w.Body.Bytes(), &result); err != nil {
					t.Fatal(err)
				}
				if string(result["status"]) != "\"processing\"" {
					break
				}
				if time.Now().After(deadline) {
					t.Fatal("job did not finish")
				}
				time.Sleep(5 * time.Millisecond)
			}
			if string(result["status"]) != "\""+string(tc.status)+"\"" {
				t.Fatal(w.Body.String())
			}
			for _, internal := range []string{"id", "created_at", "deadline_at", "result", "failure"} {
				if _, ok := result[internal]; ok {
					t.Fatalf("leaked internal field %s", internal)
				}
			}
			if tc.status == enums.JobFailed {
				var message string
				if err := json.Unmarshal(result["error"], &message); err != nil || message == "" {
					t.Fatalf("failed.error must be string: %s", w.Body.String())
				}
				if bytes.Contains(w.Body.Bytes(), []byte("private path")) || result["verdict"] != nil {
					t.Fatal(w.Body.String())
				}
			} else {
				if tc.name == "clear" && string(result["violations"]) != "[]" {
					t.Fatalf("empty array required: %s", w.Body.String())
				}
				if tc.name == "violation" {
					var events []models.Violation
					if err := json.Unmarshal(result["violations"], &events); err != nil || len(events) != 1 {
						t.Fatal(w.Body.String())
					}
					if !bytes.Equal(events[0].EvidenceFrames[0].JPEG, jpegData.Bytes()) {
						t.Fatal("JPEG/base64 roundtrip failed")
					}
					if events[0].RegulatorySource == nil {
						t.Fatal("source missing")
					}
				}
			}
		})
	}
}
func TestCORSAllowsOnlyConfiguredOrigin(t *testing.T) {
	f := newFixture(t, nil)
	for _, tc := range []struct {
		origin, method, headers string
		status                  int
	}{
		{"http://localhost:3000", "POST", "content-type,x-api-key", 204},
		{"https://untrusted.example", "POST", "x-api-key", 403},
		{"http://localhost:3000", "DELETE", "x-api-key", 403},
		{"http://localhost:3000", "POST", "authorization", 403},
	} {
		r := httptest.NewRequest("OPTIONS", "/api/v1/analyze", nil)
		r.Header.Set("Origin", tc.origin)
		r.Header.Set("Access-Control-Request-Method", tc.method)
		r.Header.Set("Access-Control-Request-Headers", tc.headers)
		w := httptest.NewRecorder()
		f.handler.ServeHTTP(w, r)
		if w.Code != tc.status {
			t.Fatalf("CORS: %+v status=%d", tc, w.Code)
		}
		if w.Code == 204 && w.Header().Get("Access-Control-Allow-Origin") != tc.origin {
			t.Fatal("missing allowed origin")
		}
	}
}
func TestManagersReuseServicesUnderConcurrentAccess(t *testing.T) {
	f := newFixture(t, nil)
	expected := f.main.GetServicesManager().Analysis()
	var wg sync.WaitGroup
	for range 32 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if got := f.main.GetServicesManager().Analysis(); got != expected {
				t.Error("different service instance")
			}
		}()
	}
	wg.Wait()
	second := newFixture(t, nil)
	if second.main.GetServicesManager().Analysis() == expected {
		t.Fatal("singleton leaked across containers")
	}
}
func TestRunStopsAndReleasesStorage(t *testing.T) {
	dir := t.TempDir()
	cfg := config.Config{HTTPAddr: "127.0.0.1:0", StorageDir: dir, APIKey: testKey, UploadTimeout: time.Second, ShutdownTimeout: time.Second}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := Run(ctx, cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected canceled startup, got %v", err)
	}
	// A reopened DB proves the failed startup released its file lock.
	m, err := managers.RegisterMainManager(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)), nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}
