package processors

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"nu-hackathon/backend/internal/models"
)

const storageJobID = "0123456789abcdef0123456789abcdef"

func movBox(kind string, data []byte) []byte {
	box := make([]byte, len(data)+8)
	binary.BigEndian.PutUint32(box, uint32(len(box)))
	copy(box[4:8], kind)
	copy(box[8:], data)
	return box
}

// This is a structural fixture, not a decodable video. Storage never claims to
// verify codecs; that belongs to the configured analyzer.
func structuralMOV() []byte {
	data := movBox("ftyp", []byte("qt  \x00\x00\x00\x00qt  "))
	data = append(data, movBox("moov", movBox("mvhd", []byte{0, 0, 0, 0}))...)
	return append(data, movBox("mdat", []byte{1, 2, 3, 4})...)
}

func testStorage(t *testing.T) (*LocalStorage, string) {
	t.Helper()
	dir := t.TempDir()
	storage, err := NewStorage(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = storage.Close() })
	return storage, dir
}

func TestStorageSavePublishAndRemove(t *testing.T) {
	storage, dir := testStorage(t)
	data := structuralMOV()
	// Both path separators are harmless because filename is never a path.
	if err := storage.Save(context.Background(), storageJobID, "../../outside\\video.MOV", bytes.NewReader(data), int64(len(data))); err != nil {
		t.Fatal(err)
	}
	path, err := storage.Path(storageJobID)
	if err != nil || path != filepath.Join(dir, storageJobID+".mov") {
		t.Fatalf("path: %q, %v", path, err)
	}
	got, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(got, data) {
		t.Fatalf("published file differs: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, storageJobID+".part")); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("partial file remains: %v", err)
	}
	if err := storage.Save(context.Background(), storageJobID, "other.mov", bytes.NewReader(data), int64(len(data))); !errors.Is(err, models.ErrConflict) {
		t.Fatalf("duplicate upload: %v", err)
	}
	if err := storage.Remove(context.Background(), storageJobID); err != nil {
		t.Fatal(err)
	}
	if err := storage.Remove(context.Background(), storageJobID); err != nil {
		t.Fatalf("remove must be retryable: %v", err)
	}
	if _, err := storage.Path(storageJobID); !errors.Is(err, models.ErrNotFound) {
		t.Fatalf("removed upload found: %v", err)
	}
}

func TestStorageRejectsInvalidUploadsAndRollsBack(t *testing.T) {
	valid := structuralMOV()
	badLength := append([]byte(nil), valid...)
	binary.BigEndian.PutUint32(badLength[:4], uint32(len(valid)+1))
	tests := []struct {
		name     string
		id       string
		filename string
		data     []byte
		limit    int64
		want     error
	}{
		{"text", storageJobID, "video.mov", []byte("plain text pretending to be a video"), 1024, models.ErrInvalidVideo},
		{"empty", storageJobID, "video.mov", nil, 1024, models.ErrInvalidVideo},
		{"missing_media", storageJobID, "video.mov", movBox("moov", []byte{1}), 1024, models.ErrInvalidVideo},
		{"oversized", storageJobID, "video.mov", valid, int64(len(valid) - 1), models.ErrTooLarge},
		{"truncated_box", storageJobID, "video.mov", badLength, 1024, models.ErrInvalidVideo},
		{"trailing_junk", storageJobID, "video.mov", append(append([]byte(nil), valid...), 0), 1024, models.ErrInvalidVideo},
		{"wrong_extension", storageJobID, "video.mp4", valid, 1024, models.ErrInvalidUpload},
		{"zero_limit", storageJobID, "video.mov", valid, 0, models.ErrInvalidUpload},
		{"traversal", "../" + storageJobID, "video.mov", valid, 1024, models.ErrInvalidUpload},
		{"uppercase_id", strings.ToUpper(storageJobID), "video.mov", valid, 1024, models.ErrInvalidUpload},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			storage, dir := testStorage(t)
			err := storage.Save(context.Background(), tc.id, tc.filename, bytes.NewReader(tc.data), tc.limit)
			if !errors.Is(err, tc.want) {
				t.Fatalf("got %v, want %v", err, tc.want)
			}
			entries, err := os.ReadDir(dir)
			if err != nil || len(entries) != 0 {
				t.Fatalf("failed upload left files: %v, %v", entries, err)
			}
		})
	}
}

type cancelingUploadReader struct {
	cancel context.CancelFunc
	data   []byte
}

func (r cancelingUploadReader) Read(p []byte) (int, error) {
	r.cancel()
	return copy(p, r.data), io.EOF
}

func TestStorageCanceledStreamLeavesNoFiles(t *testing.T) {
	storage, dir := testStorage(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	err := storage.Save(ctx, storageJobID, "video.mov", cancelingUploadReader{cancel: cancel, data: structuralMOV()}, 1024)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled stream: %v", err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 0 {
		t.Fatalf("canceled upload left files: %v, %v", entries, err)
	}
}

func TestStorageOwnsOnlyExpectedFiles(t *testing.T) {
	storage, dir := testStorage(t)
	now := time.Now()
	old := now.Add(-time.Hour)
	keepID := strings.Repeat("a", 32)
	freshID := strings.Repeat("b", 32)
	stalePartID := strings.Repeat("c", 32)
	names := []string{storageJobID + ".mov", keepID + ".mov", freshID + ".mov", stalePartID + ".part", "unrelated.mov", storageJobID + ".txt"}
	for _, name := range names {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte("fixture"), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(path, old, old); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Chtimes(filepath.Join(dir, freshID+".mov"), now, now); err != nil {
		t.Fatal(err)
	}
	nestedDir := filepath.Join(dir, strings.Repeat("d", 32)+".mov")
	if err := os.Mkdir(nestedDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := storage.Cleanup(context.Background(), now.Add(-time.Minute), map[string]bool{keepID: true}); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{storageJobID + ".mov", stalePartID + ".part"} {
		if _, err := os.Stat(filepath.Join(dir, name)); !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("expired owned file survived: %s, %v", name, err)
		}
	}
	for _, name := range []string{keepID + ".mov", freshID + ".mov", "unrelated.mov", storageJobID + ".txt", filepath.Base(nestedDir)} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Fatalf("cleanup removed protected/non-owned file: %s, %v", name, err)
		}
	}
	if err := storage.Remove(context.Background(), "../"+keepID); !errors.Is(err, models.ErrInvalidUpload) {
		t.Fatalf("Remove accepted traversal: %v", err)
	}
	if _, err := storage.Path("../" + keepID); !errors.Is(err, models.ErrNotFound) {
		t.Fatalf("Path accepted traversal: %v", err)
	}
}

func TestStorageDoesNotFollowSymlinks(t *testing.T) {
	storage, dir := testStorage(t)
	outside := filepath.Join(t.TempDir(), "preserve.mov")
	if err := os.WriteFile(outside, []byte("preserve"), 0o600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, storageJobID+".mov")
	if err := os.Symlink(outside, link); err != nil {
		t.Skipf("symbolic links unavailable: %v", err)
	}
	if _, err := storage.Path(storageJobID); !errors.Is(err, models.ErrInvalidUpload) {
		t.Fatalf("Path followed symlink: %v", err)
	}
	if err := storage.Remove(context.Background(), storageJobID); !errors.Is(err, models.ErrInvalidUpload) {
		t.Fatalf("Remove followed symlink: %v", err)
	}
	if err := storage.Cleanup(context.Background(), time.Now().Add(time.Hour), nil); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(outside)
	if err != nil || string(got) != "preserve" {
		t.Fatalf("outside file changed: %q, %v", got, err)
	}
	if _, err := os.Lstat(link); err != nil {
		t.Fatalf("cleanup removed symlink: %v", err)
	}
}

func TestStorageExtendedAndZeroSizedBoxes(t *testing.T) {
	storage, _ := testStorage(t)
	moov := make([]byte, 17)
	binary.BigEndian.PutUint32(moov[:4], 1)
	copy(moov[4:8], "moov")
	binary.BigEndian.PutUint64(moov[8:16], uint64(len(moov)))
	moov[16] = 1
	mdat := movBox("mdat", []byte{1, 2, 3})
	binary.BigEndian.PutUint32(mdat[:4], 0)
	data := append(moov, mdat...)
	if err := storage.Save(context.Background(), storageJobID, "legacy.mov", bytes.NewReader(data), int64(len(data))); err != nil {
		t.Fatalf("valid extended/EOF-sized boxes rejected: %v", err)
	}
}

type pausedUploadReader struct {
	started chan struct{}
	release chan struct{}
	data    []byte
	read    bool
}

func (r *pausedUploadReader) Read(p []byte) (int, error) {
	if r.read {
		return 0, io.EOF
	}
	r.read = true
	close(r.started)
	<-r.release
	return copy(p, r.data), io.EOF
}

func TestStorageCleanupPreservesActiveUpload(t *testing.T) {
	storage, dir := testStorage(t)
	reader := &pausedUploadReader{started: make(chan struct{}), release: make(chan struct{}), data: structuralMOV()}
	done := make(chan error, 1)
	go func() { done <- storage.Save(context.Background(), storageJobID, "video.mov", reader, 1024) }()
	<-reader.started
	err := storage.Cleanup(context.Background(), time.Now().Add(time.Hour), nil)
	_, partErr := os.Stat(filepath.Join(dir, storageJobID+".part"))
	removeErr := storage.Remove(context.Background(), storageJobID)
	close(reader.release)
	saveErr := <-done
	if err != nil || partErr != nil || !errors.Is(removeErr, models.ErrConflict) || saveErr != nil {
		t.Fatalf("cleanup interfered with active upload: cleanup=%v partial=%v remove=%v save=%v", err, partErr, removeErr, saveErr)
	}
}
