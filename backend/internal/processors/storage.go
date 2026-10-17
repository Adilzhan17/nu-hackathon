package processors

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	processor "nu-hackathon/backend/internal/interface/processors"
	"nu-hackathon/backend/internal/models"
)

// LocalStorage owns only <job-ID>.part and <job-ID>.mov inside its root.
// Client filenames are checked for an extension but never used as paths.
type LocalStorage struct {
	dir    string
	root   *os.Root
	mu     sync.Mutex
	active map[string]bool
}

var _ processor.Storage = (*LocalStorage)(nil)

func NewStorage(dir string) (*LocalStorage, error) {
	if strings.TrimSpace(dir) == "" {
		return nil, fmt.Errorf("storage directory is empty")
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("resolve storage directory: %w", err)
	}
	// Reject configured symlink directories, including existing parent links.
	for current := abs; ; current = filepath.Dir(current) {
		info, err := os.Lstat(current)
		if err == nil && (!info.IsDir() || info.Mode()&os.ModeSymlink != 0) {
			return nil, fmt.Errorf("storage path is not a regular directory: %s", current)
		}
		if err != nil && !errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("inspect storage directory: %w", err)
		}
		if filepath.Dir(current) == current {
			break
		}
	}
	if err := os.MkdirAll(abs, 0o700); err != nil {
		return nil, fmt.Errorf("create storage directory: %w", err)
	}
	root, err := os.OpenRoot(abs)
	if err != nil {
		return nil, fmt.Errorf("open storage directory: %w", err)
	}
	return &LocalStorage{dir: abs, root: root, active: make(map[string]bool)}, nil
}

// Close releases the directory handle after all uploads and workers stop.
func (s *LocalStorage) Close() error { return s.root.Close() }

func (s *LocalStorage) Save(ctx context.Context, id, filename string, src io.Reader, maxBytes int64) (err error) {
	if err := ctx.Err(); err != nil {
		return err
	}
	if !models.ValidJobID(id) || src == nil || maxBytes <= 0 || !strings.EqualFold(filepath.Ext(filename), ".mov") {
		return models.ErrInvalidUpload
	}
	s.mu.Lock()
	if s.active[id] {
		s.mu.Unlock()
		return models.ErrConflict
	}
	s.active[id] = true
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		delete(s.active, id)
		s.mu.Unlock()
	}()

	partName, finalName := id+".part", id+".mov"
	if _, err := s.root.Lstat(finalName); err == nil {
		return models.ErrConflict
	} else if !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("inspect existing upload: %w", err)
	}
	f, err := s.root.OpenFile(partName, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		if errors.Is(err, fs.ErrExist) {
			return models.ErrConflict
		}
		return fmt.Errorf("create upload: %w", err)
	}
	closed, published := false, false
	defer func() {
		if !closed {
			if closeErr := f.Close(); err == nil {
				err = closeErr
			}
		}
		if !published {
			if removeErr := s.root.Remove(partName); removeErr != nil && !errors.Is(removeErr, fs.ErrNotExist) {
				err = errors.Join(err, fmt.Errorf("remove incomplete upload: %w", removeErr))
			}
		}
	}()

	size, err := copyBounded(ctx, f, src, maxBytes)
	if err != nil {
		return err
	}
	if err := validateMOV(ctx, f, size); err != nil {
		return err
	}
	if err := f.Sync(); err != nil {
		return fmt.Errorf("sync upload: %w", err)
	}
	if err := f.Close(); err != nil {
		closed = true
		return fmt.Errorf("close upload: %w", err)
	}
	closed = true
	if err := ctx.Err(); err != nil {
		return err
	}
	if _, err := s.root.Lstat(finalName); err == nil {
		return models.ErrConflict
	} else if !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("inspect upload destination: %w", err)
	}
	if err := s.root.Rename(partName, finalName); err != nil {
		return fmt.Errorf("publish upload: %w", err)
	}
	published = true
	return nil
}

// copyBounded reads at most limit+1 bytes and never writes beyond limit.
// A reader blocked inside Read must be interrupted by its owner (HTTP timeout).
func copyBounded(ctx context.Context, dst io.Writer, src io.Reader, limit int64) (int64, error) {
	buffer := make([]byte, 32*1024)
	var written int64
	emptyReads := 0
	for {
		if err := ctx.Err(); err != nil {
			return written, err
		}
		readSize := len(buffer)
		if remaining := limit - written; remaining < int64(readSize) {
			readSize = int(remaining) + 1
		}
		n, readErr := src.Read(buffer[:readSize])
		if n < 0 || n > readSize {
			return written, fmt.Errorf("invalid upload reader result")
		}
		if err := ctx.Err(); err != nil {
			return written, err
		}
		if int64(n) > limit-written {
			return written, models.ErrTooLarge
		}
		if n > 0 {
			emptyReads = 0
			count, writeErr := dst.Write(buffer[:n])
			written += int64(count)
			if writeErr != nil {
				return written, fmt.Errorf("write upload: %w", writeErr)
			}
			if count != n {
				return written, io.ErrShortWrite
			}
		} else if readErr == nil {
			emptyReads++
			if emptyReads >= 100 {
				return written, io.ErrNoProgress
			}
		}
		if errors.Is(readErr, io.EOF) {
			return written, nil
		}
		if readErr != nil {
			return written, fmt.Errorf("read upload: %w", readErr)
		}
	}
}

// validateMOV checks top-level ISO BMFF/QuickTime box boundaries. It requires
// movie metadata and nonempty media data, but deliberately does not decode codecs.
func validateMOV(ctx context.Context, file io.ReaderAt, size int64) error {
	var header [16]byte
	var offset int64
	moov, mdat := false, false
	for offset < size {
		if err := ctx.Err(); err != nil {
			return err
		}
		if size-offset < 8 {
			return models.ErrInvalidVideo
		}
		if _, err := file.ReadAt(header[:8], offset); err != nil {
			return fmt.Errorf("%w: truncated box header", models.ErrInvalidVideo)
		}
		boxSize := uint64(binary.BigEndian.Uint32(header[:4]))
		headerSize := uint64(8)
		kind := string(header[4:8])
		switch boxSize {
		case 0:
			boxSize = uint64(size - offset)
		case 1:
			if size-offset < 16 {
				return models.ErrInvalidVideo
			}
			if _, err := file.ReadAt(header[8:16], offset+8); err != nil {
				return models.ErrInvalidVideo
			}
			boxSize = binary.BigEndian.Uint64(header[8:16])
			headerSize = 16
		}
		if boxSize < headerSize || boxSize > uint64(size-offset) {
			return models.ErrInvalidVideo
		}
		switch kind {
		case "moov":
			if moov || boxSize <= headerSize {
				return models.ErrInvalidVideo
			}
			moov = true
		case "mdat":
			mdat = mdat || boxSize > headerSize
		case "ftyp":
			if boxSize < headerSize+8 || (boxSize-headerSize)%4 != 0 {
				return models.ErrInvalidVideo
			}
		}
		offset += int64(boxSize)
	}
	if !moov || !mdat {
		return models.ErrInvalidVideo
	}
	return ctx.Err()
}

func (s *LocalStorage) Path(id string) (string, error) {
	if !models.ValidJobID(id) {
		return "", models.ErrNotFound
	}
	info, err := s.root.Lstat(id + ".mov")
	if errors.Is(err, fs.ErrNotExist) {
		return "", models.ErrNotFound
	}
	if err != nil {
		return "", fmt.Errorf("inspect upload: %w", err)
	}
	if !info.Mode().IsRegular() {
		return "", models.ErrInvalidUpload
	}
	return filepath.Join(s.dir, id+".mov"), nil
}

func (s *LocalStorage) Remove(ctx context.Context, id string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if !models.ValidJobID(id) {
		return models.ErrInvalidUpload
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.active[id] {
		return models.ErrConflict
	}
	var result error
	for _, suffix := range []string{".mov", ".part"} {
		if err := ctx.Err(); err != nil {
			return errors.Join(result, err)
		}
		name := id + suffix
		info, err := s.root.Lstat(name)
		if errors.Is(err, fs.ErrNotExist) {
			continue
		}
		if err != nil {
			result = errors.Join(result, err)
			continue
		}
		if !info.Mode().IsRegular() {
			result = errors.Join(result, fmt.Errorf("%w: upload is not a regular file", models.ErrInvalidUpload))
			continue
		}
		if err := s.root.Remove(name); err != nil && !errors.Is(err, fs.ErrNotExist) {
			result = errors.Join(result, err)
		}
	}
	return result
}

func (s *LocalStorage) Cleanup(ctx context.Context, cutoff time.Time, keep map[string]bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	dir, err := s.root.Open(".")
	if err != nil {
		return fmt.Errorf("open cleanup directory: %w", err)
	}
	defer dir.Close()
	for {
		entries, readErr := dir.ReadDir(128)
		for _, entry := range entries {
			if err := ctx.Err(); err != nil {
				return err
			}
			name := entry.Name()
			suffix := filepath.Ext(name)
			if suffix != ".mov" && suffix != ".part" {
				continue
			}
			id := strings.TrimSuffix(name, suffix)
			if !models.ValidJobID(id) || keep[id] || s.active[id] {
				continue
			}
			info, err := s.root.Lstat(name)
			if errors.Is(err, fs.ErrNotExist) {
				continue
			}
			if err != nil {
				return fmt.Errorf("inspect cleanup candidate: %w", err)
			}
			if !info.Mode().IsRegular() || !info.ModTime().Before(cutoff) {
				continue
			}
			if err := s.root.Remove(name); err != nil && !errors.Is(err, fs.ErrNotExist) {
				return fmt.Errorf("remove expired upload: %w", err)
			}
		}
		if errors.Is(readErr, io.EOF) {
			return nil
		}
		if readErr != nil {
			return fmt.Errorf("read cleanup directory: %w", readErr)
		}
	}
}
