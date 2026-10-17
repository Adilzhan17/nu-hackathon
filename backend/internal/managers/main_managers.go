package managers

import (
	"context"
	"errors"
	"log/slog"
	"nu-hackathon/backend/config"
	im "nu-hackathon/backend/internal/interface/managers"
	ip "nu-hackathon/backend/internal/interface/processors"
	ir "nu-hackathon/backend/internal/interface/repositories"
	"nu-hackathon/backend/internal/jobs"
	"nu-hackathon/backend/internal/processors"
	"nu-hackathon/backend/internal/repositories"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

type MainManager struct {
	cfg                                                        config.Config
	log                                                        *slog.Logger
	repository                                                 ir.Jobs
	storage                                                    ip.Storage
	storageClose                                               func() error
	vision                                                     ip.Vision
	servicesOnce, repositoriesOnce, processorsOnce, workerOnce sync.Once
	services                                                   *ServicesManager
	repositories                                               *RepositoriesManager
	processors                                                 *ProcessorsManager
	worker                                                     *jobs.Worker
	ready                                                      atomic.Bool
	closeOnce                                                  sync.Once
	closeErr                                                   error
}

// RegisterMainManager owns resources for one application. A nil vision installs
// the explicitly unavailable adapter; it never fabricates an analysis result.
func RegisterMainManager(cfg config.Config, log *slog.Logger, vision ip.Vision) (*MainManager, error) {
	repo, err := repositories.NewJobs(filepath.Join(cfg.StorageDir, "jobs.db"))
	if err != nil {
		return nil, err
	}
	storage, err := processors.NewStorage(filepath.Join(cfg.StorageDir, "videos"))
	if err != nil {
		_ = repo.Close()
		return nil, err
	}
	if vision == nil {
		vision = processors.NewVision()
	}
	return &MainManager{cfg: cfg, log: log, repository: repo, storage: storage, storageClose: storage.Close, vision: vision}, nil
}
func (m *MainManager) GetServicesManager() im.IServicesManager {
	m.servicesOnce.Do(func() { m.services = &ServicesManager{main: m} })
	return m.services
}
func (m *MainManager) GetRepositoriesManager() im.IRepositoriesManager {
	m.repositoriesOnce.Do(func() { m.repositories = &RepositoriesManager{jobs: m.repository} })
	return m.repositories
}
func (m *MainManager) GetProcessorsManager() im.IProcessorsManager {
	m.processorsOnce.Do(func() { m.processors = &ProcessorsManager{vision: m.vision, storage: m.storage} })
	return m.processors
}
func (m *MainManager) Worker() *jobs.Worker {
	m.workerOnce.Do(func() {
		m.worker = jobs.NewWorker(m.repository, m.storage, m.vision, jobs.WorkerOptions{
			Workers: m.cfg.Workers, QueueSize: m.cfg.QueueSize, Retention: m.cfg.JobRetention, CleanupInterval: m.cfg.CleanupInterval,
		}, m.log)
	})
	return m.worker
}
func (m *MainManager) Start(ctx context.Context) error {
	if err := m.Worker().Start(ctx); err != nil {
		return err
	}
	m.ready.Store(true)
	return nil
}
func (m *MainManager) Ready(ctx context.Context) bool {
	ctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	return m.ready.Load() && m.vision.Available() && m.repository.Ping(ctx) == nil
}
func (m *MainManager) Close(ctx context.Context) error {
	m.ready.Store(false)
	stopErr := m.Worker().Stop(ctx)
	if !m.Worker().Stopped() {
		return stopErr
	}
	m.closeOnce.Do(func() { m.closeErr = errors.Join(m.repository.Close(), m.storageClose()) })
	return errors.Join(stopErr, m.closeErr)
}

var _ im.IMainManager = (*MainManager)(nil)
