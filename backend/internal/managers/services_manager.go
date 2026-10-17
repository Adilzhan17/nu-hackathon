package managers

import (
	is "nu-hackathon/backend/internal/interface/services"
	"nu-hackathon/backend/internal/services"
	"sync"
)

type ServicesManager struct {
	main                   *MainManager
	analysisOnce, jobsOnce sync.Once
	analysis               is.Analysis
	jobs                   is.Jobs
}

func (m *ServicesManager) Analysis() is.Analysis {
	m.analysisOnce.Do(func() {
		cfg := m.main.cfg
		m.analysis = services.NewAnalysis(m.main.repository, m.main.storage, m.main.vision, m.main.Worker(),
			services.AnalysisOptions{MaxBytes: cfg.UploadMaxBytes, Timeout: cfg.AnalysisTimeout, MaxUploads: cfg.MaxUploads}, m.main.log)
	})
	return m.analysis
}
func (m *ServicesManager) Jobs() is.Jobs {
	m.jobsOnce.Do(func() { m.jobs = services.NewJobs(m.main.repository) })
	return m.jobs
}
