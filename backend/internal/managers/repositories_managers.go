package managers

import "nu-hackathon/backend/internal/interface/repositories"

// The database is opened eagerly by RegisterMainManager so startup fails safely.
type RepositoriesManager struct{ jobs repositories.Jobs }

func (m *RepositoriesManager) Jobs() repositories.Jobs { return m.jobs }
