package managers

import "nu-hackathon/backend/internal/interface/repositories"

type IRepositoriesManager interface{ Jobs() repositories.Jobs }
