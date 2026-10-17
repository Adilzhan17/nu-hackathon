package managers

import "nu-hackathon/backend/internal/interface/services"

type IServicesManager interface {
	Analysis() services.Analysis
	Jobs() services.Jobs
}
