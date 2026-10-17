package managers

import "nu-hackathon/backend/internal/interface/processors"

type IProcessorsManager interface {
	Vision() processors.Vision
	Storage() processors.Storage
}
