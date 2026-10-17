package managers

import "nu-hackathon/backend/internal/interface/processors"

type ProcessorsManager struct {
	vision  processors.Vision
	storage processors.Storage
}

func (m *ProcessorsManager) Vision() processors.Vision   { return m.vision }
func (m *ProcessorsManager) Storage() processors.Storage { return m.storage }
