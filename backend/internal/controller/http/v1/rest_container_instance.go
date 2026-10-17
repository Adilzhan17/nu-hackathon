package v1

import "sync"

// Instance keeps one controller set per application container, without process globals.
type Instance struct {
	once        sync.Once
	controllers *Container
}

func (i *Instance) Get(create func() *Container) *Container {
	i.once.Do(func() { i.controllers = create() })
	return i.controllers
}
