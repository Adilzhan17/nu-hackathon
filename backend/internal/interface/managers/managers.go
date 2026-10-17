package managers

type IMainManager interface {
	GetServicesManager() IServicesManager
	GetRepositoriesManager() IRepositoriesManager
	GetProcessorsManager() IProcessorsManager
}
