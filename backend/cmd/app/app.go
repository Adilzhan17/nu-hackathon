package app

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"

	"nu-hackathon/backend/config"
	api "nu-hackathon/backend/internal/controller/http"
	v1 "nu-hackathon/backend/internal/controller/http/v1"
	"nu-hackathon/backend/internal/managers"
	"nu-hackathon/backend/pkg/httpserver"
)

func Run(ctx context.Context, cfg config.Config, log *slog.Logger) (runErr error) {
	main, err := managers.RegisterMainManager(cfg, log, nil)
	if err != nil {
		return fmt.Errorf("initialize backend: %w", err)
	}
	requestCtx, cancelRequests := context.WithCancel(ctx)
	var server *http.Server
	var requests *requestTracker
	defer func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
		defer cancel()
		cancelRequests()
		if server != nil {
			requests.Stop()
			if err := server.Shutdown(shutdownCtx); err != nil {
				runErr = errors.Join(runErr, fmt.Errorf("shutdown HTTP: %w", err), server.Close())
			}
			// Closing connections alone does not guarantee handlers have finished their
			// file/database rollback. Keep resources alive until they actually return.
			if err := requests.Wait(shutdownCtx); err != nil {
				runErr = errors.Join(runErr, fmt.Errorf("drain HTTP handlers: %w", err))
				go func() {
					_ = requests.Wait(context.Background())
					closeCtx, closeCancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
					defer closeCancel()
					if err := main.Close(closeCtx); err != nil {
						log.Error("deferred resource close", "error", err)
					}
				}()
				return
			}
		}
		runErr = errors.Join(runErr, main.Close(shutdownCtx))
	}()
	if err := main.Start(ctx); err != nil {
		return fmt.Errorf("start worker: %w", err)
	}
	service := main.GetServicesManager()
	var instance v1.Instance
	controllers := instance.Get(func() *v1.Container {
		return v1.New(service.Analysis(), service.Jobs(), v1.Options{
			MaxBytes: cfg.UploadMaxBytes, UploadTimeout: cfg.UploadTimeout, MaxUploads: cfg.MaxUploads,
		}, log)
	})
	handler := CORS(cfg.CORSOrigins, api.NewRouter(controllers, cfg.APIKey, main.Ready, log))
	requests = newRequestTracker(handler)
	server = httpserver.New(cfg.HTTPAddr, requests, cfg.UploadTimeout)
	server.BaseContext = func(net.Listener) context.Context { return requestCtx }
	listener, err := net.Listen("tcp", cfg.HTTPAddr)
	if err != nil {
		return fmt.Errorf("listen: %w", err)
	}
	done := make(chan error, 1)
	go func() { done <- server.Serve(listener) }()
	log.Info("backend listening", "address", listener.Addr().String(), "analysis_available", false)
	select {
	case err := <-done:
		if !errors.Is(err, http.ErrServerClosed) {
			return err
		}
		return nil
	case <-ctx.Done():
		return nil
	}
}
