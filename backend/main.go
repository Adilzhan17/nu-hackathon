package main

import (
	"context"
	"flag"
	"fmt"
	"nu-hackathon/backend/cmd/app"
	"nu-hackathon/backend/config"
	"nu-hackathon/backend/pkg/logger"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	envFile := flag.String("env-file", ".env", "Optional dotenv file; environment variables take precedence")
	flag.Parse()
	cfg, err := config.Load(*envFile)
	if err != nil {
		fmt.Fprintln(os.Stderr, "configuration:", err)
		os.Exit(1)
	}
	log := logger.New(cfg.LogLevel, os.Stdout)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := app.Run(ctx, cfg, log); err != nil {
		log.Error("backend stopped", "error", err)
		os.Exit(1)
	}
}
