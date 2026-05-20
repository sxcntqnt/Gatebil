// cmd/server/main.go — KYC microservice entrypoint.
//
// Startup order:
//   1. Load & validate config
//   2. Structured logger
//   3. Prometheus metrics
//   4. PostgreSQL connection pool
//   5. Smile ID client
//   6. Worker pool (started but workers block on channel)
//   7. HTTP server (non-blocking; serves in background goroutine)
//   8. Block on OS signal (SIGINT / SIGTERM)
//   9. Graceful shutdown: HTTP drain → worker pool drain → DB close
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	smileid "github.com/nutcas3/smileid-go"
	"sxcntqunts/kyc-service/internal/config"
	"sxcntqunts/kyc-service/internal/handler"
	"sxcntqunts/kyc-service/internal/metrics"
	"sxcntqunts/kyc-service/internal/repository"
	"sxcntqunts/kyc-service/internal/usecase"
	"sxcntqunts/kyc-service/internal/worker"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	// ── 1. Config ──────────────────────────────────
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("config: %w", err)
	}

	// ── 2. Logger ──────────────────────────────────
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	logger.Info("kyc-service starting",
		"smile_env", cfg.SmileEnv,
		"workers", cfg.WorkerCount,
		"queue_depth", cfg.QueueDepth,
	)

	// ── 3. Metrics ─────────────────────────────────
	m := metrics.New()

	// ── 4. PostgreSQL ──────────────────────────────
	ctx := context.Background()
	repo, err := repository.NewPostgres(ctx, cfg.DBDSN)
	if err != nil {
		return fmt.Errorf("postgres: %w", err)
	}
	defer repo.Close()
	logger.Info("postgres connected")

	// ── 5. Smile ID client ─────────────────────────
	smileClient := smileid.NewClient(smileid.Config{
		APIKey:    cfg.SmileAPIKey,
		PartnerID: cfg.SmilePartnerID,
		Env:       cfg.SmileEnv,
		Timeout:   cfg.SmileTimeout,
	})
	logger.Info("smile id client initialised", "env", cfg.SmileEnv)

	// ── 6. Worker pool ─────────────────────────────
	// serviceCtx is cancelled when the OS sends SIGINT/SIGTERM.
	// Workers respect this context to stop retrying on shutdown.
	serviceCtx, cancelService := context.WithCancel(ctx)
	defer cancelService()

	pool := worker.New(
		worker.PoolConfig{
			WorkerCount:    cfg.WorkerCount,
			QueueDepth:     cfg.QueueDepth,
			MaxRetries:     cfg.MaxRetries,
			RetryBaseDelay: cfg.RetryBaseDelay,
			JobTimeout:     cfg.JobTimeout,
		},
		smileClient,
		repo,
		m,
		logger,
	)
	pool.Start(serviceCtx)

	// ── 7. HTTP server ─────────────────────────────
	uc := usecase.New(repo, pool, logger)
	srv := handler.New(uc, pool, m, logger)

	httpServer := &http.Server{
		Addr:         ":" + cfg.Port,
		Handler:      srv.Handler(),
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
		IdleTimeout:  120 * time.Second,
	}

	serverErr := make(chan error, 1)
	go func() {
		logger.Info("http server listening", "addr", httpServer.Addr)
		if err := httpServer.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
			serverErr <- err
		}
	}()

	// ── 8. Signal handling ─────────────────────────
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case sig := <-quit:
		logger.Info("shutdown signal received", "signal", sig.String())
	case err := <-serverErr:
		return fmt.Errorf("http server: %w", err)
	}

	// ── 9. Graceful shutdown ───────────────────────
	logger.Info("initiating graceful shutdown", "timeout", cfg.ShutdownTimeout)

	// 9a. Stop accepting new HTTP requests; drain in-flight ones.
	shutCtx, cancelShut := context.WithTimeout(ctx, cfg.ShutdownTimeout)
	defer cancelShut()

	if err := httpServer.Shutdown(shutCtx); err != nil {
		logger.Error("http shutdown error", "err", err)
	}
	logger.Info("http server stopped")

	// 9b. Cancel the service context so workers stop retrying.
	cancelService()

	// 9c. Close the job channel and wait for workers to finish in-flight jobs.
	// pool.Wait() closes the channel and blocks until all goroutines exit.
	pool.Wait()

	logger.Info("kyc-service shut down cleanly")
	return nil
}
