// cmd/server/main.go — KYC microservice entrypoint.
//
// Startup order:
//  1. Load & validate config
//  2. Structured logger
//  3. Prometheus metrics
//  4. PostgreSQL connection pool
//  5. KYC inference client (PythonClient → replaces Smile ID)
//  6. Inference readiness probe (fail fast if Python service is down)
//  7. Worker pool (started but workers block on channel)
//  8. HTTP server (non-blocking; serves in background goroutine)
//  9. Block on OS signal (SIGINT / SIGTERM)
// 10. Graceful shutdown: HTTP drain → worker pool drain → DB close
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

	"sxcntqunts/kyc-service/internal/config"
	"sxcntqunts/kyc-service/internal/handler"
	"sxcntqunts/kyc-service/internal/kycclient"
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
		"kyc_service_url", cfg.KYCServiceURL,
		"workers", cfg.WorkerCount,
		"queue_depth", cfg.QueueDepth,
		"concurrency", cfg.KYCConcurrency,
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

	// ── 5. KYC inference client ────────────────────
	// PythonClient is the sole implementation of kycclient.KYCClient.
	// To swap inference backends (ONNX, remote GPU cluster, new SaaS),
	// implement the interface and change this one line.
	kycClient := kycclient.New(kycclient.Config{
		BaseURL:     cfg.KYCServiceURL,
		Timeout:     cfg.KYCServiceTimeout,
		Concurrency: cfg.KYCConcurrency,
	}, logger)
	logger.Info("kyc inference client initialised", "url", cfg.KYCServiceURL)

	// ── 6. Inference readiness probe ───────────────
	// Block startup until the Python service is ready or we time out.
	// This prevents the worker pool from immediately draining jobs
	// into a missing inference service on fresh deploys.
	if err := waitForInference(ctx, kycClient, logger); err != nil {
		return fmt.Errorf("inference service not ready: %w", err)
	}

	// ── 7. Worker pool ─────────────────────────────
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
		kycClient,
		repo,
		m,
		logger,
	)
	pool.Start(serviceCtx)

	// ── 8. HTTP server ─────────────────────────────
	uc := usecase.New(repo, pool, logger)
	srv := handler.New(uc, pool, kycClient, m, logger)

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

	// ── 9. Signal handling ─────────────────────────
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case sig := <-quit:
		logger.Info("shutdown signal received", "signal", sig.String())
	case err := <-serverErr:
		return fmt.Errorf("http server: %w", err)
	}

	// ── 10. Graceful shutdown ──────────────────────
	logger.Info("initiating graceful shutdown", "timeout", cfg.ShutdownTimeout)

	// 10a. Stop accepting new HTTP requests; drain in-flight ones.
	shutCtx, cancelShut := context.WithTimeout(ctx, cfg.ShutdownTimeout)
	defer cancelShut()

	if err := httpServer.Shutdown(shutCtx); err != nil {
		logger.Error("http shutdown error", "err", err)
	}
	logger.Info("http server stopped")

	// 10b. Cancel the service context so workers stop retrying.
	cancelService()

	// 10c. Close the job channel and wait for all goroutines to drain.
	pool.Wait()

	logger.Info("kyc-service shut down cleanly")
	return nil
}

// waitForInference polls the Python inference service until it reports ready
// or the probe window expires. A 30s window with 3s intervals is generous
// enough for model loading on CPU; reduce on GPU deployments.
func waitForInference(ctx context.Context, client kycclient.KYCClient, logger *slog.Logger) error {
	const (
		probeInterval = 3 * time.Second
		probeWindow   = 30 * time.Second
	)

	deadline := time.Now().Add(probeWindow)
	attempt := 0

	for time.Now().Before(deadline) {
		attempt++
		probeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		err := client.Health(probeCtx)
		cancel()

		if err == nil {
			// Also log model state so the startup log shows exactly
			// what is loaded — invaluable when a model silently fails init.
			modCtx, modCancel := context.WithTimeout(ctx, 5*time.Second)
			if models, merr := client.Models(modCtx); merr == nil {
				logger.Info("inference service ready",
					"attempt", attempt,
					"mtcnn", models.MTCNN,
					"vggface2", models.VGGFace2,
					"dsnt", models.DSNT,
					"gpu", models.GPU,
					"cuda", models.CUDA,
					"model_version", models.ModelVersion,
					"liveness_version", models.LivenessVersion,
				)
			}
			modCancel()
			return nil
		}

		logger.Warn("inference service not ready, retrying",
			"attempt", attempt,
			"err", err,
			"retry_in", probeInterval,
		)
		select {
		case <-time.After(probeInterval):
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	return fmt.Errorf("inference service did not become ready within %s", probeWindow)
}
