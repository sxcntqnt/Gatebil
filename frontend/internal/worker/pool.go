// Package worker implements the async KYC processing engine.
//
// Architecture:
//   - A fixed-size pool of goroutines drains a single buffered job channel.
//   - Each worker calls the Smile ID SDK within a per-job context deadline.
//   - Transient failures are retried with exponential back-off + jitter.
//   - Terminal results are persisted via the repository and metrics updated.
//   - Graceful shutdown: when the service context is cancelled the pool drains
//     in-flight jobs then exits; the WaitGroup blocks until all workers are done.
package worker

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"math/rand"
	"sync"
	"time"

	smileid "github.com/nutcas3/smileid-go"
	"sxcntqunts/kyc-service/internal/domain"
	"sxcntqunts/kyc-service/internal/metrics"
	"sxcntqunts/kyc-service/internal/repository"
)

// ──────────────────────────────────────────────────
// Pool
// ──────────────────────────────────────────────────

// Pool manages a bounded set of goroutines that process KYC jobs from a channel.
type Pool struct {
	cfg     PoolConfig
	client  *smileid.Client
	repo    repository.KYCRepository
	metrics *metrics.KYCMetrics
	logger  *slog.Logger

	jobs chan domain.KYCJob // buffered work queue
	wg   sync.WaitGroup
}

// PoolConfig holds tuning knobs for the worker pool.
type PoolConfig struct {
	WorkerCount    int
	QueueDepth     int
	MaxRetries     int
	RetryBaseDelay time.Duration
	JobTimeout     time.Duration
}

// New allocates the pool and its internal job channel but does not start workers.
// Call Start(ctx) to begin processing.
func New(
	cfg PoolConfig,
	client *smileid.Client,
	repo repository.KYCRepository,
	m *metrics.KYCMetrics,
	logger *slog.Logger,
) *Pool {
	return &Pool{
		cfg:     cfg,
		client:  client,
		repo:    repo,
		metrics: m,
		logger:  logger,
		jobs:    make(chan domain.KYCJob, cfg.QueueDepth),
	}
}

// Start launches cfg.WorkerCount goroutines. It returns immediately; workers
// run until ctx is cancelled or the jobs channel is closed.
func (p *Pool) Start(ctx context.Context) {
	for i := range p.cfg.WorkerCount {
		p.wg.Add(1)
		go p.runWorker(ctx, i)
	}
	p.logger.Info("kyc worker pool started", "workers", p.cfg.WorkerCount, "queue_depth", p.cfg.QueueDepth)
}

// Submit enqueues a job for async processing. It returns ErrQueueFull if the
// channel buffer is exhausted so the HTTP layer can respond with 503.
func (p *Pool) Submit(job domain.KYCJob) error {
	select {
	case p.jobs <- job:
		p.metrics.JobsSubmitted.Inc()
		p.metrics.WorkerQueueLen.Set(float64(len(p.jobs)))
		return nil
	default:
		return domain.ErrQueueFull
	}
}

// Wait blocks until all workers have exited. Call after cancelling the context.
func (p *Pool) Wait() {
	close(p.jobs) // signal workers to drain and exit
	p.wg.Wait()
	p.logger.Info("kyc worker pool shut down cleanly")
}

// QueueLen returns the current number of pending jobs in the channel.
func (p *Pool) QueueLen() int { return len(p.jobs) }

// ──────────────────────────────────────────────────
// Worker goroutine
// ──────────────────────────────────────────────────

func (p *Pool) runWorker(ctx context.Context, id int) {
	defer p.wg.Done()
	log := p.logger.With("worker_id", id)
	log.Debug("worker started")

	for job := range p.jobs {
		p.metrics.WorkerQueueLen.Set(float64(len(p.jobs)))
		p.metrics.ActiveWorkers.Inc()

		p.processWithRetry(ctx, job, log)

		p.metrics.ActiveWorkers.Dec()
	}
	log.Debug("worker exiting")
}

// processWithRetry handles a single job including exponential-backoff retry.
func (p *Pool) processWithRetry(ctx context.Context, job domain.KYCJob, log *slog.Logger) {
	start := time.Now()
	log = log.With("job_id", job.ID, "user_id", job.UserID)

	// Mark as processing in the DB.
	if err := p.repo.UpdateJobStatus(ctx, job.ID, domain.StatusProcessing); err != nil {
		log.Error("failed to set job processing status", "err", err)
		// Continue anyway — the job is worth attempting.
	}

	var (
		result *domain.KYCResult
		err    error
	)

	for attempt := 1; attempt <= p.cfg.MaxRetries; attempt++ {
		if attempt > 1 {
			delay := backoff(p.cfg.RetryBaseDelay, attempt)
			log.Info("retrying kyc job", "attempt", attempt, "delay", delay)
			p.metrics.RetriesTotal.Inc()

			select {
			case <-time.After(delay):
			case <-ctx.Done():
				log.Warn("context cancelled during retry backoff")
				p.persist(ctx, job, failedResult(job, attempt, "service shutdown during retry"))
				return
			}
		}

		jobCtx, cancel := context.WithTimeout(ctx, p.cfg.JobTimeout)
		result, err = p.callSmileID(jobCtx, job, attempt)
		cancel()

		if err == nil {
			break // success
		}

		log.Warn("smile id call failed", "attempt", attempt, "err", err)

		// If service context was cancelled, stop retrying.
		if ctx.Err() != nil {
			result = failedResult(job, attempt, "service shutdown")
			break
		}
	}

	if err != nil && result == nil {
		result = failedResult(job, p.cfg.MaxRetries, err.Error())
	}

	p.persist(ctx, job, result)

	elapsed := time.Since(start).Seconds()
	p.metrics.JobDuration.Observe(elapsed)
	p.metrics.JobsProcessed.WithLabelValues(string(result.Status)).Inc()

	log.Info("kyc job finished", "status", result.Status, "elapsed_s", fmt.Sprintf("%.3f", elapsed))
}

// callSmileID dispatches the actual verification request to the Smile ID SDK.
func (p *Pool) callSmileID(ctx context.Context, job domain.KYCJob, attempt int) (*domain.KYCResult, error) {
	req := smileid.KYCRequest{
		CountryCode: job.CountryCode,
		IDType:      string(job.IDType),
		IDNumber:    job.IDNumber,
		FirstName:   job.FirstName,
		LastName:    job.LastName,
	}

	t0 := time.Now()
	resp, err := p.client.KYC.VerifyUser(ctx, req)
	p.metrics.SmileLatency.Observe(time.Since(t0).Seconds())

	if err != nil {
		return nil, fmt.Errorf("smileid.VerifyUser: %w", err)
	}

	// Map SmileID response → domain result.
	// The SDK's KYCResponse fields are read directly; adapt as the library evolves.
	status := domain.StatusRejected
	if resp.Verified {
		status = domain.StatusApproved
	}

	return &domain.KYCResult{
		JobID:       job.ID,
		UserID:      job.UserID,
		Status:      status,
		SmileJobID:  resp.SmileJobID,
		ResultText:  resp.ResultText,
		ResultCode:  resp.ResultCode,
		Confidence:  resp.Confidence,
		Attempt:     attempt,
		ProcessedAt: time.Now(),
	}, nil
}

// persist writes the final result to the database and syncs the job status.
func (p *Pool) persist(ctx context.Context, job domain.KYCJob, result *domain.KYCResult) {
	log := p.logger.With("job_id", job.ID)

	if err := p.repo.UpsertResult(ctx, result); err != nil {
		log.Error("failed to persist kyc result", "err", err)
	}
	if err := p.repo.UpdateJobStatus(ctx, job.ID, result.Status); err != nil {
		log.Error("failed to update job status after completion", "err", err)
	}
}

// ──────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────

// backoff returns exponential back-off with ±25% random jitter.
func backoff(base time.Duration, attempt int) time.Duration {
	exp := float64(base) * math.Pow(2, float64(attempt-1))
	jitter := (rand.Float64()*0.5 - 0.25) * exp // ±25%
	d := time.Duration(exp + jitter)
	// Cap at 60 seconds to avoid excessive waits.
	if d > 60*time.Second {
		d = 60 * time.Second
	}
	return d
}

func failedResult(job domain.KYCJob, attempt int, reason string) *domain.KYCResult {
	return &domain.KYCResult{
		JobID:       job.ID,
		UserID:      job.UserID,
		Status:      domain.StatusFailed,
		ErrorMsg:    reason,
		Attempt:     attempt,
		ProcessedAt: time.Now(),
	}
}
