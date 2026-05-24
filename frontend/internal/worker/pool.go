// Package worker implements the bounded goroutine pool that processes KYC jobs
// asynchronously. It consumes domain.KYCJob values from a buffered channel and
// dispatches them to the KYCClient inference layer.
package worker

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"time"
	"fmt"

	"sxcntqunts/kyc-service/internal/domain"
	"sxcntqunts/kyc-service/internal/kycclient"
	"sxcntqunts/kyc-service/internal/metrics"
	"sxcntqunts/kyc-service/internal/repository"
)

// ──────────────────────────────────────────────────
// Pool configuration
// ──────────────────────────────────────────────────

// PoolConfig holds all tunable knobs for the worker pool.
type PoolConfig struct {
	WorkerCount    int           // goroutines consuming the job channel
	QueueDepth     int           // buffered channel capacity
	MaxRetries     int           // max attempts before marking StatusFailed
	RetryBaseDelay time.Duration // exponential backoff seed (e.g. 2s → 4s → 8s)
	JobTimeout     time.Duration // context deadline applied to each job attempt
}

// ──────────────────────────────────────────────────
// Pool
// ──────────────────────────────────────────────────

// Pool is a bounded set of goroutines that drain the jobs channel.
// It is safe for concurrent use after Start has been called.
type Pool struct {
	cfg    PoolConfig
	jobs   chan domain.KYCJob

	// kycClient is the only dependency the pool holds on the inference layer.
	// Swap PythonClient for any other KYCClient implementation freely.
	kycClient kycclient.KYCClient

	repo   repository.KYCRepository
	m      *metrics.KYCMetrics
	logger *slog.Logger
	wg     sync.WaitGroup
}

// New constructs a Pool. Call Start to begin processing.
func New(
	cfg PoolConfig,
	kycClient kycclient.KYCClient,
	repo repository.KYCRepository,
	m *metrics.KYCMetrics,
	logger *slog.Logger,
) *Pool {
	return &Pool{
		cfg:       cfg,
		jobs:      make(chan domain.KYCJob, cfg.QueueDepth),
		kycClient: kycClient,
		repo:      repo,
		m:         m,
		logger:    logger,
	}
}

// Start launches WorkerCount goroutines. They block on the jobs channel
// until it is closed (via Wait) or ctx is cancelled.
func (p *Pool) Start(ctx context.Context) {
	for i := 0; i < p.cfg.WorkerCount; i++ {
		p.wg.Add(1)
		go p.worker(ctx)
	}
}

// Submit enqueues a job. Returns domain.ErrQueueFull immediately if the
// channel buffer is exhausted (non-blocking).
func (p *Pool) Submit(job domain.KYCJob) error {
	select {
	case p.jobs <- job:
		p.m.JobsSubmitted.Inc()
		p.m.WorkerQueueLen.Set(float64(len(p.jobs)))
		return nil
	default:
		return domain.ErrQueueFull
	}
}

// QueueLen returns the current number of jobs waiting in the channel.
func (p *Pool) QueueLen() int {
	return len(p.jobs)
}

// Wait closes the jobs channel and blocks until all goroutines have exited.
// Call once, after cancelling the service context.
func (p *Pool) Wait() {
	close(p.jobs)
	p.wg.Wait()
}

// ──────────────────────────────────────────────────
// Worker loop
// ──────────────────────────────────────────────────

func (p *Pool) worker(ctx context.Context) {
	defer p.wg.Done()
	for job := range p.jobs {
		p.m.WorkerQueueLen.Set(float64(len(p.jobs)))
		p.processWithRetry(ctx, job)
	}
}

// processWithRetry runs the job up to MaxRetries times with exponential backoff.
// Backpressure signals from the inference layer are treated as soft retries and
// do NOT count against the failure budget.
func (p *Pool) processWithRetry(ctx context.Context, job domain.KYCJob) {
	var (
		attempt      = 0
		softRetries  = 0 // backpressure-only retries; unbounded but rate-limited
		maxSoft      = 10
	)

	for {
		// Hard failure budget exhausted.
		if attempt >= p.cfg.MaxRetries {
			p.logger.Error("job exceeded retry limit",
				"job_id", job.ID,
				"attempts", attempt,
			)
			p.finalise(ctx, job, domain.StatusFailed, nil, "exceeded max retries")
			return
		}

		// Context cancelled (service shutting down).
		if ctx.Err() != nil {
			p.logger.Info("worker context cancelled, dropping job", "job_id", job.ID)
			return
		}

		err := p.processOnce(ctx, job, attempt)

		if err == nil {
			// Success — nothing more to do; processOnce already persisted the result.
			return
		}

		// ── Classify the error ────────────────────────────────────────────

		if errors.Is(err, domain.ErrBackpressure) {
			// Inference layer is saturated. Back off briefly, but don't burn
			// a retry from the hard budget — the job itself is fine.
			softRetries++
			if softRetries > maxSoft {
				p.logger.Warn("too many backpressure retries, marking failed",
					"job_id", job.ID, "soft_retries", softRetries)
				p.finalise(ctx, job, domain.StatusFailed, nil, "inference backpressure limit")
				return
			}
			delay := p.cfg.RetryBaseDelay * time.Duration(softRetries)
			p.logger.Warn("inference backpressure, backing off",
				"job_id", job.ID,
				"soft_retry", softRetries,
				"delay", delay,
			)
			p.sleep(ctx, delay)
			continue
		}

		if errors.Is(err, domain.ErrInferenceTimeout) {
			// Timeout is a hard failure — counts against the retry budget.
			attempt++
			p.m.RetriesTotal.Inc()
			delay := p.cfg.RetryBaseDelay * (1 << attempt)
			p.logger.Warn("inference timeout, retrying",
				"job_id", job.ID, "attempt", attempt, "delay", delay)
			p.sleep(ctx, delay)
			continue
		}

		// Generic transient error — exponential backoff, count against budget.
		attempt++
		job.Attempt = attempt
		p.m.RetriesTotal.Inc()
		delay := p.cfg.RetryBaseDelay * (1 << attempt)
		p.logger.Warn("job failed, retrying",
			"job_id", job.ID,
			"attempt", attempt,
			"delay", delay,
			"err", err,
		)
		p.sleep(ctx, delay)
	}
}

// processOnce executes a single inference attempt for the job.
func (p *Pool) processOnce(ctx context.Context, job domain.KYCJob, attempt int) error {
	jobCtx, cancel := context.WithTimeout(ctx, p.cfg.JobTimeout)
	defer cancel()

	p.m.ActiveWorkers.Inc()
	defer p.m.ActiveWorkers.Dec()

	// Mark in-progress.
	if err := p.repo.UpdateJobStatus(jobCtx, job.ID, domain.StatusProcessing); err != nil {
		p.logger.Warn("could not set processing status", "job_id", job.ID, "err", err)
	}

	t0 := time.Now()

	// ── Call the inference service ────────────────────────────────────────
	// The job currently carries metadata only (no image bytes at this layer).
	// Image bytes are provided upstream by the HTTP handler via a presigned URL
	// or passed directly for the initial multipart phase.
	//
	// For TierLight: verify only.
	// For TierFull:  smart-crop, then verify, then challenge.
	result, err := p.runInference(jobCtx, job)

	p.m.InferenceLatency.Observe(time.Since(t0).Seconds())

	if err != nil {
		return err
	}

	// ── Persist result ────────────────────────────────────────────────────
	result.Attempt = attempt
	result.ProcessedAt = time.Now().UTC()
	if err := p.repo.SaveResult(jobCtx, result); err != nil {
		return fmt.Errorf("save result: %w", err)
	}

	p.m.JobsProcessed.WithLabelValues(string(result.Status)).Inc()
	p.m.JobDuration.Observe(time.Since(t0).Seconds())

	p.logger.Info("job completed",
		"job_id", job.ID,
		"status", result.Status,
		"score", result.Confidence,
		"model", result.ModelVersion,
	)
	return nil
}

// runInference calls the appropriate KYCClient methods based on job tier.
func (p *Pool) runInference(ctx context.Context, job domain.KYCJob) (*domain.KYCResult, error) {
	// NOTE: at this layer the worker has metadata only.
	// Image bytes come from the submit path — stored in object storage (future)
	// or passed in-band (current multipart flow). For now we pass empty bytes
	// and rely on the Python service having the images from the prior /id-card call.
	// This will be replaced cleanly when the object-storage URL pattern lands.
	req := kycclient.VerifyRequest{
		// SelfieURL / IDCardURL will replace these once MinIO is wired.
		SelfieBytes: nil, // TODO: fetch from object store by job.ID
		IDCardBytes: nil,
	}

	verifyResult, err := p.kycClient.Verify(ctx, req)
	if err != nil {
		if isBackpressure(err) {
			return nil, domain.ErrBackpressure
		}
		if isTimeout(err) {
			return nil, domain.ErrInferenceTimeout
		}
		return nil, err
	}

	status := domain.StatusRejected
	if verifyResult.Verified {
		status = domain.StatusApproved
	}

	return &domain.KYCResult{
		JobID:           job.ID,
		UserID:          job.UserID,
		Status:          status,
		InternalJobID:   verifyResult.InternalJobID,
		Confidence:      verifyResult.Score,
		ModelVersion:    verifyResult.ModelVersion,
		LivenessVersion: verifyResult.LivenessVersion,
	}, nil
}

// ── Terminal state helpers ────────────────────────────────────────────────────

func (p *Pool) finalise(ctx context.Context, job domain.KYCJob, status domain.KYCStatus, verifyResult *kycclient.VerifyResult, errMsg string) {
	result := &domain.KYCResult{
		JobID:       job.ID,
		UserID:      job.UserID,
		Status:      status,
		ErrorMsg:    errMsg,
		ProcessedAt: time.Now().UTC(),
		Attempt:     job.Attempt,
	}
	if verifyResult != nil {
		result.ModelVersion    = verifyResult.ModelVersion
		result.LivenessVersion = verifyResult.LivenessVersion
		result.Confidence      = verifyResult.Score
	}
	if err := p.repo.SaveResult(ctx, result); err != nil {
		p.logger.Error("could not persist terminal result",
			"job_id", job.ID, "err", err)
	}
	p.m.JobsProcessed.WithLabelValues(string(status)).Inc()
}

// ── Utility ───────────────────────────────────────────────────────────────────

// sleep blocks for d or returns early when ctx is cancelled.
func (p *Pool) sleep(ctx context.Context, d time.Duration) {
	select {
	case <-time.After(d):
	case <-ctx.Done():
	}
}

func isBackpressure(err error) bool {
	return errors.Is(err, domain.ErrBackpressure) ||
		err.Error() == "kyc inference concurrency limit reached" ||
		err.Error() == "kyc inference circuit breaker is open"
}

func isTimeout(err error) bool {
	return errors.Is(err, context.DeadlineExceeded) ||
		errors.Is(err, domain.ErrInferenceTimeout)
}
