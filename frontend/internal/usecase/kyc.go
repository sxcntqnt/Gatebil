// Package usecase contains the KYC business logic: input validation, duplicate
// detection, idempotency enforcement, job creation, enqueuing, and status queries.
// It depends only on the domain, repository, and worker packages — never on HTTP.
package usecase

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/google/uuid"
	"sxcntqunts/kyc-service/internal/domain"
	"sxcntqunts/kyc-service/internal/repository"
	"sxcntqunts/kyc-service/internal/worker"
)

// ──────────────────────────────────────────────────
// Input / Output shapes
// ──────────────────────────────────────────────────

// SubmitRequest is the validated input from the HTTP handler.
type SubmitRequest struct {
	// IdempotencyKey is supplied by the caller to make retries safe.
	// If empty the usecase generates one; callers should supply a stable
	// key (e.g. hash of user_id + id_number + tier) for true idempotency.
	IdempotencyKey string          `json:"idempotency_key"`
	UserID         string          `json:"user_id"`
	CountryCode    string          `json:"country_code"`
	IDType         domain.IDType   `json:"id_type"`
	IDNumber       string          `json:"id_number"`
	FirstName      string          `json:"first_name"`
	LastName       string          `json:"last_name"`
	Tier           domain.KYCTier  `json:"tier"`
}

// SubmitResponse is returned immediately on a successful enqueue (HTTP 202).
type SubmitResponse struct {
	JobID          string           `json:"job_id"`
	IdempotencyKey string           `json:"idempotency_key"`
	Status         domain.KYCStatus `json:"status"`
	UserID         string           `json:"user_id"`
	SubmittedAt    time.Time        `json:"submitted_at"`
	Replayed       bool             `json:"replayed,omitempty"` // true when returning a cached result
	Message        string           `json:"message"`
}

// ──────────────────────────────────────────────────
// KYCUsecase
// ──────────────────────────────────────────────────

// KYCUsecase orchestrates KYC job submission and status retrieval.
type KYCUsecase struct {
	repo   repository.KYCRepository
	pool   *worker.Pool
	logger *slog.Logger
}

// New constructs the usecase with its required dependencies.
func New(repo repository.KYCRepository, pool *worker.Pool, logger *slog.Logger) *KYCUsecase {
	return &KYCUsecase{repo: repo, pool: pool, logger: logger}
}

// ──────────────────────────────────────────────────
// Submit
// ──────────────────────────────────────────────────

// Submit validates the request, enforces idempotency, creates a KYC job record,
// and enqueues it for async processing. Returns immediately with the job ID (202 pattern).
func (u *KYCUsecase) Submit(ctx context.Context, req SubmitRequest) (*SubmitResponse, error) {
	// 0. Ensure idempotency key exists — generate one if the caller omitted it.
	//    A generated key won't survive a retry, which is fine: callers who want
	//    true replay protection must supply their own stable key.
	if req.IdempotencyKey == "" {
		req.IdempotencyKey = uuid.New().String()
	}

	// 1. Idempotency check — must happen before domain validation and before
	//    any write, so a retried request always gets the same response.
	prior, err := u.repo.FindByIdempotencyKey(ctx, req.IdempotencyKey)
	if err == nil && prior != nil {
		u.logger.Info("idempotency replay",
			"idempotency_key", req.IdempotencyKey,
			"job_id", prior.ID,
		)
		return &SubmitResponse{
			JobID:          prior.ID,
			IdempotencyKey: req.IdempotencyKey,
			Status:         domain.StatusPending,
			UserID:         prior.UserID,
			SubmittedAt:    prior.SubmittedAt,
			Replayed:       true,
			Message:        "Idempotency replay — returning previously accepted job.",
		}, nil
	}

	// 2. Domain validation.
	if err := u.validate(req); err != nil {
		return nil, err
	}

	// 3. Duplicate detection — reject if there is already an active (non-terminal) job.
	active, err := u.repo.GetActiveJobForUser(ctx, req.UserID)
	if err != nil {
		return nil, fmt.Errorf("duplicate check: %w", err)
	}
	if active != nil {
		return nil, domain.ErrDuplicateJob
	}

	// 4. Build domain job.
	now := time.Now().UTC()
	job := domain.KYCJob{
		ID:             uuid.New().String(),
		IdempotencyKey: req.IdempotencyKey,
		UserID:         req.UserID,
		CountryCode:    req.CountryCode,
		IDType:         req.IDType,
		IDNumber:       req.IDNumber,
		FirstName:      req.FirstName,
		LastName:       req.LastName,
		Tier:           req.Tier,
		Attempt:        0,
		SubmittedAt:    now,
	}

	// 5. Persist the pending job before enqueuing (audit trail always written first).
	if err := u.repo.CreateJob(ctx, &job); err != nil {
		return nil, fmt.Errorf("persist job: %w", err)
	}

	// 6. Enqueue into the worker pool channel (non-blocking).
	if err := u.pool.Submit(job); err != nil {
		// Queue full — mark failed so callers know to retry later.
		_ = u.repo.UpdateJobStatus(ctx, job.ID, domain.StatusFailed)
		return nil, err
	}

	u.logger.Info("kyc job submitted",
		"job_id", job.ID,
		"idempotency_key", job.IdempotencyKey,
		"user_id", job.UserID,
		"country", job.CountryCode,
		"id_type", job.IDType,
		"tier", job.Tier,
	)

	return &SubmitResponse{
		JobID:          job.ID,
		IdempotencyKey: req.IdempotencyKey,
		Status:         domain.StatusPending,
		UserID:         job.UserID,
		SubmittedAt:    now,
		Message:        "KYC verification queued. Poll /status/{job_id} for updates.",
	}, nil
}

// ──────────────────────────────────────────────────
// Status queries
// ──────────────────────────────────────────────────

// GetStatus returns the current status of a specific KYC job.
func (u *KYCUsecase) GetStatus(ctx context.Context, jobID string) (*domain.KYCStatusResponse, error) {
	job, err := u.repo.GetJob(ctx, jobID)
	if err != nil {
		return nil, err
	}

	resp := &domain.KYCStatusResponse{
		JobID:       job.ID,
		UserID:      job.UserID,
		Tier:        job.Tier,
		SubmittedAt: job.SubmittedAt,
	}

	// Enrich with inference result if the job has reached a terminal state.
	result, err := u.repo.GetResult(ctx, jobID)
	if err == nil {
		resp.Status          = result.Status
		resp.ResultText      = result.ResultText
		resp.Confidence      = result.Confidence
		resp.ModelVersion    = result.ModelVersion
		resp.LivenessVersion = result.LivenessVersion
		resp.ProcessedAt     = result.ProcessedAt
	} else {
		resp.Status = domain.StatusPending
	}

	return resp, nil
}

// GetStatusByUser returns the most recent KYC status for a user across all jobs.
func (u *KYCUsecase) GetStatusByUser(ctx context.Context, userID string) (*domain.KYCStatusResponse, error) {
	job, err := u.repo.GetLatestJobForUser(ctx, userID)
	if err != nil {
		return nil, err
	}
	return u.GetStatus(ctx, job.ID)
}

// ──────────────────────────────────────────────────
// Validation (private)
// ──────────────────────────────────────────────────

func (u *KYCUsecase) validate(req SubmitRequest) error {
	if req.UserID == "" {
		return fmt.Errorf("user_id is required")
	}
	if req.IDNumber == "" {
		return fmt.Errorf("id_number is required")
	}
	allowed, ok := domain.AllowedIDTypes[req.CountryCode]
	if !ok {
		return fmt.Errorf("unsupported country: %s", req.CountryCode)
	}
	if !allowed[req.IDType] {
		return domain.ErrInvalidIDType
	}
	if req.Tier != domain.TierLight && req.Tier != domain.TierFull {
		return fmt.Errorf("tier must be kyc_light or kyc_full, got %q", req.Tier)
	}
	return nil
}
