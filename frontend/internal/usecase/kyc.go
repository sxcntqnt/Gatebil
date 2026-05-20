// Package usecase contains the KYC business logic: input validation, duplicate
// detection, job creation, enqueuing, and status queries.
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
	UserID      string          `json:"user_id"`
	CountryCode string          `json:"country_code"`
	IDType      domain.IDType   `json:"id_type"`
	IDNumber    string          `json:"id_number"`
	FirstName   string          `json:"first_name"`
	LastName    string          `json:"last_name"`
	Tier        domain.KYCTier  `json:"tier"`
}

// SubmitResponse is returned immediately on a successful enqueue (HTTP 202).
type SubmitResponse struct {
	JobID       string          `json:"job_id"`
	Status      domain.KYCStatus `json:"status"`
	UserID      string          `json:"user_id"`
	SubmittedAt time.Time       `json:"submitted_at"`
	Message     string          `json:"message"`
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

// Submit validates the request, creates a KYC job record, and enqueues it for
// async processing. It returns immediately with the new job ID (202 pattern).
func (u *KYCUsecase) Submit(ctx context.Context, req SubmitRequest) (*SubmitResponse, error) {
	// 1. Domain validation
	if err := u.validate(req); err != nil {
		return nil, err
	}

	// 2. Duplicate detection — reject if there is already an active job.
	active, err := u.repo.GetActiveJobForUser(ctx, req.UserID)
	if err != nil {
		return nil, fmt.Errorf("duplicate check: %w", err)
	}
	if active != nil {
		return nil, domain.ErrDuplicateJob
	}

	// 3. Build domain job.
	now := time.Now().UTC()
	job := domain.KYCJob{
		ID:          uuid.New().String(),
		UserID:      req.UserID,
		CountryCode: req.CountryCode,
		IDType:      req.IDType,
		IDNumber:    req.IDNumber,
		FirstName:   req.FirstName,
		LastName:    req.LastName,
		Tier:        req.Tier,
		Attempt:     0,
		SubmittedAt: now,
	}

	// 4. Persist the pending job before enqueuing (audit trail first).
	if err := u.repo.CreateJob(ctx, &job); err != nil {
		return nil, fmt.Errorf("persist job: %w", err)
	}

	// 5. Enqueue into the worker pool channel (non-blocking).
	if err := u.pool.Submit(job); err != nil {
		// Queue full — mark job as failed and surface the error.
		_ = u.repo.UpdateJobStatus(ctx, job.ID, domain.StatusFailed)
		return nil, err
	}

	u.logger.Info("kyc job submitted",
		"job_id", job.ID,
		"user_id", job.UserID,
		"country", job.CountryCode,
		"id_type", job.IDType,
		"tier", job.Tier,
	)

	return &SubmitResponse{
		JobID:       job.ID,
		Status:      domain.StatusPending,
		UserID:      job.UserID,
		SubmittedAt: now,
		Message:     "KYC verification queued. Poll /status/{job_id} for updates.",
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

	// Attempt to enrich with result if terminal.
	result, err := u.repo.GetResult(ctx, jobID)
	if err == nil {
		resp.Status = result.Status
		resp.ResultText = result.ResultText
		resp.Confidence = result.Confidence
		resp.ProcessedAt = result.ProcessedAt
	} else {
		// Job exists but result not yet written — still pending/processing.
		resp.Status = domain.StatusPending
	}

	return resp, nil
}

// GetUserStatus returns the latest terminal KYC result for a given user.
func (u *KYCUsecase) GetUserStatus(ctx context.Context, userID string) (*domain.KYCStatusResponse, error) {
	result, err := u.repo.GetLatestResultForUser(ctx, userID)
	if err != nil {
		return nil, err
	}
	job, err := u.repo.GetJob(ctx, result.JobID)
	if err != nil {
		return nil, err
	}
	return &domain.KYCStatusResponse{
		JobID:       result.JobID,
		UserID:      result.UserID,
		Status:      result.Status,
		Tier:        job.Tier,
		ResultText:  result.ResultText,
		Confidence:  result.Confidence,
		ProcessedAt: result.ProcessedAt,
		SubmittedAt: job.SubmittedAt,
	}, nil
}

// ──────────────────────────────────────────────────
// Validation
// ──────────────────────────────────────────────────

func (u *KYCUsecase) validate(req SubmitRequest) error {
	if req.UserID == "" {
		return fmt.Errorf("user_id is required")
	}
	if req.CountryCode == "" {
		return fmt.Errorf("country_code is required")
	}
	if req.IDNumber == "" {
		return fmt.Errorf("id_number is required")
	}
	if req.FirstName == "" || req.LastName == "" {
		return fmt.Errorf("first_name and last_name are required")
	}
	if req.Tier == "" {
		req.Tier = domain.TierLight
	}

	// Country + ID type cross-check.
	allowed, ok := domain.AllowedIDTypes[req.CountryCode]
	if !ok {
		return fmt.Errorf("unsupported country: %s", req.CountryCode)
	}
	if !allowed[req.IDType] {
		return domain.ErrInvalidIDType
	}

	return nil
}
