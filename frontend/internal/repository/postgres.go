/// Package repository handles persistence of KYC jobs and results via PostgreSQL.
// All queries use pgx/v5 with context propagation; no raw DB connections leak.
package repository

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"sxcntqunts/kyc-service/internal/domain"
)

// KYCRepository defines the persistence contract used by the usecase layer.
// Using an interface makes the usecase layer testable without a live DB.
type KYCRepository interface {
	CreateJob(ctx context.Context, job *domain.KYCJob) error
	GetJob(ctx context.Context, jobID string) (*domain.KYCJob, error)
	GetActiveJobForUser(ctx context.Context, userID string) (*domain.KYCJob, error)
	UpdateJobStatus(ctx context.Context, jobID string, status domain.KYCStatus) error

	UpsertResult(ctx context.Context, result *domain.KYCResult) error
	GetResult(ctx context.Context, jobID string) (*domain.KYCResult, error)
	GetLatestResultForUser(ctx context.Context, userID string) (*domain.KYCResult, error)
}

// Postgres is the production implementation backed by a pgxpool.
type Postgres struct {
	pool *pgxpool.Pool
}

// NewPostgres connects to PostgreSQL and returns a ready repository.
// It also runs a lightweight schema bootstrap so the service is self-contained.
func NewPostgres(ctx context.Context, dsn string) (*Postgres, error) {
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("pgxpool.New: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		return nil, fmt.Errorf("postgres ping: %w", err)
	}
	r := &Postgres{pool: pool}
	if err := r.migrate(ctx); err != nil {
		return nil, fmt.Errorf("schema bootstrap: %w", err)
	}
	return r, nil
}

// Close releases pool connections gracefully.
func (r *Postgres) Close() { r.pool.Close() }

// ──────────────────────────────────────────────────
// Schema bootstrap (idempotent)
// ──────────────────────────────────────────────────

func (r *Postgres) migrate(ctx context.Context) error {
	ddl := `
	CREATE TABLE IF NOT EXISTS kyc_jobs (
		id            TEXT PRIMARY KEY,
		user_id       TEXT NOT NULL,
		country_code  TEXT NOT NULL,
		id_type       TEXT NOT NULL,
		id_number     TEXT NOT NULL,
		first_name    TEXT NOT NULL,
		last_name     TEXT NOT NULL,
		tier          TEXT NOT NULL DEFAULT 'kyc_light',
		status        TEXT NOT NULL DEFAULT 'pending',
		attempt       INT  NOT NULL DEFAULT 0,
		submitted_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
		updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
	);

	CREATE INDEX IF NOT EXISTS idx_kyc_jobs_user_id ON kyc_jobs(user_id);
	CREATE INDEX IF NOT EXISTS idx_kyc_jobs_status  ON kyc_jobs(status);

	CREATE TABLE IF NOT EXISTS kyc_results (
		job_id        TEXT PRIMARY KEY REFERENCES kyc_jobs(id),
		user_id       TEXT NOT NULL,
		status        TEXT NOT NULL,
		smile_job_id  TEXT,
		result_text   TEXT,
		result_code   TEXT,
		confidence    DOUBLE PRECISION DEFAULT 0,
		error_msg     TEXT,
		attempt       INT  NOT NULL DEFAULT 1,
		processed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
	);

	CREATE INDEX IF NOT EXISTS idx_kyc_results_user_id ON kyc_results(user_id);
	`
	_, err := r.pool.Exec(ctx, ddl)
	return err
}

// ──────────────────────────────────────────────────
// Job operations
// ──────────────────────────────────────────────────

func (r *Postgres) CreateJob(ctx context.Context, job *domain.KYCJob) error {
	_, err := r.pool.Exec(ctx, `
		INSERT INTO kyc_jobs
			(id, user_id, country_code, id_type, id_number, first_name, last_name, tier, status, attempt, submitted_at)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,'pending',0,$9)`,
		job.ID, job.UserID, job.CountryCode, string(job.IDType),
		job.IDNumber, job.FirstName, job.LastName, string(job.Tier), job.SubmittedAt,
	)
	return err
}

func (r *Postgres) GetJob(ctx context.Context, jobID string) (*domain.KYCJob, error) {
	row := r.pool.QueryRow(ctx, `
		SELECT id, user_id, country_code, id_type, id_number, first_name, last_name, tier, attempt, submitted_at
		FROM kyc_jobs WHERE id = $1`, jobID)

	j := &domain.KYCJob{}
	var idType, tier string
	err := row.Scan(&j.ID, &j.UserID, &j.CountryCode, &idType,
		&j.IDNumber, &j.FirstName, &j.LastName, &tier, &j.Attempt, &j.SubmittedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, domain.ErrJobNotFound
	}
	if err != nil {
		return nil, err
	}
	j.IDType = domain.IDType(idType)
	j.Tier = domain.KYCTier(tier)
	return j, nil
}

// GetActiveJobForUser returns a pending/processing job for a user, if any.
func (r *Postgres) GetActiveJobForUser(ctx context.Context, userID string) (*domain.KYCJob, error) {
	row := r.pool.QueryRow(ctx, `
		SELECT id, user_id, country_code, id_type, id_number, first_name, last_name, tier, attempt, submitted_at
		FROM kyc_jobs
		WHERE user_id = $1 AND status IN ('pending','processing')
		ORDER BY submitted_at DESC
		LIMIT 1`, userID)

	j := &domain.KYCJob{}
	var idType, tier string
	err := row.Scan(&j.ID, &j.UserID, &j.CountryCode, &idType,
		&j.IDNumber, &j.FirstName, &j.LastName, &tier, &j.Attempt, &j.SubmittedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil // no active job — caller checks nil
	}
	if err != nil {
		return nil, err
	}
	j.IDType = domain.IDType(idType)
	j.Tier = domain.KYCTier(tier)
	return j, nil
}

func (r *Postgres) UpdateJobStatus(ctx context.Context, jobID string, status domain.KYCStatus) error {
	_, err := r.pool.Exec(ctx,
		`UPDATE kyc_jobs SET status=$1, updated_at=$2 WHERE id=$3`,
		string(status), time.Now(), jobID,
	)
	return err
}

// ──────────────────────────────────────────────────
// Result operations
// ──────────────────────────────────────────────────

func (r *Postgres) UpsertResult(ctx context.Context, res *domain.KYCResult) error {
	_, err := r.pool.Exec(ctx, `
		INSERT INTO kyc_results
			(job_id, user_id, status, smile_job_id, result_text, result_code, confidence, error_msg, attempt, processed_at)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
		ON CONFLICT (job_id) DO UPDATE SET
			status       = EXCLUDED.status,
			smile_job_id = EXCLUDED.smile_job_id,
			result_text  = EXCLUDED.result_text,
			result_code  = EXCLUDED.result_code,
			confidence   = EXCLUDED.confidence,
			error_msg    = EXCLUDED.error_msg,
			attempt      = EXCLUDED.attempt,
			processed_at = EXCLUDED.processed_at`,
		res.JobID, res.UserID, string(res.Status), res.SmileJobID,
		res.ResultText, res.ResultCode, res.Confidence,
		res.ErrorMsg, res.Attempt, res.ProcessedAt,
	)
	return err
}

func (r *Postgres) GetResult(ctx context.Context, jobID string) (*domain.KYCResult, error) {
	row := r.pool.QueryRow(ctx, `
		SELECT job_id, user_id, status, smile_job_id, result_text, result_code,
		       confidence, error_msg, attempt, processed_at
		FROM kyc_results WHERE job_id = $1`, jobID)

	res := &domain.KYCResult{}
	var status string
	err := row.Scan(&res.JobID, &res.UserID, &status, &res.SmileJobID,
		&res.ResultText, &res.ResultCode, &res.Confidence,
		&res.ErrorMsg, &res.Attempt, &res.ProcessedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, domain.ErrJobNotFound
	}
	if err != nil {
		return nil, err
	}
	res.Status = domain.KYCStatus(status)
	return res, nil
}

func (r *Postgres) GetLatestResultForUser(ctx context.Context, userID string) (*domain.KYCResult, error) {
	row := r.pool.QueryRow(ctx, `
		SELECT r.job_id, r.user_id, r.status, r.smile_job_id, r.result_text,
		       r.result_code, r.confidence, r.error_msg, r.attempt, r.processed_at
		FROM kyc_results r
		JOIN kyc_jobs j ON j.id = r.job_id
		WHERE r.user_id = $1
		ORDER BY r.processed_at DESC
		LIMIT 1`, userID)

	res := &domain.KYCResult{}
	var status string
	err := row.Scan(&res.JobID, &res.UserID, &status, &res.SmileJobID,
		&res.ResultText, &res.ResultCode, &res.Confidence,
		&res.ErrorMsg, &res.Attempt, &res.ProcessedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, domain.ErrJobNotFound
	}
	if err != nil {
		return nil, err
	}
	res.Status = domain.KYCStatus(status)
	return res, nil
}
