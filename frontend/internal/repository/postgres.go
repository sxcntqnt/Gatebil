// Package repository implements the KYCRepository interface against PostgreSQL
// using pgx/v5. Schema is embedded from schema.sql and applied automatically
// on startup if the tables don't yet exist.
package repository

import (
        _ "embed"
        "context"
        "errors"
        "fmt"
        "time"

        "github.com/jackc/pgx/v5"
        "github.com/jackc/pgx/v5/pgxpool"
        "sxcntqunts/kyc-service/internal/domain"
)

// schemaSQL is embedded verbatim from schema.sql, which lives alongside this
// file in internal/repository/. It's a self-contained BEGIN...COMMIT block,
// so it's executed via the simple query protocol (see ensureSchema) rather
// than pgx's normal extended-protocol Exec, which cannot Parse multiple
// statements in a single call.
//
//go:embed schema.sql
var schemaSQL string

// ──────────────────────────────────────────────────
// Interface — defined here so usecase/worker import
// only this package, never pgx directly.
// ──────────────────────────────────────────────────

// KYCRepository is the persistence contract the rest of the service depends on.
type KYCRepository interface {
        // Job writes
        CreateJob(ctx context.Context, job *domain.KYCJob) error
        UpdateJobStatus(ctx context.Context, jobID string, status domain.KYCStatus) error
        SaveResult(ctx context.Context, result *domain.KYCResult) error

        // Job reads
        GetJob(ctx context.Context, jobID string) (*domain.KYCJob, error)
        GetActiveJobForUser(ctx context.Context, userID string) (*domain.KYCJob, error)
        GetLatestJobForUser(ctx context.Context, userID string) (*domain.KYCJob, error)
        FindByIdempotencyKey(ctx context.Context, key string) (*domain.KYCJob, error)

        // Result reads
        GetResult(ctx context.Context, jobID string) (*domain.KYCResult, error)

        Close()
}

// ──────────────────────────────────────────────────
// PostgresRepo — implements KYCRepository
// ──────────────────────────────────────────────────

// PostgresRepo wraps a pgxpool connection pool.
type PostgresRepo struct {
        pool *pgxpool.Pool
}

// NewPostgres opens a pgxpool, pings the database, and ensures the schema
// exists — creating kyc_jobs / kyc_results / indexes on first run.
// DSN format: "postgres://user:pass@host:5432/dbname?sslmode=disable"
func NewPostgres(ctx context.Context, dsn string) (*PostgresRepo, error) {
        cfg, err := pgxpool.ParseConfig(dsn)
        if err != nil {
                return nil, fmt.Errorf("parse dsn: %w", err)
        }

        cfg.MaxConns = 20
        cfg.MinConns = 2
        cfg.MaxConnLifetime = 30 * time.Minute
        cfg.MaxConnIdleTime = 5 * time.Minute

        pool, err := pgxpool.NewWithConfig(ctx, cfg)
        if err != nil {
                return nil, fmt.Errorf("connect: %w", err)
        }

        if err := pool.Ping(ctx); err != nil {
                pool.Close()
                return nil, fmt.Errorf("ping: %w", err)
        }

        if err := ensureSchema(ctx, pool); err != nil {
                pool.Close()
                return nil, fmt.Errorf("ensure schema: %w", err)
        }

        return &PostgresRepo{pool: pool}, nil
}

// ensureSchema checks whether kyc_jobs exists and, if not, applies schema.sql
// in full. It intentionally re-checks under the connection it will run on
// rather than trusting a cached flag, since this only runs once at startup.
//
// schema.sql contains multiple statements (BEGIN ... CREATE TABLE ...
// CREATE INDEX ... COMMIT), so it can't go through pool.Exec: pgx's default
// extended-protocol Exec Parses one statement per call. Instead we acquire
// a raw connection and drop to pgconn's Exec, which uses the simple query
// protocol and runs the whole script — including its own transaction — in
// one round trip.
func ensureSchema(ctx context.Context, pool *pgxpool.Pool) error {
        var exists bool
        err := pool.QueryRow(ctx,
                `SELECT to_regclass('public.kyc_jobs') IS NOT NULL`,
        ).Scan(&exists)
        if err != nil {
                return fmt.Errorf("check existing schema: %w", err)
        }
        if exists {
                return nil
        }

        conn, err := pool.Acquire(ctx)
        if err != nil {
                return fmt.Errorf("acquire conn for migration: %w", err)
        }
        defer conn.Release()

        if _, err := conn.Conn().PgConn().Exec(ctx, schemaSQL).ReadAll(); err != nil {
                return fmt.Errorf("apply schema.sql: %w", err)
        }

        return nil
}

func (r *PostgresRepo) Close() {
        r.pool.Close()
}

// ──────────────────────────────────────────────────
// Schema reference — kept in sync with schema.sql
//
// kyc_jobs
//   id               TEXT PRIMARY KEY
//   idempotency_key  TEXT UNIQUE NOT NULL
//   user_id          TEXT NOT NULL
//   country_code     TEXT NOT NULL
//   id_type          TEXT NOT NULL
//   id_number        TEXT NOT NULL
//   first_name       TEXT NOT NULL
//   last_name        TEXT NOT NULL
//   tier             TEXT NOT NULL
//   status           TEXT NOT NULL DEFAULT 'pending'
//   attempt          INT  NOT NULL DEFAULT 0
//   submitted_at     TIMESTAMPTZ NOT NULL
//
// kyc_results
//   job_id           TEXT PRIMARY KEY REFERENCES kyc_jobs(id)
//   user_id          TEXT NOT NULL
//   status           TEXT NOT NULL
//   internal_job_id  TEXT
//   result_text      TEXT
//   result_code      TEXT
//   confidence       DOUBLE PRECISION
//   model_version    TEXT
//   liveness_version TEXT
//   error_msg        TEXT
//   processed_at     TIMESTAMPTZ NOT NULL
//   attempt          INT NOT NULL
// ──────────────────────────────────────────────────

// ── Job writes ───────────────────────────────────────────────────────────────

const createJobSQL = `
INSERT INTO kyc_jobs (
        id, idempotency_key, user_id, country_code, id_type, id_number,
        first_name, last_name, tier, status, attempt, submitted_at
) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)`

func (r *PostgresRepo) CreateJob(ctx context.Context, job *domain.KYCJob) error {
        _, err := r.pool.Exec(ctx, createJobSQL,
                job.ID,
                job.IdempotencyKey,
                job.UserID,
                job.CountryCode,
                string(job.IDType),
                job.IDNumber,
                job.FirstName,
                job.LastName,
                string(job.Tier),
                string(domain.StatusPending),
                job.Attempt,
                job.SubmittedAt,
        )
        if err != nil {
                return fmt.Errorf("create job: %w", err)
        }
        return nil
}

func (r *PostgresRepo) UpdateJobStatus(ctx context.Context, jobID string, status domain.KYCStatus) error {
        tag, err := r.pool.Exec(ctx,
                `UPDATE kyc_jobs SET status = $1 WHERE id = $2`,
                string(status), jobID,
        )
        if err != nil {
                return fmt.Errorf("update job status: %w", err)
        }
        if tag.RowsAffected() == 0 {
                return domain.ErrJobNotFound
        }
        return nil
}

const saveResultSQL = `
INSERT INTO kyc_results (
        job_id, user_id, status, internal_job_id, result_text, result_code,
        confidence, model_version, liveness_version, error_msg, processed_at, attempt
) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
ON CONFLICT (job_id) DO UPDATE SET
        status           = EXCLUDED.status,
        internal_job_id  = EXCLUDED.internal_job_id,
        result_text      = EXCLUDED.result_text,
        result_code      = EXCLUDED.result_code,
        confidence       = EXCLUDED.confidence,
        model_version    = EXCLUDED.model_version,
        liveness_version = EXCLUDED.liveness_version,
        error_msg        = EXCLUDED.error_msg,
        processed_at     = EXCLUDED.processed_at,
        attempt          = EXCLUDED.attempt`

func (r *PostgresRepo) SaveResult(ctx context.Context, result *domain.KYCResult) error {
        _, err := r.pool.Exec(ctx, saveResultSQL,
                result.JobID,
                result.UserID,
                string(result.Status),
                result.InternalJobID,
                result.ResultText,
                result.ResultCode,
                result.Confidence,
                result.ModelVersion,
                result.LivenessVersion,
                result.ErrorMsg,
                result.ProcessedAt,
                result.Attempt,
        )
        if err != nil {
                return fmt.Errorf("save result: %w", err)
        }
        // Mirror terminal status back onto the job row for simple queries.
        if result.Status.IsTerminal() {
                _ = r.UpdateJobStatus(ctx, result.JobID, result.Status)
        }
        return nil
}

// ── Job reads ────────────────────────────────────────────────────────────────

const getJobSQL = `
SELECT id, idempotency_key, user_id, country_code, id_type, id_number,
       first_name, last_name, tier, status, attempt, submitted_at
FROM kyc_jobs WHERE id = $1`

func (r *PostgresRepo) GetJob(ctx context.Context, jobID string) (*domain.KYCJob, error) {
        row := r.pool.QueryRow(ctx, getJobSQL, jobID)
        return scanJob(row)
}

const getActiveJobSQL = `
SELECT id, idempotency_key, user_id, country_code, id_type, id_number,
       first_name, last_name, tier, status, attempt, submitted_at
FROM kyc_jobs
WHERE user_id = $1
  AND status NOT IN ('approved','rejected','failed')
ORDER BY submitted_at DESC
LIMIT 1`

func (r *PostgresRepo) GetActiveJobForUser(ctx context.Context, userID string) (*domain.KYCJob, error) {
        row := r.pool.QueryRow(ctx, getActiveJobSQL, userID)
        job, err := scanJob(row)
        if errors.Is(err, domain.ErrJobNotFound) {
                return nil, nil // no active job is not an error
        }
        return job, err
}

const getLatestJobSQL = `
SELECT id, idempotency_key, user_id, country_code, id_type, id_number,
       first_name, last_name, tier, status, attempt, submitted_at
FROM kyc_jobs
WHERE user_id = $1
ORDER BY submitted_at DESC
LIMIT 1`

func (r *PostgresRepo) GetLatestJobForUser(ctx context.Context, userID string) (*domain.KYCJob, error) {
        row := r.pool.QueryRow(ctx, getLatestJobSQL, userID)
        return scanJob(row)
}

const findByIdempotencyKeySQL = `
SELECT id, idempotency_key, user_id, country_code, id_type, id_number,
       first_name, last_name, tier, status, attempt, submitted_at
FROM kyc_jobs WHERE idempotency_key = $1`

func (r *PostgresRepo) FindByIdempotencyKey(ctx context.Context, key string) (*domain.KYCJob, error) {
        row := r.pool.QueryRow(ctx, findByIdempotencyKeySQL, key)
        return scanJob(row)
}

// ── Result reads ─────────────────────────────────────────────────────────────

const getResultSQL = `
SELECT job_id, user_id, status, internal_job_id, result_text, result_code,
       confidence, model_version, liveness_version, error_msg, processed_at, attempt
FROM kyc_results WHERE job_id = $1`

func (r *PostgresRepo) GetResult(ctx context.Context, jobID string) (*domain.KYCResult, error) {
        row := r.pool.QueryRow(ctx, getResultSQL, jobID)

        var res domain.KYCResult
        var status string
        err := row.Scan(
                &res.JobID,
                &res.UserID,
                &status,
                &res.InternalJobID,
                &res.ResultText,
                &res.ResultCode,
                &res.Confidence,
                &res.ModelVersion,
                &res.LivenessVersion,
                &res.ErrorMsg,
                &res.ProcessedAt,
                &res.Attempt,
        )
        if err != nil {
                if errors.Is(err, pgx.ErrNoRows) {
                        return nil, domain.ErrJobNotFound
                }
                return nil, fmt.Errorf("get result: %w", err)
        }
        res.Status = domain.KYCStatus(status)
        return &res, nil
}

// ── Scanner helper ────────────────────────────────────────────────────────────

func scanJob(row pgx.Row) (*domain.KYCJob, error) {
        var job domain.KYCJob
        var idType, tier, status string
        err := row.Scan(
                &job.ID,
                &job.IdempotencyKey,
                &job.UserID,
                &job.CountryCode,
                &idType,
                &job.IDNumber,
                &job.FirstName,
                &job.LastName,
                &tier,
                &status,
                &job.Attempt,
                &job.SubmittedAt,
        )
        if err != nil {
                if errors.Is(err, pgx.ErrNoRows) {
                        return nil, domain.ErrJobNotFound
                }
                return nil, fmt.Errorf("scan job: %w", err)
        }
        job.IDType = domain.IDType(idType)
        job.Tier = domain.KYCTier(tier)
        job.Status = domain.KYCStatus(status)
        return &job, nil
}
