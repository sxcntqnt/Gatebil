-- ============================================================================
-- KYC Service Schema
-- PostgreSQL
-- ============================================================================
BEGIN;

CREATE TABLE IF NOT EXISTS kyc_jobs (
    id               TEXT PRIMARY KEY,
    idempotency_key  TEXT NOT NULL UNIQUE,
    user_id          TEXT NOT NULL,
    country_code     TEXT NOT NULL,
    id_type          TEXT NOT NULL,
    id_number        TEXT NOT NULL,
    first_name       TEXT NOT NULL,
    last_name        TEXT NOT NULL,
    tier             TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    attempt          INTEGER NOT NULL DEFAULT 0,
    submitted_at     TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS kyc_results (
    job_id              TEXT PRIMARY KEY
                            REFERENCES kyc_jobs(id)
                            ON DELETE CASCADE,
    user_id             TEXT NOT NULL,
    status              TEXT NOT NULL,
    internal_job_id     TEXT,
    result_text         TEXT,
    result_code         TEXT,
    confidence          DOUBLE PRECISION,
    model_version       TEXT,
    liveness_version    TEXT,
    error_msg           TEXT,
    processed_at        TIMESTAMPTZ NOT NULL,
    attempt             INTEGER NOT NULL
);

-- ============================================================================
-- Indexes
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_kyc_jobs_user_id
    ON kyc_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_kyc_jobs_status
    ON kyc_jobs(status);
CREATE INDEX IF NOT EXISTS idx_kyc_jobs_user_status
    ON kyc_jobs(user_id, status);
CREATE INDEX IF NOT EXISTS idx_kyc_jobs_submitted_at
    ON kyc_jobs(submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_kyc_jobs_user_submitted
    ON kyc_jobs(user_id, submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_kyc_results_user_id
    ON kyc_results(user_id);
CREATE INDEX IF NOT EXISTS idx_kyc_results_status
    ON kyc_results(status);

COMMIT;
