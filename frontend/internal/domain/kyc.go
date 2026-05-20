// Package domain defines the core KYC business entities, status lifecycle,
// and domain errors. No external dependencies allowed here.
package domain

import (
	"errors"
	"time"
)

// ──────────────────────────────────────────────────
// Status enum
// ──────────────────────────────────────────────────

// KYCStatus represents every possible lifecycle state of a KYC job.
type KYCStatus string

const (
	StatusPending    KYCStatus = "pending"    // queued, not yet dispatched
	StatusProcessing KYCStatus = "processing" // worker picked it up
	StatusApproved   KYCStatus = "approved"   // SmileID returned verified
	StatusRejected   KYCStatus = "rejected"   // SmileID returned not verified
	StatusFailed     KYCStatus = "failed"     // transient error after all retries
)

// ──────────────────────────────────────────────────
// ID type catalogue (BCLB / Smile ID supported)
// ──────────────────────────────────────────────────

type IDType string

const (
	IDTypeNationalID IDType = "NATIONAL_ID"
	IDTypeAlienID    IDType = "ALIEN_ID"
	IDTypePassport   IDType = "PASSPORT"
	IDTypeVoterID    IDType = "VOTER_ID"
	IDTypeDrivingLic IDType = "DRIVING_LICENSE"
)

// ──────────────────────────────────────────────────
// KYC tier — maps to BCLB light vs full
// ──────────────────────────────────────────────────

type KYCTier string

const (
	TierLight KYCTier = "kyc_light"    // passengers / low-risk
	TierFull  KYCTier = "kyc_full"     // operators / crew / NTSA-grade
)

// ──────────────────────────────────────────────────
// Core domain types
// ──────────────────────────────────────────────────

// KYCJob is the unit of work queued to the worker pool.
type KYCJob struct {
	ID          string    `json:"id"`           // UUID v4
	UserID      string    `json:"user_id"`
	CountryCode string    `json:"country_code"` // "KE", "NG", "GH"
	IDType      IDType    `json:"id_type"`
	IDNumber    string    `json:"id_number"`
	FirstName   string    `json:"first_name"`
	LastName    string    `json:"last_name"`
	Tier        KYCTier   `json:"tier"`
	Attempt     int       `json:"attempt"`      // retry counter
	SubmittedAt time.Time `json:"submitted_at"`
}

// KYCResult is the persisted outcome of a completed job.
type KYCResult struct {
	JobID       string    `json:"job_id"`
	UserID      string    `json:"user_id"`
	Status      KYCStatus `json:"status"`
	SmileJobID  string    `json:"smile_job_id,omitempty"`
	ResultText  string    `json:"result_text,omitempty"`
	ResultCode  string    `json:"result_code,omitempty"`
	Confidence  float64   `json:"confidence,omitempty"`
	ErrorMsg    string    `json:"error,omitempty"`
	ProcessedAt time.Time `json:"processed_at"`
	Attempt     int       `json:"attempt"`
}

// KYCStatusResponse is the HTTP response shape for status queries.
type KYCStatusResponse struct {
	JobID       string    `json:"job_id"`
	UserID      string    `json:"user_id"`
	Status      KYCStatus `json:"status"`
	Tier        KYCTier   `json:"tier,omitempty"`
	ResultText  string    `json:"result_text,omitempty"`
	Confidence  float64   `json:"confidence,omitempty"`
	ProcessedAt time.Time `json:"processed_at,omitempty"`
	SubmittedAt time.Time `json:"submitted_at"`
}

// ──────────────────────────────────────────────────
// Domain errors
// ──────────────────────────────────────────────────

var (
	ErrJobNotFound      = errors.New("kyc job not found")
	ErrDuplicateJob     = errors.New("active kyc job already exists for this user")
	ErrInvalidIDType    = errors.New("unsupported ID type for the given country")
	ErrQueueFull        = errors.New("kyc worker queue is at capacity, try again shortly")
	ErrJobAlreadyDone   = errors.New("kyc job is already in a terminal state")
)

// ──────────────────────────────────────────────────
// Validation helpers
// ──────────────────────────────────────────────────

// AllowedIDTypes maps country → accepted ID types for quick validation.
var AllowedIDTypes = map[string]map[IDType]bool{
	"KE": {
		IDTypeNationalID: true,
		IDTypeAlienID:    true,
		IDTypePassport:   true,
	},
	"NG": {
		IDTypeNationalID: true, // NIN
		IDTypePassport:   true,
		IDTypeVoterID:    true,
	},
	"GH": {
		IDTypeNationalID: true,
		IDTypePassport:   true,
		IDTypeVoterID:    true,
		IDTypeDrivingLic: true,
	},
}

// IsTerminal reports whether a status needs no further processing.
func (s KYCStatus) IsTerminal() bool {
	return s == StatusApproved || s == StatusRejected || s == StatusFailed
}
