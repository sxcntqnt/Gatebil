// Package config loads and validates service configuration from environment
// variables. All required vars are validated at startup; the service refuses
// to start with an incomplete config.
package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

// Config holds the full runtime configuration for the KYC microservice.
type Config struct {
	// Server
	Port            string
	ReadTimeout     time.Duration
	WriteTimeout    time.Duration
	ShutdownTimeout time.Duration

	// Internal KYC inference service (replaces Smile ID)
	KYCServiceURL     string        // e.g. "http://kyc-python:5000"
	KYCServiceTimeout time.Duration // per-request timeout to the Python service
	KYCConcurrency    int           // semaphore width — max parallel calls to Python

	// PostgreSQL
	DBDSN string

	// Redis
	RedisAddr     string
	RedisPassword string
	RedisDB       int

	// Worker pool
	WorkerCount    int           // number of concurrent goroutines
	QueueDepth     int           // buffered channel size
	MaxRetries     int           // per-job retry ceiling
	RetryBaseDelay time.Duration // base for exponential backoff
	JobTimeout     time.Duration // context deadline per KYC job
}

// Load reads configuration from environment variables and validates required fields.
func Load() (*Config, error) {
	cfg := &Config{
		Port:            getEnv("PORT", "8080"),
		ReadTimeout:     getDuration("READ_TIMEOUT", 10*time.Second),
		WriteTimeout:    getDuration("WRITE_TIMEOUT", 30*time.Second),
		ShutdownTimeout: getDuration("SHUTDOWN_TIMEOUT", 15*time.Second),

		KYCServiceURL:     getEnv("KYC_SERVICE_URL", "http://localhost:5000"),
		KYCServiceTimeout: getDuration("KYC_SERVICE_TIMEOUT", 15*time.Second),
		// Default: match WorkerCount so every goroutine can make one call.
		// Tuned down at runtime if the Python service is resource-constrained.
		KYCConcurrency: getInt("KYC_SERVICE_CONCURRENCY", 20),

		DBDSN: os.Getenv("DATABASE_DSN"),

		RedisAddr:     getEnv("REDIS_ADDR", "localhost:6379"),
		RedisPassword: os.Getenv("REDIS_PASSWORD"),
		RedisDB:       getInt("REDIS_DB", 0),

		WorkerCount:    getInt("KYC_WORKER_COUNT", 20),
		QueueDepth:     getInt("KYC_QUEUE_DEPTH", 500),
		MaxRetries:     getInt("KYC_MAX_RETRIES", 3),
		RetryBaseDelay: getDuration("KYC_RETRY_BASE_DELAY", 2*time.Second),
		JobTimeout:     getDuration("KYC_JOB_TIMEOUT", 30*time.Second),
	}

	return cfg, cfg.validate()
}

func (c *Config) validate() error {
	required := map[string]string{
		"DATABASE_DSN":    c.DBDSN,
		"KYC_SERVICE_URL": c.KYCServiceURL,
	}
	for key, val := range required {
		if val == "" {
			return fmt.Errorf("missing required config: %s", key)
		}
	}
	if c.WorkerCount < 1 || c.WorkerCount > 500 {
		return fmt.Errorf("KYC_WORKER_COUNT must be between 1 and 500, got %d", c.WorkerCount)
	}
	if c.KYCConcurrency < 1 || c.KYCConcurrency > c.WorkerCount {
		return fmt.Errorf("KYC_SERVICE_CONCURRENCY must be between 1 and KYC_WORKER_COUNT (%d), got %d",
			c.WorkerCount, c.KYCConcurrency)
	}
	return nil
}

// ──────────────────────────────────────────────────
// helpers
// ──────────────────────────────────────────────────

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}

func getDuration(key string, fallback time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return fallback
}
