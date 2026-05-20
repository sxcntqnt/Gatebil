// Package metrics registers and exposes Prometheus metrics for the KYC service.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// KYCMetrics groups all Prometheus instruments for KYC operations.
type KYCMetrics struct {
	JobsSubmitted   prometheus.Counter
	JobsProcessed   *prometheus.CounterVec   // labels: status (approved|rejected|failed)
	JobDuration     prometheus.Histogram      // end-to-end wall clock per job
	SmileLatency    prometheus.Histogram      // Smile ID API round-trip
	WorkerQueueLen  prometheus.Gauge          // current depth of the job channel
	ActiveWorkers   prometheus.Gauge          // goroutines actively processing
	RetriesTotal    prometheus.Counter
	HTTPRequests    *prometheus.CounterVec   // labels: method, path, status_code
	HTTPDuration    *prometheus.HistogramVec // labels: method, path
}

// New registers all metrics with the default Prometheus registry.
func New() *KYCMetrics {
	return &KYCMetrics{
		JobsSubmitted: promauto.NewCounter(prometheus.CounterOpts{
			Name: "kyc_jobs_submitted_total",
			Help: "Total KYC jobs submitted to the queue.",
		}),

		JobsProcessed: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "kyc_jobs_processed_total",
			Help: "Total KYC jobs processed, partitioned by final status.",
		}, []string{"status"}),

		JobDuration: promauto.NewHistogram(prometheus.HistogramOpts{
			Name:    "kyc_job_duration_seconds",
			Help:    "End-to-end time from submission to terminal status.",
			Buckets: []float64{0.5, 1, 2, 5, 10, 20, 30, 60},
		}),

		SmileLatency: promauto.NewHistogram(prometheus.HistogramOpts{
			Name:    "kyc_smileid_request_duration_seconds",
			Help:    "Smile ID API round-trip latency.",
			Buckets: []float64{0.1, 0.25, 0.5, 1, 2, 5, 10, 20},
		}),

		WorkerQueueLen: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "kyc_worker_queue_length",
			Help: "Current number of jobs waiting in the worker channel.",
		}),

		ActiveWorkers: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "kyc_active_workers",
			Help: "Number of goroutines currently processing a KYC job.",
		}),

		RetriesTotal: promauto.NewCounter(prometheus.CounterOpts{
			Name: "kyc_retries_total",
			Help: "Total number of retry attempts across all jobs.",
		}),

		HTTPRequests: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "kyc_http_requests_total",
			Help: "Total HTTP requests handled, by method, path, and status code.",
		}, []string{"method", "path", "status_code"}),

		HTTPDuration: promauto.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "kyc_http_request_duration_seconds",
			Help:    "HTTP handler latency.",
			Buckets: prometheus.DefBuckets,
		}, []string{"method", "path"}),
	}
}
