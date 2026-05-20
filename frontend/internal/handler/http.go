// Package handler wires HTTP routes to the KYC usecase and implements
// middleware for logging, metrics, request IDs, and CORS.
package handler

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"sxcntqunts/kyc-service/internal/domain"
	"sxcntqunts/kyc-service/internal/metrics"
	"sxcntqunts/kyc-service/internal/usecase"
	"sxcntqunts/kyc-service/internal/worker"
)

// ──────────────────────────────────────────────────
// Server
// ──────────────────────────────────────────────────

// Server holds the HTTP mux and its dependencies.
type Server struct {
	uc      *usecase.KYCUsecase
	pool    *worker.Pool
	metrics *metrics.KYCMetrics
	logger  *slog.Logger
	mux     *http.ServeMux
}

// New builds the server and registers all routes.
func New(
	uc *usecase.KYCUsecase,
	pool *worker.Pool,
	m *metrics.KYCMetrics,
	logger *slog.Logger,
) *Server {
	s := &Server{uc: uc, pool: pool, metrics: m, logger: logger, mux: http.NewServeMux()}
	s.routes()
	return s
}

// Handler returns the root http.Handler with middleware applied.
func (s *Server) Handler() http.Handler {
	return s.withMiddleware(s.mux)
}

func (s *Server) routes() {
	// Health
	s.mux.HandleFunc("GET /healthz", s.handleHealth)
	s.mux.HandleFunc("GET /readyz", s.handleReady)

	// Metrics (Prometheus scrape endpoint)
	s.mux.Handle("GET /metrics", promhttp.Handler())

	// KYC API
	s.mux.HandleFunc("POST /api/v1/kyc/submit", s.handleSubmit)
	s.mux.HandleFunc("GET /api/v1/kyc/status/{jobID}", s.handleGetJobStatus)
	s.mux.HandleFunc("GET /api/v1/kyc/user/{userID}/status", s.handleGetUserStatus)
}

// ──────────────────────────────────────────────────
// Handlers
// ──────────────────────────────────────────────────

// handleSubmit accepts a KYC verification request, enqueues it, and returns 202.
//
//	POST /api/v1/kyc/submit
func (s *Server) handleSubmit(w http.ResponseWriter, r *http.Request) {
	var req usecase.SubmitRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}

	// Default tier to light when omitted.
	if req.Tier == "" {
		req.Tier = domain.TierLight
	}

	resp, err := s.uc.Submit(r.Context(), req)
	if err != nil {
		switch {
		case errors.Is(err, domain.ErrDuplicateJob):
			writeError(w, http.StatusConflict, err.Error())
		case errors.Is(err, domain.ErrInvalidIDType):
			writeError(w, http.StatusUnprocessableEntity, err.Error())
		case errors.Is(err, domain.ErrQueueFull):
			writeError(w, http.StatusServiceUnavailable, err.Error())
		default:
			s.logger.Error("submit error", "err", err)
			writeError(w, http.StatusInternalServerError, "internal error")
		}
		return
	}

	writeJSON(w, http.StatusAccepted, resp)
}

// handleGetJobStatus returns the current state of a single KYC job.
//
//	GET /api/v1/kyc/status/{jobID}
func (s *Server) handleGetJobStatus(w http.ResponseWriter, r *http.Request) {
	jobID := r.PathValue("jobID")
	if jobID == "" {
		writeError(w, http.StatusBadRequest, "jobID path parameter is required")
		return
	}

	resp, err := s.uc.GetStatus(r.Context(), jobID)
	if err != nil {
		if errors.Is(err, domain.ErrJobNotFound) {
			writeError(w, http.StatusNotFound, "job not found")
			return
		}
		s.logger.Error("get status error", "job_id", jobID, "err", err)
		writeError(w, http.StatusInternalServerError, "internal error")
		return
	}

	writeJSON(w, http.StatusOK, resp)
}

// handleGetUserStatus returns the latest KYC result for a given user.
//
//	GET /api/v1/kyc/user/{userID}/status
func (s *Server) handleGetUserStatus(w http.ResponseWriter, r *http.Request) {
	userID := r.PathValue("userID")
	if userID == "" {
		writeError(w, http.StatusBadRequest, "userID path parameter is required")
		return
	}

	resp, err := s.uc.GetUserStatus(r.Context(), userID)
	if err != nil {
		if errors.Is(err, domain.ErrJobNotFound) {
			writeError(w, http.StatusNotFound, "no KYC record found for user")
			return
		}
		s.logger.Error("get user status error", "user_id", userID, "err", err)
		writeError(w, http.StatusInternalServerError, "internal error")
		return
	}

	writeJSON(w, http.StatusOK, resp)
}

// handleHealth is the liveness probe — always 200 if the process is running.
func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// handleReady is the readiness probe — checks the worker queue is not overloaded.
func (s *Server) handleReady(w http.ResponseWriter, _ *http.Request) {
	qLen := s.pool.QueueLen()
	body := map[string]any{
		"status":    "ok",
		"queue_len": qLen,
	}
	writeJSON(w, http.StatusOK, body)
}

// ──────────────────────────────────────────────────
// Middleware
// ──────────────────────────────────────────────────

func (s *Server) withMiddleware(next http.Handler) http.Handler {
	return s.requestID(
		s.logging(
			s.metricsMiddleware(
				s.cors(next),
			),
		),
	)
}

// requestID injects a unique request ID into the context and response header.
func (s *Server) requestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-ID")
		if id == "" {
			id = fmt.Sprintf("%d", time.Now().UnixNano())
		}
		w.Header().Set("X-Request-ID", id)
		next.ServeHTTP(w, r)
	})
}

// logging logs each request with method, path, status, and latency.
func (s *Server) logging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		rw := &responseWriter{ResponseWriter: w, code: http.StatusOK}
		next.ServeHTTP(rw, r)
		s.logger.Info("http request",
			"method", r.Method,
			"path", r.URL.Path,
			"status", rw.code,
			"latency_ms", time.Since(t0).Milliseconds(),
			"request_id", w.Header().Get("X-Request-ID"),
		)
	})
}

// metricsMiddleware records Prometheus RED metrics per handler.
func (s *Server) metricsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		rw := &responseWriter{ResponseWriter: w, code: http.StatusOK}
		next.ServeHTTP(rw, r)

		path := cleanPath(r.URL.Path)
		code := strconv.Itoa(rw.code)
		s.metrics.HTTPRequests.WithLabelValues(r.Method, path, code).Inc()
		s.metrics.HTTPDuration.WithLabelValues(r.Method, path).Observe(time.Since(t0).Seconds())
	})
}

// cors sets permissive CORS headers for local development.
// Tighten allowed origins in production.
func (s *Server) cors(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Request-ID")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// ──────────────────────────────────────────────────
// Response helpers
// ──────────────────────────────────────────────────

type errorResponse struct {
	Error string `json:"error"`
}

func writeJSON(w http.ResponseWriter, code int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(body)
}

func writeError(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, errorResponse{Error: msg})
}

// responseWriter wraps http.ResponseWriter to capture the status code.
type responseWriter struct {
	http.ResponseWriter
	code int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.code = code
	rw.ResponseWriter.WriteHeader(code)
}

// cleanPath strips UUID-like path segments to group Prometheus labels sensibly.
func cleanPath(path string) string {
	parts := strings.Split(path, "/")
	for i, p := range parts {
		if isUUID(p) || isNumeric(p) {
			parts[i] = "{id}"
		}
	}
	return strings.Join(parts, "/")
}

func isUUID(s string) bool {
	return len(s) == 36 && strings.Count(s, "-") == 4
}

func isNumeric(s string) bool {
	_, err := strconv.ParseInt(s, 10, 64)
	return err == nil
}
