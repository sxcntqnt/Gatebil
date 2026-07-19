// Package kycclient defines the KYCClient interface and its PythonClient
// implementation. This is the only place in the Go service that knows the
// Python inference service exists. Everything above this layer is vendor-neutral.
//
// Resilience layers (innermost → outermost):
//   context timeout  →  semaphore (concurrency cap)  →  circuit breaker
package kycclient

import (
        "bytes"
        "context"
        "encoding/json"
        "fmt"
        "io"
        "log/slog"
        "mime/multipart"
        "net"
        "net/http"
        "sync"
        "time"
)

// ──────────────────────────────────────────────────
// Interface — the sovereign protocol
// ──────────────────────────────────────────────────

// KYCClient is the only dependency the worker pool holds on the inference layer.
// Swap the implementation (Python, ONNX, remote SaaS, GPU cluster) without
// touching any other package.
type KYCClient interface {
        // Verify compares a selfie against an ID-card face.
        Verify(ctx context.Context, req VerifyRequest) (*VerifyResult, error)

        // Challenge runs a single liveness frame through the requested detector.
        Challenge(ctx context.Context, req ChallengeRequest) (*ChallengeResult, error)

        // SmartCrop extracts and perspective-corrects the ID card face region.
        SmartCrop(ctx context.Context, req SmartCropRequest) (*SmartCropResult, error)

        // Health returns nil when the Python service is ready to serve.
        Health(ctx context.Context) error

        // Models returns the current load state and version of every inference model.
        Models(ctx context.Context) (*ModelsStatus, error)
}

// ──────────────────────────────────────────────────
// Request / Response shapes
// ──────────────────────────────────────────────────

type VerifyRequest struct {
        SelfieURL string // presigned R2 URL — required
        IDCardURL string // presigned R2 URL — omit to reuse last SmartCrop result on the Python side
}

type VerifyResult struct {
        Verified        bool    `json:"verified"`
        Score           float64 `json:"score"`
        ModelVersion    string  `json:"model_version"`
        LivenessVersion string  `json:"liveness_version"`
        InternalJobID   string  `json:"internal_job_id"`
}

type ChallengeRequest struct {
        FrameBytes []byte // single video frame
        Challenge  string // "blink" | "orientation" | "emotion"
        Expected   string // expected label for orientation / emotion
}

type ChallengeResult struct {
        Passed bool   `json:"passed"`
        Result string `json:"result"`
}

type SmartCropRequest struct {
        IDCardURL string // presigned R2 URL
}

type SmartCropResult struct {
        CroppedPath string  `json:"cropped_path"`
        FinalPath   string  `json:"final_path"`
        Keypoints   [][]int `json:"keypoints"`
}

type ModelsStatus struct {
        MTCNN           bool   `json:"mtcnn"`
        VGGFace2        bool   `json:"vggface2"`
        DSNT            bool   `json:"dsnt"`
        GPU             bool   `json:"gpu"`
        CUDA            string `json:"cuda"`
        ModelVersion    string `json:"model_version"`
        LivenessVersion string `json:"liveness_version"`
}

// ──────────────────────────────────────────────────
// Sentinel errors
// ──────────────────────────────────────────────────

var (
        errCircuitOpen  = fmt.Errorf("kyc inference circuit breaker is open")
        errBackpressure = fmt.Errorf("kyc inference concurrency limit reached")

        // ErrClientError wraps any 4xx response from the Python service — e.g. a
        // rejected/unprocessable ID card photo (422 IDCardProcessingError), a
        // malformed request (400), or a missing resource. It always means the
        // *input* was the problem, not the inference service's health, so:
        //   - it must NOT trip the circuit breaker (the service is fine)
        //   - it must NOT be retried by the worker pool (retrying won't change
        //     the outcome — the photo is still bad on attempt 2 and 3)
        //
        // Exported so callers (worker.Pool) can classify it via errors.Is
        // without this package needing to know about domain.KYCStatus.
        ErrClientError = fmt.Errorf("kyc inference rejected the request")
)

// ──────────────────────────────────────────────────
// Circuit breaker — minimal, no external dependency
// ──────────────────────────────────────────────────

type cbState int

const (
        cbClosed   cbState = iota // normal operation
        cbOpen                    // fast-failing; waiting for cooldown
        cbHalfOpen                // probing with one request
)

type circuitBreaker struct {
        mu          sync.Mutex
        state       cbState
        failures    int
        threshold   int           // consecutive failures to trip
        cooldown    time.Duration // how long to stay open
        lastFailure time.Time
}

func newCircuitBreaker(threshold int, cooldown time.Duration) *circuitBreaker {
        return &circuitBreaker{threshold: threshold, cooldown: cooldown}
}

// allow returns true when the request should be forwarded to the Python service.
func (cb *circuitBreaker) allow() bool {
        cb.mu.Lock()
        defer cb.mu.Unlock()

        switch cb.state {
        case cbClosed:
                return true
        case cbOpen:
                if time.Since(cb.lastFailure) >= cb.cooldown {
                        cb.state = cbHalfOpen
                        return true
                }
                return false
        case cbHalfOpen:
                return true
        default:
                return false
        }
}

func (cb *circuitBreaker) recordSuccess() {
        cb.mu.Lock()
        defer cb.mu.Unlock()
        cb.failures = 0
        cb.state = cbClosed
}

func (cb *circuitBreaker) recordFailure() {
        cb.mu.Lock()
        defer cb.mu.Unlock()
        cb.failures++
        cb.lastFailure = time.Now()
        if cb.failures >= cb.threshold {
                cb.state = cbOpen
        }
}

// ──────────────────────────────────────────────────
// PythonClient — implements KYCClient
// ──────────────────────────────────────────────────

// Config is the constructor input for PythonClient.
type Config struct {
        BaseURL     string        // e.g. "http://kyc-python:5000"
        Timeout     time.Duration // per-request context deadline (default 15s)
        Concurrency int           // max parallel in-flight requests to Python
}

// PythonClient calls the Flask/FastAPI inference service.
// It is safe for concurrent use.
type PythonClient struct {
        base    string
        timeout time.Duration
        http    *http.Client

        // Semaphore: prevents a backlog of goroutines from piling up when Python is slow.
        sem chan struct{}

        // Circuit breaker: stops hammering a degraded Python service.
        breaker *circuitBreaker

        logger *slog.Logger
}

// New constructs a PythonClient ready for use.
func New(cfg Config, logger *slog.Logger) *PythonClient {
        if cfg.Timeout == 0 {
                cfg.Timeout = 15 * time.Second
        }
        if cfg.Concurrency < 1 {
                cfg.Concurrency = 10
        }

        httpClient := &http.Client{
                Timeout: cfg.Timeout + 2*time.Second,
                Transport: &http.Transport{
                        DialContext: (&net.Dialer{
                                Timeout:   3 * time.Second,
                                KeepAlive: 30 * time.Second,
                        }).DialContext,

                        ForceAttemptHTTP2:     false,
                        MaxIdleConns:          100,
                        IdleConnTimeout:       90 * time.Second,
                        TLSHandshakeTimeout:   5 * time.Second,
                        ResponseHeaderTimeout: 5 * time.Second,

                        // prevents "infinite connect stalls"
                        ExpectContinueTimeout: 1 * time.Second,
                },
        }

        return &PythonClient{
                base:    cfg.BaseURL,
                timeout: cfg.Timeout,
                http:    httpClient,
                sem:     make(chan struct{}, cfg.Concurrency),
                breaker: newCircuitBreaker(5, 30*time.Second),
                logger:  logger,
        }
}

// ── Resilience gate ──────────────────────────────────────────────────────────

// gate acquires the semaphore slot and checks the circuit breaker.
// Returns a release func; caller must defer it.
func (c *PythonClient) gate(ctx context.Context) (func(), error) {
        if !c.breaker.allow() {
                return func() {}, fmt.Errorf("%w: circuit open", errCircuitOpen)
        }
        select {
        case c.sem <- struct{}{}:
                return func() { <-c.sem }, nil
        case <-ctx.Done():
                return func() {}, ctx.Err()
        default:
                // Non-blocking: if semaphore is full return backpressure immediately
                // so the worker pool's retry/backoff loop handles it gracefully.
                return func() {}, errBackpressure
        }
}

// classifyStatus turns an HTTP status code into the right error, applying
// the correct side effect to the circuit breaker for each class:
//
//   2xx        → nil, no breaker effect (caller records success after decode)
//   4xx        → ErrClientError, breaker untouched — the service is healthy,
//                the request was the problem
//   5xx        → generic error, breaker.recordFailure() — the service itself
//                is unhealthy or failing to process a valid request
func (c *PythonClient) classifyStatus(resp *http.Response) error {
        switch {
        case resp.StatusCode >= 200 && resp.StatusCode < 300:
                return nil
        case resp.StatusCode >= 400 && resp.StatusCode < 500:
                raw, _ := io.ReadAll(resp.Body)
                return fmt.Errorf("%w: python service %d: %s", ErrClientError, resp.StatusCode, raw)
        default: // >= 500, or any other unexpected class
                c.breaker.recordFailure()
                raw, _ := io.ReadAll(resp.Body)
                return fmt.Errorf("python service %d: %s", resp.StatusCode, raw)
        }
}

// ── HTTP helpers ─────────────────────────────────────────────────────────────

// post sends a multipart request and decodes the JSON response into out.
func (c *PythonClient) post(ctx context.Context, path string, fields map[string][]byte, out interface{}) error {
        tctx, cancel := context.WithTimeout(ctx, c.timeout)
        defer cancel()

        body, ct, err := buildMultipart(fields)
        if err != nil {
                return fmt.Errorf("build multipart: %w", err)
        }

        req, err := http.NewRequestWithContext(tctx, http.MethodPost, c.base+path, body)
        if err != nil {
                return err
        }
        req.Header.Set("Content-Type", ct)

        resp, err := c.http.Do(req)
        if err != nil {
                c.breaker.recordFailure()
                return fmt.Errorf("http: %w", err)
        }
        defer resp.Body.Close()

        if err := c.classifyStatus(resp); err != nil {
                return err
        }

        c.breaker.recordSuccess()
        return json.NewDecoder(resp.Body).Decode(out)
}

// postJSON sends a JSON request and decodes the JSON response into out.
// Used for the URL-based routes (/verify, /id-card) now that images live
// in R2 rather than crossing the wire as multipart bytes.
func (c *PythonClient) postJSON(ctx context.Context, path string, payload interface{}, out interface{}) error {
        tctx, cancel := context.WithTimeout(ctx, c.timeout)
        defer cancel()

        body, err := json.Marshal(payload)
        if err != nil {
                return fmt.Errorf("marshal payload: %w", err)
        }

        req, err := http.NewRequestWithContext(tctx, http.MethodPost, c.base+path, bytes.NewReader(body))
        if err != nil {
                return err
        }
        req.Header.Set("Content-Type", "application/json")

        resp, err := c.http.Do(req)
        if err != nil {
                c.breaker.recordFailure()
                return fmt.Errorf("http: %w", err)
        }
        defer resp.Body.Close()

        if err := c.classifyStatus(resp); err != nil {
                return err
        }

        c.breaker.recordSuccess()
        return json.NewDecoder(resp.Body).Decode(out)
}

func (c *PythonClient) get(ctx context.Context, path string, out interface{}) error {
        tctx, cancel := context.WithTimeout(ctx, c.timeout)
        defer cancel()

        req, err := http.NewRequestWithContext(tctx, http.MethodGet, c.base+path, nil)
        if err != nil {
                return err
        }
        resp, err := c.http.Do(req)
        if err != nil {
                return err
        }
        defer resp.Body.Close()
        return json.NewDecoder(resp.Body).Decode(out)
}

// ── KYCClient implementation ─────────────────────────────────────────────────

func (c *PythonClient) Verify(ctx context.Context, req VerifyRequest) (*VerifyResult, error) {
        release, err := c.gate(ctx)
        if err != nil {
                return nil, err
        }
        defer release()

        if req.SelfieURL == "" {
                return nil, fmt.Errorf("kycclient: Verify requires SelfieURL")
        }

        payload := struct {
                SelfieURL string `json:"selfie_url"`
                IDCardURL string `json:"id_card_url,omitempty"`
        }{
                SelfieURL: req.SelfieURL,
                IDCardURL: req.IDCardURL,
        }

        var out VerifyResult
        if err := c.postJSON(ctx, "/internal/v1/verify", payload, &out); err != nil {
                c.logger.Error("verify call failed", "err", err)
                return nil, err
        }
        return &out, nil
}

func (c *PythonClient) Challenge(ctx context.Context, req ChallengeRequest) (*ChallengeResult, error) {
        release, err := c.gate(ctx)
        if err != nil {
                return nil, err
        }
        defer release()

        var out ChallengeResult
        // challenge and expected are plain text form fields alongside the binary
        // frame — postChallenge builds the full multipart body itself.
        if err := c.postChallenge(ctx, req, &out); err != nil {
                return nil, err
        }
        return &out, nil
}

// postChallenge is a small specialisation because challenge/expected are
// plain text form fields alongside the binary frame field.
func (c *PythonClient) postChallenge(ctx context.Context, req ChallengeRequest, out interface{}) error {
        tctx, cancel := context.WithTimeout(ctx, c.timeout)
        defer cancel()

        var buf bytes.Buffer
        w := multipart.NewWriter(&buf)
        _ = w.WriteField("challenge", req.Challenge)
        _ = w.WriteField("expected", req.Expected)
        fw, _ := w.CreateFormFile("frame", "frame.jpg")
        _, _ = fw.Write(req.FrameBytes)
        w.Close()

        httpReq, err := http.NewRequestWithContext(tctx, http.MethodPost,
                c.base+"/internal/v1/challenge", &buf)
        if err != nil {
                return err
        }
        httpReq.Header.Set("Content-Type", w.FormDataContentType())

        resp, err := c.http.Do(httpReq)
        if err != nil {
                c.breaker.recordFailure()
                return err
        }
        defer resp.Body.Close()

        if err := c.classifyStatus(resp); err != nil {
                return err
        }

        c.breaker.recordSuccess()
        return json.NewDecoder(resp.Body).Decode(out)
}

func (c *PythonClient) SmartCrop(ctx context.Context, req SmartCropRequest) (*SmartCropResult, error) {
        release, err := c.gate(ctx)
        if err != nil {
                return nil, err
        }
        defer release()

        if req.IDCardURL == "" {
                return nil, fmt.Errorf("kycclient: SmartCrop requires IDCardURL")
        }

        payload := struct {
                IDCardURL string `json:"id_card_url"`
        }{IDCardURL: req.IDCardURL}

        var out SmartCropResult
        if err := c.postJSON(ctx, "/internal/v1/id-card", payload, &out); err != nil {
                return nil, err
        }
        return &out, nil
}

func (c *PythonClient) Health(ctx context.Context) error {
        var out struct {
                OK bool `json:"ok"`
        }
        if err := c.get(ctx, "/internal/v1/health", &out); err != nil {
                return err
        }
        if !out.OK {
                return fmt.Errorf("python service reports not ready")
        }
        return nil
}

func (c *PythonClient) Models(ctx context.Context) (*ModelsStatus, error) {
        var out ModelsStatus
        if err := c.get(ctx, "/internal/v1/models", &out); err != nil {
                return nil, err
        }
        return &out, nil
}

// ── Multipart builder ────────────────────────────────────────────────────────

func buildMultipart(fields map[string][]byte) (io.Reader, string, error) {
        var buf bytes.Buffer
        w := multipart.NewWriter(&buf)
        for field, data := range fields {
                fw, err := w.CreateFormFile(field, field+".jpg")
                if err != nil {
                        return nil, "", err
                }
                if _, err := fw.Write(data); err != nil {
                        return nil, "", err
                }
        }
        w.Close()
        return &buf, w.FormDataContentType(), nil
}
