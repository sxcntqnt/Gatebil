# KYC Service

A self-hosted, Smile ID–free identity verification system built for East Africa. Two services work together: a Go orchestration layer that manages jobs, retries, persistence, and observability, and a Python inference layer that runs the actual computer vision models.

---

## What It Does

A user submits a photo of their national ID card and a live selfie. The system:

1. Detects the four corners of the ID card using a DSNT keypoint model and corrects the perspective with a homography warp
2. Extracts the face region from the corrected ID card
3. Computes face embeddings for both the selfie and the ID face using VGGFace2
4. Compares the embeddings via cosine similarity and returns a verified/rejected decision
5. Optionally runs a liveness challenge (blink, head orientation, or emotion detection) to confirm the selfie is a real person

The system supports two verification tiers:

- `kyc_light` — face verification only, for passengers and low-risk users
- `kyc_full` — face verification plus liveness, for operators, crew, and NTSA-grade compliance

---

## Architecture

```
                         ┌─────────────────────────────────────┐
  Client / App           │         Go KYC Service              │
  ─────────────          │  cmd/server/main.go                 │
  POST /submit    ──────▶│                                     │
  GET  /status    ──────▶│  handler    →  usecase  →  worker   │
  GET  /models    ──────▶│  (HTTP)        (logic)    (pool)    │
                         │       ↕            ↕          ↕     │
                         │  postgres    idempotency   KYCClient│
                         └──────────────────────┬──────────────┘
                                                │ /internal/v1/*
                                                │ (timeout · semaphore · circuit breaker)
                         ┌──────────────────────▼──────────────┐
                         │      Python Inference Service        │
                         │  app/main.py  (FastAPI + uvicorn)   │
                         │                                     │
                         │  /id-card   →  DSNT TF keypoints    │
                         │               + homography warp      │
                         │                                     │
                         │  /verify    →  VGGFace2 embeddings  │
                         │               + cosine similarity    │
                         │                                     │
                         │  /challenge →  blink (EAR/dlib)     │
                         │               orientation (pose)     │
                         │               emotion (PyTorch)      │
                         │                                     │
                         │  /health    →  liveness probe        │
                         │  /models    →  model load state      │
                         └─────────────────────────────────────┘
                                                │
                                         MinIO / S3
                                   (future: image storage)
```

The Go service is the only externally visible component. The Python service is an internal inference kernel — never exposed directly to clients.

---

## Repository Layout

```
kyc-service/                        Go orchestration service
├── cmd/server/main.go              Entrypoint — wires all layers, startup probe
├── internal/
│   ├── config/config.go            Environment-based configuration
│   ├── domain/kyc.go               Job lifecycle, ID types, KYC tiers, errors
│   ├── kycclient/client.go         KYCClient interface + PythonClient implementation
│   │                               Circuit breaker · semaphore · context timeout
│   ├── handler/http.go             HTTP routes, middleware, error mapping
│   ├── usecase/kyc.go              Business logic: idempotency, dedup, submission
│   ├── repository/postgres.go      pgx/v5 Postgres persistence
│   ├── worker/pool.go              Bounded goroutine pool, retry/backoff
│   └── metrics/metrics.go          Prometheus instruments
├── monitoring/
│   ├── prometheus.yml              Scrape config for Go + Python services
│   └── alerts.yml                  Alerting rules (circuit breaker, queue, latency)
└── .env.example                    All environment variables documented

kyc-python/                         Python inference service
├── app/
│   ├── main.py                     Entrypoint — lifespan, routers, middleware only
│   ├── core/
│   │   ├── config.py               Pydantic-settings, model paths, thresholds
│   │   ├── exceptions.py           Domain exception hierarchy + FastAPI handlers
│   │   └── loggin.py               Structured JSON logging
│   ├── api/
│   │   └── dependency.py           FastAPI dependencies for model singletons
│   ├── models/
│   │   ├── id_detector.py          DSNT TF frozen graph wrapper + session lifecycle
│   │   └── schemas.py              Pydantic request/response shapes (matches Go structs)
│   ├── routes/
│   │   ├── health.py               GET /health  GET /models
│   │   ├── ekyc.py                 POST /id-card
│   │   ├── verification.py         POST /verify
│   │   └── liveness.py             POST /challenge
│   ├── pipelines/                  ← in progress
│   │   ├── ekyc.py                 Crop orchestration: decode → keypoints → warp
│   │   ├── verification.py         Verify: embeddings → cosine similarity
│   │   └── liveness.py             Liveness: blink / orientation / emotion
│   ├── services/                   ← in progress
│   │   ├── id_card/
│   │   │   ├── preprocessing.py    Rotate, resize, decode for model input
│   │   │   ├── inference.py        TF session runner
│   │   │   ├── homography.py       Perspective warp from keypoints
│   │   │   └── cropper.py          End-to-end smart-crop orchestration
│   │   ├── face/
│   │   │   └── verification.py     VGGFace2 embedding extraction + comparison
│   │   └── storage/
│   │       └── temp.py             Temp file lifecycle management
│   └── utils/
│       └── image.py                Decode, resize, distance helpers
```

---

## Current Build Status

### Go Service — Complete

| File | Status | Notes |
|---|---|---|
| `config/config.go` | ✅ | Smile ID removed; `KYC_SERVICE_URL`, `KYC_SERVICE_TIMEOUT`, `KYC_SERVICE_CONCURRENCY` added |
| `domain/kyc.go` | ✅ | `IdempotencyKey`, `ModelVersion`, `LivenessVersion`, `ErrBackpressure` |
| `kycclient/client.go` | ✅ | `KYCClient` interface; `PythonClient` with semaphore + circuit breaker |
| `usecase/kyc.go` | ✅ | Idempotency check before duplicate check; `GetStatusByUser` |
| `handler/http.go` | ✅ | `Idempotency-Key` header; `/models` proxy; `ErrBackpressure` → 503 + `Retry-After` |
| `repository/postgres.go` | ✅ | `FindByIdempotencyKey`, `SaveResult` (upsert), `model_version` column |
| `worker/pool.go` | ✅ | Backpressure retries do not burn the hard retry budget |
| `metrics/metrics.go` | ✅ | `InferenceLatency`, `CircuitBreakerState`, `IdempotencyReplays` |
| `cmd/server/main.go` | ✅ | `waitForInference()` startup gate; logs full model state on boot |
| `monitoring/alerts.yml` | ✅ | Six rules: circuit breaker, latency p95, queue, worker starvation, HTTP errors |

### Python Service — Foundation Complete, Services/Pipelines Pending

| Layer | File | Status |
|---|---|---|
| Entrypoint | `main.py` | ✅ Thin: lifespan + routers only |
| Core | `core/config.py` | ✅ Pydantic-settings, path validation at startup |
| Core | `core/exceptions.py` | ✅ Domain hierarchy + FastAPI handlers |
| Core | `core/loggin.py` | ✅ Structured JSON, silences noisy libs |
| Models | `models/id_detector.py` | ✅ DSNT TF graph class, `close()` on shutdown |
| Models | `models/schemas.py` | ✅ Mirrors Go `kycclient` structs exactly |
| API | `api/dependency.py` | ✅ Model singletons via `app.state`, `LivenessDetectors` bundle |
| Utils | `utils/image.py` | ✅ Decode, resize, cosine/euclidean distance |
| Routes | `routes/health.py` | ✅ `/health` + `/models` |
| Routes | `routes/ekyc.py` | ✅ `/id-card` |
| Routes | `routes/verification.py` | ✅ `/verify`, `id_image` optional |
| Routes | `routes/liveness.py` | ✅ `/challenge`, challenge type validated by `Literal` |
| Pipelines | `pipelines/ekyc.py` | 🔲 Next |
| Pipelines | `pipelines/verification.py` | 🔲 Next |
| Pipelines | `pipelines/liveness.py` | 🔲 Next |
| Services | `services/id_card/` | 🔲 Next |
| Services | `services/face/verification.py` | 🔲 Next |
| Services | `services/storage/temp.py` | 🔲 Next |

---

## Internal API Reference

All routes are under the prefix configured by `KYC_SERVICE_URL` on the Go side. The Python service mounts everything at `/internal/v1`.

### Health

```
GET /internal/v1/health
```
Process liveness probe. Always 200 if the process is running.

```json
{ "ok": true }
```

```
GET /internal/v1/models
```
Model readiness probe. Reports each model's load state and the active GPU.

```json
{
  "ok": true,
  "mtcnn":            true,
  "vggface2":         true,
  "dsnt":             true,
  "gpu":              true,
  "cuda":             "12.2",
  "model_version":    "vggface2-2026.05",
  "liveness_version": "liveness-trinity-v2"
}
```

### ID Card Smart-Crop

```
POST /internal/v1/id-card
Content-Type: multipart/form-data

file: <image bytes>   # JPEG or PNG, landscape phone capture
```

```json
{
  "ok":           true,
  "cropped_path": "tmp/cropped.jpg",
  "final_path":   "tmp/final.jpg",
  "keypoints":    [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
}
```

### Face Verification

```
POST /internal/v1/verify
Content-Type: multipart/form-data

selfie:   <image bytes>   # required
id_image: <image bytes>   # optional — falls back to last /id-card output
```

```json
{
  "ok":               true,
  "verified":         true,
  "score":            0.87,
  "model_version":    "vggface2-2026.05",
  "liveness_version": "liveness-trinity-v2",
  "internal_job_id":  ""
}
```

### Liveness Challenge

```
POST /internal/v1/challenge
Content-Type: multipart/form-data

frame:     <image bytes>              # single video frame
challenge: blink|orientation|emotion  # validated — 422 on invalid value
expected:  <string>                   # orientation label or emotion label; ignored for blink
```

```json
{
  "ok":     true,
  "passed": true,
  "result": "happy"
}
```

---

## Go Service API Reference

### Submit a KYC Job

```
POST /api/v1/kyc/submit
Content-Type: application/json
Idempotency-Key: <uuid>    # recommended for safe client retries
```

```json
{
  "user_id":      "usr_abc123",
  "country_code": "KE",
  "id_type":      "NATIONAL_ID",
  "id_number":    "12345678",
  "first_name":   "Amina",
  "last_name":    "Wanjiru",
  "tier":         "kyc_light"
}
```

Returns `202 Accepted` immediately. The job is queued for async processing.

```json
{
  "job_id":          "550e8400-e29b-41d4-a716-446655440000",
  "idempotency_key": "your-uuid-here",
  "status":          "pending",
  "user_id":         "usr_abc123",
  "submitted_at":    "2026-05-24T10:00:00Z",
  "message":         "KYC verification queued. Poll /status/{job_id} for updates."
}
```

Sending the same `Idempotency-Key` again returns `200` with `"replayed": true` and the original job ID — no duplicate job is created.

### Poll Job Status

```
GET /api/v1/kyc/status/{job_id}
```

```json
{
  "job_id":          "550e8400-...",
  "user_id":         "usr_abc123",
  "status":          "approved",
  "tier":            "kyc_light",
  "confidence":      0.87,
  "model_version":   "vggface2-2026.05",
  "processed_at":    "2026-05-24T10:00:05Z",
  "submitted_at":    "2026-05-24T10:00:00Z"
}
```

Status values: `pending` → `processing` → `approved | rejected | failed`

### Get Latest Status for a User

```
GET /api/v1/kyc/user/{user_id}/status
```

### Inspect Loaded Models (proxied from Python service)

```
GET /api/v1/kyc/models
```

### Health and Readiness

```
GET /healthz    # liveness  — always 200 if process is up
GET /readyz     # readiness — checks queue depth + inference health
GET /metrics    # Prometheus scrape endpoint
```

---

## Supported Countries and ID Types

| Country | National ID | Alien ID | Passport | Voter ID | Driving Licence |
|---|---|---|---|---|---|
| Kenya (KE) | ✅ | ✅ | ✅ | | |
| Nigeria (NG) | ✅ (NIN) | | ✅ | ✅ | |
| Ghana (GH) | ✅ | | ✅ | ✅ | ✅ |

---

## Configuration

### Go Service (`.env`)

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8080` | HTTP listen port |
| `KYC_SERVICE_URL` | `http://kyc-python:5000` | Python inference service base URL |
| `KYC_SERVICE_TIMEOUT` | `15s` | Per-request timeout to Python; must be less than `KYC_JOB_TIMEOUT` |
| `KYC_SERVICE_CONCURRENCY` | `20` | Max parallel calls to Python; tune down for single-GPU deployments |
| `DATABASE_DSN` | — | PostgreSQL connection string (required) |
| `KYC_WORKER_COUNT` | `20` | Goroutines processing jobs concurrently |
| `KYC_QUEUE_DEPTH` | `500` | Buffered channel depth |
| `KYC_MAX_RETRIES` | `3` | Hard retry budget per job (backpressure retries are separate) |
| `KYC_RETRY_BASE_DELAY` | `2s` | Exponential backoff base (2s → 4s → 8s) |
| `KYC_JOB_TIMEOUT` | `30s` | Per-job context deadline |

### Python Service (`.env`)

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | Uvicorn listen port |
| `WORKERS` | `1` | Uvicorn worker processes (keep at 1 on GPU — models are not fork-safe) |
| `LOG_LEVEL` | `info` | `debug` also enables `/docs` Swagger UI |
| `FROZEN_MODEL_PATH` | `model/frozen_model.pb` | DSNT TF frozen graph |
| `SHAPE_PREDICTOR_PATH` | `services/liveness_detection/landmarks/shape_predictor_68_face_landmarks.dat` | dlib 68-point predictor |
| `EMOTION_WEIGHTS_PATH` | `services/liveness_detection/landmarks/emotion_weights.pt` | Emotion classifier |
| `TMP_DIR` | `tmp` | Scratch directory for intermediate images |
| `FACE_VERIFICATION_THRESHOLD` | `0.6` | Cosine distance threshold for accept/reject |
| `MODEL_VERSION` | `vggface2-2026.05` | Included in every verification response for audit |
| `LIVENESS_VERSION` | `liveness-trinity-v2` | Included in every verification response for audit |

---

## Resilience Design

The Go service protects the Python inference service with three concentric layers:

```
Worker goroutine
    │
    ├── context.WithTimeout(15s)      ← kills a hung TF session
    │
    ├── semaphore (KYC_SERVICE_CONCURRENCY)
    │       └── returns ErrBackpressure immediately if full
    │           worker pool backs off without burning retry budget
    │
    └── circuit breaker
            ├── trips after 5 consecutive 5xx responses
            ├── stays open for 30s (fast-fails during that window)
            └── half-opens to probe with one request
```

Backpressure from the inference layer (semaphore full or breaker open) is treated as a soft retry with its own counter — up to 10 soft retries before the job is failed. This prevents a momentarily slow Python service from consuming the hard retry budget of valid jobs.

---

## Observability

### Prometheus Metrics

| Metric | Type | Description |
|---|---|---|
| `kyc_jobs_submitted_total` | Counter | Jobs accepted into the queue |
| `kyc_jobs_processed_total{status}` | Counter | Terminal outcomes by status |
| `kyc_inference_request_duration_seconds` | Histogram | Python service round-trip |
| `kyc_job_duration_seconds` | Histogram | End-to-end submission → terminal |
| `kyc_circuit_breaker_state` | Gauge | 0=closed 1=open 2=half-open |
| `kyc_worker_queue_length` | Gauge | Jobs waiting in the channel |
| `kyc_active_workers` | Gauge | Goroutines actively processing |
| `kyc_retries_total` | Counter | Hard retry attempts |
| `kyc_idempotency_replays_total` | Counter | Requests served from idempotency cache |
| `kyc_http_requests_total{method,path,status_code}` | Counter | HTTP RED |
| `kyc_http_request_duration_seconds{method,path}` | Histogram | HTTP latency |

### Alerts

| Alert | Severity | Condition |
|---|---|---|
| `KYCCircuitBreakerOpen` | critical | Breaker state = 1 (fires immediately) |
| `KYCCircuitBreakerHalfOpen` | warning | Breaker state = 2 for > 1 min |
| `KYCInferenceHighLatency` | warning | p95 inference > 10s for 2 min |
| `KYCQueueNearCapacity` | warning | Queue > 80% full for 1 min |
| `KYCWorkerStarvation` | critical | No active workers with jobs queued for 2 min |
| `KYCHighRetryRate` | warning | > 2 retries/sec for 3 min |
| `KYCHTTPErrorRate` | warning | 5xx rate > 5% for 2 min |

---

## Roadmap

### Immediate (pipelines and services layer)
- `app/pipelines/ekyc.py` — decode → rotate → keypoints → homography → write temp
- `app/pipelines/verification.py` — embeddings → cosine comparison
- `app/pipelines/liveness.py` — blink / orientation / emotion dispatch
- `app/services/id_card/` — preprocessing, TF inference, homography, cropper
- `app/services/face/verification.py` — VGGFace2 wrapper
- `app/services/storage/temp.py` — temp file lifecycle

### Near-term
- Object storage (MinIO/S3) — `SubmitRequest.SelfieURL` and `IDCardURL` already in the Go struct; Python service reads images by URL instead of multipart bytes
- ONNX export — `frozen_model.pb` → `dsnt.onnx`, VGGFace2 PyTorch → `vggface2.onnx`; removes TF and cuts the container from ~4GB to ~600MB

### Future
- Hyperledger Fabric enrollment trigger on `StatusApproved`
- H3 geospatial tagging of KYC jobs for regional analytics
- Multi-GPU routing — Go semaphore routes to specific Python replicas by GPU affinity
