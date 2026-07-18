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
                         │  main.py  (FastAPI + uvicorn)        │
                         │                                     │
                         │  /id-card   →  keypoints             │
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
                         └──────────────────┬──────────────────┘
                                            │
                                     Cloudflare R2
                              (selfie + ID card images,
                               fetched via presigned URLs)
```

The Go service is the only externally visible component. The Python service is an internal inference kernel — never exposed directly to clients.

Image bytes never cross the Go ↔ Python boundary. The SvelteKit frontend uploads selfie and ID card images directly to Cloudflare R2, then passes presigned GET URLs through Go to Python, which fetches the bytes itself.

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
├── main.py                         Entrypoint — lifespan, routers, middleware only
├── core/
│   ├── config.py                   Pydantic-settings, model paths, thresholds
│   ├── exceptions.py               Domain exception hierarchy + FastAPI handlers
│   └── loggin.py                   Structured JSON logging
├── api/
│   └── dependency.py               FastAPI dependencies for model singletons
├── model/
│   ├── id_detector.py              PyTorch ID card keypoint detector
│   └── schemas.py                  Pydantic request/response shapes (URL-based)
├── routes/
│   ├── health.py                   GET /health  GET /models
│   ├── ekyc.py                     POST /id-card   (JSON, R2 URL)
│   ├── verification.py             POST /verify    (JSON, R2 URLs)
│   └── liveness.py                 POST /challenge (multipart — live video frame)
├── tasks/                          Pipeline orchestration (decode → infer → format)
│   ├── dsnt.py                     Keypoint model task wrapper
│   ├── liveliness.py               Liveness challenge dispatch
│   └── verification.py             verify_faces(): decode → verify_pair → format
├── services/
│   ├── id_card/
│   │   ├── ekyc.py                 process_id_card(): decode → keypoints → homography → EKYCResult
│   │   ├── id_cropper.py
│   │   ├── id_homography.py
│   │   └── id_preprocessing.py
│   ├── face_detection/             MTCNN (facenet-pytorch, vendored)
│   ├── face_verification/
│   │   └── face_verification.py    extract_embedding / compare_faces / verify_pair
│   ├── liveness_detection/
│   │   ├── blink_detection.py
│   │   ├── emotion_prediction.py
│   │   └── face_orientation.py
│   └── verification_models/
│       ├── VGGFace.py
│       └── VGGFace2.py             VGGFace2.load_model()
└── utils/
    ├── image.py                    Decode, resize, distance helpers, bytes_to_bgr
    ├── temp.py                     Slot storage + fetch_bytes() for R2 URLs
    ├── distance.py
    ├── functions.py
    └── plot.py
```

---

## Current Build Status

### Go Service — Complete

| File | Status | Notes |
|---|---|---|
| `config/config.go` | ✅ | Smile ID removed; `KYC_SERVICE_URL`, `KYC_SERVICE_TIMEOUT`, `KYC_SERVICE_CONCURRENCY` added |
| `domain/kyc.go` | ✅ | `IdempotencyKey`, `ModelVersion`, `LivenessVersion`, `ErrBackpressure`, `SelfieURL`/`IDCardURL` on `KYCJob` |
| `kycclient/client.go` | ✅ | `KYCClient` interface; `PythonClient` with semaphore + circuit breaker; `Verify`/`SmartCrop` are JSON+URL, `Challenge` stays multipart |
| `usecase/kyc.go` | ✅ | Idempotency check before duplicate check; `GetStatusByUser`; `SelfieURL`/`IDCardURL` required on submit |
| `handler/http.go` | ✅ | `Idempotency-Key` header; `/models` proxy; `ErrBackpressure` → 503 + `Retry-After` |
| `repository/postgres.go` | ✅ | `FindByIdempotencyKey`, `SaveResult` (upsert), `model_version` column |
| `worker/pool.go` | ✅ | Backpressure retries do not burn the hard retry budget; `runInference` passes URLs straight from `KYCJob`, no object-store fetch on the Go side |
| `metrics/metrics.go` | ✅ | `InferenceLatency`, `CircuitBreakerState`, `IdempotencyReplays` |
| `cmd/server/main.go` | ✅ | `waitForInference()` startup gate; logs full model state on boot |
| `monitoring/alerts.yml` | ✅ | Six rules: circuit breaker, latency p95, queue, worker starvation, HTTP errors |

### Python Service — Core Complete

| Layer | File | Status |
|---|---|---|
| Entrypoint | `main.py` | ✅ Thin: lifespan + routers only |
| Core | `core/config.py` | ✅ Pydantic-settings, path validation at startup |
| Core | `core/exceptions.py` | ✅ Domain hierarchy + FastAPI handlers |
| Core | `core/loggin.py` | ✅ Structured JSON, silences noisy libs |
| Model | `model/id_detector.py` | ✅ PyTorch keypoint detector, `close()` on shutdown |
| Model | `model/schemas.py` | ✅ URL-based request schemas (`SmartCropRequest`, `VerifyRequest`) |
| API | `api/dependency.py` | ✅ Model singletons via `app.state`, `LivenessDetectors` bundle |
| Utils | `utils/image.py` | ✅ Decode, resize, cosine/euclidean distance, `bytes_to_bgr` |
| Utils | `utils/temp.py` | ✅ Slot storage + `fetch_bytes()` for presigned R2 URLs |
| Routes | `routes/health.py` | ✅ `/health` + `/models` |
| Routes | `routes/ekyc.py` | ✅ `/id-card` — JSON `{id_card_url}`, fetches from R2, standalone preview/debug endpoint |
| Routes | `routes/verification.py` | ✅ `/verify` — JSON `{selfie_url, id_card_url}`, rectifies fresh per request, no cached-fallback path |
| Routes | `routes/liveness.py` | ✅ `/challenge`, challenge type validated by `Literal`, stays multipart (live video frame) |
| Task | `tasks/verification.py` | ✅ `verify_faces()` — decode → `verify_pair` → format; temp-slot fallback removed |
| Service | `services/id_card/ekyc.py` | ✅ `process_id_card()` — decode → keypoints → homography → `EKYCResult` |
| Service | `services/face_verification/face_verification.py` | ✅ `extract_embedding` / `compare_faces` / `verify_pair` (BGR in, cosine similarity out) |
| Pipeline | `tasks/liveliness.py` | 🔲 Next |

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

Standalone preview/debug endpoint. Not used by `/verify`, which rectifies its own copy per request.

```
POST /internal/v1/id-card
Content-Type: application/json
```

```json
{
  "id_card_url": "https://<r2-presigned-url>"
}
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
Content-Type: application/json
```

```json
{
  "selfie_url":  "https://<r2-presigned-url>",
  "id_card_url": "https://<r2-presigned-url>"
}
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

Both `selfie_url` and `id_card_url` are required. The ID card is rectified fresh on every call — there is no fallback to a cached prior result, since the worker pool calls this endpoint concurrently across jobs and shared temp state isn't request-scoped.

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

This route stays multipart — it's a live video frame from the liveness SDK, not an object stored in R2.

---

## Go Service API Reference

### Submit a KYC Job

The client (SvelteKit frontend) uploads the selfie and ID card images directly to Cloudflare R2, then submits presigned GET URLs to Go — the Go service never receives image bytes.

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
  "tier":         "kyc_light",
  "selfie_url":   "https://<r2-presigned-url>",
  "id_card_url":  "https://<r2-presigned-url>"
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
| `WORKERS` | `1` | Uvicorn worker processes (keep at 1 on GPU — models are not fork-safe). Note: this limits multiprocessing, not async concurrency — FastAPI still serves multiple in-flight requests per worker, which is why `/verify` never relies on shared temp state |
| `LOG_LEVEL` | `info` | `debug` also enables `/docs` Swagger UI |
| `FROZEN_MODEL_PATH` | `model/frozen_model.pb` | Legacy TF keypoint model path, if still in use |
| `SHAPE_PREDICTOR_PATH` | `services/liveness_detection/landmarks/shape_predictor_68_face_landmarks.dat` | dlib 68-point predictor |
| `EMOTION_WEIGHTS_PATH` | `services/liveness_detection/landmarks/emotion_weights.pt` | Emotion classifier |
| `TMP_DIR` | `tmp` | Scratch directory for intermediate images |
| `FACE_VERIFICATION_THRESHOLD` | `0.6` | Cosine similarity threshold for accept/reject |
| `MODEL_VERSION` | `vggface2-2026.05` | Included in every verification response for audit |
| `LIVENESS_VERSION` | `liveness-trinity-v2` | Included in every verification response for audit |

The Python service holds no object-storage credentials of its own — it only ever fetches a presigned URL it's handed, via `utils/temp.fetch_bytes()`.

### Object Storage (Cloudflare R2)

R2 credentials and bucket config live in the **SvelteKit frontend repo's** `.env` — neither gatebill (Go) nor the Python inference service ever holds R2 credentials or constructs an R2 endpoint. They only ever receive short-lived presigned GET URLs.

| Variable | Description |
|---|---|
| `PRIVATE_R2_ACCOUNT_ID` | Cloudflare account ID. Used to build the S3-compatible endpoint: `https://<PRIVATE_R2_ACCOUNT_ID>.r2.cloudflarestorage.com`. This endpoint is only ever used server-side inside `src/lib/server/r2.ts` — it is never sent to the browser. |
| `PRIVATE_R2_ACCESS_KEY_ID` | R2 API access key |
| `PRIVATE_R2_SECRET_ACCESS_KEY` | R2 API secret key |
| `PRIVATE_R2_BUCKET` | Bucket name (private — not publicly readable) |

Flow: the SvelteKit `+page.server.ts` action uploads the selfie and ID card to the bucket via `PutObjectCommand`, then calls `getSignedUrl` to generate a presigned `GetObjectCommand` URL (default 1hr TTL). That URL — not the account ID, not the bucket name — is what actually gets sent to gatebill and, from there, to the Python service. Neither downstream service needs to know the R2 endpoint exists.

---

## Resilience Design

The Go service protects the Python inference service with three concentric layers:

```
Worker goroutine
    │
    ├── context.WithTimeout(15s)      ← kills a hung inference call
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

### Immediate
- `tasks/liveliness.py` — finish blink / orientation / emotion dispatch wiring for the `kyc_full` tier

### Recently completed
- **Object storage (Cloudflare R2)** — client uploads directly to R2 via the SvelteKit backend; presigned GET URLs flow through Go's `KYCJob` and `kycclient.VerifyRequest`/`SmartCropRequest` as strings, never as bytes; Python's `/id-card` and `/verify` fetch bytes server-side via `utils/temp.fetch_bytes()` instead of accepting multipart uploads
- **Concurrency-safe `/verify`** — removed the "fall back to last `/id-card` output" temp-slot pattern, since the worker pool calls Python concurrently and that shared state wasn't request-scoped; `/verify` now rectifies the ID card fresh on every call

### Future
- ONNX export — legacy TF keypoint model → ONNX, VGGFace2 PyTorch → `vggface2.onnx`; removes any remaining TF dependency and shrinks the container
- Hyperledger Fabric enrollment trigger on `StatusApproved`
- H3 geospatial tagging of KYC jobs for regional analytics
- Multi-GPU routing — Go semaphore routes to specific Python replicas by GPU affinity

### Citation
Please cite this paper, if using midv dataset, link for dataset provided in paper

    @article{DBLP:journals/corr/abs-1807-05786,
      author    = {Vladimir V. Arlazarov and
                   Konstantin Bulatov and
                   Timofey S. Chernov and
                   Vladimir L. Arlazarov},
      title     = {{MIDV-500:} {A} Dataset for Identity Documents Analysis and Recognition
                   on Mobile Devices in Video Stream},
      journal   = {CoRR},
      volume    = {abs/1807.05786},
      year      = {2018},
      url       = {http://arxiv.org/abs/1807.05786},
      archivePrefix = {arXiv},
      eprint    = {1807.05786},
      timestamp = {Mon, 13 Aug 2018 16:46:35 +0200},
      biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1807-05786},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
