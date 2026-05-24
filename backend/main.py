# main.py
"""
app.main
────────
FastAPI application entrypoint.

This file is ONLY allowed to:
  1. Configure logging
  2. Define the lifespan (load models → serve → unload)
  3. Create the FastAPI app
  4. Register exception handlers
  5. Add middleware
  6. Include routers
  7. Expose a uvicorn __main__ block

No inference logic, no model code, no business rules belong here.
Every import from app.services.* or app.pipelines.* lives behind
a function call so the import graph stays readable.

──────────────────────────────────────────────────────────────────────
CHANGELOG
──────────────────────────────────────────────────────────────────────

- Replaced TF1 IDCardDetector with PyTorch IDDetector.
    Previously:
        from app.models.id_detector import IDCardDetector
        app.state.id_detector = IDCardDetector(settings.frozen_model_path)
    Now:
        from app.models.id_detector import IDDetector
        app.state.id_detector = IDDetector.from_checkpoint(
            settings.id_detector_checkpoint
        )
    This eliminates the TensorFlow runtime dependency, the frozen-graph
    load (~3s), and all GPU/CUDA driver warnings emitted by TF on CPU
    machines. The TF session is no longer opened and no longer needs
    to be explicitly closed in _unload_models.

- Device resolution centralised.
    `device` is resolved once at the top of _load_models and passed
    to every model constructor. Added Apple MPS (M-series) fallback
    between CUDA and CPU so the service runs natively on dev machines.

- Warmup call added after IDDetector load.
    Runs 3 dummy forward passes to compile CUDA kernels before the
    first real request bears that cost.

- _unload_models updated.
    IDDetector.close() moves the model to CPU and clears the CUDA
    cache. PyTorch models have no session to close; the comment
    referencing a TF session is removed.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.exceptions import register_exception_handlers
from core.loggin import configure_logging

log = logging.getLogger(__name__)


# ── Device resolution ──────────────────────────────────────────────────────────

def _resolve_device() -> torch.device:
    """
    Pick the best available compute device.

    Priority:
        CUDA  → NVIDIA GPU, full production path.
        MPS   → Apple Silicon, native acceleration on dev machines.
        CPU   → fallback; all models support CPU inference.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load every model singleton into app.state before the first request,
    and cleanly release resources on shutdown.

    app.state is the only place models live — route handlers retrieve
    them via the dependency functions in app.api.dependency.
    """
    device = _resolve_device()
    _load_models(app, device)
    log.info(
        "all models ready",
        extra={
            "model_version":    settings.model_version,
            "liveness_version": settings.liveness_version,
            "device":           str(device),
            "gpu":              torch.cuda.is_available(),
            "cuda_version":     torch.version.cuda or "n/a",
        },
    )
    yield
    _unload_models(app)
    log.info("inference service shut down cleanly")


def _load_models(app: FastAPI, device: torch.device) -> None:
    """
    Initialise every model and attach it to app.state.

    Import order: lighter models first so a failure in a lightweight model
    (MTCNN, dlib) is reported before spending time loading heavy ones
    (VGGFace2, IDDetector backbone).
    """

    # ── MTCNN face detector ────────────────────────────────────────────────
    log.info("loading MTCNN …")
    from services.face_detection.mtcnn import MTCNN
    app.state.mtcnn = MTCNN(device=device)

    # ── Liveness detectors (dlib + PyTorch) ───────────────────────────────
    # Loaded before VGGFace2 / IDDetector: dlib is lightweight and
    # its failure should not be shadowed by a heavier model timeout.
    log.info("loading liveness detectors …")
    from services.liveness_detection.blink_detection import BlinkDetector
    from services.liveness_detection.face_orientation import FaceOrientationDetector
    from services.liveness_detection.emotion_prediction import EmotionPredictor

    app.state.blink_detector  = BlinkDetector()
    app.state.orient_detector = FaceOrientationDetector()
    app.state.emotion_pred    = EmotionPredictor(device=device)

    # ── VGGFace2 verification model ────────────────────────────────────────
    log.info("loading VGGFace2 …")
    from services.verification_models import VGGFace2
    app.state.verif_model = VGGFace2.load_model(device=device)

    # ── PyTorch ID card keypoint detector ─────────────────────────────────
    # Replaces the previous TF1 frozen-graph IDCardDetector.
    # Loads a checkpoint containing IDCardModel weights + metadata.
    # warmup() pre-compiles CUDA kernels so the first real request
    # does not bear the kernel compilation overhead (~200-400ms on GPU).
    log.info("loading IDDetector (PyTorch) …")
    from app.models.id_detector import IDDetector
    app.state.id_detector = IDDetector.from_checkpoint(
        checkpoint_path=settings.id_detector_checkpoint,
        device=device,
    )
    app.state.id_detector.warmup()


def _unload_models(app: FastAPI) -> None:
    """
    Release resources that require explicit cleanup on shutdown.

    IDDetector.close() moves the model to CPU, deletes the module,
    and calls torch.cuda.empty_cache() to release VRAM immediately.

    All other PyTorch models (VGGFace2, EmotionPredictor, MTCNN) are
    released when app.state is garbage-collected — no explicit step needed.

    dlib C++ predictors are managed by Python GC automatically.
    """
    detector: object = getattr(app.state, "id_detector", None)
    if detector is not None:
        detector.close()


# ── Application factory ────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Build and return the configured FastAPI application.
    Separated from module level so tests can call create_app() cleanly.
    """
    configure_logging(settings.log_level.upper())

    app = FastAPI(
        title="eKYC Inference Service",
        version=settings.model_version,
        description=(
            "Internal inference API. All routes are under "
            f"{settings.api_prefix} and are not exposed publicly."
        ),
        lifespan=lifespan,
        # /docs is only available in debug mode; never exposed in production.
        docs_url="/docs" if settings.log_level.lower() == "debug" else None,
        redoc_url=None,
    )

    # ── Exception handlers ─────────────────────────────────────────────────
    register_exception_handlers(app)

    # ── Middleware ─────────────────────────────────────────────────────────
    # CORS is permissive: the Go service is the only caller and this service
    # is never exposed directly to the public internet.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST", "GET"],
        allow_headers=["Content-Type"],
    )

    # ── Routers ────────────────────────────────────────────────────────────
    _register_routers(app)

    return app


def _register_routers(app: FastAPI) -> None:
    """
    Mount every route blueprint under the configured API prefix.
    This is the only place route modules are imported, keeping the
    import graph flat and the startup sequence obvious.
    """
    from routes.health       import router as health_router
    from routes.ekyc         import router as ekyc_router
    from routes.verification import router as verification_router
    from routes.liveness     import router as liveness_router

    prefix = settings.api_prefix  # "/internal/v1"

    app.include_router(health_router,       prefix=prefix, tags=["health"])
    app.include_router(ekyc_router,         prefix=prefix, tags=["id-card"])
    app.include_router(verification_router, prefix=prefix, tags=["verification"])
    app.include_router(liveness_router,     prefix=prefix, tags=["liveness"])


# ── Module-level app instance (for uvicorn / gunicorn) ────────────────────────
app = create_app()


# ── Direct run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
        reload=False,  # never reload in production
    )
