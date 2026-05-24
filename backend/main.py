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


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load every model singleton into app.state before the first request,
    and cleanly release resources on shutdown.

    app.state is the only place models live — route handlers retrieve
    them via the dependency functions in app.api.dependency.
    """
    _load_models(app)
    log.info(
        "all models ready",
        extra={
            "model_version":    settings.model_version,
            "liveness_version": settings.liveness_version,
            "gpu":              torch.cuda.is_available(),
            "cuda":             torch.version.cuda,
        },
    )
    yield
    _unload_models(app)
    log.info("inference service shut down cleanly")


def _load_models(app: FastAPI) -> None:
    """
    Initialise every model and attach it to app.state.
    Import order matters: heavier models (VGGFace2, TF graph) load last
    so a failure in a lighter model (MTCNN, dlib) is reported first.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── MTCNN face detector ────────────────────────────────────────────────
    log.info("loading MTCNN …")
    from services.face_detection.mtcnn import MTCNN  # old path, kept until service layer migrates
    app.state.mtcnn = MTCNN(device=device)

    # ── VGGFace2 verification model ────────────────────────────────────────
    log.info("loading VGGFace2 …")
    from services.verification_models import VGGFace2
    app.state.verif_model = VGGFace2.load_model(device=device)

    # ── Liveness detectors (dlib + PyTorch) ───────────────────────────────
    log.info("loading liveness detectors …")
    from services.liveness_detection.blink_detection import BlinkDetector
    from services.liveness_detection.face_orientation import FaceOrientationDetector
    from services.liveness_detection.emotion_prediction import EmotionPredictor

    app.state.blink_detector  = BlinkDetector()
    app.state.orient_detector = FaceOrientationDetector()
    app.state.emotion_pred    = EmotionPredictor(device=device)

    # ── DSNT TF keypoint detector for ID card corners ─────────────────────
    log.info("loading IDCardDetector (TF frozen graph) …")
    from app.models.id_detector import IDCardDetector
    app.state.id_detector = IDCardDetector(settings.frozen_model_path)


def _unload_models(app: FastAPI) -> None:
    """Release resources that require explicit cleanup."""
    detector: object = getattr(app.state, "id_detector", None)
    if detector is not None:
        detector.close()  # closes the TF session

    # PyTorch models are GC'd; nothing explicit needed.
    # dlib predictors are C++ objects; Python GC handles them.


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
        # Disable the default /docs in prod; enable via env if needed.
        docs_url="/docs" if settings.log_level.lower() == "debug" else None,
        redoc_url=None,
    )

    # ── Exception handlers ─────────────────────────────────────────────────
    register_exception_handlers(app)

    # ── Middleware ─────────────────────────────────────────────────────────
    # CORS is permissive here because the Go service is the only caller and
    # the Python service is never exposed to the public internet.
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
    This is the only place route modules are imported — keeping the
    import graph flat and the startup sequence obvious.
    """
    from app.routes.health       import router as health_router
    from app.routes.ekyc         import router as ekyc_router
    from app.routes.verification import router as verification_router
    from app.routes.liveness     import router as liveness_router

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
        # Reload only makes sense in development; never in production.
        reload=False,
    )
