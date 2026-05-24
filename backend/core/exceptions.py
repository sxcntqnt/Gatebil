"""
app.core.exceptions
───────────────────
Domain exception hierarchy and the FastAPI exception handlers that
translate them into structured JSON responses.

Register all handlers in main.py via register_exception_handlers(app).
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


# ── Domain exceptions ──────────────────────────────────────────────────────────

class KYCBaseError(Exception):
    """Root for all inference-service errors."""
    status_code: int = 500
    detail: str = "Internal inference error"

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.__class__.detail
        super().__init__(self.detail)


class ModelNotReadyError(KYCBaseError):
    """Raised when a model singleton has not been initialised yet."""
    status_code = 503
    detail = "Inference model is not ready"


class NoFaceDetectedError(KYCBaseError):
    """Raised when MTCNN finds no face in the supplied image."""
    status_code = 422
    detail = "No face detected in the image"


class MultipleFacesError(KYCBaseError):
    """Raised when MTCNN finds more than one face where exactly one is required."""
    status_code = 422
    detail = "Multiple faces detected; exactly one face is required"


class ImageDecodeError(KYCBaseError):
    """Raised when an uploaded file cannot be decoded as an image."""
    status_code = 400
    detail = "Could not decode the uploaded file as an image"


class IDCardProcessingError(KYCBaseError):
    """Raised when keypoint detection or homography fails on an ID card."""
    status_code = 422
    detail = "ID card processing failed"


class LivenessError(KYCBaseError):
    """Raised when a liveness challenge frame cannot be evaluated."""
    status_code = 422
    detail = "Liveness evaluation failed"


class StorageError(KYCBaseError):
    """Raised when temp file read/write fails."""
    status_code = 500
    detail = "Temporary storage operation failed"


# ── FastAPI exception handlers ─────────────────────────────────────────────────

def _kyc_error_handler(request: Request, exc: KYCBaseError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "error": exc.detail},
    )


def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Never let a raw traceback reach the client.
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": "An unexpected error occurred"},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to the FastAPI application instance."""
    app.add_exception_handler(KYCBaseError, _kyc_error_handler)          # type: ignore[arg-type]
    app.add_exception_handler(Exception, _unhandled_error_handler)        # type: ignore[arg-type]
