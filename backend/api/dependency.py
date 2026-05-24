"""
app.api.dependency
──────────────────
FastAPI dependency functions for all model singletons.

Every model is stored in app.state during the lifespan startup and
retrieved here via a typed dependency. Route handlers never import
models directly — they declare a dependency and receive the object.

Pattern
-------
    @router.post("/verify")
    async def verify(
        verif_model: Annotated[VGGFace2Model, Depends(get_verif_model)],
        ...
    ):
        ...

If a model failed to initialise, get_* raises ModelNotReadyError → 503,
which the exception handler in core/exceptions.py serialises cleanly.
"""
from __future__ import annotations

from typing import Annotated

import torch
from fastapi import Depends, Request

from app.core.exceptions import ModelNotReadyError
from app.models.id_detector import IDCardDetector


# ── Generic state accessor ─────────────────────────────────────────────────────

def _require(request: Request, attr: str, label: str):
    """Pull an attribute from app.state; raise 503 if it is None or missing."""
    model = getattr(request.app.state, attr, None)
    if model is None:
        raise ModelNotReadyError(f"{label} is not initialised")
    return model


# ── Per-model dependencies ─────────────────────────────────────────────────────

def get_mtcnn(request: Request):
    """MTCNN face detector (facenet-pytorch)."""
    return _require(request, "mtcnn", "MTCNN")


def get_verif_model(request: Request):
    """VGGFace2 verification model."""
    return _require(request, "verif_model", "VGGFace2")


def get_id_detector(request: Request) -> IDCardDetector:
    """DSNT TF keypoint detector for ID card corners."""
    return _require(request, "id_detector", "IDCardDetector")


def get_blink_detector(request: Request):
    """dlib-based blink (EAR) detector."""
    return _require(request, "blink_detector", "BlinkDetector")


def get_orient_detector(request: Request):
    """Face orientation classifier."""
    return _require(request, "orient_detector", "FaceOrientationDetector")


def get_emotion_predictor(request: Request):
    """Emotion prediction model."""
    return _require(request, "emotion_pred", "EmotionPredictor")


# ── Convenience bundle for routes that need the full liveness stack ────────────

class LivenessDetectors:
    """Groups the three liveness detectors for routes that use all of them."""
    def __init__(
        self,
        blink:   Annotated[object, Depends(get_blink_detector)],
        orient:  Annotated[object, Depends(get_orient_detector)],
        emotion: Annotated[object, Depends(get_emotion_predictor)],
    ) -> None:
        self.blink   = blink
        self.orient  = orient
        self.emotion = emotion


def get_liveness_detectors(
    blink:   Annotated[object, Depends(get_blink_detector)],
    orient:  Annotated[object, Depends(get_orient_detector)],
    emotion: Annotated[object, Depends(get_emotion_predictor)],
) -> LivenessDetectors:
    ld = LivenessDetectors.__new__(LivenessDetectors)
    ld.blink   = blink
    ld.orient  = orient
    ld.emotion = emotion
    return ld


# ── GPU / device info (no failure — informational only) ───────────────────────

def get_device_info(request: Request) -> dict:
    """Returns GPU availability for the /models endpoint."""
    gpu  = torch.cuda.is_available()
    cuda = torch.version.cuda if gpu else None
    return {"gpu": gpu, "cuda": cuda}
