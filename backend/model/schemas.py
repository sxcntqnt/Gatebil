"""
app.models.schemas
──────────────────
Pydantic v2 request and response schemas for every inference route.

These are the shapes the Go KYCClient deserialises — keep them in sync
with the structs in internal/kycclient/client.go.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ── Shared ─────────────────────────────────────────────────────────────────────

class BaseResponse(BaseModel):
    ok: bool = True


class ErrorResponse(BaseModel):
    ok: bool = False
    error: str


# ── Health / Models ────────────────────────────────────────────────────────────

class HealthResponse(BaseResponse):
    pass  # { "ok": true }


class ModelsResponse(BaseResponse):
    mtcnn:            bool
    vggface2:         bool
    dsnt:             bool
    gpu:              bool
    cuda:             str | None = None
    model_version:    str
    liveness_version: str


# ── ID-card smart-crop (/internal/v1/id-card) ─────────────────────────────────

class SmartCropResponse(BaseResponse):
    cropped_path: str
    final_path:   str
    keypoints:    list[list[int]]


# ── Face verification (/internal/v1/verify) ───────────────────────────────────

class VerifyResponse(BaseResponse):
    verified:         bool
    score:            float = Field(..., ge=0.0, le=1.0,
                                   description="Cosine similarity score")
    model_version:    str
    liveness_version: str
    internal_job_id:  str = Field(
        default="",
        description="Opaque reference for audit; populated when object-store path lands",
    )


# ── Liveness challenge (/internal/v1/challenge) ───────────────────────────────

class ChallengeResponse(BaseResponse):
    passed: bool
    result: str = Field(..., description="Detected value (blink bool, orientation label, emotion label)")
