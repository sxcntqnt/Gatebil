#core/config.py
"""
app.core.config
───────────────
Single source of truth for all runtime configuration.
Loaded once at import time; every module that needs a value imports `settings`.

Environment variables are read automatically by pydantic-settings.
No value is hard-coded anywhere else in the codebase.
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ────────────────────────────────────────────────────────────
    host: str = Field("0.0.0.0", description="Bind address")
    port: int = Field(5000, description="Listen port")
    workers: int = Field(1, description="Uvicorn worker processes (1 on GPU)")
    log_level: str = Field("info", description="Uvicorn / application log level")

    # ── Model paths ───────────────────────────────────────────────────────
    # Resolved relative to the project root so Docker WORKDIR works cleanly.
    frozen_model_path: Path = Field(
        Path("../model/frozen_model.pb"),
        description="TF1 frozen protobuf for DSNT keypoint detection",
    )
    shape_predictor_path: Path = Field(
        Path("../services/liveness_detection/landmarks/shape_predictor_68_face_landmarks.dat"),
        description="dlib 68-point landmark predictor",
    )
    emotion_weights_path: Path = Field(
        Path("../services/liveness_detection/landmarks/emotion_weights.pt"),
        description="Emotion classifier weights",
    )

    # ── Storage ───────────────────────────────────────────────────────────
    tmp_dir: Path = Field(
        Path("tmp"),
        description="Scratch directory for intermediate images",
    )
    tmp_max_age_seconds: int = Field(
        300,
        description="Temp files older than this are eligible for cleanup",
    )

    # ── Inference thresholds ──────────────────────────────────────────────
    face_verification_threshold: float = Field(
        0.6,
        description="Cosine distance threshold for VGGFace2 accept/reject",
    )
    liveness_blink_ear_threshold: float = Field(
        0.25,
        description="Eye aspect ratio below which a blink is detected",
    )

    # ── Model versioning (included in every response for audit) ──────────
    model_version: str = Field(
        "vggface2-2026.05",
        description="Human-readable version tag for the face model",
    )
    liveness_version: str = Field(
        "liveness-trinity-v2",
        description="Human-readable version tag for the liveness stack",
    )

    # ── Internal API prefix ───────────────────────────────────────────────
    api_prefix: str = Field(
        "/internal/v1",
        description="All inference routes mount under this prefix",
    )

    @field_validator("frozen_model_path", "shape_predictor_path", "emotion_weights_path", mode="after")
    @classmethod
    def _path_must_exist(cls, v: Path) -> Path:
        # Validated at startup so a missing model file fails loudly,
        # not silently at first request.
        if not v.exists():
            raise ValueError(f"Model file not found: {v.resolve()}")
        return v

    @field_validator("tmp_dir", mode="after")
    @classmethod
    def _ensure_tmp(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance. Cached after first call."""
    return Settings()


# Module-level alias — most imports just do `from app.core.config import settings`.
settings: Settings = get_settings()
