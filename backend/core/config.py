#core/config.py
"""
app.core.config
───────────────
Single source of truth for all runtime configuration.
Loaded once at import time; every module that needs a value imports `settings`.

Environment variables are read automatically by pydantic-settings.
No value is hard-coded anywhere else in the codebase.
"""

"""
app.core.config
────────────────
Centralized runtime configuration for the entire application.

Features:
- Environment-variable driven via pydantic-settings
- Safe filesystem path resolution using absolute project-relative paths
- Startup validation for required ML model assets
- Automatic temporary directory creation
- Cached singleton settings object
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================================
# Project Root Resolution
# ============================================================================
# config.py lives in:
#   backend/core/config.py
#
# parent       -> backend/core
# parent.parent -> backend
#
BASE_DIR = Path(__file__).resolve().parent.parent

# ML weights directory
WEIGHTS_DIR = BASE_DIR / "resources" / "weights"

# Temporary working directory
TMP_DIR = BASE_DIR / "tmp"


class Settings(BaseSettings):
    """
    Global application settings.

    Values may be overridden by environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # Server
    # =========================================================================
    host: str = Field(
        default="0.0.0.0",
        description="Bind address",
    )

    port: int = Field(
        default=5000,
        description="Uvicorn listen port",
    )

    workers: int = Field(
        default=1,
        description="Worker processes (keep at 1 on GPU systems)",
    )

    log_level: str = Field(
        default="info",
        description="Application logging level",
    )

    # =========================================================================
    # Model Paths
    # =========================================================================
    frozen_model_path: Path = Field(
        default=WEIGHTS_DIR / "frozen_model.pb",
        description="TensorFlow frozen graph for DSNT keypoint detection",
    )

    shape_predictor_path: Path = Field(
        default=WEIGHTS_DIR / "shape_predictor_68_face_landmarks.dat",
        description="dlib 68-point facial landmark predictor",
    )

    emotion_weights_path: Path = Field(
        default=WEIGHTS_DIR / "emotion_weights.pt",
        description="Emotion classifier PyTorch weights",
    )

    # =========================================================================
    # Storage
    # =========================================================================
    tmp_dir: Path = Field(
        default=TMP_DIR,
        description="Scratch directory for temporary images/files",
    )

    tmp_max_age_seconds: int = Field(
        default=300,
        description="Temporary file cleanup threshold",
    )

    # =========================================================================
    # Inference Thresholds
    # =========================================================================
    face_verification_threshold: float = Field(
        default=0.6,
        description="Cosine distance threshold for face verification",
    )

    liveness_blink_ear_threshold: float = Field(
        default=0.25,
        description="Blink EAR threshold",
    )

    # =========================================================================
    # Version Metadata
    # =========================================================================
    model_version: str = Field(
        default="vggface2-2026.05",
        description="Face model version identifier",
    )

    liveness_version: str = Field(
        default="liveness-trinity-v2",
        description="Liveness system version identifier",
    )

    # =========================================================================
    # API
    # =========================================================================
    api_prefix: str = Field(
        default="/internal/v1",
        description="Internal API route prefix",
    )

    # =========================================================================
    # Validators
    # =========================================================================
    @field_validator(
        "frozen_model_path",
        "shape_predictor_path",
        "emotion_weights_path",
        mode="after",
    )
    @classmethod
    def validate_model_paths(cls, v: Path) -> Path:
        """
        Ensure all required model assets exist at startup.
        """
        resolved = v.resolve()

        if not resolved.exists():
            raise ValueError(f"Model file not found: {resolved}")

        if not resolved.is_file():
            raise ValueError(f"Expected a file but got: {resolved}")

        return resolved

    @field_validator("tmp_dir", mode="after")
    @classmethod
    def ensure_tmp_dir(cls, v: Path) -> Path:
        """
        Create temp directory automatically if missing.
        """
        resolved = v.resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved


# ============================================================================
# Cached Singleton
# ============================================================================

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return singleton settings instance.
    """
    return Settings()


# Global settings object
settings: Settings = get_settings()
