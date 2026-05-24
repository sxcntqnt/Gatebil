# core/config.py
"""
app.core.config
───────────────
Single source of truth for all runtime configuration.
Loaded once at import time; every module that needs a value imports `settings`.

Environment variables are read automatically by pydantic-settings.
No value is hard-coded anywhere else in the codebase.

──────────────────────────────────────────────────────────────────────
CHANGELOG
──────────────────────────────────────────────────────────────────────

- Replaced `frozen_model_path` (TensorFlow .pb frozen graph) with
  `id_detector_checkpoint` (PyTorch .pt checkpoint).

  Old:
      frozen_model_path: Path = WEIGHTS_DIR / "frozen_model.pb"

  New:
      id_detector_checkpoint: Path = WEIGHTS_DIR / "id_detector.pt"

  The .pb file is no longer loaded anywhere in the codebase.
  The validator list is updated accordingly — frozen_model_path is
  removed so startup does not fail if the old .pb file is absent.

- `id_detector_version` metadata field added alongside the existing
  model_version / liveness_version fields, so checkpoint provenance
  is visible in the /health response and startup logs.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================================
# Project Root Resolution
# ============================================================================
# config.py lives at:
#   backend/core/config.py
#
#   .parent       -> backend/core/
#   .parent.parent -> backend/
#
BASE_DIR = Path(__file__).resolve().parent.parent

# All ML weight files live under resources/weights/.
WEIGHTS_DIR = BASE_DIR / "resources" / "weights"

# Scratch directory for temporary images and working files.
TMP_DIR = BASE_DIR / "tmp"


class Settings(BaseSettings):
    """
    Global application settings.

    All fields can be overridden via environment variables or a .env file.
    Field names map 1:1 to env var names (case-insensitive).

    Example .env:
        ID_DETECTOR_CHECKPOINT=/mnt/models/id_detector_v2.pt
        LOG_LEVEL=debug
        PORT=8080
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
        description="Bind address for uvicorn.",
    )

    port: int = Field(
        default=5000,
        description="Uvicorn listen port.",
    )

    workers: int = Field(
        default=1,
        description=(
            "Uvicorn worker processes. "
            "Keep at 1 on GPU systems — multiple workers each allocate VRAM, "
            "and PyTorch models are not fork-safe under CUDA."
        ),
    )

    log_level: str = Field(
        default="info",
        description="Application logging level (debug / info / warning / error).",
    )

    # =========================================================================
    # Model Paths
    # =========================================================================

    id_detector_checkpoint: Path = Field(
        default=WEIGHTS_DIR / "id_detector.pt",
        description=(
            "PyTorch checkpoint for the DSNT-based ID card keypoint detector. "
            "Must contain keys: model_state_dict, keypoint_names, input_size. "
            "Replaces the deprecated frozen_model.pb TF1 frozen graph."
        ),
    )

    shape_predictor_path: Path = Field(
        default=WEIGHTS_DIR / "shape_predictor_68_face_landmarks.dat",
        description="dlib 68-point facial landmark predictor (.dat file).",
    )

    emotion_weights_path: Path = Field(
        default=WEIGHTS_DIR / "emotion_weights.pt",
        description="PyTorch weights for the emotion classification head.",
    )

    # =========================================================================
    # Storage
    # =========================================================================

    tmp_dir: Path = Field(
        default=TMP_DIR,
        description="Scratch directory for temporary images and working files.",
    )

    tmp_max_age_seconds: int = Field(
        default=300,
        description=(
            "Maximum age in seconds before a temporary file is eligible "
            "for cleanup. Used by the background cleanup task."
        ),
    )

    # =========================================================================
    # Inference Thresholds
    # =========================================================================

    face_verification_threshold: float = Field(
        default=0.6,
        description=(
            "Cosine distance threshold for face verification. "
            "Pairs below this distance are considered the same identity."
        ),
    )

    liveness_blink_ear_threshold: float = Field(
        default=0.25,
        description=(
            "Eye Aspect Ratio (EAR) threshold for blink detection. "
            "Values below this are classified as a closed eye."
        ),
    )

    id_detector_confidence_threshold: float = Field(
        default=0.15,
        description=(
            "Minimum DSNT heatmap peak confidence for a keypoint to be "
            "considered valid. Detections below this are flagged as uncertain "
            "and will cause the eKYC pipeline to reject the image."
        ),
    )

    # =========================================================================
    # Version Metadata
    # =========================================================================

    model_version: str = Field(
        default="vggface2-2026.05",
        description="Face verification model version identifier.",
    )

    liveness_version: str = Field(
        default="liveness-trinity-v2",
        description="Liveness detection system version identifier.",
    )

    id_detector_version: str = Field(
        default="id-dsnt-v1",
        description=(
            "ID card keypoint detector version identifier. "
            "Surfaced in /health and startup logs for checkpoint traceability."
        ),
    )

    # =========================================================================
    # API
    # =========================================================================

    api_prefix: str = Field(
        default="/internal/v1",
        description="Route prefix for all internal API endpoints.",
    )

    # =========================================================================
    # Validators
    # =========================================================================

    @field_validator(
        "id_detector_checkpoint",
        "shape_predictor_path",
        "emotion_weights_path",
        mode="after",
    )
    @classmethod
    def validate_model_paths(cls, v: Path) -> Path:
        """
        Verify that every required model asset exists on disk at startup.

        Failing here at import time (before any requests are accepted) is
        intentional — a missing weight file is a hard deployment error, not a
        recoverable runtime condition.

        Raises
        ------
        ValueError
            If the resolved path does not exist or is not a regular file.
        """
        resolved = v.resolve()

        if not resolved.exists():
            raise ValueError(
                f"Model file not found: {resolved}\n"
                "Check that the file exists and that WEIGHTS_DIR is correct. "
                "Override the path via environment variable if the file is "
                "stored outside the default resources/weights/ directory."
            )

        if not resolved.is_file():
            raise ValueError(
                f"Expected a file but got a directory: {resolved}"
            )

        return resolved

    @field_validator("tmp_dir", mode="after")
    @classmethod
    def ensure_tmp_dir(cls, v: Path) -> Path:
        """
        Create the temporary directory automatically if it does not exist.
        Uses parents=True so intermediate directories are created as needed.
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
    Return the singleton Settings instance.

    lru_cache(maxsize=1) ensures the .env file is read and all validators
    run exactly once per process lifetime. Tests that need different settings
    should call get_settings.cache_clear() before constructing a new instance.
    """
    return Settings()


# Module-level alias imported by all other modules.
settings: Settings = get_settings()
