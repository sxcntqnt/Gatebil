"""
app.pipelines.verification
───────────────────────────
Face verification pipeline — called by routes/verification.py.

Flow
----
    selfie bytes   +   id_image bytes (or temp slot fallback)
        │
        ▼
    decode both to BGR
        │
        ▼
    services.face.verification.verify_pair()
        │  MTCNN detect → VGGFace2 embed → cosine similarity
        ▼
    { verified, score, model_version, liveness_version }

The id_image is optional: if the caller omits it (id_image_bytes=None),
the pipeline reads the "id_face" slot written by id_detect.py, or falls
back to the "final" slot written by ekyc.py. This mirrors the original
Flask fallback behaviour while keeping the preference order explicit.
"""
from __future__ import annotations

import logging

import numpy as np

from core.config import settings
from core.exceptions import StorageError
from services.face_verification.face_verification import verify_pair
from utils import temp
from utils.image import bytes_to_bgr

log = logging.getLogger(__name__)

# Preferred fallback slot order when id_image_bytes is not supplied.
_ID_FALLBACK_SLOTS = ("id_face", "final")


def verify_faces(
    selfie_bytes: bytes,
    id_image_bytes: bytes | None,
    mtcnn,
    verif_model,
) -> dict:
    """
    Compare a selfie against an ID card face and return a verification result.

    Parameters
    ----------
    selfie_bytes : bytes
        Raw bytes of the live selfie.
    id_image_bytes : bytes or None
        Raw bytes of the ID face image.
        If None, the pipeline reads from temp storage (id_face → final slot).
    mtcnn : MTCNN
        Face detector from app.state.mtcnn.
    verif_model : InceptionResnetV1
        Embedding model from app.state.verif_model.

    Returns
    -------
    dict
        {
            "verified":         bool,
            "score":            float,   [0, 1]
            "model_version":    str,
            "liveness_version": str,
            "internal_job_id":  str,
        }
        Maps directly to VerifyResponse.

    Raises
    ------
    NoFaceDetectedError
        If MTCNN finds no face in either image.
    StorageError
        If id_image_bytes is None and no temp fallback exists.
    """
    # ── Decode selfie ─────────────────────────────────────────────────────
    selfie_bgr = bytes_to_bgr(selfie_bytes)

    # ── Decode or load ID face ────────────────────────────────────────────
    id_bgr: np.ndarray
    if id_image_bytes is not None:
        id_bgr = bytes_to_bgr(id_image_bytes)
        log.debug("verification: using uploaded id_image")
    else:
        id_bgr = _load_id_from_temp()
        log.debug("verification: using temp slot fallback")

    # ── Compare ───────────────────────────────────────────────────────────
    verified, score = verify_pair(selfie_bgr, id_bgr, mtcnn, verif_model)

    log.info(
        "verification complete",
        extra={
            "verified": verified,
            "score":    round(score, 4),
            "threshold": settings.face_verification_threshold,
        },
    )

    return {
        "verified":         verified,
        "score":            round(score, 4),
        "model_version":    settings.model_version,
        "liveness_version": settings.liveness_version,
        "internal_job_id":  "",
    }


def _load_id_from_temp() -> np.ndarray:
    """
    Try each fallback slot in preference order.
    Raises StorageError with a clear message if none exist.
    """
    for slot in _ID_FALLBACK_SLOTS:
        if temp.exists(slot):
            return temp.read_bgr(slot)

    raise StorageError(
        f"No ID image available. Tried slots: {_ID_FALLBACK_SLOTS}. "
        "Call /id-card (or /id-card + face extraction) before /verify."
    )
