"""
app.pipelines.verification
───────────────────────────
Face verification pipeline — called by routes/verification.py.

Flow
----
    selfie bytes + id_image bytes
        │
        ▼
    decode both to BGR
        │
        ▼
    services.face_verification.face_verification.verify_pair()
        │  MTCNN detect → VGGFace2 embed → cosine similarity
        ▼
    { verified, score, model_version, liveness_version }

id_image_bytes is required. The /verify route always rectifies the ID
card fresh per-request and passes the result here — no temp-slot
fallback, since that shared mutable state isn't safe under the worker
pool's concurrent request load.
"""
from __future__ import annotations

import logging

from core.config import settings
from services.face_verification.face_verification import verify_pair
from utils.image import bytes_to_bgr

log = logging.getLogger(__name__)


def verify_faces(
    selfie_bytes: bytes,
    id_image_bytes: bytes,
    mtcnn,
    verif_model,
) -> dict:
    """
    Compare a selfie against an ID card face and return a verification result.

    Parameters
    ----------
    selfie_bytes : bytes
        Raw bytes of the live selfie.
    id_image_bytes : bytes
        Raw bytes of the rectified ID card image.
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
    """
    selfie_bgr = bytes_to_bgr(selfie_bytes)
    id_bgr = bytes_to_bgr(id_image_bytes)

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
