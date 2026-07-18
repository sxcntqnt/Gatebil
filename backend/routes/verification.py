"""
app.routes.verification
───────────────────────
Mounted at  POST /internal/v1/verify

Fetches selfie + ID card by URL, rectifies the ID card fresh on every
call, then compares faces. No reliance on cached temp-slot fallback —
the worker pool calls this concurrently across jobs, and that fallback
reads whatever the last request happened to write.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Annotated

import cv2
from fastapi import APIRouter, Depends

from api.dependency import get_mtcnn, get_verif_model, get_id_detector
from core.exceptions import StorageError
from model.id_detector import IDDetector
from model.schemas import VerifyRequest, VerifyResponse
from services.id_card.ekyc import process_id_card  # NOTE: confirm this path — routes/ekyc.py
                                            # imports from services.id_card.ekyc instead;
                                            # only one of these matches your real layout
from tasks.verification import verify_faces
from utils import temp

log = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/verify",
    response_model=VerifyResponse,
    summary="Verify selfie against ID card face",
    description=(
        "Fetches selfie and ID card by URL, rectifies the card fresh, "
        "then compares embeddings via VGGFace2 cosine similarity."
    ),
)
async def verify(
    body: VerifyRequest,
    mtcnn:       Annotated[object, Depends(get_mtcnn)]           = None,
    verif_model: Annotated[object, Depends(get_verif_model)]     = None,
    detector:    Annotated[IDDetector, Depends(get_id_detector)] = None,
) -> VerifyResponse:
    selfie_bytes, id_card_bytes = await asyncio.gather(
        temp.fetch_bytes(str(body.selfie_url)),
        temp.fetch_bytes(str(body.id_card_url)),
    )

    ekyc_result = await asyncio.to_thread(
        process_id_card,
        id_card_bytes,
        detector,
    )
    if not ekyc_result.success:
        raise StorageError(f"ID card rectification failed: {ekyc_result.error}")

    # rectified_image is RGB uint8 — re-encode to JPEG bytes since
    # verify_faces/bytes_to_bgr expects an encoded image, not a raw array.
    id_face_bytes = _encode_rgb_to_jpeg_bytes(ekyc_result.rectified_image)

    result = await asyncio.to_thread(
        verify_faces,
        selfie_bytes=selfie_bytes,
        id_image_bytes=id_face_bytes,
        mtcnn=mtcnn,
        verif_model=verif_model,
    )

    log.info(
        "verification complete",
        extra={"verified": result["verified"], "score": result["score"]},
    )
    return VerifyResponse(**result)


def _encode_rgb_to_jpeg_bytes(rgb_image) -> bytes:
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise StorageError("Failed to re-encode rectified ID card image")
    return buf.tobytes()
