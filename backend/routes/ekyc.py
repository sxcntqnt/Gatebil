"""
app.routes.ekyc
───────────────
Mounted at  POST /internal/v1/id-card

Accepts a raw ID card image and returns the smart-cropped face region
using the DSNT TF keypoint model + homography warp.

The cropped result is written to the temp store so the downstream
/verify endpoint can reference it without re-sending the image.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from api.dependency import get_id_detector
from model.id_detector import IDDetector
from model.schemas import SmartCropResponse
from services.ekyc import process_id_card

log = logging.getLogger(__name__)

router = APIRouter()
# No prefix — /internal/v1 is applied by main.py's _register_routers.


@router.post(
    "/id-card",
    response_model=SmartCropResponse,
    summary="Smart-crop ID card",
    description=(
        "Upload a raw ID card photo (landscape, phone capture). "
        "Returns paths to the perspective-corrected crop and the final scaled image, "
        "plus the raw corner keypoints for debugging."
    ),
)
async def upload_id_card(
    file: UploadFile = File(..., description="ID card image (JPEG or PNG)"),
    detector: Annotated[IDDetector, Depends(get_id_detector)] = None,
) -> SmartCropResponse:
    """
    Pipeline: decode → rotate → resize → DSNT keypoints → homography → write temp.

    Heavy numpy / cv2 / TF work is pushed off the event loop via asyncio.to_thread
    so FastAPI's async machinery is not blocked during inference.
    """
    image_bytes = await file.read()

    result = await asyncio.to_thread(
        process_id_card,
        image_bytes=image_bytes,
        detector=detector,
    )

    log.info("id-card processed", extra={"final_path": result["final_path"]})
    return SmartCropResponse(**result)
