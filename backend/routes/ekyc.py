"""
app.routes.ekyc
───────────────
Mounted at  POST /internal/v1/id-card

Standalone preview/debug endpoint: fetches an ID card image by URL and
returns the perspective-corrected crop. Not used by /verify anymore —
that route rectifies its own copy per-request rather than depending on
this endpoint's temp output, since shared temp state isn't safe under
concurrent requests.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Annotated

import cv2
from fastapi import APIRouter, Depends

from api.dependency import get_id_detector
from core.exceptions import IDCardProcessingError  
from model.id_detector import IDDetector
from model.schemas import SmartCropRequest, SmartCropResponse
from services.id_card.ekyc import process_id_card
from utils import temp

log = logging.getLogger(__name__)
router = APIRouter()
# No prefix — /internal/v1 is applied by main.py's _register_routers.


@router.post(
    "/id-card",
    response_model=SmartCropResponse,
    summary="Smart-crop ID card",
    description="Fetch an ID card image by URL and return the perspective-corrected crop.",
)
async def upload_id_card(
    body: SmartCropRequest,
    detector: Annotated[IDDetector, Depends(get_id_detector)] = None,
) -> SmartCropResponse:
    image_bytes = await temp.fetch_bytes(str(body.id_card_url))

    result = await asyncio.to_thread(
        process_id_card,
        image_bytes,
        detector,
    )

    if not result.success:
        raise IDCardProcessingError(f"ID card rectification failed: {result.error}")

    # rectified_image is RGB (per ekyc.py's _load_image contract);
    # temp.write_bgr expects BGR (cv2.imwrite convention) — convert first.
    bgr_image = cv2.cvtColor(result.rectified_image, cv2.COLOR_RGB2BGR)

    cropped_path = temp.write_bgr("cropped", bgr_image)
    final_path = temp.write_bgr("final", bgr_image)

    keypoints = [
        [int(x), int(y)]
        for x, y in result.keypoints.corner_points().tolist()
    ]

    log.info("id-card processed", extra={"final_path": str(final_path)})
    return SmartCropResponse(
        cropped_path=str(cropped_path),
        final_path=str(final_path),
        keypoints=keypoints,
    )
