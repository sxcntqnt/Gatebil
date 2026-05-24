"""
app.routes.liveness
────────────────────
Mounted at  POST /internal/v1/challenge

Evaluates a single video frame against one of three liveness challenges:
  blink       — eye aspect ratio drops below threshold
  orientation — head pose label matches expected ("left" | "right" | "straight")
  emotion     — predicted emotion label matches expected ("happy" | "neutral" | …)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, UploadFile

from api.dependency import get_liveness_detectors, LivenessDetectors
from model.schemas import ChallengeResponse
from pipelines.liveness import run_liveness

log = logging.getLogger(__name__)

router = APIRouter()
# No prefix — /internal/v1 is applied by main.py's _register_routers.

# Valid challenge types — validated by FastAPI before the handler runs.
ChallengeType = Literal["blink", "orientation", "emotion"]


@router.post(
    "/challenge",
    response_model=ChallengeResponse,
    summary="Evaluate a liveness challenge frame",
    description=(
        "Submit a single video frame and a challenge type. "
        "For 'blink' the expected field is ignored. "
        "For 'orientation' and 'emotion', expected must match the label "
        "the model should detect for the challenge to pass."
    ),
)
async def challenge(
    frame:     UploadFile    = File(...,  description="Single video frame (JPEG or PNG)"),
    challenge: ChallengeType = Form(...,  description="Challenge type: blink | orientation | emotion"),
    expected:  str           = Form("",   description="Expected label (orientation / emotion only)"),
    detectors: Annotated[LivenessDetectors, Depends(get_liveness_detectors)] = None,
) -> ChallengeResponse:
    """
    Pipeline: decode frame → run detector → compare result to expected.

    Using Literal["blink"|"orientation"|"emotion"] means FastAPI returns
    422 with a clear message before any inference code runs — no manual
    validation needed in the pipeline.

    Heavy dlib / PyTorch inference pushed off the event loop.
    """
    frame_bytes = await frame.read()

    result = await asyncio.to_thread(
        run_liveness,
        frame_bytes=frame_bytes,
        challenge=challenge,
        expected=expected,
        detectors=detectors,
    )

    log.info(
        "challenge evaluated",
        extra={
            "challenge": challenge,
            "passed":    result["passed"],
            "result":    result["result"],
        },
    )
    return ChallengeResponse(**result)
