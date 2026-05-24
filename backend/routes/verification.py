"""
app.routes.verification
───────────────────────
Mounted at  POST /internal/v1/verify

Compares a selfie against an ID card face using VGGFace2 embeddings.

id_image is optional: if omitted the pipeline falls back to the
temp file written by the most recent /id-card call on this instance.
This mirrors the original Flask behaviour while keeping the route clean.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from api.dependency import get_mtcnn, get_verif_model
from model.schemas import VerifyResponse
from pipelines.verification import verify_faces

log = logging.getLogger(__name__)

router = APIRouter()
# No prefix — /internal/v1 is applied by main.py's _register_routers.


@router.post(
    "/verify",
    response_model=VerifyResponse,
    summary="Verify selfie against ID card face",
    description=(
        "Compares the selfie embedding against the ID card face embedding "
        "using VGGFace2 cosine similarity. "
        "id_image is optional — omit it to reuse the output of the last /id-card call."
    ),
)
async def verify(
    selfie:   UploadFile = File(...,  description="Live selfie (JPEG or PNG)"),
    id_image: UploadFile = File(None, description="Cropped ID face — optional, falls back to last smart-crop"),
    mtcnn:       Annotated[object, Depends(get_mtcnn)]       = None,
    verif_model: Annotated[object, Depends(get_verif_model)] = None,
) -> VerifyResponse:
    """
    Pipeline: decode selfie → decode/load id image → extract embeddings → compare.

    id_image=None triggers the temp-file fallback path in the pipeline.
    Heavy inference pushed off the event loop via asyncio.to_thread.
    """
    selfie_bytes   = await selfie.read()
    id_image_bytes = await id_image.read() if id_image else None

    result = await asyncio.to_thread(
        verify_faces,
        selfie_bytes=selfie_bytes,
        id_image_bytes=id_image_bytes,
        mtcnn=mtcnn,
        verif_model=verif_model,
    )

    log.info(
        "verification complete",
        extra={"verified": result["verified"], "score": result["score"]},
    )
    return VerifyResponse(**result)
