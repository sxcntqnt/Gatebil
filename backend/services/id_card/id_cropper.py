"""
app.services.id_card.cropper
──────────────────────────────
End-to-end smart-crop for ID cards.

Orchestrates:
  1. preprocessing  — decode, rotate, resize
  2. id_detector    — DSNT keypoint detection
  3. homography     — perspective warp and crop
  4. temp storage   — write intermediates for downstream /verify

This is the only function pipelines/ekyc.py and pipelines/id_detect.py
need to call. Everything else is internal.
"""
from __future__ import annotations

import logging

import numpy as np

from app.models.id_detector import IDCardDetector
from app.services.id_card import preprocessing as prep
from app.services.id_card import homography as hom
from app.services.storage import temp

log = logging.getLogger(__name__)


def smart_crop(
    image_bytes: bytes,
    detector: IDCardDetector,
) -> dict:
    """
    Run the full ID card smart-crop pipeline.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of the uploaded ID card image.
    detector : IDCardDetector
        The DSNT TF model loaded in app.state.

    Returns
    -------
    dict
        {
            "cropped_path": str,
            "final_path":   str,
            "keypoints":    list[list[int]],
        }
        Matches SmartCropResponse schema exactly.
    """
    # ── 1. Decode & rotate ────────────────────────────────────────────────
    bgr_orig, pil_rotated = prep.decode_and_rotate(image_bytes)
    rw, rh = prep.scale_factors(bgr_orig)

    temp.write_bgr("original", bgr_orig)

    # ── 2. Prepare model input ────────────────────────────────────────────
    img_nd = prep.prepare_for_model(pil_rotated)

    # ── 3. DSNT keypoint detection ────────────────────────────────────────
    keypoints = detector.detect(img_nd)

    log.debug(
        "keypoints detected",
        extra={"pixels": keypoints.pixels.tolist()},
    )

    # ── 4. Homography warp + crop ─────────────────────────────────────────
    cropped_bgr, kp_px = hom.warp_and_crop(img_nd, keypoints.pixels)

    # ── 5. Write intermediates ────────────────────────────────────────────
    cropped_path = temp.write_bgr("cropped", cropped_bgr)

    final_bgr  = hom.scale_to_original(cropped_bgr, rw, rh)
    final_path = temp.write_bgr("final", final_bgr)

    log.info(
        "smart crop complete",
        extra={
            "cropped": str(cropped_path),
            "final":   str(final_path),
        },
    )

    return {
        "cropped_path": str(cropped_path),
        "final_path":   str(final_path),
        "keypoints":    kp_px.tolist(),
    }
