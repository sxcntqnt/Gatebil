"""
app.services.id_card.preprocessing
────────────────────────────────────
Image preparation for the DSNT keypoint model.

Phone cameras capture ID cards in landscape orientation. This module
rotates the image to portrait and prepares two representations:

  original  — full-resolution BGR array for scale-back after warp
  model_nd  — RGB uint8 array resized to IDCardDetector.MODEL_W × MODEL_H
              ready to feed directly to the TF session
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

from app.core.exceptions import ImageDecodeError
from app.models.id_detector import IDCardDetector

log = logging.getLogger(__name__)

# Target resolution the DSNT model was trained on.
MODEL_W = IDCardDetector.MODEL_W   # 600
MODEL_H = IDCardDetector.MODEL_H   # 800


def decode_and_rotate(image_bytes: bytes) -> tuple[np.ndarray, Image.Image]:
    """
    Decode raw image bytes and rotate 270° to correct phone landscape capture.

    Returns
    -------
    bgr_orig : np.ndarray
        Full-resolution BGR array (OpenCV convention).
    pil_rotated : PIL.Image.Image
        RGB PIL image after rotation, used for model preprocessing.
    """
    try:
        pil_img = Image.open(__import__("io").BytesIO(image_bytes))
    except Exception as exc:
        raise ImageDecodeError(f"PIL could not open image: {exc}") from exc

    pil_rotated = pil_img.rotate(270, expand=True)
    bgr_orig    = cv2.cvtColor(np.array(pil_rotated), cv2.COLOR_RGB2BGR)

    log.debug(
        "id card decoded",
        extra={"original_size": bgr_orig.shape[:2]},
    )
    return bgr_orig, pil_rotated


def prepare_for_model(pil_rotated: Image.Image) -> np.ndarray:
    """
    Resize the rotated PIL image to (MODEL_W, MODEL_H) for the DSNT model.

    Returns
    -------
    np.ndarray
        RGB uint8 array of shape (MODEL_H, MODEL_W, 3).
        Do NOT convert to BGR — the TF model expects RGB.
    """
    resized = pil_rotated.resize((MODEL_W, MODEL_H))
    return np.array(resized, dtype=np.uint8)


def scale_factors(bgr_orig: np.ndarray) -> tuple[float, float]:
    """
    Compute width/height scale factors from model resolution back to original.

    Used after the warp to resize the cropped region to original-image scale.

    Returns
    -------
    (rw, rh) : tuple[float, float]
        rw = original_width  / MODEL_W
        rh = original_height / MODEL_H
    """
    h, w = bgr_orig.shape[:2]
    return w / MODEL_W, h / MODEL_H
