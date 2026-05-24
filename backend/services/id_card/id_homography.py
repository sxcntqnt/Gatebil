"""
app.services.id_card.homography
────────────────────────────────
Perspective correction for ID card images using DSNT keypoints.

The DSNT model returns four corner keypoints of the ID card in pixel space.
This module computes a homography that maps those corners to an axis-aligned
rectangle, warps the image, and crops the result.

All functions are pure (no side effects, no file I/O) so they are easy to
test in isolation.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from app.core.exceptions import IDCardProcessingError
from app.models.id_detector import Keypoints

log = logging.getLogger(__name__)


def compute_target_corners(keypoints: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    """
    Compute the axis-aligned target rectangle from four detected corner keypoints.

    The keypoints come back from DSNT in (x, y) pixel coords at model resolution.
    We average adjacent corners to produce a clean rectangle.

    Parameters
    ----------
    keypoints : np.ndarray, shape (4, 2)
        Pixel-space corner coordinates [top-left, top-right, bottom-left, bottom-right].

    Returns
    -------
    new_kps : np.ndarray, shape (4, 2), float32
        Target rectangle corners.
    x1, y1, x2, y2 : float
        Bounding coordinates of the target rectangle.
    """
    if keypoints.shape != (4, 2):
        raise IDCardProcessingError(
            f"Expected keypoints shape (4, 2), got {keypoints.shape}"
        )

    x1 = (keypoints[0, 0] + keypoints[2, 0]) / 2.0
    y1 = (keypoints[0, 1] + keypoints[1, 1]) / 2.0
    x2 = (keypoints[1, 0] + keypoints[3, 0]) / 2.0
    y2 = (keypoints[2, 1] + keypoints[3, 1]) / 2.0

    if x2 <= x1 or y2 <= y1:
        raise IDCardProcessingError(
            f"Degenerate keypoints: x1={x1:.1f} x2={x2:.1f} y1={y1:.1f} y2={y2:.1f}. "
            "The ID card may not be visible in the frame."
        )

    new_kps = np.array(
        [[x1, y1], [x2, y1], [x1, y2], [x2, y2]],
        dtype=np.float32,
    )
    return new_kps, x1, y1, x2, y2


def warp_and_crop(
    img_rgb: np.ndarray,
    keypoints_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a homography warp and crop the ID card region.

    Parameters
    ----------
    img_rgb : np.ndarray
        RGB uint8 array at model resolution (MODEL_H × MODEL_W × 3).
    keypoints_px : np.ndarray, shape (4, 2)
        Detected corner keypoints in pixel space (from IDCardDetector).

    Returns
    -------
    cropped_bgr : np.ndarray
        BGR crop of the perspective-corrected ID card.
    keypoints_px : np.ndarray
        The input keypoints (passed through for storage in the response).
    """
    new_kps, x1, y1, x2, y2 = compute_target_corners(keypoints_px)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    H, status = cv2.findHomography(
        keypoints_px.astype(np.float32),
        new_kps,
        cv2.RANSAC,
        5.0,
    )
    if H is None:
        raise IDCardProcessingError(
            "Homography computation failed — not enough inlier correspondences."
        )

    inliers = int(status.sum()) if status is not None else 0
    log.debug("homography computed", extra={"inliers": inliers})

    h, w = img_bgr.shape[:2]
    resize_factor = w / (x2 - x1)
    new_w = int(w * resize_factor)
    new_h = int(h * resize_factor)

    warped  = cv2.warpPerspective(img_bgr, H, (new_w, new_h))
    cropped = warped[int(y1):int(y2), int(x1):int(x2)]

    if cropped.size == 0:
        raise IDCardProcessingError(
            "Crop region is empty after warp — keypoints may be outside image bounds."
        )

    return cropped, keypoints_px


def scale_to_original(
    cropped: np.ndarray,
    rw: float,
    rh: float,
) -> np.ndarray:
    """
    Scale the model-resolution crop back to original-image scale.

    Parameters
    ----------
    cropped : np.ndarray
        BGR crop at model resolution.
    rw, rh : float
        Scale factors from preprocessing.scale_factors().

    Returns
    -------
    np.ndarray
        BGR image at original-image scale.
    """
    dim = (
        int(cropped.shape[1] * rw),
        int(cropped.shape[0] * rh),
    )
    return cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
