# pipeline/ekyc.py
"""
pipeline.ekyc
=============

eKYC (electronic Know Your Customer) processing pipeline.

This module resolves the ModuleNotFoundError raised at startup:

    from pipeline.ekyc import process_id_card

Responsibilities
----------------
    - Accept a raw uploaded image (bytes, numpy array, or PIL Image).
    - Run ID card keypoint detection via IDDetector.
    - Validate that all four corners were detected above confidence threshold.
    - Compute the perspective transform to produce a rectified, top-down
      ID card crop.
    - Return a structured EKYCResult to the route handler.

The pipeline is stateless — the IDDetector is injected (or resolved from
the app state) rather than instantiated here. This keeps the pipeline
testable without a GPU and decoupled from the FastAPI request lifecycle.

Typical call site (routes/ekyc.py):
-------------------------------------
    from pipeline.ekyc import process_id_card

    @router.post('/ekyc/scan')
    async def scan_id(
        file: UploadFile,
        request: Request,
    ) -> JSONResponse:
        image_bytes = await file.read()
        result = await process_id_card(
            image_bytes,
            detector=request.app.state.id_detector,
        )
        if not result.success:
            raise HTTPException(400, detail=result.error)
        return JSONResponse(result.to_dict())

FastAPI lifespan integration:
------------------------------
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.id_detector = IDDetector.from_checkpoint(
            settings.ID_DETECTOR_CHECKPOINT
        )
        app.state.id_detector.warmup()
        yield
        app.state.id_detector.close()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

from app.models.id_detector import IDDetector, KeypointResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------

@dataclass
class EKYCResult:
    """
    Result returned by process_id_card().

    Attributes
    ----------
    success : bool
        True if the ID card was detected and rectified without errors.

    keypoints : KeypointResult or None
        Raw keypoint detection output. Always present even on failure,
        so callers can inspect confidence scores and debug detections.

    rectified_image : np.ndarray or None, shape (H, W, 3) uint8
        Perspective-corrected crop of the ID card in RGB.
        None if detection failed or all_valid was False.

    error : str or None
        Human-readable failure reason. None on success.

    metadata : dict
        Supplementary info: detection time, input resolution, etc.
    """
    success: bool
    keypoints: Optional[KeypointResult]
    rectified_image: Optional[np.ndarray]
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """
        Serializable representation for JSON API responses.

        Does not include rectified_image (binary); encode separately if needed.
        """
        return {
            'success':  self.success,
            'error':    self.error,
            'keypoints': self.keypoints.as_dict() if self.keypoints else None,
            'metadata':  self.metadata,
        }

    def __repr__(self) -> str:
        return f"EKYCResult(success={self.success}, error={self.error!r})"


# ---------------------------------------------------------------------------
# Output card dimensions
# ---------------------------------------------------------------------------

# Standard ID-1 card aspect ratio (ISO/IEC 7810): 85.6mm x 53.98mm ≈ 1.586:1
# The rectified crop is produced at this aspect ratio regardless of how the
# card appears in the input image.
_CARD_OUTPUT_W = 640
_CARD_OUTPUT_H = 404   # 640 / 1.586 ≈ 403.5


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_image(
    source: Union[bytes, np.ndarray, Image.Image, str],
) -> np.ndarray:
    """
    Normalize any supported input type to a uint8 RGB numpy array.

    Parameters
    ----------
    source : bytes, np.ndarray, PIL Image, or str (file path)

    Returns
    -------
    np.ndarray, shape (H, W, 3), dtype uint8, RGB channel order.

    Raises
    ------
    ValueError
        If the source cannot be decoded as an image.
    """
    if isinstance(source, bytes):
        arr = np.frombuffer(source, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(
                "Could not decode image from bytes. "
                "Ensure the uploaded file is a valid JPEG, PNG, or WebP."
            )
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            # Grayscale — convert to RGB.
            return cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        if source.shape[2] == 4:
            # RGBA — drop alpha channel.
            return cv2.cvtColor(source, cv2.COLOR_RGBA2RGB)
        return source.astype(np.uint8)

    if isinstance(source, Image.Image):
        return np.array(source.convert('RGB'), dtype=np.uint8)

    if isinstance(source, str):
        bgr = cv2.imread(source)
        if bgr is None:
            raise ValueError(f"Could not read image file: {source}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    raise ValueError(
        f"Unsupported image source type: {type(source)}. "
        "Pass bytes, np.ndarray, PIL Image, or a file path string."
    )


# ---------------------------------------------------------------------------
# Perspective rectification
# ---------------------------------------------------------------------------

def _rectify(
    image: np.ndarray,
    keypoints: KeypointResult,
) -> np.ndarray:
    """
    Apply perspective transform to produce a top-down ID card crop.

    The four corners detected by IDDetector (TL, TR, BR, BL) define the
    source quadrilateral. The destination is a rectangle of fixed output size.

    Parameters
    ----------
    image : np.ndarray (H, W, 3) uint8
    keypoints : KeypointResult
        Must have all_valid == True before calling.

    Returns
    -------
    np.ndarray (H_out, W_out, 3) uint8
        Rectified card crop in RGB.

    Raises
    ------
    ValueError
        If fewer than 4 valid corners are available.
    """
    src = keypoints.corner_points()
    if src is None:
        raise ValueError(
            "Cannot rectify: not all four corners are valid. "
            f"Valid mask: {keypoints.valid}, confidences: {keypoints.confidence}"
        )

    # Destination corners in output card space.
    # Order: TL, TR, BR, BL — must match keypoint_names order.
    dst = np.array([
        [0,                 0                ],
        [_CARD_OUTPUT_W - 1, 0               ],
        [_CARD_OUTPUT_W - 1, _CARD_OUTPUT_H - 1],
        [0,                 _CARD_OUTPUT_H - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    rectified = cv2.warpPerspective(
        image, M,
        (_CARD_OUTPUT_W, _CARD_OUTPUT_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return rectified


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def process_id_card(
    image: Union[bytes, np.ndarray, Image.Image, str],
    detector: IDDetector,
    require_all_keypoints: bool = True,
) -> EKYCResult:
    """
    Full eKYC ID card processing pipeline.

    Steps:
        1. Load and decode the input image.
        2. Run IDDetector to locate card corners.
        3. Validate keypoint confidence.
        4. Rectify the card via perspective transform.
        5. Return EKYCResult.

    Parameters
    ----------
    image : bytes, np.ndarray, PIL Image, or str
        Input image containing an ID card.

    detector : IDDetector
        Initialized and warmed-up detector instance.
        Typically injected from request.app.state.id_detector.

    require_all_keypoints : bool
        If True (default), reject detections where any corner is below the
        detector's confidence_threshold. Set False to return partial results
        for debugging.

    Returns
    -------
    EKYCResult
        .success is True only if detection + rectification both succeeded.
        Always check .success before using .rectified_image.

    Notes
    -----
    This function is synchronous. In FastAPI async routes, run it in a
    thread pool to avoid blocking the event loop:

        result = await asyncio.get_event_loop().run_in_executor(
            None, process_id_card, image_bytes, detector
        )
    """
    import time
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1: Load image.
    # ------------------------------------------------------------------
    try:
        image_np = _load_image(image)
    except ValueError as exc:
        log.warning("eKYC image load failed: %s", exc)
        return EKYCResult(
            success=False,
            keypoints=None,
            rectified_image=None,
            error=f"Image load error: {exc}",
        )

    h, w = image_np.shape[:2]

    # ------------------------------------------------------------------
    # Step 2: Keypoint detection.
    # ------------------------------------------------------------------
    try:
        keypoints = detector.detect(image_np)
    except Exception as exc:
        log.exception("eKYC detection failed")
        return EKYCResult(
            success=False,
            keypoints=None,
            rectified_image=None,
            error=f"Detection error: {exc}",
        )

    log.debug(
        "eKYC detection: %d/%d keypoints valid, confidences=%s",
        int(keypoints.valid.sum()), keypoints.n_keypoints,
        keypoints.confidence.tolist(),
    )

    # ------------------------------------------------------------------
    # Step 3: Validate confidence.
    # ------------------------------------------------------------------
    if require_all_keypoints and not keypoints.all_valid:
        invalid_names = [
            name for name, v in zip(keypoints.keypoint_names, keypoints.valid) if not v
        ]
        msg = (
            f"Low-confidence keypoints: {invalid_names}. "
            "Ensure the ID card is fully visible, well-lit, and not occluded."
        )
        log.info("eKYC validation failed: %s", msg)
        return EKYCResult(
            success=False,
            keypoints=keypoints,
            rectified_image=None,
            error=msg,
            metadata={
                'input_size': (w, h),
                'confidences': keypoints.confidence.tolist(),
                'elapsed_ms': round((time.perf_counter() - t0) * 1000, 2),
            },
        )

    # ------------------------------------------------------------------
    # Step 4: Perspective rectification.
    # ------------------------------------------------------------------
    try:
        rectified = _rectify(image_np, keypoints)
    except Exception as exc:
        log.exception("eKYC rectification failed")
        return EKYCResult(
            success=False,
            keypoints=keypoints,
            rectified_image=None,
            error=f"Rectification error: {exc}",
        )

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    log.info("eKYC pipeline complete in %s ms.", elapsed_ms)

    return EKYCResult(
        success=True,
        keypoints=keypoints,
        rectified_image=rectified,
        metadata={
            'input_size':       (w, h),
            'output_size':      (_CARD_OUTPUT_W, _CARD_OUTPUT_H),
            'confidences':      keypoints.confidence.tolist(),
            'elapsed_ms':       elapsed_ms,
        },
    )


async def process_id_card_async(
    image: Union[bytes, np.ndarray, Image.Image, str],
    detector: IDDetector,
    require_all_keypoints: bool = True,
) -> EKYCResult:
    """
    Async wrapper for process_id_card.

    Runs the synchronous pipeline in a thread pool executor so it does not
    block the FastAPI event loop. Use this in async route handlers.

    Example
    -------
        @router.post('/ekyc/scan')
        async def scan_id(file: UploadFile, request: Request):
            data = await file.read()
            result = await process_id_card_async(
                data,
                detector=request.app.state.id_detector,
            )
            ...
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        process_id_card,
        image,
        detector,
        require_all_keypoints,
    )
