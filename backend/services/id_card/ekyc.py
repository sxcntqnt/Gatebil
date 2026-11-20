# services/id_card/ekyc.py
"""
services.id_card.ekyc
=====================

eKYC (electronic Know Your Customer) ID-card processing pipeline.

Responsibilities
----------------
    - Decode an uploaded image (bytes, numpy array, PIL Image or path).
    - Run ID card keypoint detection via IDDetector.
    - Order the four detected corners *geometrically* (TL, TR, BR, BL),
      so the pipeline does not depend on the checkpoint's channel order.
    - Validate the quadrilateral (convex, sane area, angles, aspect ratio)
      before warping, and say *which stage* failed when something does.
    - Produce a landscape, perspective-corrected card crop.
    - Optionally write an annotated debug overlay (see KYC_DEBUG_DIR).

What changed versus the previous version
----------------------------------------
    1. Corners are ordered from their positions, not trusted by channel.
       The old TF model emitted TL, TR, BL, BR while this module assumed
       TL, TR, BR, BL; a swapped pair produces a self-intersecting quad
       (see the logged coords_norm). Geometric ordering makes the result
       independent of that convention.
    2. A geometry gate (`validate_quad`) rejects non-convex, tiny, huge,
       badly-angled or wrongly-proportioned quads before cv2 warps them.
    3. The quad is rotated so its long side is the top edge, so a card
       held sideways still rectifies to a landscape crop.
    4. Failures carry a `stage` (decode / detect / confidence / geometry /
       warp) and the error text is prefixed with it.
    5. On a confidence failure the geometry verdict is still computed and
       reported in metadata, so you can tell "model unsure" apart from
       "model confidently wrong".
    6. `orient_card_by_face` fixes the remaining 180-degree ambiguity by
       asking MTCNN which orientation contains a face.

PII warning
-----------
    KYC_DEBUG_DIR, when set, makes this module write images of ID cards to
    disk. Use it only on a dev machine with test cards, never in production.

Torch is only imported for type checking, so the geometry helpers can be
imported and unit-tested without a GPU stack.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover - typing only
    from model.id_detector import IDDetector, KeypointResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stages (reported in EKYCResult.stage and prefixed onto error messages)
# ---------------------------------------------------------------------------

STAGE_DECODE = "decode"
STAGE_DETECT = "detect"
STAGE_CONFIDENCE = "confidence"
STAGE_GEOMETRY = "geometry"
STAGE_WARP = "warp"


# ---------------------------------------------------------------------------
# Output card dimensions and geometry-gate limits
# ---------------------------------------------------------------------------

# ISO/IEC 7810 ID-1: 85.6 x 53.98 mm, about 1.586:1.
_CARD_OUTPUT_W = 640
_CARD_OUTPUT_H = 404   # 640 / 1.586 = 403.5

# Fraction of the frame the card quad may cover.
_MIN_AREA_FRAC = 0.08
_MAX_AREA_FRAC = 0.98

# Measured from edge lengths of the (landscape-normalised) quad. Perspective
# foreshortening skews this, so the range is deliberately loose around 1.586
# (passport data pages are about 1.42, ID-1 cards 1.586).
_MIN_ASPECT = 1.15
_MAX_ASPECT = 2.20

# Interior angle limits in degrees. A card seen at a steep angle can reach
# roughly 50-130; anything outside 35-145 is almost certainly a bad quad.
_MIN_ANGLE_DEG = 35.0
_MAX_ANGLE_DEG = 145.0

# Opposite edges may differ by at most this factor (perspective).
_MAX_OPPOSITE_EDGE_RATIO = 2.5

# Corners may sit slightly outside the frame (card partly cropped).
_FRAME_MARGIN_FRAC = 0.05


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
        True if the card was detected, validated and rectified.

    keypoints : KeypointResult or None
        Raw detector output (channel order, original-image pixels).

    rectified_image : np.ndarray or None, shape (H, W, 3) uint8
        Perspective-corrected, landscape card crop in RGB.

    error : str or None
        Failure reason, prefixed with the stage, e.g. "[geometry] ...".

    metadata : dict
        Timing, sizes, confidences and the geometry verdict.

    stage : str or None
        Stage that failed (STAGE_*). None on success.

    quad : np.ndarray or None, shape (4, 2) float32
        Corners ordered TL, TR, BR, BL (landscape-normalised), in
        original-image pixels. Use this, not keypoints.coords, for display.
    """

    success: bool
    keypoints: Optional["KeypointResult"]
    rectified_image: Optional[np.ndarray]
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    stage: Optional[str] = None
    quad: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Serializable representation for JSON API responses."""
        return {
            "success": self.success,
            "error": self.error,
            "stage": self.stage,
            "keypoints": self.keypoints.as_dict() if self.keypoints else None,
            "quad": self.quad.tolist() if self.quad is not None else None,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"EKYCResult(success={self.success}, stage={self.stage!r}, error={self.error!r})"


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_image(source: Union[bytes, np.ndarray, Image.Image, str]) -> np.ndarray:
    """
    Normalize any supported input to a uint8 RGB numpy array (H, W, 3).

    Raises
    ------
    ValueError
        If the source cannot be decoded.
    """
    if isinstance(source, bytes):
        arr = np.frombuffer(source, dtype=np.uint8)
        # cv2.imdecode applies the EXIF orientation tag by default.
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(
                "Could not decode image from bytes. "
                "Ensure the uploaded file is a valid JPEG, PNG, or WebP."
            )
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            return cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        if source.shape[2] == 4:
            return cv2.cvtColor(source, cv2.COLOR_RGBA2RGB)
        return source.astype(np.uint8)

    if isinstance(source, Image.Image):
        return np.array(source.convert("RGB"), dtype=np.uint8)

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
# Geometry helpers (pure functions, no torch)
# ---------------------------------------------------------------------------

def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order four points as TL, TR, BR, BL (clockwise on screen, y down),
    using only their positions.

    Points are sorted by angle around their centroid, which yields a
    clockwise ring for any four points in convex position; the ring is then
    rotated so it starts at the point nearest the top-left (smallest x + y).

    Works for any input order, including the swapped bottom pair that makes
    a channel-ordered quad self-intersect. Cards tilted much past 45 degrees
    are handled afterwards by `landscape_quad`.
    """
    p = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    p = p[np.argsort(ang)]
    start = int(np.argmin(p.sum(axis=1)))
    return np.roll(p, -start, axis=0).astype(np.float32)


def edge_lengths(q: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (top, right, bottom, left) edge lengths of an ordered quad."""
    top = float(np.linalg.norm(q[1] - q[0]))
    right = float(np.linalg.norm(q[2] - q[1]))
    bottom = float(np.linalg.norm(q[2] - q[3]))
    left = float(np.linalg.norm(q[3] - q[0]))
    return top, right, bottom, left


def landscape_quad(q: np.ndarray) -> np.ndarray:
    """
    Rotate an ordered quad's labels so the longer pair of edges is on top/bottom.

    A card photographed sideways has its long edges running vertically in
    the frame; cycling the corner labels by one makes the warp output a
    landscape card instead of a sideways one.
    """
    top, right, bottom, left = edge_lengths(q)
    if (top + bottom) < (left + right):
        q = np.roll(q, -1, axis=0)
    return q


def card_aspect(q: np.ndarray) -> float:
    """Mean horizontal edge length over mean vertical edge length."""
    top, right, bottom, left = edge_lengths(q)
    v = (left + right) / 2.0
    return ((top + bottom) / 2.0) / v if v > 1e-6 else float("inf")


def _interior_angles(q: np.ndarray) -> List[float]:
    """Interior angle in degrees at each vertex of an ordered quad."""
    angles = []
    for i in range(4):
        a, b, c = q[i - 1], q[i], q[(i + 1) % 4]
        v1, v2 = a - b, c - b
        n = np.linalg.norm(v1) * np.linalg.norm(v2)
        if n < 1e-9:
            angles.append(0.0)
            continue
        cosang = float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cosang))))
    return angles


def validate_quad(q: np.ndarray, width: int, height: int) -> Optional[str]:
    """
    Check an ordered, landscape-normalised quad. Returns None if it passes,
    otherwise a short human-readable reason.

    Checks, in order: finite coordinates, corners inside the frame (with a
    small margin), convexity, area fraction, interior angles, opposite-edge
    ratios, and the card aspect ratio.
    """
    if not np.all(np.isfinite(q)):
        return "non-finite coordinates"

    mx, my = _FRAME_MARGIN_FRAC * width, _FRAME_MARGIN_FRAC * height
    if (q[:, 0] < -mx).any() or (q[:, 0] > width + mx).any() \
            or (q[:, 1] < -my).any() or (q[:, 1] > height + my).any():
        return "corners fall outside the image frame"

    top, right, bottom, left = edge_lengths(q)
    if min(top, right, bottom, left) < 1.0:
        return "degenerate quad (an edge is shorter than 1 px)"

    contour = q.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        return "quad is not convex"

    area_frac = abs(float(cv2.contourArea(contour))) / float(width * height)
    if not (_MIN_AREA_FRAC <= area_frac <= _MAX_AREA_FRAC):
        return f"quad covers {area_frac:.0%} of the frame (expected {_MIN_AREA_FRAC:.0%}-{_MAX_AREA_FRAC:.0%})"

    angles = _interior_angles(q)
    if min(angles) < _MIN_ANGLE_DEG or max(angles) > _MAX_ANGLE_DEG:
        return "interior angles out of range: " + ", ".join(f"{a:.0f}" for a in angles)

    for name, a, b in (("top/bottom", top, bottom), ("left/right", left, right)):
        ratio = max(a, b) / min(a, b)
        if ratio > _MAX_OPPOSITE_EDGE_RATIO:
            return f"{name} edges differ by {ratio:.1f}x"

    aspect = card_aspect(q)
    if not (_MIN_ASPECT <= aspect <= _MAX_ASPECT):
        return f"aspect ratio {aspect:.2f} is not card-like (expected {_MIN_ASPECT}-{_MAX_ASPECT})"

    return None


def analyze_corners(coords: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, Optional[str]]:
    """Order raw detector coords, normalise to landscape, and validate."""
    quad = landscape_quad(order_corners(coords))
    return quad, validate_quad(quad, width, height)


# ---------------------------------------------------------------------------
# Perspective rectification
# ---------------------------------------------------------------------------

def _rectify(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """
    Warp an ordered (TL, TR, BR, BL) quad to a fixed landscape rectangle.

    Returns an (H_out, W_out, 3) uint8 RGB crop.
    """
    dst = np.array(
        [
            [0, 0],
            [_CARD_OUTPUT_W - 1, 0],
            [_CARD_OUTPUT_W - 1, _CARD_OUTPUT_H - 1],
            [0, _CARD_OUTPUT_H - 1],
        ],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(
        image,
        m,
        (_CARD_OUTPUT_W, _CARD_OUTPUT_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ---------------------------------------------------------------------------
# Orientation: resolve the remaining 180 degree ambiguity with a face check
# ---------------------------------------------------------------------------

def _face_score(rgb: np.ndarray, mtcnn) -> float:
    """Best MTCNN face probability in an RGB image, 0.0 if none."""
    try:
        boxes, probs = mtcnn.detect(Image.fromarray(rgb))
    except Exception:  # detection must never break the request
        log.exception("face check failed during orientation search")
        return 0.0
    if boxes is None or probs is None:
        return 0.0
    vals = [float(p) for p in np.atleast_1d(probs) if p is not None]
    return max(vals) if vals else 0.0


def orient_card_by_face(rectified_rgb: np.ndarray, mtcnn) -> Tuple[np.ndarray, int, float]:
    """
    Return the rectified card rotated so a face is upright, if one is found.

    Tries 0 and 180 degrees first (the quad is already landscape), then 90
    and 270 as a last resort. Returns (image, rotation_degrees, face_score).
    If no face is found at any rotation the original image is returned with
    score 0.0, so the caller's normal "no face" error still fires.
    """
    base = _face_score(rectified_rgb, mtcnn)
    if base >= 0.95:
        return rectified_rgb, 0, base

    best_img, best_rot, best_score = rectified_rgb, 0, base
    for rot, k in ((180, 2), (90, 1), (270, 3)):
        if rot in (90, 270) and best_score > 0.0:
            break  # a landscape orientation already has a face
        cand = np.ascontiguousarray(np.rot90(rectified_rgb, k))
        score = _face_score(cand, mtcnn)
        if score > best_score:
            best_img, best_rot, best_score = cand, rot, score

    if best_rot:
        log.info("card rotated %d deg to bring face upright (score %.2f)", best_rot, best_score)
    return best_img, best_rot, best_score


# ---------------------------------------------------------------------------
# Debug overlay (opt-in, writes PII to disk)
# ---------------------------------------------------------------------------

_CHANNEL_COLORS = [(255, 64, 64), (64, 200, 64), (64, 64, 255), (255, 200, 0)]  # RGB per channel


def _write_debug_overlay(
    debug_dir: Union[str, Path],
    request_id: str,
    image_rgb: np.ndarray,
    keypoints: "KeypointResult",
    quad: np.ndarray,
    verdict: Optional[str],
) -> None:
    """
    Save <request_id>_overlay.jpg: raw channel points (numbered, with
    confidence), the ordered quad, and the strongest heatmap as a tint.
    Never raises.
    """
    try:
        out_dir = Path(debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        vis = image_rgb.copy()
        h, w = vis.shape[:2]

        hm = keypoints.heatmaps.max(axis=0)
        hm = cv2.resize(hm / (hm.max() + 1e-12), (w, h))
        tint = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        tint = cv2.cvtColor(tint, cv2.COLOR_BGR2RGB)
        vis = cv2.addWeighted(vis, 0.65, tint, 0.35, 0)

        thick = max(2, w // 400)
        cv2.polylines(vis, [quad.astype(np.int32).reshape(-1, 1, 2)], True, (255, 255, 255), thick)
        for i, ((x, y), conf) in enumerate(zip(keypoints.coords, keypoints.confidence)):
            color = _CHANNEL_COLORS[i % len(_CHANNEL_COLORS)]
            cv2.circle(vis, (int(x), int(y)), thick * 4, color, -1)
            cv2.putText(vis, f"{i}:{conf:.3f}", (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 1600), color, thick)
        label = "OK" if verdict is None else f"REJECT: {verdict}"
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, max(0.7, w / 1400),
                    (255, 255, 0), thick)

        path = out_dir / f"{request_id}_overlay.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        log.warning("KYC debug overlay written (contains PII): %s", path)
    except Exception:
        log.exception("could not write debug overlay")


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def _fail(stage: str, msg: str, **kw) -> EKYCResult:
    return EKYCResult(
        success=False,
        keypoints=kw.pop("keypoints", None),
        rectified_image=None,
        error=f"[{stage}] {msg}",
        stage=stage,
        **kw,
    )


def process_id_card(
    image: Union[bytes, np.ndarray, Image.Image, str],
    detector: "IDDetector",
    require_all_keypoints: bool = True,
    debug_dir: Optional[Union[str, Path]] = None,
) -> EKYCResult:
    """
    Full eKYC ID card processing pipeline.

    Steps:
        1. Decode the input image.                         (stage: decode)
        2. Detect the four card corners.                   (stage: detect)
        3. Order the corners geometrically and validate
           the quad; the verdict is always computed.
        4. Confidence gate (all corners above threshold).  (stage: confidence)
        5. Geometry gate (convex, area, angles, aspect).   (stage: geometry)
        6. Perspective-warp to a landscape crop.           (stage: warp)

    Parameters
    ----------
    require_all_keypoints : bool
        If True (default), reject when any corner is below the detector's
        confidence threshold. The geometry gate always applies.
    debug_dir : path or None
        Where to write an annotated overlay. Defaults to $KYC_DEBUG_DIR;
        unset means no files are written. The overlay contains PII.

    Notes
    -----
    Synchronous. In async routes use `process_id_card_async` or
    `asyncio.to_thread`.
    """
    t0 = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    debug_dir = debug_dir or os.environ.get("KYC_DEBUG_DIR") or None

    def elapsed() -> float:
        return round((time.perf_counter() - t0) * 1000, 2)

    # 1. Decode ------------------------------------------------------------
    try:
        image_np = _load_image(image)
    except ValueError as exc:
        log.warning("eKYC image load failed: %s", exc)
        return _fail(STAGE_DECODE, str(exc))

    h, w = image_np.shape[:2]

    # 2. Detect ------------------------------------------------------------
    try:
        keypoints = detector.detect(image_np)
    except Exception as exc:
        log.exception("eKYC detection failed")
        return _fail(STAGE_DETECT, str(exc))

    # 3. Order + validate (always, so failures report geometry too) ----------
    quad, verdict = analyze_corners(keypoints.coords, w, h)
    geometry = {
        "valid": verdict is None,
        "reason": verdict,
        "aspect": round(card_aspect(quad), 3),
        "quad": quad.round(1).tolist(),
    }
    meta = {
        "request_id": request_id,
        "input_size": (w, h),
        "confidences": [round(float(c), 5) for c in keypoints.confidence],
        "geometry": geometry,
    }

    log.debug(
        "eKYC detection: %d/%d keypoints valid, confidences=%s, geometry=%s",
        int(keypoints.valid.sum()), keypoints.n_keypoints,
        keypoints.confidence.tolist(), verdict or "ok",
    )

    # 4. Confidence gate -----------------------------------------------------
    if require_all_keypoints and not keypoints.all_valid:
        invalid = [n for n, v in zip(keypoints.keypoint_names, keypoints.valid) if not v]
        msg = (
            f"Low-confidence keypoints: {invalid}. "
            "Ensure the ID card is fully visible, well-lit, and not occluded."
        )
        log.info("eKYC validation failed: %s (geometry: %s)", msg, verdict or "ok")
        if debug_dir:
            _write_debug_overlay(debug_dir, request_id, image_np, keypoints, quad, verdict)
        meta["elapsed_ms"] = elapsed()
        return _fail(STAGE_CONFIDENCE, msg, keypoints=keypoints, quad=quad, metadata=meta)

    # 5. Geometry gate ---------------------------------------------------------
    if verdict is not None:
        msg = f"Detected card outline is implausible: {verdict}."
        log.info("eKYC geometry rejected: %s", verdict)
        if debug_dir:
            _write_debug_overlay(debug_dir, request_id, image_np, keypoints, quad, verdict)
        meta["elapsed_ms"] = elapsed()
        return _fail(STAGE_GEOMETRY, msg, keypoints=keypoints, quad=quad, metadata=meta)

    # 6. Rectify ----------------------------------------------------------------
    try:
        rectified = _rectify(image_np, quad)
    except Exception as exc:
        log.exception("eKYC rectification failed")
        return _fail(STAGE_WARP, str(exc), keypoints=keypoints, quad=quad, metadata=meta)

    if debug_dir:
        _write_debug_overlay(debug_dir, request_id, image_np, keypoints, quad, None)

    meta["output_size"] = (_CARD_OUTPUT_W, _CARD_OUTPUT_H)
    meta["elapsed_ms"] = elapsed()
    log.info("eKYC pipeline complete in %s ms.", meta["elapsed_ms"])

    return EKYCResult(
        success=True,
        keypoints=keypoints,
        rectified_image=rectified,
        metadata=meta,
        quad=quad,
    )


async def process_id_card_async(
    image: Union[bytes, np.ndarray, Image.Image, str],
    detector: "IDDetector",
    require_all_keypoints: bool = True,
    debug_dir: Optional[Union[str, Path]] = None,
) -> EKYCResult:
    """Async wrapper: runs `process_id_card` in a worker thread."""
    return await asyncio.to_thread(
        process_id_card, image, detector, require_all_keypoints, debug_dir
    )
