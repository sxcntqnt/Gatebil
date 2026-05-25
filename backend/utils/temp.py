"""
app.services.storage.temp
─────────────────────────
Manages the scratch directory for intermediate images.

Replaces the raw TMP_DIR global and scattered os.path.join(TMP_DIR, ...) calls.
All paths go through this module so the storage strategy can change (e.g. to
object storage) in one place without touching any pipeline code.

Slots
-----
Named slots provide stable filenames for the within-request handoff between
/id-card and /verify:
    "original"  — rotated, full-resolution ID card
    "cropped"   — perspective-corrected crop at model resolution
    "final"     — cropped image scaled back to original resolution
    "id_face"   — face region extracted from the corrected ID card
    "selfie"    — live selfie uploaded for verification

Each slot maps to a fixed filename. Concurrent requests on a single-worker
deployment are fine; multi-worker deployments should use object storage
(the slot names already appear in SubmitRequest.SelfieURL / IDCardURL).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from core.config import settings
from core.exceptions import StorageError

log = logging.getLogger(__name__)

# ── Slot registry ──────────────────────────────────────────────────────────────

_SLOTS: dict[str, str] = {
    "original": "original.jpg",
    "cropped":  "cropped.jpg",
    "final":    "final.jpg",
    "id_face":  "id_face.jpg",
    "selfie":   "verify_selfie.jpg",
}


def slot_path(slot: str) -> Path:
    """Return the absolute path for a named temp slot."""
    if slot not in _SLOTS:
        raise StorageError(f"Unknown temp slot '{slot}'. Valid: {list(_SLOTS)}")
    return settings.tmp_dir / _SLOTS[slot]


# ── Write ──────────────────────────────────────────────────────────────────────

def write_bgr(slot: str, img: np.ndarray) -> Path:
    """
    Write a BGR numpy array to a named temp slot as JPEG.

    Parameters
    ----------
    slot : str
        One of the registered slot names.
    img : np.ndarray
        BGR image array (OpenCV convention).

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    path = slot_path(slot)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise StorageError(f"cv2.imwrite failed for slot '{slot}' at {path}")
    log.debug("temp write", extra={"slot": slot, "path": str(path), "shape": img.shape})
    return path


def write_bytes(slot: str, data: bytes) -> Path:
    """Write raw bytes directly to a named temp slot."""
    path = slot_path(slot)
    try:
        path.write_bytes(data)
    except OSError as exc:
        raise StorageError(f"Failed to write slot '{slot}': {exc}") from exc
    return path


# ── Read ───────────────────────────────────────────────────────────────────────

def read_bgr(slot: str) -> np.ndarray:
    """
    Read a named temp slot and return a BGR numpy array.

    Raises StorageError if the file does not exist.
    """
    path = slot_path(slot)
    if not path.exists():
        raise StorageError(
            f"Temp slot '{slot}' has no file at {path}. "
            "Call /id-card before /verify."
        )
    img = cv2.imread(str(path))
    if img is None:
        raise StorageError(f"Could not decode temp file for slot '{slot}' at {path}")
    return img


def exists(slot: str) -> bool:
    """Return True if a named slot file exists on disk."""
    return slot_path(slot).exists()


# ── Cleanup ────────────────────────────────────────────────────────────────────

def purge_stale() -> int:
    """
    Delete temp files older than settings.tmp_max_age_seconds.
    Returns the number of files deleted.
    Called periodically — not per-request.
    """
    cutoff = time.time() - settings.tmp_max_age_seconds
    deleted = 0
    for path in settings.tmp_dir.iterdir():
        if path.is_file() and path.stat().st_mtime < cutoff:
            try:
                path.unlink()
                deleted += 1
            except OSError:
                pass
    if deleted:
        log.info("purged stale temp files", extra={"count": deleted})
    return deleted
