"""
app.utils.image
───────────────
Low-level image helpers with no ML dependencies.

Consolidates:
  backend/utils/functions.py  — get_image, file decode helpers
  backend/utils/distance.py   — cosine / euclidean distance for embeddings
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from app.core.exceptions import ImageDecodeError


# ── Decode ─────────────────────────────────────────────────────────────────────

def bytes_to_bgr(data: bytes) -> np.ndarray:
    """
    Decode raw image bytes to a BGR numpy array (OpenCV convention).
    Raises ImageDecodeError if the bytes are not a valid image.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError("cv2.imdecode returned None — not a valid image")
    return img


def bytes_to_rgb(data: bytes) -> np.ndarray:
    """Decode raw image bytes to RGB numpy array (PIL / model convention)."""
    return cv2.cvtColor(bytes_to_bgr(data), cv2.COLOR_BGR2RGB)


def upload_to_bgr(file_bytes: bytes) -> np.ndarray:
    """Alias kept for compatibility with old `_decode_upload` call sites."""
    return bytes_to_bgr(file_bytes)


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL Image (RGB) to a BGR numpy array."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def bgr_to_rgb_array(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR numpy array to RGB (for model input)."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── Resize ─────────────────────────────────────────────────────────────────────

def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize a numpy image array to (w, h) using INTER_AREA."""
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def resize_pil(pil_img: Image.Image, w: int, h: int) -> Image.Image:
    return pil_img.resize((w, h))


# ── Embedding distance ──────────────────────────────────────────────────────────
# Migrated from backend/utils/distance.py

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine distance in [0, 1] between two embedding vectors.
    0 = identical, 1 = orthogonal.
    """
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    dot    = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0, 1]. Complement of cosine_distance."""
    return 1.0 - cosine_distance(a, b)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between two embedding vectors."""
    return float(np.linalg.norm(a.flatten() - b.flatten()))


# ── IO helpers ─────────────────────────────────────────────────────────────────

def load_image_bgr(path: str) -> np.ndarray:
    """
    Read an image from disk and return a BGR numpy array.
    Mirrors old `get_image()` from backend/utils/functions.py.
    """
    img = cv2.imread(path)
    if img is None:
        raise ImageDecodeError(f"Could not read image from path: {path}")
    return img
