"""
app.services.face.verification
───────────────────────────────
Face embedding extraction and similarity comparison using VGGFace2
(InceptionResnetV1 backbone).

Design
------
- All inference runs under torch.inference_mode() — no gradient tape
- Cosine similarity replaces euclidean distance (more robust across lighting
  and compression conditions)
- extract_embedding() accepts BGR numpy arrays (OpenCV convention) and handles
  the RGB conversion, MTCNN detection, and normalization internally
- Returns (bool, float) so callers get both the decision and the raw score
  for audit logging and the confidence field in KYCResult

FaceDetector (the assembled class from services/face_detection) is used
rather than raw MTCNN + InceptionResnetV1 so detection and embedding share
a single consistent preprocessing path.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.config import settings
from core.exceptions import NoFaceDetectedError
from utils.image import cosine_similarity

log = logging.getLogger(__name__)


# ── Preprocessing ──────────────────────────────────────────────────────────────

def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert BGR numpy array to PIL RGB image for MTCNN input."""
    rgb = bgr[:, :, ::-1].copy()   # BGR → RGB without a full cv2 call
    return Image.fromarray(rgb.astype(np.uint8))


def _face_transform(
    face_crop: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Apply VGGFace2 fixed standardization and move to device.
    face_crop: (3, 160, 160) float tensor from MTCNN.
    """
    # fixed_image_standardization: (pixel - 127.5) / 128.0
    face = (face_crop - 127.5) / 128.0
    return face.unsqueeze(0).to(device)   # → (1, 3, 160, 160)


# ── Core functions ─────────────────────────────────────────────────────────────

@torch.inference_mode()
def extract_embedding(
    bgr: np.ndarray,
    mtcnn,
    verif_model,
) -> torch.Tensor:
    """
    Detect the primary face in a BGR image and return its L2-normalized
    512-d VGGFace2 embedding.

    Parameters
    ----------
    bgr : np.ndarray
        BGR uint8 image (OpenCV convention).
    mtcnn : MTCNN
        Face detector (from app.state.mtcnn).
    verif_model : InceptionResnetV1
        Embedding model (from app.state.verif_model).

    Returns
    -------
    torch.Tensor, shape (512,)
        L2-normalized embedding vector on CPU.

    Raises
    ------
    NoFaceDetectedError
        When MTCNN finds no face in the image.
    """
    pil_img = _bgr_to_pil(bgr)

    # MTCNN returns a (3, 160, 160) float tensor, or None if no face detected.
    face_crop = mtcnn(pil_img)
    if face_crop is None:
        raise NoFaceDetectedError("No face detected in the supplied image")

    device = next(verif_model.parameters()).device
    face_t = _face_transform(face_crop, device)

    embedding = verif_model(face_t)           # (1, 512), already L2-normalized
    return embedding.squeeze(0).cpu()         # (512,)


def compare_faces(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    threshold: float | None = None,
) -> tuple[bool, float]:
    """
    Compare two 512-d embeddings using cosine similarity.

    Parameters
    ----------
    emb1, emb2 : torch.Tensor, shape (512,)
    threshold : float or None
        Accept/reject threshold. Defaults to settings.face_verification_threshold.
        Higher = stricter. Sensible range: 0.5 – 0.8.

    Returns
    -------
    (verified, score) : tuple[bool, float]
        verified — True if similarity >= threshold
        score    — raw cosine similarity in [0, 1]
    """
    if threshold is None:
        threshold = settings.face_verification_threshold

    score = float(F.cosine_similarity(
        emb1.unsqueeze(0),
        emb2.unsqueeze(0),
    ).item())

    # Clamp to [0, 1] — cosine similarity can theoretically be negative
    # for very different faces; we floor at 0 for a clean confidence score.
    score = max(0.0, score)

    return score >= threshold, score


def verify_pair(
    bgr1: np.ndarray,
    bgr2: np.ndarray,
    mtcnn,
    verif_model,
    threshold: float | None = None,
) -> tuple[bool, float]:
    """
    Full pipeline: two BGR images → (verified, score).

    Raises NoFaceDetectedError if either image has no detectable face.
    """
    emb1 = extract_embedding(bgr1, mtcnn, verif_model)
    emb2 = extract_embedding(bgr2, mtcnn, verif_model)
    return compare_faces(emb1, emb2, threshold)
