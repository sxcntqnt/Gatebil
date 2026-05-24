# face_detector.py
"""
FaceDetector — Main Assembler
==============================

Single entry point for the face detection + embedding pipeline.

Replaces setup.py as the module orchestrator. Composes MTCNN (detection)
with InceptionResnetV1 (embedding) into a unified, device-aware API.

Pipeline
--------
    image(s)
        |
        v
    MTCNN                  ->  boxes, probs, landmarks
        |
        v
    face crop + align
        |
        v
    InceptionResnetV1      ->  512-d L2-normalized embeddings

Usage
-----
    from face_detector import FaceDetector

    detector = FaceDetector(device='cuda', pretrained='vggface2')

    # Detection only
    result = detector.detect(img)
    result.boxes        # (N, 4)  xyxy pixel coords
    result.probs        # (N,)    detection confidence
    result.landmarks    # (N, 5, 2)

    # Detection + embedding in one pass
    result = detector.detect_and_embed(img)
    result.embeddings   # (N, 512) L2-normalized

    # Embedding from pre-cropped faces
    embeddings = detector.embed(face_tensor)  # (N, 3, 160, 160) -> (N, 512)

Notes
-----
- All models are set to eval() on init; no gradients are computed during
  inference. Use detector.train_mode() to re-enable for fine-tuning.
- Passing device=None auto-selects CUDA if available, else CPU.
- Results are dataclasses; all fields may be None if no face was detected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, Optional, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .mtcnn import MTCNN
from .inception_resnet_v1 import InceptionResnetV1


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """
    Output from a single detection pass.

    Attributes
    ----------
    boxes : np.ndarray or None, shape (N, 4)
        Detected bounding boxes in xyxy pixel coordinates.
        None if no face detected.

    probs : np.ndarray or None, shape (N,)
        Detection confidence per box in [0, 1].

    landmarks : np.ndarray or None, shape (N, 5, 2)
        Five facial landmarks per detection:
        [left_eye, right_eye, nose, left_mouth, right_mouth]
        in (x, y) pixel coordinates.
    """
    boxes: Optional[np.ndarray] = None
    probs: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None

    @property
    def n_faces(self) -> int:
        """Number of detected faces. 0 if none."""
        return 0 if self.boxes is None else len(self.boxes)

    def __repr__(self) -> str:
        return f"DetectionResult(n_faces={self.n_faces})"


@dataclass
class EmbeddingResult:
    """
    Output from a detection + embedding pass.

    Attributes
    ----------
    embeddings : torch.Tensor or None, shape (N, 512)
        L2-normalized 512-d face embeddings.
        None if no face detected.

    faces : torch.Tensor or None, shape (N, 3, 160, 160)
        Cropped and standardized face crops fed to the embedding model.

    boxes : np.ndarray or None, shape (N, 4)
        Source bounding boxes (same as DetectionResult.boxes).

    probs : np.ndarray or None, shape (N,)
        Detection confidences (same as DetectionResult.probs).

    landmarks : np.ndarray or None, shape (N, 5, 2)
        Facial landmarks (same as DetectionResult.landmarks).
    """
    embeddings: Optional[torch.Tensor] = None
    faces: Optional[torch.Tensor] = None
    boxes: Optional[np.ndarray] = None
    probs: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None

    @property
    def n_faces(self) -> int:
        return 0 if self.embeddings is None else len(self.embeddings)

    def __repr__(self) -> str:
        return f"EmbeddingResult(n_faces={self.n_faces})"


# ---------------------------------------------------------------------------
# FaceDetector — main assembler
# ---------------------------------------------------------------------------

class FaceDetector:
    """
    Unified face detection and embedding pipeline.

    Composes MTCNN (P/R/O-Net cascade) with InceptionResnetV1.
    All inference runs under torch.inference_mode() — no gradients computed.

    Parameters
    ----------
    device : str, torch.device, or None
        Target device. Passing None auto-selects CUDA if available.

    pretrained : str or None
        Pretrained weights for the embedding model.
        One of: 'vggface2', 'casia-webface', or None (random init).

    min_face_size : int
        Minimum face size in pixels to detect. Default 20.

    mtcnn_thresholds : list[float]
        MTCNN stage confidence thresholds [P, R, O]. Default [0.6, 0.7, 0.7].

    keep_all_faces : bool
        If True, all detected faces are returned per image.
        If False, only the most prominent face is returned.

    selection_method : str
        Face selection heuristic when keep_all_faces=False.
        One of: 'largest', 'probability', 'center_weighted_size',
        'largest_over_threshold'.
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        pretrained: Optional[str] = 'vggface2',
        min_face_size: int = 20,
        mtcnn_thresholds: List[float] = None,
        keep_all_faces: bool = False,
        selection_method: str = 'largest',
    ) -> None:

        # ------------------------------------------------------------------
        # Device resolution — never store raw strings; always torch.device.
        # Auto-selection prefers CUDA, falls back to MPS (Apple Silicon),
        # then CPU.
        # ------------------------------------------------------------------
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        self.device = torch.device(device)

        if mtcnn_thresholds is None:
            mtcnn_thresholds = [0.6, 0.7, 0.7]

        # ------------------------------------------------------------------
        # MTCNN — face detection cascade.
        # Always runs in eval mode; weights are fixed pretrained MTCNN weights.
        # ------------------------------------------------------------------
        self.mtcnn = MTCNN(
            min_face_size=min_face_size,
            thresholds=mtcnn_thresholds,
            keep_all=keep_all_faces,
            selection_method=selection_method,
            device=self.device,
        ).eval()

        # ------------------------------------------------------------------
        # InceptionResnetV1 — face embedding backbone.
        # classify=False returns 512-d L2-normalized embeddings, not logits.
        # ------------------------------------------------------------------
        self.embedder = InceptionResnetV1(
            pretrained=pretrained,
            classify=False,
        ).eval().to(self.device)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def detect(
        self,
        img: Union[Image.Image, np.ndarray, torch.Tensor, List],
    ) -> DetectionResult:
        """
        Run MTCNN face detection.

        Parameters
        ----------
        img : PIL Image, np.ndarray, torch.Tensor, or list of these.
            Input image(s). RGB channel order expected.

        Returns
        -------
        DetectionResult
        """
        with torch.inference_mode():
            boxes, probs, landmarks = self.mtcnn.detect(img, landmarks=True)

        return DetectionResult(
            boxes=boxes,
            probs=probs,
            landmarks=landmarks,
        )

    @torch.inference_mode()
    def embed(
        self,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute L2-normalized embeddings from pre-cropped face crops.

        Parameters
        ----------
        faces : torch.Tensor, shape (N, 3, 160, 160)
            Face crops standardized by fixed_image_standardization.
            Use MTCNN.extract() or the extract_face() utility to produce these.

        Returns
        -------
        torch.Tensor, shape (N, 512)
            L2-normalized embedding vectors.
        """
        if faces.ndim == 3:
            # Single face without batch dimension — add it.
            faces = faces.unsqueeze(0)

        faces = faces.to(self.device)
        return self.embedder(faces)

    def detect_and_embed(
        self,
        img: Union[Image.Image, np.ndarray, torch.Tensor, List],
    ) -> EmbeddingResult:
        """
        Detect all faces and return embeddings in one pass.

        Parameters
        ----------
        img : PIL Image, np.ndarray, torch.Tensor, or list.
            Input image(s). RGB channel order expected.

        Returns
        -------
        EmbeddingResult
            Contains embeddings, crops, boxes, probs, landmarks.
            All fields are None if no face is detected.
        """
        with torch.inference_mode():
            # Detect and crop simultaneously.
            faces, probs = self.mtcnn(img, return_prob=True)
            boxes, _, landmarks = self.mtcnn.detect(img, landmarks=True)

        if faces is None:
            return EmbeddingResult()

        # Ensure batch dimension exists.
        if isinstance(faces, torch.Tensor) and faces.ndim == 3:
            faces = faces.unsqueeze(0)

        with torch.inference_mode():
            embeddings = self.embedder(faces.to(self.device))

        return EmbeddingResult(
            embeddings=embeddings,
            faces=faces,
            boxes=boxes,
            probs=probs if isinstance(probs, np.ndarray) else np.array(probs),
            landmarks=landmarks,
        )

    def train_mode(
        self,
        embed: bool = True,
        detect: bool = False,
    ) -> 'FaceDetector':
        """
        Switch components between eval and train mode for fine-tuning.

        MTCNN weights are typically frozen; embed=True, detect=False is the
        standard setup for fine-tuning only the embedding backbone.

        Parameters
        ----------
        embed : bool
            If True, put embedding model in train mode.
        detect : bool
            If True, put MTCNN in train mode. Rarely needed.

        Returns
        -------
        self (for chaining)
        """
        if embed:
            self.embedder.train()
        else:
            self.embedder.eval()

        if detect:
            self.mtcnn.train()
        else:
            self.mtcnn.eval()

        return self

    def to(self, device: Union[str, torch.device]) -> 'FaceDetector':
        """Move all models to a new device."""
        self.device = torch.device(device)
        self.mtcnn = self.mtcnn.to(self.device)
        self.embedder = self.embedder.to(self.device)
        return self

    def __repr__(self) -> str:
        return (
            f"FaceDetector(\n"
            f"  device={self.device},\n"
            f"  mtcnn={self.mtcnn.__class__.__name__},\n"
            f"  embedder={self.embedder.__class__.__name__},\n"
            f")"
        )
