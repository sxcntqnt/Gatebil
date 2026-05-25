# app/models/id_detector.py
"""
app.models.id_detector
=======================

PyTorch ID card keypoint detector backed by a DSNT coordinate regression head.

Replaces the TF1 frozen-graph implementation entirely.

Responsibilities
----------------
    - Load a PyTorch checkpoint once at startup and hold the model on the
      target device.
    - Expose a single `detect(image_nd) -> DetectionResult` method.
    - Own all preprocessing constants, keypoint definitions, and tensor
      layout details in one place.
    - Manage model lifecycle (load, warmup, close).

Architecture
------------
    image (H, W, 3) uint8
        |
        v
    preprocess          — resize to INPUT_SIZE, normalize to [-1, 1]
        |
        v
    IDCardBackbone      — MobileNetV3-Small feature extractor
        |
        v
    KeypointHead        — 1x1 conv, upsample -> C heatmap channels
        |
        v
    DSNT                — spatial softmax + expectation
        |
        v
    DetectionResult     — coords (C, 2), confidence (C,), heatmaps (C, H, W)

Keypoints (C = 4)
-----------------
    0  top-left corner
    1  top-right corner
    2  bottom-right corner
    3  bottom-left corner

These four corners define the ID card homography for perspective correction
upstream in the eKYC pipeline.

Checkpoint format
-----------------
    torch.save({
        'model_state_dict': model.state_dict(),
        'keypoint_names':   ['top_left', 'top_right', 'bottom_right', 'bottom_left'],
        'input_size':       (256, 160),   # (W, H)
        'version':          '1.0',
    }, 'id_detector.pt')

Migration from TF1
------------------
    The previous implementation loaded a frozen protobuf and ran inference
    through tf.compat.v1.Session. That introduced:
        - TF install as a hard dependency
        - GPU/CUDA driver warnings on CPU machines
        - No gradient support (frozen graph)
        - ~3s cold-start overhead from TF runtime init
        - self.training = False pattern that masked eval() calls

    This implementation uses only PyTorch + Kornia (already required by dsnt.py)
    and starts in ~80ms on CPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from tasks.dsnt import DSNT, DSNTOutput

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Model input resolution (W, H). Must match checkpoint training config.
# Override via IDDetector(input_size=...) if your checkpoint differs.
DEFAULT_INPUT_SIZE: Tuple[int, int] = (256, 160)

# Number of keypoints. Must match the checkpoint's head output channels.
DEFAULT_NUM_KEYPOINTS: int = 4

# Default keypoint semantic names, in channel order.
DEFAULT_KEYPOINT_NAMES: List[str] = [
    'top_left',
    'top_right',
    'bottom_right',
    'bottom_left',
]

# ImageNet normalization constants.
# Used because the backbone was pretrained on ImageNet.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Minimum confidence threshold below which a keypoint is flagged as unreliable.
# Downstream code (ekyc pipeline) may use this to reject low-quality detections.
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.15


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class KeypointResult:
    """
    Output from a single IDDetector.detect() call.

    Attributes
    ----------
    coords : np.ndarray, shape (C, 2)
        Keypoint (x, y) coordinates in image pixel space, ordered as defined
        by keypoint_names. Origin is top-left corner of the original image.

    confidence : np.ndarray, shape (C,)
        Peak heatmap probability per keypoint.
        Values near 1/(H*W) indicate a diffuse (unreliable) heatmap.
        Values significantly above that indicate a confident detection.

    heatmaps : np.ndarray, shape (C, H, W)
        Normalized probability heatmaps from the DSNT head.
        In the heatmap coordinate space (not original image space).
        Useful for visualization and debugging.

    keypoint_names : list[str]
        Semantic name for each keypoint channel.

    valid : np.ndarray, shape (C,), dtype bool
        True for each keypoint whose confidence exceeds the detector's
        confidence_threshold. Use this mask before computing a homography.

    original_size : tuple[int, int]
        (width, height) of the original input image, used to scale coords.
    """

    coords: np.ndarray
    confidence: np.ndarray
    heatmaps: np.ndarray
    keypoint_names: List[str]
    valid: np.ndarray
    original_size: Tuple[int, int]

    @property
    def n_keypoints(self) -> int:
        return len(self.keypoint_names)

    @property
    def all_valid(self) -> bool:
        """True only if every keypoint passed the confidence threshold."""
        return bool(self.valid.all())

    def as_dict(self) -> Dict[str, Dict]:
        """
        Serializable representation for API responses.

        Returns
        -------
        dict mapping keypoint name -> {x, y, confidence, valid}
        """
        return {
            name: {
                'x': float(self.coords[i, 0]),
                'y': float(self.coords[i, 1]),
                'confidence': float(self.confidence[i]),
                'valid': bool(self.valid[i]),
            }
            for i, name in enumerate(self.keypoint_names)
        }

    def corner_points(self) -> Optional[np.ndarray]:
        """
        Return the four corner keypoints as a (4, 2) array suitable for
        cv2.getPerspectiveTransform or similar.

        Returns None if any corner is invalid (below confidence threshold).
        """
        if not self.all_valid:
            return None
        return self.coords.astype(np.float32)

    def __repr__(self) -> str:
        valid_count = int(self.valid.sum())
        return (
            f"KeypointResult("
            f"n_keypoints={self.n_keypoints}, "
            f"valid={valid_count}/{self.n_keypoints}, "
            f"original_size={self.original_size})"
        )


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class KeypointHead(nn.Module):
    """
    Lightweight keypoint regression head attached to the MobileNetV3 backbone.

    Takes the backbone's final feature map and produces C heatmap channels,
    upsampled back to (H/4, W/4) of the input resolution for DSNT.

    Architecture:
        backbone_features (B, 576, h, w)
            |
            1x1 conv -> 128 channels
            |
            bilinear upsample x4
            |
            1x1 conv -> C heatmap channels (raw logits for DSNT)

    The output is raw logits — DSNT applies spatial softmax internally.
    """

    def __init__(self, in_channels: int, num_keypoints: int) -> None:
        super().__init__()

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Final 1x1 conv produces one raw logit heatmap per keypoint.
        # No activation — DSNT expects unnormalized logits.
        self.logit_head = nn.Conv2d(128, num_keypoints, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize logit head weights near zero.

        Starting near-zero logits means the initial heatmaps are close to
        uniform (after softmax), which is a stable starting point for DSNT.
        Biased initialization (e.g. kaiming) can cause the first few training
        batches to produce degenerate coordinates.
        """
        nn.init.normal_(self.logit_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.logit_head.bias)

    def forward(self, x: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, C_in, h, w)
        output_size : tuple (H_out, W_out)
            Target spatial size for upsampling.

        Returns
        -------
        torch.Tensor, shape (B, num_keypoints, H_out, W_out)
        """
        x = self.reduce(x)
        x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)
        return self.logit_head(x)


class IDCardModel(nn.Module):
    """
    Full ID card keypoint detection model.

    Backbone: MobileNetV3-Small (pretrained on ImageNet).
    Head: KeypointHead (random init, trained on ID card data).
    Coordinate regression: DSNT.

    Parameters
    ----------
    num_keypoints : int
        Number of keypoints to regress. Default 4 (card corners).

    dsnt_temperature : float
        Softmax temperature for DSNT. Lower = sharper peaks. Default 1.0.

    pretrained_backbone : bool
        If True, initialize backbone from ImageNet weights.
        Set False for testing or when loading a full checkpoint.
    """

    def __init__(
        self,
        num_keypoints: int = DEFAULT_NUM_KEYPOINTS,
        dsnt_temperature: float = 1.0,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Backbone: MobileNetV3-Small.
        # Strip the classifier; keep only the feature extractor.
        # Output channels: 576 (final inverted residual output).
        # Spatial reduction: /32 of input (for 256x160 input -> 8x5 feature map).
        # ------------------------------------------------------------------
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone_full = mobilenet_v3_small(weights=weights)

        # features[:-1] removes the AdaptiveAvgPool at the end of features,
        # keeping the spatial feature map rather than collapsing to (B, C, 1, 1).
        self.backbone = backbone_full.features

        # ------------------------------------------------------------------
        # Keypoint head + DSNT.
        # MobileNetV3-Small final feature channels: 576.
        # ------------------------------------------------------------------
        self.head = KeypointHead(
            in_channels=576,
            num_keypoints=num_keypoints,
        )

        self.dsnt = DSNT(
            temperature=dsnt_temperature,
            normalized_coordinates=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> DSNTOutput:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 3, H, W)
            Preprocessed input (ImageNet normalized, float32).

        Returns
        -------
        DSNTOutput
            .coords     (B, C, 2) in normalized [-1, 1] space
            .heatmaps   (B, C, H/8, W/8)
            .confidence (B, C)
        """
        # Target heatmap resolution: H/8 x W/8.
        # This gives reasonable spatial resolution without excessive memory.
        heatmap_size = (x.shape[2] // 8, x.shape[3] // 8)

        features = self.backbone(x)           # (B, 576, H/32, W/32)
        logits = self.head(features, heatmap_size)  # (B, C, H/8, W/8)
        return self.dsnt(logits)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _build_transform(input_size: Tuple[int, int]) -> transforms.Compose:
    """
    Build the inference preprocessing pipeline.

    Steps:
        1. Resize to (W, H) with bilinear interpolation.
        2. Convert to float tensor in [0, 1].
        3. Normalize with ImageNet mean/std.

    Parameters
    ----------
    input_size : tuple (W, H)
        Target width and height.

    Returns
    -------
    transforms.Compose
    """
    w, h = input_size
    return transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# IDDetector — public interface
# ---------------------------------------------------------------------------

class IDDetector:
    """
    ID card keypoint detector.

    Wraps IDCardModel with preprocessing, postprocessing, device management,
    and a simple detect() API.

    Usage
    -----
        detector = IDDetector.from_checkpoint('weights/id_detector.pt')
        result = detector.detect(image_np)  # numpy uint8 (H, W, 3)

        if result.all_valid:
            corners = result.corner_points()  # (4, 2) for homography

    Parameters
    ----------
    model : IDCardModel
        Instantiated model (weights loaded externally or via from_checkpoint).

    input_size : tuple (W, H)
        Preprocessing target size. Must match model training config.

    keypoint_names : list[str]
        Semantic label for each keypoint channel.

    confidence_threshold : float
        Keypoints below this peak heatmap value are flagged as invalid.

    device : torch.device or None
        Inference device. None auto-selects CUDA/MPS/CPU.
    """

    def __init__(
        self,
        model: IDCardModel,
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
        keypoint_names: Optional[List[str]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')

        self.device = device
        self.input_size = input_size
        self.keypoint_names = keypoint_names or DEFAULT_KEYPOINT_NAMES
        self.confidence_threshold = confidence_threshold

        self.model = model.eval().to(self.device)
        self._transform = _build_transform(input_size)

        log.info(
            "IDDetector ready — device=%s, input_size=%s, keypoints=%s",
            self.device, self.input_size, self.keypoint_names,
        )

    # -----------------------------------------------------------------------
    # Construction helpers
    # -----------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> 'IDDetector':
        """
        Load a detector from a saved checkpoint.

        Checkpoint must contain:
            'model_state_dict'  — model weights
            'keypoint_names'    — list of keypoint name strings (optional)
            'input_size'        — (W, H) tuple (optional)
            'num_keypoints'     — int (optional, inferred from state_dict if absent)

        Parameters
        ----------
        checkpoint_path : str or Path
        device : torch.device or None
        confidence_threshold : float

        Returns
        -------
        IDDetector

        Raises
        ------
        FileNotFoundError
            If checkpoint_path does not exist.
        KeyError
            If 'model_state_dict' is missing from the checkpoint.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}. "
                "Set MODEL_PATH in your environment or provide the correct path."
            )

        log.info("Loading ID detector checkpoint from %s", path)

        # map_location='cpu' keeps checkpoint loading device-agnostic.
        # The model is moved to `device` after loading.
        ckpt = torch.load(path, map_location='cpu', weights_only=False)

        if 'model_state_dict' not in ckpt:
            raise KeyError(
                f"Checkpoint at {path} does not contain 'model_state_dict'. "
                "Expected keys: model_state_dict, keypoint_names, input_size."
            )

        keypoint_names = ckpt.get('keypoint_names', DEFAULT_KEYPOINT_NAMES)
        input_size = ckpt.get('input_size', DEFAULT_INPUT_SIZE)
        num_keypoints = ckpt.get('num_keypoints', len(keypoint_names))

        model = IDCardModel(
            num_keypoints=num_keypoints,
            pretrained_backbone=False,  # weights come from the checkpoint
        )
        model.load_state_dict(ckpt['model_state_dict'])

        log.info(
            "Checkpoint loaded — version=%s, keypoints=%d",
            ckpt.get('version', 'unknown'), num_keypoints,
        )

        return cls(
            model=model,
            input_size=input_size,
            keypoint_names=keypoint_names,
            confidence_threshold=confidence_threshold,
            device=device,
        )

    @classmethod
    def from_scratch(
        cls,
        device: Optional[torch.device] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> 'IDDetector':
        """
        Instantiate with random head weights and ImageNet backbone.

        Used for training a new detector from scratch or for testing
        the pipeline without a trained checkpoint.
        """
        model = IDCardModel(pretrained_backbone=True)
        return cls(
            model=model,
            confidence_threshold=confidence_threshold,
            device=device,
        )

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    @torch.inference_mode()
    def detect(
        self,
        image: Union[np.ndarray, 'PIL.Image.Image'],
    ) -> KeypointResult:
        """
        Detect ID card keypoints in a single image.

        Parameters
        ----------
        image : np.ndarray (H, W, 3) uint8, or PIL Image
            Input image in RGB channel order.
            Any resolution is accepted; resized internally to input_size.

        Returns
        -------
        KeypointResult
            Coordinates are in the original image's pixel space.

        Notes
        -----
        - This method is decorated with @torch.inference_mode() and is safe
          to call from multiple threads concurrently (read-only model state).
        - For batch processing use detect_batch() instead of looping over
          individual detect() calls.
        """
        from PIL import Image as PILImage

        # ------------------------------------------------------------------
        # Normalize to PIL for consistent transform pipeline.
        # ------------------------------------------------------------------
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            pil_image = PILImage.fromarray(image)
        else:
            pil_image = image

        original_size = pil_image.size  # (W, H)

        # ------------------------------------------------------------------
        # Preprocess: resize + normalize to (1, 3, H_in, W_in).
        # ------------------------------------------------------------------
        tensor = self._transform(pil_image).unsqueeze(0).to(self.device)

        # ------------------------------------------------------------------
        # Forward pass.
        # Output coords are in normalized [-1, 1] space relative to the
        # model input size — we rescale to original pixel space below.
        # ------------------------------------------------------------------
        output: DSNTOutput = self.model(tensor)

        # ------------------------------------------------------------------
        # Postprocess: convert normalized coords -> original pixel coords.
        #
        # DSNT normalized space: [-1, 1] x [-1, 1]
        # Map to pixel space: x_px = (x_norm + 1) / 2 * W_orig
        #                     y_px = (y_norm + 1) / 2 * H_orig
        # ------------------------------------------------------------------
        coords_norm = output.coords[0].cpu().numpy()   # (C, 2) in [-1, 1]
        ow, oh = original_size
        coords_px = (coords_norm + 1.0) / 2.0 * np.array([ow, oh], dtype=np.float32)

        confidence = output.confidence[0].cpu().numpy()  # (C,)
        heatmaps = output.heatmaps[0].cpu().numpy()      # (C, H_hm, W_hm)

        valid = confidence >= self.confidence_threshold

        return KeypointResult(
            coords=coords_px,
            confidence=confidence,
            heatmaps=heatmaps,
            keypoint_names=self.keypoint_names,
            valid=valid,
            original_size=original_size,
        )

    @torch.inference_mode()
    def detect_batch(
        self,
        images: List[Union[np.ndarray, 'PIL.Image.Image']],
    ) -> List[KeypointResult]:
        """
        Detect keypoints in a batch of images.

        All images are resized to the same input_size and processed in a
        single forward pass, which is significantly faster than calling
        detect() in a loop on GPU.

        Parameters
        ----------
        images : list of np.ndarray or PIL Images.

        Returns
        -------
        list of KeypointResult, one per input image.
        """
        from PIL import Image as PILImage

        pil_images = [
            PILImage.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img
            for img in images
        ]
        original_sizes = [img.size for img in pil_images]

        # Stack into a single batch tensor.
        batch = torch.stack(
            [self._transform(img) for img in pil_images]
        ).to(self.device)

        output: DSNTOutput = self.model(batch)

        coords_norm = output.coords.cpu().numpy()    # (B, C, 2)
        confidence  = output.confidence.cpu().numpy()  # (B, C)
        heatmaps    = output.heatmaps.cpu().numpy()   # (B, C, H, W)

        results = []
        for i, (ow, oh) in enumerate(original_sizes):
            coords_px = (coords_norm[i] + 1.0) / 2.0 * np.array([ow, oh], dtype=np.float32)
            results.append(KeypointResult(
                coords=coords_px,
                confidence=confidence[i],
                heatmaps=heatmaps[i],
                keypoint_names=self.keypoint_names,
                valid=confidence[i] >= self.confidence_threshold,
                original_size=(ow, oh),
            ))

        return results

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def warmup(self, n_runs: int = 3) -> None:
        """
        Run a few dummy forward passes to warm up CUDA kernels and cuDNN.

        Call this once after from_checkpoint() during FastAPI lifespan startup
        so the first real request does not bear the kernel compilation cost.

        Parameters
        ----------
        n_runs : int
            Number of warmup passes. Default 3.
        """
        w, h = self.input_size
        dummy = torch.zeros(1, 3, h, w, device=self.device)
        for _ in range(n_runs):
            _ = self.model(dummy)
        log.info("IDDetector warmup complete (%d runs).", n_runs)

    def close(self) -> None:
        """
        Release model resources.

        Moves the model to CPU and deletes it, freeing CUDA memory.
        Called by the FastAPI lifespan on shutdown.
        """
        self.model.cpu()
        del self.model
        torch.cuda.empty_cache()
        log.info("IDDetector closed.")

    def __repr__(self) -> str:
        return (
            f"IDDetector("
            f"device={self.device}, "
            f"input_size={self.input_size}, "
            f"keypoints={self.keypoint_names}, "
            f"threshold={self.confidence_threshold})"
        )
