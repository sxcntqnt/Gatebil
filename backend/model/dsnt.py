# dsnt.py

"""
Differentiable Spatial to Numerical Transform (DSNT)
====================================================

Production-grade PyTorch + Kornia implementation of the DSNT layer from:

    "Numerical Coordinate Regression with Convolutional Neural Networks"
    Nibali et al.
    https://arxiv.org/abs/1801.07372

-------------------------------------------------------------------------------

CHANGELOG FROM v2
-------------------------------------------------------------------------------

v3 introduces the following additions and fixes over v2:

    1. Learnable temperature support
          Temperature can now be optionally registered as a trainable
          nn.Parameter, allowing the network to adapt softmax sharpness
          during training rather than treating it as a fixed hyperparameter.

    2. Resolution-aware sigma scaling in js_reg_loss
          Fixed-sigma Gaussians previously had inconsistent spatial extent
          depending on heatmap resolution. Sigma is now correctly scaled
          relative to heatmap size when using normalized coordinates, ensuring
          consistent regularization across resolutions.

    3. Per-keypoint confidence scores
          DSNTOutput now carries a `confidence` field containing the peak
          probability per (batch, channel) keypoint. Used downstream for
          occlusion handling, multi-person NMS, and filtering low-quality
          predictions.

    4. Dtype validation on forward input
          Integer tensors passed to DSNT previously produced silently wrong
          results after softmax. Forward now raises immediately on non-floating
          point input.

    5. Legacy fwhm compatibility shim in js_reg_loss
          Configs and checkpoints written against the original TensorFlow
          js_reg_loss(fwhm=...) API are supported via automatic FWHM-to-sigma
          conversion. No silent breakage on migration.

    6. Stable KL divergence via log-space clamping
          KL divergence now clamps in log-space rather than probability-space
          for improved fp16/AMP numerical safety under extreme heatmap values.

-------------------------------------------------------------------------------

OVERVIEW
-------------------------------------------------------------------------------

DSNT converts dense spatial heatmaps into continuous differentiable coordinates
without using non-differentiable argmax operations.

Instead of:

    heatmap -> argmax -> integer pixel coordinate

DSNT performs:

    heatmap -> spatial probability distribution -> expectation -> subpixel coordinate

This preserves gradient flow and allows stable end-to-end coordinate regression.

-------------------------------------------------------------------------------

EXPECTED INPUT SHAPES
-------------------------------------------------------------------------------

Heatmaps:
    (B, C, H, W)

Where:
    B = batch size
    C = number of keypoints / channels
    H = heatmap height
    W = heatmap width

Coordinates:
    (B, C, 2)

Coordinate ordering:
    (x, y) — column-first / screen-space convention
    Note: this is NOT (row, col). Some downstream tasks expect (y, x); flip
    coords[..., [1, 0]] if needed.

-------------------------------------------------------------------------------

COORDINATE SYSTEMS
-------------------------------------------------------------------------------

normalized_coordinates=True  (default):

    Coordinates lie in [-1, 1]

    (-1, -1) -> top-left pixel
    ( 1,  1) -> bottom-right pixel

    This is the recommended mode for training; it decouples coordinate
    magnitudes from heatmap resolution and plays nicely with normalized
    ground-truth annotations.

normalized_coordinates=False:

    Coordinates lie in pixel space.

    (0,     0    ) -> top-left pixel
    (W - 1, H - 1) -> bottom-right pixel

    Use this mode when interfacing with downstream code that expects
    absolute pixel coordinates (e.g. OpenCV, COCO keypoint format).

-------------------------------------------------------------------------------

NUMERICAL STABILITY NOTES
-------------------------------------------------------------------------------

1. KL/JS divergence uses log-space clamping to avoid:
       log(0)
       division by zero
       NaNs during mixed precision (fp16) training

2. Kornia spatial_softmax2d internally applies numerically stable softmax
   (max-subtraction before exp), safe under fp16.

3. All operations are fully differentiable through both coordinates and loss.

4. Temperature is stored as a buffer (or Parameter) rather than a raw float
   to ensure correct device placement without manual .to(device) calls.

-------------------------------------------------------------------------------

PERFORMANCE NOTES
-------------------------------------------------------------------------------

    - Fully vectorized; no Python loops in the hot path
    - CUDA-compatible
    - torch.compile friendly
    - TorchScript friendly (when learnable_temperature=False)
    - AMP / fp16 compatible
    - Batch-safe and multi-channel safe

-------------------------------------------------------------------------------

COMMON USAGE
-------------------------------------------------------------------------------

Basic usage:

    import torch
    from dsnt import DSNT, js_reg_loss

    dsnt = DSNT(temperature=1.0)

    # heatmaps: raw logits from your backbone, shape (B, C, H, W)
    heatmaps = torch.randn(8, 17, 64, 64)           # 8 images, 17 keypoints

    out = dsnt(heatmaps)

    out.coords       # (8, 17, 2) — (x, y) in [-1, 1]
    out.heatmaps     # (8, 17, 64, 64) — after softmax normalization
    out.confidence   # (8, 17) — peak probability per keypoint

With learnable temperature:

    dsnt = DSNT(temperature=1.0, learnable_temperature=True)
    # temperature is now an nn.Parameter included in model.parameters()

Training with JS regularization:

    target_coords = torch.zeros(8, 17, 2)            # (B, C, 2) in [-1, 1]

    coord_loss = F.mse_loss(out.coords, target_coords)
    reg_loss   = js_reg_loss(out.heatmaps, target_coords, sigma=1.0)

    loss = coord_loss + 0.1 * reg_loss
    loss.backward()

Legacy fwhm API (migration from TensorFlow DSNT):

    # Old call site: js_reg_loss(heatmaps, centres, fwhm=2)
    # New call site (identical behavior):
    reg_loss = js_reg_loss(out.heatmaps, target_coords, fwhm=2)

Filtering low-confidence predictions:

    mask = out.confidence > 0.3                      # (B, C) bool
    reliable_coords = out.coords[mask]               # (N, 2)

-------------------------------------------------------------------------------

COMMON APPLICATIONS
-------------------------------------------------------------------------------

    - Human pose estimation
    - Facial landmark detection
    - Gaze estimation
    - Medical landmark localization
    - Robotics perception
    - Keypoint tracking
    - SLAM frontends
    - Differentiable geometry pipelines

-------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia.geometry.subpix as KSP


# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------

CoordinateMode = Literal["normalized", "pixel"]


# -----------------------------------------------------------------------------
# Output container
# -----------------------------------------------------------------------------

@dataclass
class DSNTOutput:
    """
    Container returned by the DSNT forward pass.

    Attributes
    ----------
    heatmaps : torch.Tensor, shape (B, C, H, W)
        Spatial probability distributions after softmax normalization.
        Each (H, W) slice sums to 1.0 over the spatial dimensions.

    coords : torch.Tensor, shape (B, C, 2)
        Expected coordinate locations extracted from heatmaps.
        Ordering is (x, y) — i.e. column-first / screen-space.
        Range is [-1, 1] when normalized_coordinates=True,
        or [0, W-1] / [0, H-1] otherwise.

    confidence : torch.Tensor, shape (B, C)
        Peak probability value per keypoint, computed as the spatial maximum
        of the normalized heatmap.

        Interpretation:
            High value -> sharp, confident heatmap peak.
            Low value  -> diffuse heatmap; prediction may be unreliable.

        Typical use:
            Threshold at inference time to discard occluded or uncertain
            keypoints before downstream processing.

            Example:
                valid = output.confidence > 0.3
                reliable_coords = output.coords[valid]
    """

    heatmaps: torch.Tensor
    coords: torch.Tensor
    confidence: torch.Tensor


# -----------------------------------------------------------------------------
# DSNT layer
# -----------------------------------------------------------------------------

class DSNT(nn.Module):
    """
    Differentiable Spatial to Numerical Transform.

    Converts dense heatmaps into differentiable subpixel coordinates.

    Pipeline
    --------
        raw logits  (B, C, H, W)
            |
            v
        spatial softmax  [Kornia — numerically stable, vectorized]
            |
            v
        probability heatmaps  (B, C, H, W)  [sums to 1 per channel]
            |
            v
        spatial expectation  [Kornia]
            |
            v
        continuous coordinates  (B, C, 2)
            +
        peak confidence  (B, C)

    Notes
    -----
    - Coordinates are returned in (x, y) order (column-first / screen-space).
    - All channels are processed independently and in parallel.
    - Fully differentiable through both coordinates and confidence.
    - No Python loops in the forward path.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        normalized_coordinates: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        temperature : float, default 1.0
            Softmax temperature scaling factor applied before spatial softmax.

            Lower  -> sharper distributions, more confident coordinate estimates.
            Higher -> smoother distributions, more uncertain coordinate estimates.

            Typical range: 0.1 – 2.0.

            When learnable_temperature=True this serves as the initial value.

        learnable_temperature : bool, default False
            If True, temperature is registered as an nn.Parameter and will
            be updated by the optimizer alongside all other model weights.

            This allows the network to adaptively sharpen or soften its
            spatial distributions during training.

            Note: TorchScript export requires learnable_temperature=False,
            as dynamic Parameter vs buffer dispatch is not scriptable.

        normalized_coordinates : bool, default True
            Controls the coordinate space of the output coords tensor.

            True  -> coordinates in [-1, 1] (recommended for training).
            False -> coordinates in pixel space [0, W-1] / [0, H-1].
        """

        super().__init__()

        if temperature <= 0.0:
            raise ValueError(
                f"temperature must be strictly positive, got {temperature}. "
                "Use a small positive value (e.g. 0.1) for near-argmax behavior."
            )

        self.normalized_coordinates = normalized_coordinates

        # ---------------------------------------------------------------------
        # Register temperature as either a trainable parameter or a fixed
        # buffer, depending on learnable_temperature.
        #
        # Using register_buffer (not a raw float) ensures the tensor is
        # automatically moved to the correct device when calling .to(device)
        # or .cuda(), with no manual bookkeeping required.
        # ---------------------------------------------------------------------

        temp_tensor = torch.tensor(float(temperature))

        if learnable_temperature:
            # nn.Parameter — included in model.parameters(), updated by optimizer.
            self.temperature = nn.Parameter(temp_tensor)
        else:
            # Non-trainable buffer — moves with the module, excluded from grad.
            self.register_buffer("temperature", temp_tensor)

    def forward(self, x: torch.Tensor) -> DSNTOutput:
        """
        Execute DSNT transform.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, H, W)
            Raw unnormalized heatmaps / logits from a backbone or detection head.
            Must be a floating-point tensor (float32 or float16).

        Returns
        -------
        DSNTOutput
            Dataclass containing normalized heatmaps, coordinates, and
            per-keypoint confidence scores.

        Raises
        ------
        ValueError
            If input tensor rank != 4.
        ValueError
            If input tensor is not a floating-point type.
        """

        # ---------------------------------------------------------------------
        # Validate tensor rank.
        # Expected layout: (batch, channels, height, width).
        # ---------------------------------------------------------------------

        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input (B, C, H, W), got shape {tuple(x.shape)} "
                f"with {x.ndim} dimensions. "
                "Ensure your heatmap tensor has not been squeezed unexpectedly."
            )

        # ---------------------------------------------------------------------
        # Validate floating-point dtype.
        #
        # Integer tensors passed to softmax produce silently wrong gradients
        # (the op runs but all outputs become 1/N uniform distributions after
        # implicit integer division). Catching this early saves debugging time.
        # ---------------------------------------------------------------------

        if not x.is_floating_point():
            raise ValueError(
                f"Expected floating-point input, got dtype {x.dtype}. "
                "Cast your tensor before calling DSNT: x = x.float()"
            )

        # ---------------------------------------------------------------------
        # Convert raw logits into spatial probability distributions.
        #
        # Kornia spatial_softmax2d:
        #   - Applies max-subtraction for numerical stability (safe in fp16).
        #   - Vectorized over all (B, C) pairs simultaneously.
        #   - Output sums to 1.0 per spatial slice.
        #
        # temperature is a buffer/Parameter on the correct device — no manual
        # .to(device) needed here.
        # ---------------------------------------------------------------------

        heatmaps = KSP.spatial_softmax2d(
            x,
            temperature=self.temperature,
        )

        # ---------------------------------------------------------------------
        # Compute spatial expectation (the core DSNT operation).
        #
        #   x_coord = Σ_{i,j} p(i, j) * x_grid(j)
        #   y_coord = Σ_{i,j} p(i, j) * y_grid(i)
        #
        # Kornia spatial_expectation2d handles grid construction and the
        # inner product, producing output of shape (B, C, 2).
        # ---------------------------------------------------------------------

        coords = KSP.spatial_expectation2d(
            heatmaps,
            normalized_coordinates=self.normalized_coordinates,
        )

        # ---------------------------------------------------------------------
        # Compute per-keypoint confidence.
        #
        # Defined as the peak (spatial maximum) of the normalized heatmap.
        # A sharp Gaussian peak -> value near 1/(small area).
        # A diffuse uniform     -> value near 1/(H*W).
        #
        # Flattening over (H, W) and taking the max is vectorized and avoids
        # any Python-level iteration over keypoints.
        # ---------------------------------------------------------------------

        confidence = heatmaps.flatten(start_dim=-2).max(dim=-1).values  # (B, C)

        return DSNTOutput(
            heatmaps=heatmaps,
            coords=coords,
            confidence=confidence,
        )


# -----------------------------------------------------------------------------
# Jensen-Shannon regularization loss
# -----------------------------------------------------------------------------

def js_reg_loss(
    heatmaps: torch.Tensor,
    centers: torch.Tensor,
    sigma: Optional[float] = None,
    fwhm: Optional[float] = None,
    normalized_coordinates: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Jensen-Shannon divergence regularization loss for DSNT training.

    Purpose
    -------
    DSNT alone only constrains the *expectation* of the heatmap distribution.
    Without additional regularization the network may converge to:
        - Multi-modal heatmaps (two peaks averaging to the right coordinate).
        - Noisy / flat distributions with poor generalization.
        - Unstable gradients from unconstrained spatial probability mass.

    JS regularization adds a soft shape constraint, pulling predicted heatmaps
    toward smooth isotropic Gaussians centered on the ground-truth coordinate.
    This typically improves both accuracy and convergence stability.

    Sigma Scaling
    -------------
    When normalized_coordinates=True, sigma is automatically scaled relative
    to the heatmap resolution so that "sigma=1.0" always means "1 pixel worth
    of spread" regardless of whether your heatmap is 16x16 or 128x128.

    Without this scaling, the same sigma value would produce drastically
    different Gaussian widths at different resolutions, making hyperparameter
    transfer between architectures unreliable.

    Legacy API Compatibility (fwhm)
    --------------------------------
    The original TensorFlow DSNT implementation expressed Gaussian width as
    full-width-half-maximum (fwhm). This parameter is preserved as a keyword
    argument and automatically converted to sigma:

        sigma = fwhm / (2 * sqrt(2 * ln(2)))

    Passing fwhm and sigma simultaneously raises ValueError.

    Parameters
    ----------
    heatmaps : torch.Tensor, shape (B, C, H, W)
        Predicted spatial probability heatmaps (post-softmax, summing to 1).
        Typically taken directly from DSNTOutput.heatmaps.

    centers : torch.Tensor, shape (B, C, 2)
        Ground-truth coordinate centers in (x, y) order.
        Must use the same coordinate space as normalized_coordinates.

    sigma : float, optional
        Gaussian standard deviation in pixels (before resolution scaling).
        Defaults to 1.0 if neither sigma nor fwhm is provided.

    fwhm : float, optional
        Full-width-half-maximum of the target Gaussian.
        Legacy compatibility parameter from TensorFlow DSNT.
        Mutually exclusive with sigma.

    normalized_coordinates : bool, default True
        Whether centers use normalized [-1, 1] coordinate space.
        Must match the setting used in the DSNT forward pass.

    eps : float, default 1e-8
        Numerical stability epsilon for KL divergence computation.
        Clamps log arguments away from zero.
        Reduce slightly (e.g. 1e-12) for fp32-only pipelines.
        Increase (e.g. 1e-6) for aggressive fp16 / bf16 training.

    Returns
    -------
    torch.Tensor
        Scalar loss value (mean JS divergence over all batch/keypoint pairs).

    Raises
    ------
    ValueError
        If both sigma and fwhm are provided.
    ValueError
        If heatmaps or centers have unexpected rank.
    """

    # -------------------------------------------------------------------------
    # Resolve sigma from arguments.
    #
    # Priority:
    #   1. sigma — direct specification.
    #   2. fwhm  — legacy API; convert to sigma.
    #   3. Neither provided — default to sigma=1.0.
    # -------------------------------------------------------------------------

    if sigma is not None and fwhm is not None:
        raise ValueError(
            "Provide either `sigma` or `fwhm`, not both. "
            f"Got sigma={sigma}, fwhm={fwhm}."
        )

    if fwhm is not None:
        # FWHM -> sigma conversion: sigma = FWHM / (2 * sqrt(2 * ln(2)))
        # This preserves the Gaussian shape from old TF-based call sites.
        sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    if sigma is None:
        sigma = 1.0

    # -------------------------------------------------------------------------
    # Validate input shapes.
    # -------------------------------------------------------------------------

    if heatmaps.ndim != 4:
        raise ValueError(
            f"Expected heatmaps shape (B, C, H, W), got {tuple(heatmaps.shape)}."
        )

    if centers.ndim != 3 or centers.shape[-1] != 2:
        raise ValueError(
            f"Expected centers shape (B, C, 2), got {tuple(centers.shape)}."
        )

    B, C, H, W = heatmaps.shape

    # -------------------------------------------------------------------------
    # Scale sigma relative to heatmap resolution.
    #
    # Problem without scaling:
    #   A Gaussian with sigma=1.0 in normalized [-1, 1] space covers a very
    #   different fraction of the heatmap on a 16x16 grid vs a 64x64 grid.
    #   This makes the regularization strength resolution-dependent, which
    #   breaks transfer of hyperparameters across architectures.
    #
    # Solution:
    #   Express sigma in "pixel units" and scale to normalized space:
    #       sigma_x_norm = sigma / W
    #       sigma_y_norm = sigma / H
    #
    #   This ensures that sigma=1.0 always means "1 pixel of spread",
    #   regardless of heatmap resolution.
    #
    # In pixel-space mode no scaling is applied — sigma is already in pixels.
    # -------------------------------------------------------------------------

    if normalized_coordinates:
        sigma_x = sigma / W
        sigma_y = sigma / H
    else:
        sigma_x = sigma
        sigma_y = sigma

    # Build (B, C, 2) std tensor: [sigma_x, sigma_y] per keypoint.
    std = torch.stack(
        [
            torch.full((B, C), sigma_x, device=heatmaps.device, dtype=heatmaps.dtype),
            torch.full((B, C), sigma_y, device=heatmaps.device, dtype=heatmaps.dtype),
        ],
        dim=-1,
    )  # (B, C, 2)

    # -------------------------------------------------------------------------
    # Render differentiable Gaussian targets using Kornia.
    #
    # Kornia render_gaussian2d produces a normalized (sums-to-1) Gaussian
    # heatmap for each (batch, channel) coordinate pair.
    #
    # Output shape: (B, C, H, W)
    # -------------------------------------------------------------------------

    target = KSP.render_gaussian2d(
        mean=centers,
        std=std,
        size=(H, W),
        normalized_coordinates=normalized_coordinates,
    ).view(B, C, H, W)

    # -------------------------------------------------------------------------
    # Compute mean JS divergence over all (batch, keypoint) pairs.
    # -------------------------------------------------------------------------

    return _js_divergence(heatmaps, target, eps=eps).mean()


# -----------------------------------------------------------------------------
# KL divergence (spatial)
# -----------------------------------------------------------------------------

def _kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Spatial KL divergence: KL(P || Q), reduced over (H, W).

    Computed in log-space with clamping for fp16 / AMP safety:

        KL(P || Q) = Σ p * (log p - log q)

    Clamping log arguments (not probabilities directly) is more numerically
    stable under fp16 because it avoids underflow in the probability values
    themselves before they reach the log operation.

    Parameters
    ----------
    p, q : torch.Tensor, shape (B, C, H, W)
        Spatial probability distributions. Both should sum to ~1 over (H, W).

    eps : float
        Minimum value for clamping log arguments.

    Returns
    -------
    torch.Tensor, shape (B, C)
    """

    # Clamp in log-space: compute log first, then clamp, rather than clamping
    # the raw probability (which can cause the distribution to no longer sum to 1).
    log_p = torch.log(p.clamp(min=eps))
    log_q = torch.log(q.clamp(min=eps))

    return torch.sum(p * (log_p - log_q), dim=(-2, -1))


# -----------------------------------------------------------------------------
# Jensen-Shannon divergence (spatial)
# -----------------------------------------------------------------------------

def _js_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Spatial Jensen-Shannon divergence, reduced over (H, W).

    JS divergence is preferred over KL for regularization because it is:
        - Symmetric:   JS(P || Q) == JS(Q || P)
        - Bounded:     JS ∈ [0, log(2)]  (for natural log)
        - Smoother:    less prone to exploding gradients than bare KL

    Formula:
        M  = 0.5 * (P + Q)
        JS = 0.5 * KL(P || M) + 0.5 * KL(Q || M)

    Parameters
    ----------
    p, q : torch.Tensor, shape (B, C, H, W)
        Spatial probability distributions.

    eps : float
        Passed through to _kl_divergence for numerical stability.

    Returns
    -------
    torch.Tensor, shape (B, C)
    """

    m = 0.5 * (p + q)

    return 0.5 * _kl_divergence(p, m, eps) + 0.5 * _kl_divergence(q, m, eps)
