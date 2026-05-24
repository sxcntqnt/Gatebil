# inception_resnet_v1.py
"""
InceptionResnetV1 — Face Embedding Backbone
=============================================

PyTorch implementation of Inception-ResNet-V1, ported from the original
FaceNet TensorFlow implementation (davidsandberg/facenet).

Pretrained weights are available for two datasets:
    'vggface2'      — 8,631 identity classes, general-purpose
    'casia-webface' — 10,575 identity classes, research-grade

Usage
-----
    # Embedding mode (default) — returns 512-d L2-normalized vectors
    model = InceptionResnetV1(pretrained='vggface2').eval()
    embeddings = model(face_batch)  # (B, 512)

    # Classification mode — returns raw logits
    model = InceptionResnetV1(pretrained='vggface2', classify=True).eval()
    logits = model(face_batch)  # (B, 8631)

    # Fine-tuning on a new identity set
    model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=512).eval()

Changes from legacy version
----------------------------
    - Removed self.device attribute (anti-pattern; use .to(device) externally)
    - Replaced custom download_url_to_file with torch.hub.load_state_dict_from_url
      which handles caching, integrity checking, and progress display natively
    - Type hints throughout
    - Cleaner pretrained class count resolution
    - load_weights() is now a module-level function not dependent on module internals
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Pretrained weight registry
# ---------------------------------------------------------------------------

# Weights are hosted as GitHub release assets from the original facenet-pytorch
# repo. torch.hub.load_state_dict_from_url caches them to
# ~/.cache/torch/hub/checkpoints/ on first download.

_PRETRAINED_URLS: dict[str, str] = {
    'vggface2': (
        'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/'
        '20180402-114759-vggface2.pt'
    ),
    'casia-webface': (
        'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/'
        '20180408-102900-casia-webface.pt'
    ),
}

# Number of identity classes each pretrained checkpoint was trained with.
_PRETRAINED_CLASSES: dict[str, int] = {
    'vggface2': 8631,
    'casia-webface': 10575,
}


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class BasicConv2d(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU building block."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
    ) -> None:
        super().__init__()

        # bias=False because BatchNorm provides the bias via its affine params.
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False,
        )
        # eps=0.001 matches the original TensorFlow checkpoint's epsilon.
        # momentum=0.1 is PyTorch default; matches expected running stat behavior.
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        # inplace=False avoids in-place modification that can confuse autograd
        # when the same tensor is referenced elsewhere in the graph.
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Block35(nn.Module):
    """Inception-ResNet-A block (35x35 spatial scale)."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )
        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.branch0(x), self.branch1(x), self.branch2(x)], dim=1)
        return self.relu(self.conv2d(out) * self.scale + x)


class Block17(nn.Module):
    """Inception-ResNet-B block (17x17 spatial scale)."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(128, 128, kernel_size=(7, 1), stride=1, padding=(3, 0)),
        )
        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.branch0(x), self.branch1(x)], dim=1)
        return self.relu(self.conv2d(out) * self.scale + x)


class Block8(nn.Module):
    """
    Inception-ResNet-C block (8x8 spatial scale).

    Parameters
    ----------
    no_relu : bool
        If True, skip the final ReLU. Used for the last Block8 before
        global average pooling, per the original architecture.
    """

    def __init__(self, scale: float = 1.0, no_relu: bool = False) -> None:
        super().__init__()
        self.scale = scale
        self.no_relu = no_relu

        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(192, 192, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        )
        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.no_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2d(torch.cat([self.branch0(x), self.branch1(x)], dim=1))
        out = out * self.scale + x
        return out if self.no_relu else self.relu(out)


class Mixed_6a(nn.Module):
    """Reduction-A block (256 -> 896 channels)."""

    def __init__(self) -> None:
        super().__init__()
        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2),
        )
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch0(x), self.branch1(x), self.branch2(x)], dim=1)


class Mixed_7a(nn.Module):
    """Reduction-B block (896 -> 1792 channels)."""

    def __init__(self) -> None:
        super().__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2),
        )
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)],
            dim=1,
        )


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class InceptionResnetV1(nn.Module):
    """
    Inception-ResNet-V1 face embedding model.

    Input
    -----
    Batch of face crops: (B, 3, 160, 160), float32.
    Apply fixed_image_standardization before passing to this model.

    Output
    ------
    classify=False (default):
        (B, 512) — L2-normalized embedding vectors.
        Use cosine similarity or L2 distance for face matching.

    classify=True:
        (B, num_classes) — raw logits for classification.

    Parameters
    ----------
    pretrained : str or None
        'vggface2', 'casia-webface', or None.
        Weights are downloaded and cached on first use.

    classify : bool
        If True, output logits. If False, output embeddings.

    num_classes : int or None
        Number of output classes when classify=True.
        If pretrained is set and num_classes differs from the checkpoint's
        class count, the logits layer is freshly initialized (transfer learning).

    dropout_prob : float
        Dropout probability before the bottleneck linear. Default 0.6.
    """

    def __init__(
        self,
        pretrained: Optional[str] = None,
        classify: bool = False,
        num_classes: Optional[int] = None,
        dropout_prob: float = 0.6,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Validate argument combinations before building any layers.
        # ------------------------------------------------------------------
        if pretrained is not None and pretrained not in _PRETRAINED_URLS:
            raise ValueError(
                f"Unknown pretrained dataset '{pretrained}'. "
                f"Choose from: {list(_PRETRAINED_URLS.keys())}"
            )

        if pretrained is None and classify and num_classes is None:
            raise ValueError(
                "When pretrained=None and classify=True, num_classes must be set."
            )

        self.pretrained = pretrained
        self.classify = classify
        self.num_classes = num_classes

        # Determine how many classes the pretrained checkpoint head expects.
        # Used only for loading pretrained weights, not for the final output head.
        pretrained_classes = _PRETRAINED_CLASSES.get(pretrained) if pretrained else None

        # ------------------------------------------------------------------
        # Stem
        # ------------------------------------------------------------------
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)

        # ------------------------------------------------------------------
        # Inception-ResNet blocks
        # ------------------------------------------------------------------
        self.repeat_1 = nn.Sequential(*[Block35(scale=0.17) for _ in range(5)])
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(*[Block17(scale=0.10) for _ in range(10)])
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(*[Block8(scale=0.20) for _ in range(5)])
        self.block8 = Block8(no_relu=True)

        # ------------------------------------------------------------------
        # Head
        # ------------------------------------------------------------------
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

        # ------------------------------------------------------------------
        # Pretrained weight loading.
        #
        # The checkpoint includes a logits layer trained on pretrained_classes.
        # We load it fully first, then optionally replace the logits layer
        # with a new randomly initialized one for transfer learning.
        # ------------------------------------------------------------------
        if pretrained is not None:
            # Temporarily add the pretrained logits layer so state_dict keys match.
            self.logits = nn.Linear(512, pretrained_classes)
            load_weights(self, pretrained)

        # ------------------------------------------------------------------
        # Output logits layer (for classify=True).
        #
        # If num_classes matches pretrained, reuse the loaded weights.
        # If num_classes differs (transfer learning), reinitialize.
        # ------------------------------------------------------------------
        if classify and num_classes is not None:
            if num_classes != pretrained_classes:
                # Transfer learning: fresh logits head.
                self.logits = nn.Linear(512, num_classes)
            # else: loaded logits layer is already correct, keep it.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 3, 160, 160)
            Standardized face crops (apply fixed_image_standardization first).

        Returns
        -------
        torch.Tensor
            (B, 512) L2-normalized embeddings, or (B, num_classes) logits.
        """
        # Stem
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)

        # Inception-ResNet blocks
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)

        # Head
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)

        if self.classify:
            return self.logits(x)
        else:
            # L2 normalize for embedding similarity matching.
            return F.normalize(x, p=2, dim=1)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_weights(model: InceptionResnetV1, name: str) -> None:
    """
    Download (if needed) and load pretrained weights into model.

    Weights are cached to ~/.cache/torch/hub/checkpoints/ by
    torch.hub.load_state_dict_from_url. Subsequent calls load from cache.

    Parameters
    ----------
    model : InceptionResnetV1
    name : str
        'vggface2' or 'casia-webface'
    """
    if name not in _PRETRAINED_URLS:
        raise ValueError(
            f"No pretrained weights for '{name}'. "
            f"Available: {list(_PRETRAINED_URLS.keys())}"
        )

    state_dict = torch.hub.load_state_dict_from_url(
        _PRETRAINED_URLS[name],
        progress=True,
        # map_location keeps the download on CPU regardless of model device.
        # The caller moves the model to the right device via .to(device).
        map_location='cpu',
    )
    model.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Image preprocessing utilities
# ---------------------------------------------------------------------------

def fixed_image_standardization(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply fixed standardization matching the FaceNet training preprocessing.

    Formula:
        output = (pixel - 127.5) / 128.0

    This maps uint8 [0, 255] values to approximately [-1, 1].

    Parameters
    ----------
    image_tensor : torch.Tensor
        Raw image tensor with pixel values in [0, 255].

    Returns
    -------
    torch.Tensor
        Standardized tensor ready for InceptionResnetV1 forward pass.
    """
    return (image_tensor - 127.5) / 128.0


def prewhiten(x: torch.Tensor) -> torch.Tensor:
    """
    Per-image whitening (zero mean, unit variance).

    Alternative to fixed_image_standardization. Adapts to each image's
    own statistics rather than a fixed global scale. Useful when input
    pixel distributions vary widely.

    Parameters
    ----------
    x : torch.Tensor
        Single image or batch tensor.

    Returns
    -------
    torch.Tensor
        Whitened tensor.
    """
    mean = x.mean()
    std = x.std()
    # Clamp denominator to avoid divide-by-zero on constant-value images.
    std_adj = std.clamp(min=1.0 / (float(x.numel()) ** 0.5))
    return (x - mean) / std_adj
