# utils/detect_face.py
"""
Low-level MTCNN cascade implementation.
========================================

This module implements the three-stage MTCNN detection cascade and supporting
geometry utilities. It is called by MTCNN.detect() and should not typically be
used directly — go through the MTCNN class instead.

Stage summary:
    Stage 1 (PNet)  — build image pyramid, run sliding-window fully-conv net,
                       generate initial box candidates with NMS per scale.
    Stage 2 (RNet)  — crop and resize candidates to 24x24, refine and filter.
    Stage 3 (ONet)  — crop and resize survivors to 48x48, final regression +
                       five-point landmark localization.
"""

from __future__ import annotations

import math
import os
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F
from torchvision.ops.boxes import batched_nms
from PIL import Image

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Batch processing helper
# ---------------------------------------------------------------------------

def _fixed_batch_process(
    im_data: torch.Tensor,
    model: torch.nn.Module,
    batch_size: int = 512,
) -> Tuple[torch.Tensor, ...]:
    """
    Run model inference in fixed-size batches to avoid OOM on large inputs.

    Splits im_data along dim=0, runs model, concatenates outputs.
    batch_size=512 is a safe default for most GPUs with 6GB+ VRAM.
    Reduce if hitting OOM during the RNet or ONet stages.

    Parameters
    ----------
    im_data : torch.Tensor
        Input batch, shape (N, C, H, W).
    model : nn.Module
        Model returning a tuple of tensors.
    batch_size : int
        Maximum number of samples per forward pass.

    Returns
    -------
    Tuple of concatenated output tensors.
    """
    outputs = []
    for start in range(0, len(im_data), batch_size):
        outputs.append(model(im_data[start : start + batch_size]))
    return tuple(torch.cat(v, dim=0) for v in zip(*outputs))


# ---------------------------------------------------------------------------
# Main cascade
# ---------------------------------------------------------------------------

@torch.inference_mode()
def detect_face(
    imgs: Union['PIL.Image.Image', np.ndarray, torch.Tensor, list],
    minsize: int,
    pnet: torch.nn.Module,
    rnet: torch.nn.Module,
    onet: torch.nn.Module,
    threshold: List[float],
    factor: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run full P/R/O-Net detection cascade on one or more images.

    Parameters
    ----------
    imgs : PIL Image, np.ndarray (uint8), torch.Tensor, or list of these.
        For batch processing pass a list of equal-dimension images.

    minsize : int
        Minimum face size in pixels.

    pnet, rnet, onet : nn.Module
        Pretrained MTCNN stage networks, already on `device`.

    threshold : list[float]
        Per-stage confidence thresholds [PNet, RNet, ONet].

    factor : float
        Image pyramid downscaling factor (0 < factor < 1).

    device : torch.device
        Device for all tensor operations.

    Returns
    -------
    batch_boxes : np.ndarray of object dtype, shape (B,)
        Each element is an (N, 5) array [x1, y1, x2, y2, score] or empty.

    batch_points : np.ndarray of object dtype, shape (B,)
        Each element is an (N, 5, 2) array of landmark (x, y) coords or empty.
    """
    # ------------------------------------------------------------------
    # Normalize inputs to a single NCHW float tensor on `device`.
    # ------------------------------------------------------------------
    if isinstance(imgs, (np.ndarray, torch.Tensor)):
        if isinstance(imgs, np.ndarray):
            imgs = torch.as_tensor(imgs.copy(), device=device)
        else:
            imgs = imgs.to(device)
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
    else:
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        if any(img.size != imgs[0].size for img in imgs):
            raise ValueError(
                "Batch processing requires all images to have equal dimensions. "
                "Process variable-size images individually or resize first."
            )
        imgs = torch.as_tensor(
            np.stack([np.uint8(img) for img in imgs]).copy(),
            device=device,
        )

    # Reorder from NHWC to NCHW and cast to model dtype.
    model_dtype = next(pnet.parameters()).dtype
    imgs = imgs.permute(0, 3, 1, 2).to(model_dtype)

    batch_size = len(imgs)
    h, w = imgs.shape[2], imgs.shape[3]

    # ------------------------------------------------------------------
    # Stage 1 — PNet: image pyramid + sliding-window proposals
    # ------------------------------------------------------------------

    # Build scale pyramid such that the smallest detectable face (minsize px)
    # maps to the PNet receptive field size of 12x12.
    m = 12.0 / minsize
    scales = []
    scale = m
    min_side = min(h, w) * m
    while min_side >= 12:
        scales.append(scale)
        scale *= factor
        min_side *= factor

    all_boxes: List[torch.Tensor] = []
    all_image_inds: List[torch.Tensor] = []
    scale_picks: List[torch.Tensor] = []
    offset = 0

    for scale in scales:
        scaled_h = int(h * scale + 1)
        scaled_w = int(w * scale + 1)
        im_data = imresample(imgs, (scaled_h, scaled_w))
        im_data = (im_data - 127.5) * 0.0078125  # fixed standardization

        reg, probs = pnet(im_data)

        boxes_scale, image_inds_scale = _generate_bounding_box(
            reg, probs[:, 1], scale, threshold[0]
        )
        all_boxes.append(boxes_scale)
        all_image_inds.append(image_inds_scale)

        # NMS within each (scale, image) pair.
        pick = batched_nms(
            boxes_scale[:, :4], boxes_scale[:, 4], image_inds_scale, 0.5
        )
        scale_picks.append(pick + offset)
        offset += boxes_scale.shape[0]

    boxes = torch.cat(all_boxes, dim=0)
    image_inds = torch.cat(all_image_inds, dim=0)
    scale_picks = torch.cat(scale_picks, dim=0)

    boxes, image_inds = boxes[scale_picks], image_inds[scale_picks]

    # NMS within each image across all scales.
    pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
    boxes, image_inds = boxes[pick], image_inds[pick]

    # Bounding box regression: apply offsets from PNet regression head.
    regw = boxes[:, 2] - boxes[:, 0]
    regh = boxes[:, 3] - boxes[:, 1]
    qq1 = boxes[:, 0] + boxes[:, 5] * regw
    qq2 = boxes[:, 1] + boxes[:, 6] * regh
    qq3 = boxes[:, 2] + boxes[:, 7] * regw
    qq4 = boxes[:, 3] + boxes[:, 8] * regh
    boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]], dim=1)

    # Square the boxes (MTCNN convention: aspect ratio is always 1:1).
    boxes = _rerec(boxes)
    y, ey, x, ex = _pad(boxes, w, h)

    # ------------------------------------------------------------------
    # Stage 2 — RNet: 24x24 crop refinement
    # ------------------------------------------------------------------
    if len(boxes) > 0:
        crops = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                crop = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]]
                crops.append(imresample(crop.unsqueeze(0), (24, 24)))

        im_data = (torch.cat(crops, dim=0) - 127.5) * 0.0078125
        out = _fixed_batch_process(im_data, rnet)

        out0 = out[0].permute(1, 0)
        score = out[1].permute(1, 0)[1, :]
        ipass = score > threshold[1]
        boxes = torch.cat([boxes[ipass, :4], score[ipass].unsqueeze(1)], dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, ipass].permute(1, 0)

        pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
        boxes = _bbreg(boxes, mv)
        boxes = _rerec(boxes)

    # ------------------------------------------------------------------
    # Stage 3 — ONet: 48x48 final regression + landmark localization
    # ------------------------------------------------------------------
    points = torch.zeros(0, 5, 2, device=device)

    if len(boxes) > 0:
        y, ey, x, ex = _pad(boxes, w, h)
        crops = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                crop = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]]
                crops.append(imresample(crop.unsqueeze(0), (48, 48)))

        im_data = (torch.cat(crops, dim=0) - 127.5) * 0.0078125
        out = _fixed_batch_process(im_data, onet)

        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        score = out[2].permute(1, 0)[1, :]
        raw_points = out1

        ipass = score > threshold[2]
        raw_points = raw_points[:, ipass]
        boxes = torch.cat([boxes[ipass, :4], score[ipass].unsqueeze(1)], dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, ipass].permute(1, 0)

        # Scale landmark coordinates from [0, 1] to pixel space.
        w_i = boxes[:, 2] - boxes[:, 0] + 1
        h_i = boxes[:, 3] - boxes[:, 1] + 1
        pts_x = w_i.repeat(5, 1) * raw_points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
        pts_y = h_i.repeat(5, 1) * raw_points[5:10, :] + boxes[:, 1].repeat(5, 1) - 1
        points = torch.stack([pts_x, pts_y]).permute(2, 1, 0)

        boxes = _bbreg(boxes, mv)

        # Final NMS with Min-area strategy (less aggressive than union IoU).
        pick = _batched_nms_numpy(boxes[:, :4], boxes[:, 4], image_inds, 0.7, 'Min')
        boxes, image_inds, points = boxes[pick], image_inds[pick], points[pick]

    # ------------------------------------------------------------------
    # Move results to CPU and split by image index.
    # ------------------------------------------------------------------
    boxes_np = boxes.cpu().numpy()
    points_np = points.cpu().numpy()
    image_inds_cpu = image_inds.cpu()

    batch_boxes_out = []
    batch_points_out = []
    for b in range(batch_size):
        inds = np.where(image_inds_cpu == b)[0]
        batch_boxes_out.append(boxes_np[inds].copy())
        batch_points_out.append(points_np[inds].copy())

    return (
        np.array(batch_boxes_out, dtype=object),
        np.array(batch_points_out, dtype=object),
    )


# ---------------------------------------------------------------------------
# Geometry utilities (module-private)
# ---------------------------------------------------------------------------

def _bbreg(
    boundingbox: torch.Tensor,
    reg: torch.Tensor,
) -> torch.Tensor:
    """Apply bounding box regression offsets."""
    if reg.shape[1] == 1:
        reg = reg.reshape(reg.shape[2], reg.shape[3])

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, :4] = torch.stack([b1, b2, b3, b4], dim=1)
    return boundingbox


def _generate_bounding_box(
    reg: torch.Tensor,
    probs: torch.Tensor,
    scale: float,
    thresh: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert PNet spatial output maps into candidate bounding boxes.

    Parameters
    ----------
    reg : torch.Tensor
        Bounding box regression output from PNet.
    probs : torch.Tensor
        Face probability map from PNet.
    scale : float
        Current pyramid scale factor.
    thresh : float
        Confidence threshold.

    Returns
    -------
    boundingbox : torch.Tensor, shape (N, 9)
        [x1, y1, x2, y2, score, reg_dx1, reg_dy1, reg_dx2, reg_dy2]
    image_inds : torch.Tensor, shape (N,)
        Batch index per box.
    """
    stride = 2
    cellsize = 12

    reg = reg.permute(1, 0, 2, 3)
    mask = probs >= thresh
    mask_inds = mask.nonzero()
    image_inds = mask_inds[:, 0]
    score = probs[mask]
    reg = reg[:, mask].permute(1, 0)
    # mask_inds[:, 1:] is (row, col); flip to (col, row) for (x, y).
    bb = mask_inds[:, 1:].to(reg.dtype).flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
    boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
    return boundingbox, image_inds


def _nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    method: str,
) -> np.ndarray:
    """
    Non-Maximum Suppression in numpy.

    Used only for the final ONet stage where 'Min' IoU strategy is needed.
    torchvision.ops.batched_nms uses the union-IoU strategy, which is too
    aggressive for the final stage landmark-preserving NMS.

    Parameters
    ----------
    method : str
        'Min' — IoU denominator is min(area_i, area_j).
        Anything else — standard union IoU.
    """
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int16)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = np.argsort(scores)
    pick = np.zeros(len(scores), dtype=np.int16)
    counter = 0

    while order.size > 0:
        i = order[-1]
        pick[counter] = i
        counter += 1
        idx = order[:-1]

        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])

        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)

        if method == 'Min':
            iou = inter / np.minimum(area[i], area[idx])
        else:
            iou = inter / (area[i] + area[idx] - inter)

        order = order[iou <= threshold]

    return pick[:counter]


def _batched_nms_numpy(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    idxs: torch.Tensor,
    threshold: float,
    method: str,
) -> torch.Tensor:
    """
    Batched NMS using numpy backend (for Min-IoU strategy).

    Separates boxes by class index via large offset trick, then applies
    _nms_numpy. Returns selected indices as a torch.Tensor on original device.
    """
    device = boxes.device
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)

    # Offset boxes by class index so boxes from different images never overlap.
    offsets = idxs.to(boxes) * (boxes.max() + 1)
    shifted = (boxes + offsets[:, None]).cpu().numpy()
    keep = _nms_numpy(shifted, scores.cpu().numpy(), threshold, method)
    return torch.as_tensor(keep, dtype=torch.long, device=device)


def _pad(
    boxes: torch.Tensor,
    w: int,
    h: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clamp box coordinates to image boundaries."""
    b = boxes.trunc().int().cpu().numpy()
    x, y, ex, ey = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    x = np.clip(x, 1, None)
    y = np.clip(y, 1, None)
    ex = np.clip(ex, None, w)
    ey = np.clip(ey, None, h)
    return y, ey, x, ex


def _rerec(bbox: torch.Tensor) -> torch.Tensor:
    """
    Convert rectangular boxes to squares by extending the shorter side.

    MTCNN convention: all crops are square to match the fixed-size networks.
    """
    h = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2] - bbox[:, 0]
    side = torch.max(w, h)
    bbox[:, 0] = bbox[:, 0] + w * 0.5 - side * 0.5
    bbox[:, 1] = bbox[:, 1] + h * 0.5 - side * 0.5
    bbox[:, 2:4] = bbox[:, :2] + side.unsqueeze(1).repeat(1, 2)
    return bbox


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def imresample(img: torch.Tensor, sz: Tuple[int, int]) -> torch.Tensor:
    """
    Resize a batch of images using area interpolation.

    'area' mode is equivalent to OpenCV's INTER_AREA — it correctly
    anti-aliases when downsampling, which matters for the pyramid stages.
    """
    return interpolate(img, size=sz, mode='area')


def crop_resize(
    img: Union['PIL.Image.Image', np.ndarray, torch.Tensor],
    box: list,
    image_size: int,
) -> Union[np.ndarray, torch.Tensor, 'PIL.Image.Image']:
    """Crop and resize an image to image_size x image_size."""
    if isinstance(img, np.ndarray):
        if not _CV2_AVAILABLE:
            raise ImportError(
                "OpenCV (cv2) is required for numpy array input. "
                "Install with: pip install opencv-python"
            )
        crop = img[box[1]:box[3], box[0]:box[2]]
        return cv2.resize(
            crop, (image_size, image_size), interpolation=cv2.INTER_AREA
        ).copy()

    elif isinstance(img, torch.Tensor):
        crop = img[box[1]:box[3], box[0]:box[2]]
        return (
            imresample(
                crop.permute(2, 0, 1).unsqueeze(0).float(),
                (image_size, image_size),
            )
            .byte()
            .squeeze(0)
            .permute(1, 2, 0)
        )

    else:
        # PIL Image
        return img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)


def save_img(
    img: Union[np.ndarray, 'PIL.Image.Image'],
    path: str,
) -> None:
    """Save an image to disk. Handles both numpy arrays and PIL Images."""
    if isinstance(img, np.ndarray):
        if not _CV2_AVAILABLE:
            raise ImportError("cv2 required to save numpy array images.")
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        img.save(path)


def get_size(
    img: Union[np.ndarray, torch.Tensor, 'PIL.Image.Image'],
) -> Tuple[int, int]:
    """Return (width, height) for any supported image type."""
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return img.shape[1], img.shape[0]
    return img.size


def extract_face(
    img: Union['PIL.Image.Image', np.ndarray, torch.Tensor],
    box: np.ndarray,
    image_size: int = 160,
    margin: int = 0,
    save_path: Optional[str] = None,
) -> torch.Tensor:
    """
    Extract and resize a face region from an image.

    The margin is applied proportionally so that it represents the same
    fraction of the output image regardless of the original box size.

    Parameters
    ----------
    img : PIL Image, np.ndarray, or torch.Tensor
    box : np.ndarray, shape (4,)
        Bounding box [x1, y1, x2, y2] in pixel coordinates.
    image_size : int
        Output crop size (square). Default 160.
    margin : int
        Margin to add around the box, in output-image pixels. Default 0.
    save_path : str or None
        If provided, save the extracted face to disk.

    Returns
    -------
    torch.Tensor, shape (3, image_size, image_size), dtype float32.
    """
    # Compute margin in original-image pixels, proportional to box size.
    margin_x = margin * (box[2] - box[0]) / (image_size - margin)
    margin_y = margin * (box[3] - box[1]) / (image_size - margin)

    raw_w, raw_h = get_size(img)
    box_padded = [
        int(max(box[0] - margin_x / 2, 0)),
        int(max(box[1] - margin_y / 2, 0)),
        int(min(box[2] + margin_x / 2, raw_w)),
        int(min(box[3] + margin_y / 2, raw_h)),
    ]

    face = crop_resize(img, box_padded, image_size)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
        save_img(face, save_path)

    return F.to_tensor(np.float32(face))
