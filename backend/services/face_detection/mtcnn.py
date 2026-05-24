# mtcnn.py
"""
MTCNN — Multi-Task Cascaded Convolutional Networks
====================================================

Three-stage face detection cascade:
    PNet (Proposal Network)    — fast sliding-window candidate generation
    RNet (Refinement Network)  — candidate refinement and filtering
    ONet (Output Network)      — final bounding box + landmark regression

Reference:
    "Joint Face Detection and Alignment using Multi-task Cascaded
     Convolutional Networks", Zhang et al. 2016

Usage
-----
    mtcnn = MTCNN(device='cuda')

    # Detect + crop faces (torch.Tensor outputs)
    faces = mtcnn(img)                        # (3, 160, 160)
    faces, probs = mtcnn(img, return_prob=True)

    # Detect only — bounding boxes + landmarks
    boxes, probs = mtcnn.detect(img)
    boxes, probs, points = mtcnn.detect(img, landmarks=True)

Changes from legacy version
----------------------------
    - Removed `self.training = False` in PNet/RNet/ONet.
      This was overriding nn.Module's training attribute (a managed property)
      with a plain instance attribute, masking calls to .eval()/.train().
      Models now properly call .eval() after instantiation.
    - Removed self.device instance attribute from PNet/RNet/ONet
      (anti-pattern; device is managed by .to(device) on the parent MTCNN).
    - torch.inference_mode() replaces torch.no_grad() in detect().
    - Type hints throughout.
    - selection_method validated at init.
"""

from __future__ import annotations

import os
from typing import Optional, Union, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .utils.detect_face import detect_face, extract_face
from .inception_resnet_v1 import fixed_image_standardization


# ---------------------------------------------------------------------------
# Valid selection methods
# ---------------------------------------------------------------------------

_SELECTION_METHODS = frozenset({
    'largest',
    'probability',
    'center_weighted_size',
    'largest_over_threshold',
})


# ---------------------------------------------------------------------------
# PNet — Proposal Network
# ---------------------------------------------------------------------------

class PNet(nn.Module):
    """
    Proposal Network: fast fully-convolutional face candidate generator.

    Operates at multiple scales via image pyramid.
    Outputs bounding box regressions and face/non-face probabilities
    as spatial feature maps (no fully-connected layers).
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

        if pretrained:
            state_dict_path = os.path.join(
                os.path.dirname(__file__), 'data', 'pnet.pt'
            )
            self.load_state_dict(
                torch.load(state_dict_path, map_location='cpu', weights_only=True)
            )

        # eval() is the correct way to disable dropout/BN train behavior.
        # Do NOT use self.training = False (overrides nn.Module property).
        self.eval()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        b = self.conv4_2(x)
        a = self.softmax4_1(self.conv4_1(x))
        return b, a


# ---------------------------------------------------------------------------
# RNet — Refinement Network
# ---------------------------------------------------------------------------

class RNet(nn.Module):
    """
    Refinement Network: rejects false positives from PNet candidates.

    Operates on fixed 24x24 crops. Outputs refined bounding boxes
    and updated face/non-face probabilities.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)

        if pretrained:
            state_dict_path = os.path.join(
                os.path.dirname(__file__), 'data', 'rnet.pt'
            )
            self.load_state_dict(
                torch.load(state_dict_path, map_location='cpu', weights_only=True)
            )

        self.eval()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.pool1(self.prelu1(self.conv1(x)))
        x = self.pool2(self.prelu2(self.conv2(x)))
        x = self.prelu3(self.conv3(x))
        # permute matches the original TF weight layout (NHWC -> NCHW order swap).
        x = self.prelu4(self.dense4(x.permute(0, 3, 2, 1).contiguous().view(x.shape[0], -1)))
        a = self.softmax5_1(self.dense5_1(x))
        b = self.dense5_2(x)
        return b, a


# ---------------------------------------------------------------------------
# ONet — Output Network
# ---------------------------------------------------------------------------

class ONet(nn.Module):
    """
    Output Network: final precise bounding box regression + landmark localization.

    Operates on fixed 48x48 crops. Outputs refined boxes, five facial
    landmark coordinates, and final face/non-face probabilities.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)

        if pretrained:
            state_dict_path = os.path.join(
                os.path.dirname(__file__), 'data', 'onet.pt'
            )
            self.load_state_dict(
                torch.load(state_dict_path, map_location='cpu', weights_only=True)
            )

        self.eval()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.pool1(self.prelu1(self.conv1(x)))
        x = self.pool2(self.prelu2(self.conv2(x)))
        x = self.pool3(self.prelu3(self.conv3(x)))
        x = self.prelu4(self.conv4(x))
        x = self.prelu5(self.dense5(x.permute(0, 3, 2, 1).contiguous().view(x.shape[0], -1)))
        a = self.softmax6_1(self.dense6_1(x))
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        return b, c, a


# ---------------------------------------------------------------------------
# MTCNN — full detection pipeline
# ---------------------------------------------------------------------------

class MTCNN(nn.Module):
    """
    Multi-Task Cascaded CNN face detector.

    Accepts PIL Images, numpy arrays (uint8), torch.Tensors, or lists of any of these.
    Returns face crops as standardized torch.Tensors and/or bounding boxes.

    Parameters
    ----------
    image_size : int
        Output face crop size in pixels (square). Default 160.

    margin : int
        Margin added to the bounding box before cropping, in pixels of the
        final image_size. Applied proportionally to the original box size,
        so margin is scale-invariant. Default 0.

    min_face_size : int
        Minimum face height/width in pixels to detect. Default 20.

    thresholds : list[float]
        Detection confidence thresholds for [PNet, RNet, ONet] stages.
        Default [0.6, 0.7, 0.7].

    factor : float
        Scaling factor for the image pyramid. Default 0.709.

    post_process : bool
        If True, apply fixed_image_standardization to face crops before
        returning. Default True.

    keep_all : bool
        If True, return all detected faces. If False, return only the
        face selected by selection_method. Default False.

    selection_method : str
        Face selection heuristic when keep_all=False:
            'largest'               — largest bounding box area
            'probability'           — highest detection confidence
            'center_weighted_size'  — largest box penalized by offset from image center
            'largest_over_threshold'— largest box above a confidence threshold
        Default 'largest'.

    device : torch.device or str or None
        Device for all subnetwork inference. Default None (CPU).
    """

    def __init__(
        self,
        image_size: int = 160,
        margin: int = 0,
        min_face_size: int = 20,
        thresholds: Optional[List[float]] = None,
        factor: float = 0.709,
        post_process: bool = True,
        keep_all: bool = False,
        selection_method: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()

        if thresholds is None:
            thresholds = [0.6, 0.7, 0.7]

        # Resolve and validate selection_method before touching any submodules.
        if selection_method is None:
            selection_method = 'largest'
        if selection_method not in _SELECTION_METHODS:
            raise ValueError(
                f"Unknown selection_method '{selection_method}'. "
                f"Choose from: {sorted(_SELECTION_METHODS)}"
            )

        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.post_process = post_process
        self.keep_all = keep_all
        self.selection_method = selection_method

        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

        # Device management: store as torch.device, move subnetworks via .to().
        # This is the correct pattern — do NOT assign self.device to each submodule.
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.to(self.device)

    def forward(
        self,
        img: Union['PIL.Image.Image', np.ndarray, torch.Tensor, list],
        save_path: Optional[Union[str, List[str]]] = None,
        return_prob: bool = False,
    ) -> Union[Optional[torch.Tensor], Tuple[Optional[torch.Tensor], Optional[np.ndarray]]]:
        """
        Detect and return cropped face tensors.

        Parameters
        ----------
        img : PIL Image, np.ndarray, torch.Tensor, or list of these.

        save_path : str, list[str], or None
            If provided, save cropped faces to disk at these paths.

        return_prob : bool
            If True, also return detection probabilities.

        Returns
        -------
        faces : torch.Tensor or None
            (3, H, W) for single image / single face.
            (N, 3, H, W) for keep_all=True.
            (B, 3, H, W) for list input.
            None if no face detected.

        probs : np.ndarray or None (only when return_prob=True)
        """
        batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)

        if not self.keep_all:
            batch_boxes, batch_probs, batch_points = self.select_boxes(
                batch_boxes, batch_probs, batch_points,
                img, method=self.selection_method,
            )

        faces = self.extract(img, batch_boxes, save_path)

        if return_prob:
            return faces, batch_probs
        return faces

    @torch.inference_mode()
    def detect(
        self,
        img: Union['PIL.Image.Image', np.ndarray, torch.Tensor, list],
        landmarks: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        Detect all faces and return bounding boxes with optional landmarks.

        Parameters
        ----------
        img : PIL Image, np.ndarray, torch.Tensor, or list.

        landmarks : bool
            If True, also return facial landmark points. Default False.

        Returns
        -------
        boxes : np.ndarray, shape (N, 4) or (B, N, 4) for batch input
            Bounding boxes in xyxy pixel coordinates.

        probs : np.ndarray, shape (N,)
            Detection confidence per box.

        points : np.ndarray, shape (N, 5, 2) — only if landmarks=True
            Five facial landmark coordinates (x, y) per detection.
        """
        batch_boxes, batch_points = detect_face(
            img,
            self.min_face_size,
            self.pnet, self.rnet, self.onet,
            self.thresholds,
            self.factor,
            self.device,
        )

        boxes_out, probs_out, points_out = [], [], []

        for box, point in zip(batch_boxes, batch_points):
            box = np.array(box)
            point = np.array(point)

            if len(box) == 0:
                boxes_out.append(None)
                probs_out.append([None])
                points_out.append(None)
            else:
                if self.selection_method == 'largest':
                    # Sort by descending box area.
                    order = np.argsort(
                        (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
                    )[::-1]
                    box = box[order]
                    point = point[order]
                boxes_out.append(box[:, :4])
                probs_out.append(box[:, 4])
                points_out.append(point)

        boxes_out = np.array(boxes_out, dtype=object)
        probs_out = np.array(probs_out, dtype=object)
        points_out = np.array(points_out, dtype=object)

        # Unwrap single-image results (no batch dimension).
        is_batch = (
            isinstance(img, (list, tuple))
            or (isinstance(img, np.ndarray) and img.ndim == 4)
            or (isinstance(img, torch.Tensor) and img.ndim == 4)
        )
        if not is_batch:
            boxes_out = boxes_out[0]
            probs_out = probs_out[0]
            points_out = points_out[0]

        if landmarks:
            return boxes_out, probs_out, points_out
        return boxes_out, probs_out

    def select_boxes(
        self,
        all_boxes: np.ndarray,
        all_probs: np.ndarray,
        all_points: np.ndarray,
        imgs: Union['PIL.Image.Image', np.ndarray, torch.Tensor, list],
        method: str = 'probability',
        threshold: float = 0.9,
        center_weight: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select a single bounding box per image from multiple candidates.

        Parameters
        ----------
        all_boxes, all_probs, all_points : np.ndarray
            Outputs from detect() with landmarks=True.

        imgs : image or list of images
            Used to compute image center for center_weighted_size method.

        method : str
            Selection heuristic. One of _SELECTION_METHODS.

        threshold : float
            Minimum confidence for 'largest_over_threshold'. Default 0.9.

        center_weight : float
            Penalty weight for offset from image center in
            'center_weighted_size'. Default 2.0.

        Returns
        -------
        Tuple of selected boxes, probs, points (one per image).
        """
        is_batch = (
            isinstance(imgs, (list, tuple))
            or (isinstance(imgs, np.ndarray) and imgs.ndim == 4)
            or (isinstance(imgs, torch.Tensor) and imgs.ndim == 4)
        )

        if not is_batch:
            imgs = [imgs]
            all_boxes = [all_boxes]
            all_probs = [all_probs]
            all_points = [all_points]

        selected_boxes, selected_probs, selected_points = [], [], []

        for boxes, points, probs, img in zip(all_boxes, all_points, all_probs, imgs):
            if boxes is None:
                selected_boxes.append(None)
                selected_probs.append([None])
                selected_points.append(None)
                continue

            boxes = np.array(boxes)
            probs = np.array(probs)
            points = np.array(points)

            if method == 'largest':
                order = np.argsort(
                    (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                )[::-1]

            elif method == 'probability':
                order = np.argsort(probs)[::-1]

            elif method == 'center_weighted_size':
                sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                img_center = (img.width / 2, img.height / 2)
                centers = np.column_stack([
                    (boxes[:, 0] + boxes[:, 2]) / 2,
                    (boxes[:, 1] + boxes[:, 3]) / 2,
                ])
                offset_sq = np.sum((centers - img_center) ** 2, axis=1)
                order = np.argsort(sizes - offset_sq * center_weight)[::-1]

            elif method == 'largest_over_threshold':
                mask = probs > threshold
                if not mask.any():
                    selected_boxes.append(None)
                    selected_probs.append([None])
                    selected_points.append(None)
                    continue
                boxes = boxes[mask]
                points = points[mask]
                order = np.argsort(
                    (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                )[::-1]

            selected_boxes.append(boxes[order][[0]])
            selected_probs.append(probs[order][[0]])
            selected_points.append(points[order][[0]])

        if is_batch:
            return (
                np.array(selected_boxes),
                np.array(selected_probs),
                np.array(selected_points),
            )
        return selected_boxes[0], selected_probs[0][0], selected_points[0]

    def extract(
        self,
        img: Union['PIL.Image.Image', np.ndarray, torch.Tensor, list],
        batch_boxes: np.ndarray,
        save_path: Optional[Union[str, List[str]]],
    ) -> Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]]:
        """
        Extract and standardize face crops from detected bounding boxes.

        Parameters
        ----------
        img : image or list of images
        batch_boxes : np.ndarray
            Bounding boxes from detect() or select_boxes().
        save_path : str, list[str], or None

        Returns
        -------
        torch.Tensor or list of torch.Tensor or None
        """
        is_batch = (
            isinstance(img, (list, tuple))
            or (isinstance(img, np.ndarray) and img.ndim == 4)
            or (isinstance(img, torch.Tensor) and img.ndim == 4)
        )

        if not is_batch:
            img = [img]
            batch_boxes = [batch_boxes]

        if save_path is None:
            save_paths = [None] * len(img)
        elif isinstance(save_path, str):
            save_paths = [save_path]
        else:
            save_paths = save_path

        faces = []
        for im, boxes, path in zip(img, batch_boxes, save_paths):
            if boxes is None:
                faces.append(None)
                continue

            if not self.keep_all:
                boxes = boxes[[0]]

            faces_im = []
            for i, box in enumerate(boxes):
                face_path = path
                if path is not None and i > 0:
                    name, ext = os.path.splitext(path)
                    face_path = f"{name}_{i + 1}{ext}"

                face = extract_face(im, box, self.image_size, self.margin, face_path)

                if self.post_process:
                    face = fixed_image_standardization(face)
                faces_im.append(face)

            faces.append(torch.stack(faces_im) if self.keep_all else faces_im[0])

        return faces if is_batch else faces[0]
