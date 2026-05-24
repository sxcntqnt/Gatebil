"""
app.models.id_detector
───────────────────────
Wraps the TF1 frozen-graph DSNT keypoint model behind a clean Python object.

Responsibilities
  - Load the frozen protobuf once and hold the tf.Session.
  - Expose a single `detect(image_nd) -> Keypoints` method.
  - Own all tensor name strings in one place.

The session is closed when the object is garbage-collected or explicitly
via close(). The FastAPI lifespan calls close() on shutdown.

TF1 API note
  Uses tensorflow.compat.v1 with v2 behaviour disabled so this file
  works under TF 2.x installs without requiring a separate TF1 install.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ── TF1 compat shim ────────────────────────────────────────────────────────────
try:
    import tensorflow.compat.v1 as tf  # TF 2.x with compat layer
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf            # Actual TF 1.x install


# ── DSNT helper (imported from old model/dsnt.py, kept local to this module) ──

def _dsnt(activation_map: "tf.Tensor"):
    """
    Differentiable Spatial to Numerical Transform.
    Returns (heatmap, keypoint) tensors for a single channel.
    """
    heatmap = tf.nn.softmax(tf.reshape(activation_map, [tf.shape(activation_map)[0], -1]))
    heatmap = tf.reshape(heatmap, tf.shape(activation_map))

    # Coordinate grids in [-1, 1]
    h = tf.shape(activation_map)[1]
    w = tf.shape(activation_map)[2]
    xs = tf.cast(tf.range(w), tf.float32) / tf.cast(w - 1, tf.float32) * 2 - 1
    ys = tf.cast(tf.range(h), tf.float32) / tf.cast(h - 1, tf.float32) * 2 - 1

    x_coords = tf.reduce_sum(heatmap * xs[tf.newaxis, tf.newaxis, :], axis=[1, 2])
    y_coords = tf.reduce_sum(heatmap * ys[tf.newaxis, :, tf.newaxis], axis=[1, 2])
    keypoint  = tf.stack([x_coords, y_coords], axis=-1)

    return heatmap, keypoint


@dataclass
class Keypoints:
    """Four corner keypoints of the ID card, in pixel space at (600, 800)."""
    raw:     np.ndarray   # shape (4, 2) — normalised coords from DSNT
    pixels:  np.ndarray   # shape (4, 2) — pixel coords at model input resolution


# ── IDCardDetector ─────────────────────────────────────────────────────────────

class IDCardDetector:
    """
    Loads the DSNT frozen graph and runs keypoint detection.

    Usage
    -----
        detector = IDCardDetector(path)
        kps = detector.detect(img_nd)   # img_nd: (H, W, 3) uint8 RGB at 600×800
    """

    # Input resolution expected by the frozen model.
    MODEL_W: int = 600
    MODEL_H: int = 800

    # Tensor names inside the frozen graph.
    _INPUT_NAME:     str = "input:0"
    _ACTIVATION_MAP: str = "heats_map_regression/pred_keypoints/BiasAdd:0"

    def __init__(self, model_path: Path) -> None:
        self._graph   = self._load_graph(model_path)
        self._session = tf.Session(graph=self._graph)
        self._build_output_tensors()
        log.info("IDCardDetector loaded", extra={"path": str(model_path)})

    # ── Public ────────────────────────────────────────────────────────────

    def detect(self, image_nd: np.ndarray) -> Keypoints:
        """
        Run DSNT on a single image.

        Parameters
        ----------
        image_nd : np.ndarray
            RGB uint8 array of shape (MODEL_H, MODEL_W, 3).

        Returns
        -------
        Keypoints
            raw  — DSNT output in [-1, 1]
            pixels — rescaled to MODEL_W × MODEL_H pixel space
        """
        batch = np.expand_dims(image_nd, 0)   # (1, H, W, 3)

        kp1, kp2, kp3, kp4 = self._session.run(
            [self._kp1, self._kp2, self._kp3, self._kp4],
            feed_dict={self._input_tensor: batch},
        )

        raw = np.array([kp1[0], kp2[0], kp3[0], kp4[0]])

        # Map from [-1, 1] to pixel space.
        pixels = ((raw + 1) / 2 * np.array([self.MODEL_W, self.MODEL_H])).astype("int")

        return Keypoints(raw=raw, pixels=pixels)

    def close(self) -> None:
        """Release the TF session. Called from the FastAPI lifespan shutdown."""
        if self._session is not None:
            self._session.close()
            log.info("IDCardDetector TF session closed")

    # ── Private ───────────────────────────────────────────────────────────

    @staticmethod
    def _load_graph(path: Path) -> "tf.Graph":
        with tf.gfile.GFile(str(path), "rb") as fh:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fh.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        return graph

    def _build_output_tensors(self) -> None:
        self._input_tensor = self._graph.get_tensor_by_name(self._INPUT_NAME)
        activation_map     = self._graph.get_tensor_by_name(self._ACTIVATION_MAP)

        _, self._kp1 = _dsnt(activation_map[..., 0])
        _, self._kp2 = _dsnt(activation_map[..., 1])
        _, self._kp3 = _dsnt(activation_map[..., 2])
        _, self._kp4 = _dsnt(activation_map[..., 3])
