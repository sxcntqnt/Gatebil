"""
app.pipelines.liveness
───────────────────────
Liveness challenge evaluation pipeline — called by routes/liveness.py.

Fixes
-----
The original BlinkDetector uses self.counter / self.total as INSTANCE STATE,
meaning concurrent requests share blink counts — a correctness bug in any
multi-threaded server.

This pipeline evaluates blink as a SINGLE-FRAME EAR (Eye Aspect Ratio) check:
if the EAR is below threshold in the submitted frame, the blink is considered
detected. The caller (Go service or client) is responsible for the temporal
sequencing — they submit frames until the challenge passes.

This is the correct API-native design:
  client sends frame → server says "blink detected: yes/no" → client moves on

Flow
----
    frame bytes
        │
        ▼
    decode to BGR
        │
        ▼
    MTCNN detect face → box, landmarks
        │
        ├── blink      → EAR check on landmarks (stateless)
        ├── orientation → vector angle on MTCNN 5-pt landmarks
        └── emotion     → VGG-based emotion classification on face crop
        │
        ▼
    { passed, result }
"""
from __future__ import annotations

import logging
import math

import cv2
import dlib
import numpy as np
import torch
from imutils import face_utils
from PIL import Image

from api.dependency import LivenessDetectors
from core.config import settings
from core.exceptions import LivenessError, NoFaceDetectedError

log = logging.getLogger(__name__)

# ── EAR computation ────────────────────────────────────────────────────────────

def _eye_aspect_ratio(eye: np.ndarray) -> float:
    """
    Compute Eye Aspect Ratio from 6 dlib landmark points.
    EAR = (A + B) / (2 * C) where A, B are vertical distances, C is horizontal.
    """
    A = math.dist(eye[1], eye[5])
    B = math.dist(eye[2], eye[4])
    C = math.dist(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C > 0 else 1.0


def _check_blink_ear(
    frame_bgr: np.ndarray,
    box: np.ndarray,
    shape_predictor,
    threshold: float,
) -> bool:
    """
    Stateless EAR blink check for a single frame.

    Loads 68-point landmarks via dlib, computes EAR for both eyes,
    returns True if the averaged EAR is below the threshold.
    This replaces the stateful BlinkDetector.eye_blink() counter approach.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = box.astype(int)
    rect = dlib.rectangle(x1, y1, x2, y2)

    shape = shape_predictor(gray, rect)
    shape_np = face_utils.shape_to_np(shape)

    (lS, lE) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rS, rE) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    left_ear  = _eye_aspect_ratio(shape_np[lS:lE])
    right_ear = _eye_aspect_ratio(shape_np[rS:rE])
    ear       = (left_ear + right_ear) / 2.0

    log.debug("EAR computed", extra={"ear": round(ear, 3), "threshold": threshold})
    return ear < threshold


# ── Liveness dispatch ──────────────────────────────────────────────────────────

def run_liveness(
    frame_bytes: bytes,
    challenge: str,
    expected: str,
    detectors: LivenessDetectors,
) -> dict:
    """
    Evaluate a single liveness frame.

    Parameters
    ----------
    frame_bytes : bytes
        Raw bytes of the video frame.
    challenge : str
        "blink" | "orientation" | "emotion"
    expected : str
        For orientation: "left" | "right" | "front"
        For emotion:     "smile" | "surprise" | "neutral"
        For blink:       ignored
    detectors : LivenessDetectors
        Bundle of blink / orient / emotion detectors from app.state.

    Returns
    -------
    dict
        { "passed": bool, "result": str }
        Maps directly to ChallengeResponse.

    Raises
    ------
    NoFaceDetectedError
        If MTCNN finds no face in the frame.
    LivenessError
        If a detector raises an unexpected exception.
    """
    # ── Decode ────────────────────────────────────────────────────────────
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise LivenessError("Could not decode the video frame")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_frame  = Image.fromarray(frame_rgb)

    # ── MTCNN detection — boxes + landmarks ───────────────────────────────
    mtcnn = detectors.blink   # blink_detector holds no MTCNN; we use orient.mtcnn
    # NOTE: MTCNN is on app.state — detectors hold the liveness models, not MTCNN.
    # The route passes detectors; MTCNN is accessed separately. We reconstruct
    # the box + landmarks here using the orient_detector's embedded logic.

    try:
        if challenge == "blink":
            result, passed = _eval_blink(frame_bgr, pil_frame, detectors)
        elif challenge == "orientation":
            result, passed = _eval_orientation(pil_frame, expected, detectors)
        else:  # emotion
            result, passed = _eval_emotion(frame_rgb, expected, detectors)
    except (NoFaceDetectedError, LivenessError):
        raise
    except Exception as exc:
        raise LivenessError(f"Liveness evaluation failed: {exc}") from exc

    log.info(
        "liveness evaluated",
        extra={"challenge": challenge, "passed": passed, "result": result},
    )
    return {"passed": passed, "result": str(result)}


# ── Per-challenge evaluators ───────────────────────────────────────────────────

def _get_mtcnn_box_and_landmarks(pil_frame: Image.Image, mtcnn_ref):
    """
    Run MTCNN detection and return (box, landmarks_5pt).
    Raises NoFaceDetectedError if no face is found.
    """
    with torch.inference_mode():
        boxes, _, landmarks = mtcnn_ref.detect(pil_frame, landmarks=True)

    if boxes is None or len(boxes) == 0:
        raise NoFaceDetectedError("No face detected in the liveness frame")

    return boxes[0], landmarks[0]   # take the primary face


def _eval_blink(
    frame_bgr: np.ndarray,
    pil_frame: Image.Image,
    detectors: LivenessDetectors,
) -> tuple[str, bool]:
    """
    Stateless blink detection via EAR.
    Uses dlib 68-point landmarks via the BlinkDetector's predictor.
    """
    blink_det = detectors.blink
    shape_predictor = blink_det.predictor_eyes

    # Get MTCNN box for the face region to pass to dlib.
    # We use orient_detector's MTCNN since blink_detector doesn't hold one.
    # Both are the same model singleton from app.state.
    # The box is extracted from orient_detector's detection.
    orient_det = detectors.orient

    try:
        # Orient detector's detect() expects landmarks, not BGR.
        # Use blink_detector's own dlib detector for the face rect.
        gray      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        dlib_det  = dlib.get_frontal_face_detector()
        rects     = dlib_det(gray, 0)

        if len(rects) == 0:
            raise NoFaceDetectedError("dlib found no face for blink detection")

        rect = rects[0]
        shape = shape_predictor(gray, rect)
        shape_np = face_utils.shape_to_np(shape)

        (lS, lE) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rS, rE) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        ear = (
            _eye_aspect_ratio(shape_np[lS:lE]) +
            _eye_aspect_ratio(shape_np[rS:rE])
        ) / 2.0

        passed = ear < settings.liveness_blink_ear_threshold
        return f"ear={ear:.3f}", passed

    except NoFaceDetectedError:
        raise
    except Exception as exc:
        raise LivenessError(f"Blink EAR computation failed: {exc}") from exc


def _eval_orientation(
    pil_frame: Image.Image,
    expected: str,
    detectors: LivenessDetectors,
) -> tuple[str, bool]:
    """
    Head orientation detection using MTCNN 5-point landmarks.
    FaceOrientationDetector.detect() takes the 5 landmark points directly.
    """
    orient_det = detectors.orient

    # We need MTCNN landmarks — use the blink_detector's associated MTCNN.
    # Both share app.state.mtcnn; here we call detect() directly.
    with torch.inference_mode():
        try:
            boxes, probs, landmarks = orient_det.mtcnn.detect(pil_frame, landmarks=True) \
                if hasattr(orient_det, "mtcnn") else (None, None, None)
        except Exception:
            boxes, landmarks = None, None

    if landmarks is None or len(landmarks) == 0:
        raise NoFaceDetectedError("No face landmarks detected for orientation check")

    orientation = orient_det.detect(landmarks[0])   # 5-point landmarks
    passed      = str(orientation).lower() == expected.lower()
    return orientation, passed


def _eval_emotion(
    frame_rgb: np.ndarray,
    expected: str,
    detectors: LivenessDetectors,
) -> tuple[str, bool]:
    """
    Emotion classification using the VGG-based EmotionPredictor.
    """
    emotion_pred = detectors.emotion
    emotion      = emotion_pred.predict(frame_rgb)
    passed       = str(emotion).lower() == expected.lower()
    return emotion, passed
