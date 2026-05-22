"""
eKYC Flask Server
-----------------
Unified entry-point replacing the PyQt5 GUI.

Endpoints
  GET  /                  → serves index.html
  POST /api/id-card       → upload ID card; returns smart-cropped image path
  POST /api/verify        → compare selfie vs cropped ID face; returns verified bool
  POST /api/challenge     → liveness challenge-response (blink / orientation / emotion)
"""

import io
import os
import logging

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

# ── Internal modules ────────────────────────────────────────────────────────
from model.dsnt import dsnt
from pipeline.face_verification import verify
from services.facenet.models.mtcnn import MTCNN
from services.liveness_detection.blink_detection import BlinkDetector
from services.liveness_detection.emotion_prediction import EmotionPredictor
from services.liveness_detection.face_orientation import FaceOrientationDetector
from services.verification_models import VGGFace2
from utils.functions import get_image

import tensorflow as tf

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ekyc")

# ── App & paths ──────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="bil/templates", static_folder="bil/static")

TMP_DIR            = "tmp"
FROZEN_MODEL_PATH  = "model/frozen_model.pb"

os.makedirs(TMP_DIR, exist_ok=True)

# ── Singletons (loaded once at startup) ──────────────────────────────────────
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn          = None
verif_model    = None
blink_detector = None
orient_detector= None
emotion_pred   = None
tf_graph       = None
tf_sess        = None

# TF tensor handles (populated in init)
tf_inputs      = None
tf_hm1 = tf_hm2 = tf_hm3 = tf_hm4 = None
tf_kp1 = tf_kp2 = tf_kp3 = tf_kp4 = None


# ── Initialisation ────────────────────────────────────────────────────────────

def _load_tf_graph(path: str):
    """Load a TF1 frozen protobuf and return the graph."""
    with tf.gfile.GFile(path, "rb") as fh:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fh.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph


def init():
    """Initialise all models exactly once before the server starts serving."""
    global mtcnn, verif_model, blink_detector, orient_detector, emotion_pred
    global tf_graph, tf_sess
    global tf_inputs, tf_hm1, tf_hm2, tf_hm3, tf_hm4, tf_kp1, tf_kp2, tf_kp3, tf_kp4

    log.info("Loading MTCNN …")
    mtcnn = MTCNN(device=device)

    log.info("Loading VGGFace2 …")
    verif_model = VGGFace2.load_model(device=device)

    log.info("Loading liveness detectors …")
    blink_detector  = BlinkDetector()
    orient_detector = FaceOrientationDetector()
    emotion_pred    = EmotionPredictor(device=device)

    log.info("Loading TF frozen graph …")
    tf_graph  = _load_tf_graph(FROZEN_MODEL_PATH)
    tf_sess   = tf.Session(graph=tf_graph)

    tf_inputs        = tf_graph.get_tensor_by_name("input:0")
    activation_map   = tf_graph.get_tensor_by_name(
        "heats_map_regression/pred_keypoints/BiasAdd:0"
    )

    tf_hm1, tf_kp1 = dsnt(activation_map[..., 0])
    tf_hm2, tf_kp2 = dsnt(activation_map[..., 1])
    tf_hm3, tf_kp3 = dsnt(activation_map[..., 2])
    tf_hm4, tf_kp4 = dsnt(activation_map[..., 3])

    log.info("All models ready — device=%s", device)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_upload(file_storage) -> np.ndarray:
    """FileStorage → BGR numpy array (in-memory, no temp write needed)."""
    buf  = io.BytesIO()
    file_storage.save(buf)
    arr  = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _smart_crop_id(id_file) -> dict:
    """
    Run the TF keypoint model on an ID-card image.
    Returns a dict with paths to the cropped and final images plus raw keypoints.
    """
    # ── Read & rotate (phone captures landscape) ───────────────────────────
    pil_orig = Image.open(id_file.stream).rotate(270, expand=True)
    img_orig = cv2.cvtColor(np.array(pil_orig), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(TMP_DIR, "original.jpg"), img_orig)

    rw = img_orig.shape[1] / 600.0
    rh = img_orig.shape[0] / 800.0

    # ── Resize for model feed ──────────────────────────────────────────────
    id_file.stream.seek(0)
    img_nd = np.array(
        Image.open(id_file.stream).rotate(270, expand=True).resize((600, 800))
    )

    # ── Run TF session ─────────────────────────────────────────────────────
    hm1, hm2, hm3, hm4, kp1, kp2, kp3, kp4 = tf_sess.run(
        [tf_hm1, tf_hm2, tf_hm3, tf_hm4, tf_kp1, tf_kp2, tf_kp3, tf_kp4],
        feed_dict={tf_inputs: np.expand_dims(img_nd, 0)},
    )

    # ── Decode keypoints ───────────────────────────────────────────────────
    keypoints = np.array([kp1[0], kp2[0], kp3[0], kp4[0]])
    keypoints = ((keypoints + 1) / 2 * np.array([600, 800])).astype("int")

    x1 = (keypoints[0, 0] + keypoints[2, 0]) / 2.0
    y1 = (keypoints[0, 1] + keypoints[1, 1]) / 2.0
    x2 = (keypoints[1, 0] + keypoints[3, 0]) / 2.0
    y2 = (keypoints[2, 1] + keypoints[3, 1]) / 2.0

    new_kps = np.array(
        [[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype="float32"
    )

    # ── Homography + warp ──────────────────────────────────────────────────
    img_bgr = cv2.cvtColor(img_nd, cv2.COLOR_RGB2BGR)
    H, _    = cv2.findHomography(keypoints.astype("float32"), new_kps, cv2.RANSAC, 5.0)

    resize_factor     = img_bgr.shape[1] / (x2 - x1)
    h, w              = img_bgr.shape[:2]
    new_h, new_w      = int(h * resize_factor), int(w * resize_factor)
    warped            = cv2.warpPerspective(img_bgr, H, (new_w, new_h))
    cropped           = warped[int(y1):int(y2), int(x1):int(x2)]

    # ── Write intermediates ────────────────────────────────────────────────
    cropped_path = os.path.join(TMP_DIR, "cropped.jpg")
    cv2.imwrite(cropped_path, cropped)

    dim   = (int(cropped.shape[1] * rw), int(cropped.shape[0] * rh))
    final = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)

    final_path = os.path.join(TMP_DIR, "final.jpg")
    cv2.imwrite(final_path, final)

    return {
        "cropped_path": cropped_path,
        "final_path":   final_path,
        "keypoints":    keypoints.tolist(),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/id-card")
def api_id_card():
    """
    Accepts multipart form with field 'file' (ID card image).
    Returns JSON with paths to the smart-cropped output.
    """
    if "file" not in request.files:
        return jsonify(error="Missing 'file' field"), 400

    try:
        result = _smart_crop_id(request.files["file"])
        return jsonify(ok=True, **result)
    except Exception as exc:
        log.exception("ID card processing failed")
        return jsonify(ok=False, error=str(exc)), 500


@app.post("/api/verify")
def api_verify():
    """
    Accepts multipart form:
      - 'selfie'    : live photo of the person
      - 'id_image'  : cropped ID card face (or use the last smart-cropped tmp file)

    Returns JSON { ok, verified, distance }.
    """
    selfie_file = request.files.get("selfie")

    # Support passing the raw selfie and relying on the previously cropped ID
    if selfie_file is None:
        return jsonify(error="Missing 'selfie' field"), 400

    selfie_path = os.path.join(TMP_DIR, "verify_selfie.jpg")
    selfie_img  = _decode_upload(selfie_file)
    cv2.imwrite(selfie_path, selfie_img)

    # Use explicit ID image if supplied, otherwise fall back to last smart-crop
    if "id_image" in request.files:
        id_img = _decode_upload(request.files["id_image"])
        id_path = os.path.join(TMP_DIR, "verify_id.jpg")
        cv2.imwrite(id_path, id_img)
    else:
        id_path = os.path.join(TMP_DIR, "final.jpg")
        if not os.path.exists(id_path):
            return jsonify(error="No ID image available; call /api/id-card first"), 400

    try:
        id_img_arr   = get_image(id_path)
        selfie_arr   = get_image(selfie_path)
        verified     = verify(id_img_arr, selfie_arr, mtcnn, verif_model, model_name="VGG-Face2")
        return jsonify(ok=True, verified=bool(verified))
    except Exception as exc:
        log.exception("Face verification failed")
        return jsonify(ok=False, error=str(exc)), 500


@app.post("/api/challenge")
def api_challenge():
    """
    Liveness challenge-response.

    Accepts multipart form:
      - 'frame'     : single video frame (JPEG/PNG)
      - 'challenge' : one of 'blink' | 'orientation' | 'emotion'
      - 'expected'  : expected value (e.g. orientation label or emotion label)

    Returns JSON { ok, passed, result }.
    """
    frame_file = request.files.get("frame")
    challenge  = request.form.get("challenge", "").lower()
    expected   = request.form.get("expected", "")

    if frame_file is None:
        return jsonify(error="Missing 'frame' field"), 400
    if challenge not in ("blink", "orientation", "emotion"):
        return jsonify(error="'challenge' must be blink | orientation | emotion"), 400

    frame = _decode_upload(frame_file)

    try:
        if challenge == "blink":
            result = blink_detector.detect(frame)
            passed = bool(result)

        elif challenge == "orientation":
            result = orient_detector.detect(frame)
            passed = (str(result).lower() == expected.lower())

        else:  # emotion
            result = emotion_pred.predict(frame)
            passed = (str(result).lower() == expected.lower())

        return jsonify(ok=True, passed=passed, result=str(result))

    except Exception as exc:
        log.exception("Challenge-response failed")
        return jsonify(ok=False, error=str(exc)), 500


# ── Entry-point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init()
    app.run(host="0.0.0.0", port=5000, debug=False)
