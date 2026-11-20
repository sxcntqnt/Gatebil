"""
Tests for services/id_card/ekyc.py.

No torch, GPU or checkpoint needed: the detector and MTCNN are faked.
Run from backend/:   python -m pytest tests/ekyc_geometry_test.py
"""
import itertools
import tempfile
from pathlib import Path

import cv2
import numpy as np

from services.id_card import ekyc
from services.id_card.ekyc import (
    STAGE_CONFIDENCE, STAGE_DECODE, STAGE_DETECT, STAGE_GEOMETRY,
    analyze_corners, card_aspect, landscape_quad, order_corners,
    orient_card_by_face, process_id_card, validate_quad, _rectify,
)

# coords_norm from the real failing request (channel order as the model emitted it)
LOGGED_NORM = np.array([
    [0.028716783970594406, -0.35123324394226074],
    [0.7447912693023682, -0.23957045376300812],
    [-0.42713236808776855, 0.6037882566452026],
    [0.5321786999702454, 0.7045505046844482],
], dtype=np.float32)


def norm_to_px(norm, w, h):
    return (norm + 1.0) / 2.0 * np.array([w, h], dtype=np.float32)


# ---- fakes -----------------------------------------------------------------

class FakeKP:
    """Duck-types model.id_detector.KeypointResult."""
    def __init__(self, coords, conf, size, thr=0.15):
        self.coords = np.asarray(coords, dtype=np.float32)
        self.confidence = np.asarray(conf, dtype=np.float32)
        self.valid = self.confidence >= thr
        self.heatmaps = np.full((4, 20, 32), 1 / 640, dtype=np.float32)
        self.keypoint_names = ["top_left", "top_right", "bottom_right", "bottom_left"]
        self.original_size = size

    n_keypoints = 4
    @property
    def all_valid(self): return bool(self.valid.all())


class FakeDetector:
    def __init__(self, coords, conf):
        self.coords, self.conf = coords, conf
    def detect(self, img):
        h, w = img.shape[:2]
        return FakeKP(self.coords, self.conf, (w, h))


class BoomDetector:
    def detect(self, img): raise RuntimeError("cuda exploded")


def jpeg_bytes(w=800, h=600):
    img = np.full((h, w, 3), 120, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


GOOD_QUAD = np.array([[150, 120], [650, 140], [630, 430], [130, 400]], np.float32)  # TL TR BR BL


# ---- ordering --------------------------------------------------------------

def test_order_corners_is_permutation_invariant():
    expected = order_corners(GOOD_QUAD)
    for perm in itertools.permutations(range(4)):
        assert np.allclose(order_corners(GOOD_QUAD[list(perm)]), expected)
    assert np.allclose(expected, GOOD_QUAD)


def test_swapped_bottom_pair_is_repaired():
    swapped = GOOD_QUAD[[0, 1, 3, 2]]            # TL TR BL BR: old TF channel order
    assert np.allclose(order_corners(swapped), GOOD_QUAD)


def test_logged_coords_crossed_in_channel_order_and_gate_depends_on_frame_shape():
    """
    The real failing request: crossed as emitted, a plausible convex quad once
    ordered. Whether it then passes depends on the (unknown) frame shape, because
    the card-aspect check is real: in a 4:3 landscape frame the quad measures
    1.06 (not a card), in portrait or 16:9 frames it is card-like.
    """
    px = norm_to_px(LOGGED_NORM, 4000, 3000)
    assert not cv2.isContourConvex(px.astype(np.float32).reshape(-1, 1, 2))
    expect = {(4000, 3000): "aspect", (3000, 4000): None, (1600, 900): None, (900, 1600): None}
    for (w, h), want in expect.items():
        quad, verdict = analyze_corners(norm_to_px(LOGGED_NORM, w, h), w, h)
        assert card_aspect(quad) >= 1.0           # landscape-normalised
        if want is None:
            assert verdict is None, (w, h, verdict)
        else:
            assert verdict is not None and want in verdict, (w, h, verdict)


def test_sideways_card_becomes_landscape():
    # tall quad: 200 wide, 320 high
    tall = np.array([[100, 50], [300, 60], [295, 380], [105, 370]], np.float32)
    q = landscape_quad(order_corners(tall))
    top, right, bottom, left = ekyc.edge_lengths(q)
    assert top + bottom > left + right


# ---- validation ------------------------------------------------------------

def test_valid_card_passes():
    assert validate_quad(GOOD_QUAD, 800, 600) is None


def test_rejections():
    w, h = 800, 600
    assert "not convex" in validate_quad(
        order_corners(np.array([[100, 100], [700, 100], [400, 160], [100, 500]], np.float32)), w, h)
    tiny = np.array([[10, 10], [60, 10], [60, 40], [10, 40]], np.float32)
    assert "covers" in validate_quad(tiny, w, h)
    skinny = np.array([[0, 250], [800, 250], [800, 330], [0, 330]], np.float32)
    assert "aspect" in validate_quad(skinny, w, h)
    outside = GOOD_QUAD + np.array([900, 0], np.float32)
    assert "outside" in validate_quad(outside, w, h)
    line = np.array([[100, 100], [200, 100], [300, 100], [400, 100]], np.float32)
    assert validate_quad(order_corners(line), w, h) is not None
    assert "non-finite" in validate_quad(np.full((4, 2), np.nan, np.float32), w, h)


# ---- rectification ---------------------------------------------------------

def test_rectify_recovers_card_texture():
    cw, ch = ekyc._CARD_OUTPUT_W, ekyc._CARD_OUTPUT_H
    card = np.zeros((ch, cw, 3), np.uint8)
    card[:, :, 0] = np.linspace(0, 255, cw, dtype=np.uint8)[None, :]
    card[:, :, 1] = np.linspace(0, 255, ch, dtype=np.uint8)[:, None]
    cv2.rectangle(card, (40, 40), (200, 160), (255, 255, 255), -1)
    src = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], np.float32)
    m = cv2.getPerspectiveTransform(src, GOOD_QUAD)
    frame = cv2.warpPerspective(card, m, (800, 600), borderValue=(30, 30, 30))
    quad, verdict = analyze_corners(GOOD_QUAD[[2, 0, 3, 1]], 800, 600)   # shuffled input
    assert verdict is None
    out = _rectify(frame, quad)
    assert out.shape == (ch, cw, 3)
    inner = (slice(15, ch - 15), slice(15, cw - 15))
    assert np.abs(out[inner].astype(int) - card[inner].astype(int)).mean() < 10


# ---- pipeline --------------------------------------------------------------

def test_pipeline_success_even_with_swapped_channels():
    det = FakeDetector(GOOD_QUAD[[0, 1, 3, 2]], [0.4] * 4)
    r = process_id_card(jpeg_bytes(), det)
    assert r.success and r.stage is None and r.error is None
    assert r.rectified_image.shape == (ekyc._CARD_OUTPUT_H, ekyc._CARD_OUTPUT_W, 3)
    assert np.allclose(r.quad, GOOD_QUAD)


def test_pipeline_low_confidence_reports_geometry_verdict():
    det = FakeDetector(GOOD_QUAD, [0.008, 0.029, 0.011, 0.019])
    r = process_id_card(jpeg_bytes(), det)
    assert not r.success and r.stage == STAGE_CONFIDENCE
    assert r.error.startswith("[confidence] Low-confidence keypoints")
    assert r.metadata["geometry"]["valid"] is True   # geometry fine, model just unsure


def test_pipeline_geometry_gate():
    bad = np.array([[100, 100], [700, 100], [400, 160], [100, 500]], np.float32)
    r = process_id_card(jpeg_bytes(), FakeDetector(bad, [0.9] * 4))
    assert not r.success and r.stage == STAGE_GEOMETRY and "not convex" in r.error


def test_pipeline_decode_and_detect_stages():
    r = process_id_card(b"not an image", FakeDetector(GOOD_QUAD, [0.9] * 4))
    assert r.stage == STAGE_DECODE and r.error.startswith("[decode]")
    r = process_id_card(jpeg_bytes(), BoomDetector())
    assert r.stage == STAGE_DETECT and "cuda exploded" in r.error


def test_debug_overlay_written_only_when_asked():
    with tempfile.TemporaryDirectory() as d:
        det = FakeDetector(GOOD_QUAD, [0.4] * 4)
        process_id_card(jpeg_bytes(), det)
        assert list(Path(d).iterdir()) == []
        r = process_id_card(jpeg_bytes(), det, debug_dir=d)
        files = list(Path(d).glob("*_overlay.jpg"))
        assert len(files) == 1 and r.metadata["request_id"] in files[0].name


# ---- orientation -----------------------------------------------------------

class MarkerMTCNN:
    """Pretends a face exists only when the bright marker is in the top-left."""
    def detect(self, pil):
        a = np.asarray(pil)
        if a[:30, :30].mean() > 200:
            return np.array([[1, 1, 20, 20]], np.float32), np.array([0.99])
        return None, [None]


def test_orient_card_by_face_flips_upside_down_card():
    img = np.zeros((404, 640, 3), np.uint8)
    img[-30:, -30:] = 255                          # marker bottom-right = upside down
    out, rot, score = orient_card_by_face(img, MarkerMTCNN())
    assert rot == 180 and score > 0.9 and out[:30, :30].mean() > 200
    upright = np.zeros_like(img); upright[:30, :30] = 255
    assert orient_card_by_face(upright, MarkerMTCNN())[1] == 0


def test_orient_card_by_face_leaves_image_alone_when_no_face():
    img = np.zeros((404, 640, 3), np.uint8)
    out, rot, score = orient_card_by_face(img, MarkerMTCNN())
    assert rot == 0 and score == 0.0 and out is img
