"""Fast CPU sanity checks for camtrolpp.metrics."""

import numpy as np
import pytest
from PIL import Image

from camtrolpp.metrics import (
    EVAL_SIZE, FlowEstimator, PerceptualMetrics, epipolar_errors, fpa_zmr,
    fundamental_matrix, resize_for_eval,
)

K = np.array([[200.0, 0.0, 128.0], [0.0, 200.0, 96.0], [0.0, 0.0, 1.0]])


def rotation_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def project(K: np.ndarray, pose: np.ndarray, X: np.ndarray) -> np.ndarray:
    x = (K @ (pose[:, :3] @ X.T + pose[:, 3:])).T
    return x[:, :2] / x[:, 2:]


def pose_pair():
    ref = np.hstack([np.eye(3), np.zeros((3, 1))])
    tgt = np.hstack([rotation_y(10), np.array([[0.3], [0.05], [0.1]])])
    return ref, tgt


def test_epipolar_errors_vanish_for_exact_correspondences():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-1, 1, 200), rng.uniform(-1, 1, 200), rng.uniform(3, 6, 200)])
    ref, tgt = pose_pair()
    F = fundamental_matrix(K, ref, tgt)
    eae, sampson = epipolar_errors(project(K, ref, X), project(K, tgt, X), F, np.hypot(*EVAL_SIZE))
    assert eae.max() < 1e-9
    assert sampson.max() < 1e-12


def test_epipolar_errors_grow_for_wrong_pose():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-1, 1, 200), rng.uniform(-1, 1, 200), rng.uniform(3, 6, 200)])
    ref, tgt = pose_pair()
    wrong = np.hstack([rotation_y(-10), np.array([[0.0], [0.3], [0.0]])])
    F = fundamental_matrix(K, ref, wrong)
    eae, _ = epipolar_errors(project(K, ref, X), project(K, tgt, X), F, np.hypot(*EVAL_SIZE))
    assert np.median(eae) > 1e-3


def test_fpa_zmr_on_synthetic_flow():
    h, w = EVAL_SIZE[1], EVAL_SIZE[0]
    moving = [np.full((h, w, 2), 2.0, dtype=np.float32)] * 3
    still = [np.zeros((h, w, 2), dtype=np.float32)] * 3
    assert fpa_zmr(moving) == pytest.approx({"fpa": 1.0, "zmr": 0.0})
    assert fpa_zmr(still)["zmr"] == 1.0


def test_resize_rescales_intrinsics():
    images = [Image.new("RGB", (960, 704))]
    K_full = np.array([[800.0, 0.0, 480.0], [0.0, 800.0, 352.0], [0.0, 0.0, 1.0]])
    frames, K_eval = resize_for_eval(images, K_full)
    assert frames[0].shape == (EVAL_SIZE[1], EVAL_SIZE[0], 3)
    assert K_eval[0, 2] == pytest.approx(128.0) and K_eval[1, 2] == pytest.approx(96.0)
    assert K_eval[0, 0] == pytest.approx(800 * 256 / 960) and K_eval[1, 1] == pytest.approx(800 * 192 / 704)


def test_static_sequence_has_no_motion_and_no_change():
    rng = np.random.default_rng(0)
    frame = (rng.uniform(0, 255, (EVAL_SIZE[1], EVAL_SIZE[0], 3))).astype(np.uint8)
    frames = [frame] * 3
    flows = FlowEstimator("cpu").sequence_flows(frames)
    assert fpa_zmr(flows)["zmr"] == 1.0
    scores = PerceptualMetrics("cpu").compute(frames)
    assert scores["lpips_next"] == pytest.approx(0.0, abs=1e-6)
    assert scores["clipsim_next"] == pytest.approx(1.0, abs=1e-5)
