"""Evaluation metrics for camera-controlled novel view synthesis.

All metrics are computed at 256x192 (following WAVE), with the intrinsics
rescaled to that resolution, so pixel-unit numbers (Sampson) are comparable
across papers. Poses are 3x4 world-to-camera matrices (OpenCV), x_cam = R X + t.

Reported keys (see `evaluate_sequence`):
    lpips_next, lpips_first, clipsim_next, clipsim_first   appearance consistency
    eae_1e3, sampson_1e4                                   agreement with the commanded camera
    fpa, zmr                                               motion smoothness / frozen pixels
"""

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

EVAL_SIZE = (256, 192)  # (width, height)
ZMR_THRESHOLD = 0.75  # pixels at EVAL_SIZE
GRID_STEP = 12  # spacing of the tracked point grid, pixels
FB_THRESHOLD = 1.0  # forward-backward error limit, percent of the longer image side
MIN_CORRESPONDENCES = 8


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resize_for_eval(images: List[Image.Image], K: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """Resize frames to EVAL_SIZE (bicubic) and rescale K to match."""
    w, h = EVAL_SIZE
    K = K.astype(np.float64).copy()
    K[0, :] *= w / images[0].width
    K[1, :] *= h / images[0].height
    frames = [np.array(im.convert("RGB").resize((w, h), Image.BICUBIC)) for im in images]
    return frames, K


# ---------------------------------------------------------------------------
# Appearance: LPIPS and CLIP similarity
# ---------------------------------------------------------------------------

class PerceptualMetrics:
    """LPIPS (AlexNet) and CLIP ViT-B/32 image similarity."""

    def __init__(self, device: str):
        import lpips
        import open_clip

        self.device = device
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.clip, _, self.clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        self.clip = self.clip.to(device).eval()

    def _to_lpips_input(self, frame: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
        return t.to(self.device)

    @torch.no_grad()
    def lpips_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return self.lpips(self._to_lpips_input(a), self._to_lpips_input(b)).item()

    @torch.no_grad()
    def clip_embedding(self, frame: np.ndarray) -> torch.Tensor:
        x = self.clip_preprocess(Image.fromarray(frame)).unsqueeze(0).to(self.device)
        feat = self.clip.encode_image(x)
        return feat / feat.norm(dim=-1, keepdim=True)

    def compute(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """"next" compares each frame with the previous one (temporal flicker);
        "first" compares every frame, frame 0 included, with frame 0 (drift)."""
        emb = [self.clip_embedding(f) for f in frames]
        n = len(frames)
        return {
            "lpips_next": float(np.mean([self.lpips_distance(frames[t], frames[t - 1]) for t in range(1, n)])),
            "clipsim_next": float(np.mean([(emb[t] @ emb[t - 1].T).item() for t in range(1, n)])),
            "lpips_first": float(np.mean([self.lpips_distance(frames[t], frames[0]) for t in range(n)])),
            "clipsim_first": float(np.mean([(emb[t] @ emb[0].T).item() for t in range(n)])),
        }


# ---------------------------------------------------------------------------
# Optical flow
# ---------------------------------------------------------------------------

class FlowEstimator:
    """RAFT-large (torchvision, default weights) optical flow."""

    def __init__(self, device: str):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        self.device = device
        self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()

    @torch.no_grad()
    def flow(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Flow from frame a to frame b, shape (H, W, 2) as (dx, dy)."""
        h, w = a.shape[:2]
        assert h % 8 == 0 and w % 8 == 0, "RAFT needs sides divisible by 8"

        def prep(x: np.ndarray) -> torch.Tensor:
            return (torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0).to(self.device)

        return self.raft(prep(a), prep(b))[-1][0].permute(1, 2, 0).cpu().numpy()

    def sequence_flows(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        return [self.flow(frames[t], frames[t + 1]) for t in range(len(frames) - 1)]


# ---------------------------------------------------------------------------
# Motion: FPA and ZMR
# ---------------------------------------------------------------------------

def fpa_zmr(flows: List[np.ndarray], tau: float = ZMR_THRESHOLD, eps: float = 1e-8) -> Dict[str, float]:
    """FPA (flow persistence agreement, higher is smoother): how well each
    pixel's flow keeps its direction and magnitude from one frame pair to the
    next. ZMR (zero-motion ratio, lower is better): fraction of pixels that
    move less than `tau` pixels, i.e. regions that stay frozen while the
    camera moves."""
    dir_terms, mag_terms = [], []
    for f0, f1 in zip(flows[:-1], flows[1:]):
        n0, n1 = np.linalg.norm(f0, axis=-1), np.linalg.norm(f1, axis=-1)
        dir_terms.append(np.mean((f0 * f1).sum(-1) / (n0 * n1 + eps)))
        mag_terms.append(1.0 - np.mean(np.abs(n0 - n1) / (n0 + n1 + eps)))
    zmr = float(np.mean([np.mean(np.linalg.norm(f, axis=-1) < tau) for f in flows]))
    fpa = 0.5 * (float(np.mean(dir_terms)) + float(np.mean(mag_terms))) if dir_terms else float("nan")
    return {"fpa": fpa, "zmr": zmr}


# ---------------------------------------------------------------------------
# Geometry: EAE and Sampson distance against the commanded poses
# ---------------------------------------------------------------------------

def skew(t: np.ndarray) -> np.ndarray:
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def fundamental_matrix(K: np.ndarray, pose_ref: np.ndarray, pose_tgt: np.ndarray) -> np.ndarray:
    """F such that x_tgt^T F x_ref = 0, from shared intrinsics and two w2c poses."""
    R_ref, t_ref = pose_ref[:, :3], pose_ref[:, 3]
    R_tgt, t_tgt = pose_tgt[:, :3], pose_tgt[:, 3]
    R_rel = R_tgt @ R_ref.T
    t_rel = t_tgt - R_rel @ t_ref
    K_inv = np.linalg.inv(K)
    return K_inv.T @ skew(t_rel) @ R_rel @ K_inv


def epipolar_errors(x_ref: np.ndarray, x_tgt: np.ndarray, F: np.ndarray, image_diag: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per-point errors for (N, 2) pixel correspondences.

    EAE: symmetric point-to-epipolar-line distance, divided by the image diagonal.
    Sampson: first-order approximation of the squared reprojection error (pixels^2).
    """
    ones = np.ones((len(x_ref), 1))
    x1, x2 = np.hstack([x_ref, ones]), np.hstack([x_tgt, ones])
    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    residual = np.sum(x2 * Fx1, axis=1)

    d1 = np.sqrt(Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + 1e-12)
    d2 = np.sqrt(Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-12)
    eae = 0.5 * (np.abs(residual) / d1 + np.abs(residual) / d2) / image_diag

    sampson = residual ** 2 / (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-12)
    return eae, sampson


def _sample(field: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinearly sample an (H, W, 2) field at (N, 2) float32 points."""
    fx = cv2.remap(field[..., 0], pts[:, 0], pts[:, 1], cv2.INTER_LINEAR).ravel()
    fy = cv2.remap(field[..., 1], pts[:, 0], pts[:, 1], cv2.INTER_LINEAR).ravel()
    return np.stack([fx, fy], axis=1)


def _farneback(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ga, gb = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 21, 3, 5, 1.2, 0)


def track_correspondences(frames: List[np.ndarray], flows: List[np.ndarray],
                          ref: int, tgt: int) -> Tuple[np.ndarray, np.ndarray]:
    """Track a regular point grid from frame `ref` to frame `tgt` by chaining
    RAFT flow, keeping points that stay in view and pass a forward-backward
    check. The check uses an independent Farneback flow between the two
    frames, so a point is kept only if that flow maps it back to where it began."""
    h, w = frames[ref].shape[:2]
    ys, xs = np.mgrid[GRID_STEP // 2:h:GRID_STEP, GRID_STEP // 2:w:GRID_STEP]
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)

    cur = pts.copy()
    for t in range(ref, tgt):
        cur = cur + _sample(flows[t], cur)
    valid = (cur[:, 0] >= 0) & (cur[:, 0] < w) & (cur[:, 1] >= 0) & (cur[:, 1] < h)

    fwd, bwd = _farneback(frames[ref], frames[tgt]), _farneback(frames[tgt], frames[ref])
    fwd_pts = pts + _sample(fwd, pts)
    fwd_pts_in = np.ascontiguousarray(np.clip(fwd_pts, [0, 0], [w - 1, h - 1]).astype(np.float32))
    fb_err = np.linalg.norm(fwd_pts + _sample(bwd, fwd_pts_in) - pts, axis=1)
    valid &= fb_err < FB_THRESHOLD * max(h, w) / 100.0
    return pts[valid], cur[valid]


def epipolar_metrics(frames: List[np.ndarray], flows: List[np.ndarray],
                     K: np.ndarray, poses: np.ndarray) -> Dict[str, float]:
    """EAE and Sampson distance over consecutive frame pairs. The fundamental
    matrix comes from the commanded cameras, not from the pixels, so the score
    says how closely the generated motion follows the requested trajectory.
    Per pair we take the median over tracked points, then average over pairs.
    Pairs without camera translation (pan, tilt, roll) have no epipolar
    geometry, so they are skipped and a pure rotation scores NaN."""
    h, w = frames[0].shape[:2]
    diag = math.hypot(h, w)
    eae, sampson = [], []
    for i in range(len(frames) - 1):
        R_rel = poses[i + 1][:, :3] @ poses[i][:, :3].T
        if np.linalg.norm(poses[i + 1][:, 3] - R_rel @ poses[i][:, 3]) < 1e-8:
            continue
        x_ref, x_tgt = track_correspondences(frames, flows, i, i + 1)
        if len(x_ref) < MIN_CORRESPONDENCES:
            continue
        e, s = epipolar_errors(x_ref, x_tgt, fundamental_matrix(K, poses[i], poses[i + 1]), diag)
        eae.append(np.median(e))
        sampson.append(np.median(s))
    return {
        "eae": float(np.mean(eae)) if eae else float("nan"),
        "sampson": float(np.mean(sampson)) if sampson else float("nan"),
    }


# ---------------------------------------------------------------------------
# Full suite
# ---------------------------------------------------------------------------

def evaluate_sequence(frames: List[np.ndarray], K: np.ndarray, poses: np.ndarray,
                      perceptual: PerceptualMetrics, flow: FlowEstimator) -> Dict[str, float]:
    """Score one sequence already at EVAL_SIZE (see `resize_for_eval`).
    EAE is reported x1e3 and Sampson x1e4, the units used in the paper tables."""
    flows = flow.sequence_flows(frames)
    geo = epipolar_metrics(frames, flows, K, poses)
    return {
        **perceptual.compute(frames),
        "eae_1e3": geo["eae"] * 1e3,
        "sampson_1e4": geo["sampson"] * 1e4,
        **fpa_zmr(flows),
        "n_frames": len(frames),
    }
