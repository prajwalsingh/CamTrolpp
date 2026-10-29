"""Camera paths.

A path is an array of world-to-camera poses with shape (N, 3, 4) in the OpenCV
convention (x right, y down, z forward). A world point X maps to camera
coordinates R @ X + t. Frame 0 is always the input view.

Every camera action is defined over the whole path, so splitting it into
passes never changes the curve, only where the point cloud is rebuilt.
Rotations are given in degrees. Translations are given as a fraction of the
depth at the image centre, so the same setting suits near and far scenes.
"""

import math

import numpy as np

HEIGHT, WIDTH = 704, 960   # generation resolution
FOCAL = 5.8269e02          # focal length in pixels at 960x704
FRAMES_PER_PASS = 14       # native clip length of Stable Video Diffusion


def intrinsics(height: int = HEIGHT, width: int = WIDTH, focal: float = FOCAL) -> np.ndarray:
    return np.array([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]], dtype=np.float32)


def _rot_y(deg):
    a = math.radians(deg)
    return np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])


def _rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _rot_z(deg):
    a = math.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _pose(R=np.eye(3), t=(0.0, 0.0, 0.0)):
    P = np.zeros((3, 4))
    P[:, :3] = R
    P[:, 3] = t
    return P


def _orbit(deg):
    """Turn around the scene while sliding sideways, 1 unit per 15 degrees (paper setting)."""
    th = np.radians(deg)
    E = np.eye(4)
    E[:3, :3] = [[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]
    E[:3, 3] = [-deg / 15.0, 0, np.sin(th)]
    return np.linalg.inv(E)[:3, :]


# Each action maps an amount a (degrees, or fraction of centre depth d) to a pose.
ACTIONS = {
    "orbit_left":    ("deg", lambda a, d: _orbit(a)),
    "orbit_right":   ("deg", lambda a, d: _orbit(-a)),
    "pan_left":      ("deg", lambda a, d: _pose(_rot_y(-a))),
    "pan_right":     ("deg", lambda a, d: _pose(_rot_y(a))),
    "tilt_up":       ("deg", lambda a, d: _pose(_rot_x(-a))),
    "tilt_down":     ("deg", lambda a, d: _pose(_rot_x(a))),
    "roll_left":     ("deg", lambda a, d: _pose(_rot_z(a))),
    "roll_right":    ("deg", lambda a, d: _pose(_rot_z(-a))),
    "zoom_in":       ("depth", lambda a, d: _pose(t=(0, 0, -a * d))),
    "zoom_out":      ("depth", lambda a, d: _pose(t=(0, 0, a * d))),
    "truck_left":    ("depth", lambda a, d: _pose(t=(a * d, 0, 0))),
    "truck_right":   ("depth", lambda a, d: _pose(t=(-a * d, 0, 0))),
    "pedestal_up":   ("depth", lambda a, d: _pose(t=(0, a * d, 0))),
    "pedestal_down": ("depth", lambda a, d: _pose(t=(0, -a * d, 0))),
}


def num_passes(total: float, per_pass: float) -> int:
    """Fewest passes that keep the motion of every pass within per_pass."""
    return max(1, math.ceil(abs(total) / per_pass - 1e-9))


def camera_path(action: str, total: float, n_frames: int, center_depth: float = 1.0) -> np.ndarray:
    """Poses of n_frames evenly spaced along the action, from 0 to total."""
    if action not in ACTIONS:
        raise ValueError(f"unknown camera action {action!r}, choose from {sorted(ACTIONS)}")
    _, pose_fn = ACTIONS[action]
    return np.stack([pose_fn(total * g / (n_frames - 1), center_depth) for g in range(n_frames)]).astype(np.float64)
