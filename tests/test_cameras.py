"""CPU tests for camera paths and the warping geometry."""

import numpy as np
import pytest
from PIL import Image

from camtrolpp.cameras import ACTIONS, camera_path, intrinsics, num_passes
from camtrolpp.warp import reproject


def e1_orbit(total_deg, n_frames):
    """The orbit used for every result in the paper (copied from the experiment code)."""
    poses = []
    for g in range(n_frames):
        theta_deg = total_deg * g / (n_frames - 1)
        th = np.radians(theta_deg)
        E = np.eye(4)
        E[:3, :3] = [[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]
        E[:3, 3] = [-1.0 * theta_deg / 15.0, 0, 1.0 * np.sin(th)]
        poses.append(np.linalg.inv(E)[:3, :])
    return np.stack(poses)


@pytest.mark.parametrize("action", sorted(ACTIONS))
def test_paths_are_valid_rigid_motions(action):
    poses = camera_path(action, 30 if ACTIONS[action][0] == "deg" else 0.5, 27, center_depth=3.0)
    assert poses.shape == (27, 3, 4)
    np.testing.assert_allclose(poses[0], np.eye(3, 4), atol=1e-12)        # frame 0 is the input view
    for P in poses:
        np.testing.assert_allclose(P[:, :3] @ P[:, :3].T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(P[:, :3]), 1.0)
    assert not np.allclose(poses[-1], poses[0])                            # the camera actually moves


@pytest.mark.parametrize("passes", [1, 2, 3, 4, 5, 6])
def test_orbit_matches_paper_trajectory_exactly(passes):
    n = 13 * passes + 1
    assert np.array_equal(camera_path("orbit_left", 60.0, n), e1_orbit(60.0, n))


def test_number_of_passes():
    assert num_passes(60, 15) == 4
    assert num_passes(10, 15) == 1
    assert num_passes(50.5, 15) == 4
    assert num_passes(0.5, 0.25) == 2


def test_rotation_amounts():
    deg = lambda R: np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    for action in ["pan_left", "tilt_up", "roll_left", "orbit_right"]:
        assert np.isclose(deg(camera_path(action, 45.0, 10)[-1][:, :3]), 45.0)


def test_identity_warp_keeps_the_image():
    # A linear ramp is unchanged by the symmetric 3x3 splat kernel away from the border.
    ramp = np.linspace(20, 230, 96)[None, :, None].repeat(64, 0).repeat(3, 2)
    image = Image.fromarray(ramp.astype(np.uint8))
    depth = np.full((64, 96), 2.0, dtype=np.float32)
    K = intrinsics(64, 96, focal=80.0)
    poses = np.stack([np.eye(3, 4), np.eye(3, 4)])
    warped, keep = next(reproject(image, depth, K, poses))
    assert keep.all()
    np.testing.assert_allclose(warped[2:-2, 2:-2] * 255, np.array(image)[2:-2, 2:-2], atol=1.0)


def test_sideways_move_opens_a_hole_on_one_side():
    image = Image.fromarray(np.full((64, 96, 3), 128, dtype=np.uint8))
    depth = np.full((64, 96), 2.0, dtype=np.float32)
    K = intrinsics(64, 96, focal=80.0)
    poses = camera_path("truck_right", 0.25, 2, center_depth=2.0)
    _, keep = next(reproject(image, depth, K, poses))
    left, right = keep[:, :20].mean(), keep[:, -20:].mean()
    assert min(left, right) < 0.5 < max(left, right)
