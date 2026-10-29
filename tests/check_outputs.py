"""Sanity checks for generated runs: python tests/check_outputs.py outputs/<image>

Every camera action must give 13K + 1 frames that are valid images, poses that
start at the input view, and a camera that visibly moves along the path.
"""

import glob
import json
import os
import sys

import numpy as np
from PIL import Image


def check_run(run_dir):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    frames = np.stack([np.array(Image.open(f)) for f in sorted(glob.glob(os.path.join(run_dir, "frames", "*.png")))]).astype(np.float32)
    poses = np.load(os.path.join(run_dir, "poses_w2c.npy"))
    n = 13 * meta["n_passes"] + 1
    problems = []
    if len(frames) != n or meta["n_frames"] != n or len(poses) != n:
        problems.append(f"expected {n} frames and poses, got {len(frames)} frames, {len(poses)} poses")
    if not os.path.exists(os.path.join(run_dir, "video.mp4")):
        problems.append("video.mp4 missing")
    if len(meta["eval_idx"]) != 14:
        problems.append("eval_idx does not hold 14 frames")
    if not np.allclose(poses[0], np.eye(3, 4), atol=1e-6):
        problems.append("first pose is not the input view")
    if frames.std(axis=(1, 2, 3)).min() < 5:
        problems.append("a frame is nearly blank")
    step = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2, 3))     # mean change between consecutive frames
    end = np.abs(frames[-1] - frames[0]).mean()                      # change over the whole path
    if end < 5 or (step < 0.05).mean() > 0.2:
        problems.append(f"camera barely moves (first to last frame {end:.1f}, frozen steps {(step < 0.05).mean():.0%})")
    return problems, end, step.mean(), meta["time_s"]


def main():
    ok = True
    for run_dir in sorted(glob.glob(os.path.join(sys.argv[1], "*", "meta.json"))):
        run_dir = os.path.dirname(run_dir)
        problems, end, step, t = check_run(run_dir)
        ok &= not problems
        status = "OK" if not problems else "FAIL: " + "; ".join(problems)
        print(f"{os.path.basename(run_dir):>14}: change first-to-last {end:5.1f}, per frame {step:4.2f}, {t:6.1f}s  {status}")
    print("OUTPUTS_OK" if ok else "OUTPUTS_FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
