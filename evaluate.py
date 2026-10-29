"""Score generated sequences with the CamTrol++ metric suite.

    python evaluate.py <run_dir> [<run_dir> ...] [--device cuda]

Every folder under the given run dirs that holds frames, poses and intrinsics
is scored. Two layouts are recognised:

    release   <out>/<image_stem>/<camera_action>/
                  frames/00000.png ...   all generated frames
                  poses_w2c.npy          (N, 3, 4) world-to-camera
                  K.npy                  (3, 3) at the generation resolution
                  meta.json              {"eval_idx": [...]} frames to score
    legacy    <scene>/eval_frames/*.png, camera_Rt_eval.npy, camera_K.npy

Each sequence gets a metrics.json next to its frames, and a table with the
mean of every metric per group is printed (release: one group per camera
action, legacy: one group per parent folder).
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from camtrolpp.metrics import (
    FlowEstimator, PerceptualMetrics, default_device, evaluate_sequence, resize_for_eval,
)

COLUMNS = ["lpips_next", "lpips_first", "clipsim_next", "clipsim_first", "eae_1e3", "sampson_1e4", "fpa", "zmr"]


def layout_of(seq_dir: str) -> Optional[str]:
    has = lambda name: os.path.exists(os.path.join(seq_dir, name))
    if has("frames") and has("poses_w2c.npy") and has("K.npy"):
        return "release"
    if has("eval_frames") and has("camera_Rt_eval.npy") and has("camera_K.npy"):
        return "legacy"
    return None


def find_sequences(run_dir: str) -> List[Tuple[str, str]]:
    found = []
    for root, _, _ in os.walk(run_dir):
        layout = layout_of(root)
        if layout:
            found.append((root, layout))
    return sorted(found)


def load_sequence(seq_dir: str, layout: str) -> Tuple[List[Image.Image], np.ndarray, np.ndarray]:
    """Return the frames to score, their w2c poses and K at frame resolution."""
    if layout == "release":
        files = sorted(glob.glob(os.path.join(seq_dir, "frames", "*.png")))
        poses = np.load(os.path.join(seq_dir, "poses_w2c.npy"))
        with open(os.path.join(seq_dir, "meta.json")) as f:
            idx = json.load(f).get("eval_idx", list(range(len(files))))
        files, poses = [files[i] for i in idx], poses[idx]
        K = np.load(os.path.join(seq_dir, "K.npy"))
    else:
        files = sorted(glob.glob(os.path.join(seq_dir, "eval_frames", "*.png")))
        poses = np.load(os.path.join(seq_dir, "camera_Rt_eval.npy"))
        K = np.load(os.path.join(seq_dir, "camera_K.npy"))
    assert len(files) == len(poses), f"{seq_dir}: {len(files)} frames but {len(poses)} poses"
    return [Image.open(f).convert("RGB") for f in files], poses, K


def group_of(run_dir: str, seq_dir: str, layout: str) -> str:
    # release: average over input images for each camera action
    rel = os.path.relpath(seq_dir, run_dir).split(os.sep)
    if layout == "release" and len(rel) >= 2:
        return os.path.join(run_dir, *rel[:-2], "*", rel[-1])
    return os.path.dirname(os.path.normpath(seq_dir))


def print_summary(groups: Dict[str, List[dict]]) -> None:
    name_w = max(len(g) for g in groups) + 2
    print("\n" + "group".ljust(name_w) + f"{'n':>4}" + "".join(f"{c:>14}" for c in COLUMNS))
    for g, results in sorted(groups.items()):
        means = [np.nanmean([r[c] for r in results]) for c in COLUMNS]
        print(g.ljust(name_w) + f"{len(results):>4}" + "".join(f"{m:>14.4f}" for m in means))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--device", default=default_device())
    args = p.parse_args()

    sequences = [(run_dir, seq, layout) for run_dir in args.run_dirs for seq, layout in find_sequences(run_dir)]
    if not sequences:
        raise SystemExit("no sequence folders found")

    perceptual = PerceptualMetrics(args.device)
    flow = FlowEstimator(args.device)

    groups = defaultdict(list)
    for run_dir, seq_dir, layout in sequences:
        images, poses, K = load_sequence(seq_dir, layout)
        frames, K = resize_for_eval(images, K)
        result = evaluate_sequence(frames, K, poses, perceptual, flow)
        with open(os.path.join(seq_dir, "metrics.json"), "w") as f:
            json.dump(result, f, indent=2)
        groups[group_of(run_dir, seq_dir, layout)].append(result)
        print(f"{seq_dir}: LPIPS-next {result['lpips_next']:.4f}  EAE(e-3) {result['eae_1e3']:.3f}  "
              f"ZMR {result['zmr']:.3f}", flush=True)

    print_summary(groups)


if __name__ == "__main__":
    main()
