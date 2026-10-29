"""CamTrol++ generation: split the camera path into short passes and refine each.

For each pass the latest frame is reprojected along the poses of that pass,
the warps are refined by SVD, and the last refined frame (colour matched to
the input image) starts the next pass. Consecutive passes share their
boundary frame, so K passes of 14 frames give 13K + 1 frames.
"""

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field

import imageio
import numpy as np
import torch
from PIL import Image

from .cameras import ACTIONS, FRAMES_PER_PASS, HEIGHT, WIDTH, camera_path, intrinsics, num_passes
from .color import match_lab
from .refine import RefineSettings, load_svd, refine_pass
from .warp import Reprojector


@dataclass
class Settings:
    seed: int = 1
    color_match: bool = True     # LAB anchoring of inpainted frames and pass boundaries
    refine: RefineSettings = field(default_factory=RefineSettings)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def center_depth(depth: np.ndarray) -> float:
    h, w = depth.shape
    return float(np.mean(depth[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10]))


class CamTrolPP:
    def __init__(self, checkpoints: str = "checkpoints", device: str = "cuda"):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.device = device
        self.svd = load_svd(os.path.join(root, "sgm", "svd.yaml"), FRAMES_PER_PASS, RefineSettings.num_steps, device)
        inpaint = [os.path.join(checkpoints, f"sd-v1-5-inpainting.{ext}") for ext in ("safetensors", "ckpt")]
        inpaint = next((f for f in inpaint if os.path.exists(f)), inpaint[-1])
        self.reprojector = Reprojector(
            inpaint,
            os.path.join(root, "third_party", "ZoeDepth"), device)

    def generate(self, image_path: str, action: str, total: float, per_pass: float,
                 settings: Settings = None) -> dict:
        """Generate the camera move `action` of size `total` with at most `per_pass` per pass."""
        settings = settings or Settings()
        image = Image.open(image_path).convert("RGB").resize((WIDTH, HEIGHT))
        reference = np.array(image)
        K = intrinsics()

        k_passes = num_passes(total, per_pass)
        n_frames = (FRAMES_PER_PASS - 1) * k_passes + 1
        depth = center_depth(self.reprojector.depth(image)) if ACTIONS[action][0] == "depth" else 1.0
        poses = camera_path(action, total, n_frames, depth)

        seed_everything(settings.seed)
        anchor = reference if settings.color_match else None
        frames, start = [], time.time()
        for k in range(k_passes):
            lo = (FRAMES_PER_PASS - 1) * k
            pass_poses = poses[lo:lo + FRAMES_PER_PASS]
            warped = self.reprojector.warp_pass(image, pass_poses, K, settings.refine.num_steps, anchor)
            cond = warped[:1]  # the first conditioning frame is the pass input itself
            clip = refine_pass(self.svd, warped, cond, torch.tensor(K), torch.tensor(pass_poses).float(),
                               settings.refine, self.device)
            frames.append(clip if k == 0 else clip[1:])  # drop the shared boundary frame
            last = clip[-1] if anchor is None else match_lab(clip[-1], anchor)
            image = Image.fromarray(last)
            print(f"  pass {k + 1}/{k_passes} done", flush=True)

        frames = np.concatenate(frames)
        return {
            "frames": frames, "poses": poses, "K": K,
            "eval_idx": list(range(0, n_frames, k_passes)),  # 14 frames evenly spaced along the path
            "meta": {"image": image_path, "action": action, "total": total, "per_pass": per_pass,
                     "n_passes": k_passes, "n_frames": n_frames, "center_depth": depth,
                     "time_s": round(time.time() - start, 1), "settings": asdict(settings)},
        }


def save_result(result: dict, out_dir: str, fps: int = 8):
    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
    for i, frame in enumerate(result["frames"]):
        Image.fromarray(frame).save(os.path.join(out_dir, "frames", f"{i:05d}.png"))
    imageio.mimwrite(os.path.join(out_dir, "video.mp4"), list(result["frames"]), fps=fps)
    np.save(os.path.join(out_dir, "poses_w2c.npy"), result["poses"].astype(np.float32))
    np.save(os.path.join(out_dir, "K.npy"), result["K"])
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({**result["meta"], "eval_idx": result["eval_idx"]}, f, indent=2)
