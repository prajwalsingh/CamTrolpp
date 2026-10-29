"""Registration-free reprojection.

The first frame of a pass is lifted to a point cloud with monocular depth
(ZoeDepth) and splatted into every other pose of the pass. Small gaps are
filled from the nearest valid pixel, and large disocclusions are filled by
Stable Diffusion inpainting. No point cloud registration is needed.
"""

import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter
from torchvision.transforms import ToTensor

from .color import match_lab

INPAINT_PROMPT = "consistent coherent background scene outpainting"
INPAINT_NEGATIVE = "blurry, low quality, distorted, artifacts"


def splat(pixels: np.ndarray, colors: np.ndarray, height: int, width: int) -> np.ndarray:
    """Forward-splat colours to (x, y) pixel positions with a 3x3 kernel, then
    fill pixels that received nothing from their nearest filled neighbour."""
    out = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)
    px = np.round(pixels).astype(np.int32)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            w = 1.0 / (1 + abs(dx) + abs(dy))
            ty, tx = px[:, 1] + dy, px[:, 0] + dx
            ok = (ty >= 0) & (ty < height) & (tx >= 0) & (tx < width)
            out[ty[ok], tx[ok]] += colors[ok] * w
            weight[ty[ok], tx[ok]] += w
    out = np.where(weight[..., None] > 0, out / (weight[..., None] + 1e-8), 0)
    empty = weight == 0
    if empty.any():
        out = out[tuple(distance_transform_edt(empty, return_distances=False, return_indices=True))]
    return out


def reproject(image: Image.Image, depth: np.ndarray, K: np.ndarray, poses: np.ndarray):
    """Warp `image` (seen from poses[0]) into every later pose.

    Yields (warped, keep) per target pose: warped is an HxWx3 float image in
    [0, 1] with holes set to 0, keep is 1 where the warp is valid.
    """
    H, W = depth.shape
    x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy")
    edge = np.pad(np.ones((H - 4, W - 4)), ((2, 2), (2, 2)))

    R0, T0 = poses[0, :3, :3], poses[0, :3, 3:4]
    pts_cam = np.matmul(np.linalg.inv(K), np.stack((x * depth, y * depth, 1 * depth), axis=0).reshape(3, -1))
    pts_world = (np.linalg.inv(R0).dot(pts_cam) - np.linalg.inv(R0).dot(T0)).astype(np.float32)
    colors = np.array(image).reshape(-1, 3).astype(np.float32) / 255.0

    for pose in poses[1:]:
        R, T = pose[:3, :3], pose[:3, 3:4]
        pix = np.matmul(K, R.dot(pts_world) + T)
        valid = np.where(np.logical_and.reduce((
            pix[2] > 0,
            pix[0] / pix[2] >= 0, pix[0] / pix[2] <= W - 1,
            pix[1] / pix[2] >= 0, pix[1] / pix[2] <= H - 1,
        )))[0]
        pix = pix[:2, valid] / pix[-1:, valid]

        warped = splat(pix.transpose(1, 0), colors[valid, :], H, W)
        warped = edge[..., None] * warped + (1 - edge[..., None]) * np.pad(warped[1:-1, 1:-1], ((1, 1), (1, 1), (0, 0)), mode="edge")

        # Pixels far from any projected point are holes, then shrink the valid area
        # so the inpainter also redraws the unreliable rim around each hole.
        covered = np.zeros((H, W), dtype=np.float32)
        hit = np.round(pix).astype(np.int32)
        covered[hit[1], hit[0]] = 1
        covered = maximum_filter(covered, size=(9, 9), axes=(0, 1))
        warped = covered[..., None] * warped + (1 - covered[..., None]) * (-1)
        keep = minimum_filter((warped.sum(-1) != -3) * 1, size=(11, 11), axes=(0, 1))
        warped = keep[..., None] * warped + (1 - keep[..., None]) * 0
        yield warped, keep


class Reprojector:
    """Depth estimation, warping and inpainting for one pass."""

    def __init__(self, inpaint_ckpt: str, zoedepth_dir: str, device: str = "cuda"):
        self.device = device
        self.inpainter = StableDiffusionInpaintPipeline.from_single_file(
            inpaint_ckpt, torch_dtype=torch.float16, local_files_only=True).to(device)
        self.inpainter.set_progress_bar_config(disable=True)
        self.depth_model = torch.hub.load(zoedepth_dir, "ZoeD_N", source="local", pretrained=True).to(device)

    def depth(self, image: Image.Image) -> np.ndarray:
        return self.depth_model.infer_pil(image)

    def inpaint(self, warped, keep, generator, steps, reference=None) -> Image.Image:
        H, W = keep.shape
        result = self.inpainter(
            prompt=INPAINT_PROMPT, negative_prompt=INPAINT_NEGATIVE, generator=generator,
            num_inference_steps=steps, height=H, width=W,
            image=Image.fromarray(np.round(warped * 255.0).astype(np.uint8)),
            mask_image=Image.fromarray(np.round((1 - keep) * 255.0).astype(np.uint8)),
        ).images[0]
        result = np.array(result)
        if reference is not None:
            result = match_lab(result, reference)
        return Image.fromarray(result)

    def warp_pass(self, image: Image.Image, poses: np.ndarray, K: np.ndarray,
                  steps: int = 25, reference: np.ndarray = None) -> torch.Tensor:
        """Conditioning frames for one pass, shape (N, 3, H, W) in [-1, 1].

        Frame 0 is `image` itself. With a reference image, inpainted frames are
        colour matched to it so newly filled regions start with its colours.
        """
        to_model = lambda im: (ToTensor()(im) * 2.0 - 1.0).unsqueeze(0).to(self.device)
        generator = torch.Generator(device=self.device).manual_seed(-1)
        frames = [to_model(image)]
        for warped, keep in reproject(image, self.depth(image), K, poses):
            frames.append(to_model(self.inpaint(warped, keep, generator, steps, reference)))
        return torch.cat(frames)
