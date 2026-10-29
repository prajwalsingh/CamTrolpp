"""Refinement of one pass with Stable Video Diffusion.

The warped frames are encoded, noised to an intermediate level and denoised
with the frozen SVD model, conditioned on the first frame of the pass. The
camera poses of the pass drive the epipolar attention in the decoder.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf

from sgm.util import instantiate_from_config


@dataclass
class RefineSettings:
    num_steps: int = 25          # sampler steps of the full schedule
    add_index: int = 11          # start denoising at this step (higher keeps more of the warp)
    fps_id: int = 6
    motion_bucket_id: int = 115
    cond_aug: float = 0.02
    decoding_t: int = 4          # frames decoded at a time (lower saves memory)
    epipolar_layers: tuple = (23,)  # attention blocks with epipolar attention, () turns it off
    epipolar_alpha: float = 0.9     # weight of epipolar against standard attention
    epipolar_temperature: float = 20.0


def load_svd(config_path: str, num_frames: int, num_steps: int, device: str = "cuda"):
    config = OmegaConf.load(config_path)
    params = config.model.params
    params.conditioner_config.params.emb_models[0].params.open_clip_embedding_config.params.init_device = device
    params.sampler_config.params.verbose = False
    params.sampler_config.params.num_steps = num_steps
    params.sampler_config.params.guider_config.params.num_frames = num_frames
    with torch.device(device):
        return instantiate_from_config(config.model).to(device).eval()


def _conditioning(model, cond_image, settings, n, device):
    """Image, fps, motion and augmentation conditioning for an n-frame clip."""
    batch = {
        "cond_frames_without_noise": cond_image,
        "cond_frames": cond_image + 1e-10 * torch.randn_like(cond_image),
        "fps_id": torch.tensor([settings.fps_id]).to(device).repeat(n),
        "motion_bucket_id": torch.tensor([settings.motion_bucket_id]).to(device).repeat(n),
        "cond_aug": repeat(torch.tensor([settings.cond_aug]).to(device), "1 -> b", b=n),
    }
    batch_uc = {k: v.clone() for k, v in batch.items()}
    batch["num_video_frames"] = n
    c, uc = model.conditioner.get_unconditional_conditioning(
        batch, batch_uc=batch_uc, force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"])
    for k in ["crossattn", "concat"]:
        uc[k] = rearrange(repeat(uc[k], "b ... -> b t ...", t=n), "b t ... -> (b t) ...", t=n)
        c[k] = rearrange(repeat(c[k], "b ... -> b t ...", t=n), "b t ... -> (b t) ...", t=n)
    return c, uc


@torch.no_grad()
def refine_pass(model, warped: torch.Tensor, cond_image: torch.Tensor, K: torch.Tensor,
                poses: torch.Tensor, settings: RefineSettings, device: str = "cuda") -> np.ndarray:
    """Refine warped frames (N, 3, H, W) in [-1, 1]. Returns N uint8 RGB frames."""
    n, _, H, W = warped.shape
    with torch.autocast(device):
        c, uc = _conditioning(model, cond_image, settings, n, device)
        extra = {
            "image_only_indicator": torch.zeros(2, n).to(device),
            "num_video_frames": n,
            "Cam_K": K.unsqueeze(0).to(device),
            "Cam_Rt": poses.to(device),
            "Attn_Vis": [],
            "c2w": False,
            "epipolar_layers": tuple(settings.epipolar_layers),
            "epipolar_alpha": settings.epipolar_alpha,
            "epipolar_temperature": settings.epipolar_temperature,
        }

        def denoiser(x, sigma, cond):
            return model.denoiser(model.model, x, sigma, cond, **extra)

        model.en_and_decode_n_samples_a_time = settings.decoding_t
        latents = model.encode_first_stage(warped)
        shape = (n, 4, H // 8, W // 8)
        randn = torch.randn(shape, device=device)
        sampler = model.sampler
        _, s_in, sigmas, num_sigmas, cond, uc = sampler.prepare_sampling_loop(
            randn, cond=c, uc=uc, num_steps=settings.num_steps)

        # Start from the warped latents at an intermediate noise level.
        x = latents + torch.randn(shape, device=device) * sigmas[settings.add_index]
        for i in sampler.get_sigma_gen(num_sigmas)[settings.add_index:]:
            gamma = (min(sampler.s_churn / (num_sigmas - 1), 2 ** 0.5 - 1)
                     if sampler.s_tmin <= sigmas[i] <= sampler.s_tmax else 0.0)
            x = sampler.sampler_step(s_in * sigmas[i], s_in * sigmas[i + 1], denoiser, x, cond, uc, gamma)
        frames = model.decode_first_stage(x)

    frames = torch.clamp((frames + 1.0) / 2.0, 0.0, 1.0)
    return (rearrange(frames, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8)
