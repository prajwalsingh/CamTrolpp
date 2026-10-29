
import logging
import math
from inspect import isfunction
from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from packaging import version
from torch import nn
from torch.utils.checkpoint import checkpoint

import math
import xformers
import xformers.ops

logpy = logging.getLogger(__name__)

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = {}
    logpy.warn(
        f"No SDP backend available, likely because you are running in pytorch "
        f"versions < 2.0. In fact, you are using PyTorch {torch.__version__}. "
        f"You might want to consider upgrading."
    )

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    logpy.warn("no module 'xformers'. Processing without...")

@torch.jit.script
def invert_rt_matrix(Rt: torch.Tensor) -> torch.Tensor:
    """
    Inverts a 3x4 Camera-to-World (C2W) matrix to a World-to-Camera (W2C) matrix.
    Input Rt = [R | t]
    Output Rt_inv = [R.T | -R.T @ t]
    """
    # 1. Extract R and t
    R = Rt[:, :3] # [3, 3]
    t = Rt[:, 3]  # [3]
    
    # 2. Calculate R_inv (which is R.T for an orthonormal matrix)
    R_inv = R.T # [3, 3]
    
    # 3. Calculate t_inv (-R.T @ t)
    # We need to unsqueeze t to be [3, 1] for matmul, then squeeze back to [3]
    t_inv = -R_inv @ t.unsqueeze(-1) # [3, 1]
    
    # 4. Combine back into a 3x4 matrix
    # Create an output tensor to avoid in-place modification
    Rt_inv = torch.zeros_like(Rt)
    Rt_inv[:, :3] = R_inv
    Rt_inv[:, 3] = t_inv.squeeze() # Squeeze back to [3]
    
    return Rt_inv

@torch.jit.script
def skew_symmetric(t):
    # ... (your JIT code) ...
    out = torch.zeros((3, 3), device=t.device, dtype=t.dtype)
    out[0, 1] = -t[2]; out[0, 2] =  t[1]
    out[1, 0] =  t[2]; out[1, 2] = -t[0]
    out[2, 0] = -t[1]; out[2, 1] =  t[0]
    return out

@torch.jit.script
def get_pixel_grid(height: int, width: int, device: torch.device):
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    return torch.stack([x.float(), y.float(), torch.ones_like(x.float())], dim=-1).view(-1, 3)

@torch.jit.script
def get_epipolar_lines(K, Rt_ref, Rt_tgt, height: int, width: int):
    # FIX (verified against synthetic + real-code tests, see epipolar_bug_full_explanation.md
    # Sec. 2): the line in the REFERENCE image for a TARGET pixel is l_ref = F^T @ x_tgt.
    # As a row-vector product that is p_tgt @ F, NOT p_tgt @ F.T (the original had the
    # extra transpose, which is only correct for the target-pose-inversion bug this
    # patch also fixes below -- the two must be fixed together, never independently).
    R_ref, t_ref = Rt_ref[:, :3], Rt_ref[:, 3]
    R_tgt, t_tgt = Rt_tgt[:, :3], Rt_tgt[:, 3]
    R_rel = R_tgt @ R_ref.T
    t_rel = t_tgt - R_rel @ t_ref
    t_skew = skew_symmetric(t_rel)
    E = t_skew @ R_rel
    K_inv = torch.inverse(K)
    F = K_inv.T @ E @ K_inv
    p_tgt = get_pixel_grid(height, width, K.device)
    epipolar_lines = p_tgt @ F          # FIX: was `p_tgt @ F.T`
    return epipolar_lines.view(height, width, 3)


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class SelfAttention(nn.Module):
    ATTENTION_MODES = ("xformers", "torch", "math")

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_mode: str = "xformers",
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        assert attn_mode in self.ATTENTION_MODES
        self.attn_mode = attn_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        B, L, C = x.shape

        qkv = self.qkv(x)
        if self.attn_mode == "torch":
            qkv = rearrange(
                qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
            ).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = rearrange(x, "B H L D -> B L (H D)")
        elif self.attn_mode == "xformers":
            qkv = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = rearrange(x, "B L H D -> B L (H D)", H=self.num_heads)
        elif self.attn_mode == "math":
            qkv = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = torch.einsum("bij,bjk->bik", q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        h_ = self.proj_out(h_)

        return x + h_


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.backend = backend

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        
        h = self.heads

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        ## old
        """
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        """
        ## new
        with sdp_kernel(**BACKEND_MAP[self.backend]):
            # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )  # scale is dim_head ** -0.5 per default

        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0,
        epipolar_temperature=20.0, epipolar_layers=(23,), epipolar_alpha=0.9, **kwargs
    ):
        super().__init__()
        logpy.debug(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, "
            f"context_dim is {context_dim} and using {heads} heads with a "
            f"dimension of {dim_head}."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        # FIX (temperature): the original scale (1/sqrt(dim_head), applied AFTER
        # L2-normalizing q and k) confines logits to [-1/sqrt(d), 1/sqrt(d)] and
        # produces a near-uniform softmax over the 60 sampled points (measured
        # entropy ratio 1.0000 -- i.e. a plain average, not correspondence
        # selection). This value (20.0) was chosen so the softmax's entropy ratio
        # drops to ~0.56 (vs. 1.0 shipped) on a conservative random-vector floor;
        # real, correlated features will be even more selective. Sweep as an
        # ablation axis (e.g. {15, 20, 25, 30}) rather than treating this as final.
        self.epipolar_temperature = epipolar_temperature
        # FIX: was hardcoded `if level in [23]:` inline in forward(); exposed here
        # so it's configurable without editing the method body.
        self.epipolar_layers = set(epipolar_layers)
        # NEW: was hardcoded `alpha = 0.9` inline in forward(); exposed here as the
        # construction-time default. All three of epipolar_layers/epipolar_alpha/
        # epipolar_temperature can ALSO be overridden per forward() call (threaded
        # as kwargs from sample() -> ... -> here, same path as Cam_K/level/c2w) --
        # see forward()'s signature below. These self.X values are only the
        # fallback used when a forward-time override isn't provided.
        self.epipolar_alpha = epipolar_alpha

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op = None

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
        Cam_K=None,
        Cam_Rt=None,
        Attn_Vis = None,
        height=88,
        width=120,
        level=0,
        vis_target_frame_idx=7,
        vis_target_pixel_coords=(60, 44),
        c2w=False,
        epipolar_layers=None,
        epipolar_alpha=None,
        epipolar_temperature=None,
    ):
        # NEW: forward-time overrides for epipolar config, threaded the same way
        # as Cam_K/level/c2w (sample() -> ... -> here). None means "use the
        # construction-time self.X default" -- fully backward compatible with
        # any caller that doesn't pass these.
        active_epipolar_layers = set(epipolar_layers) if epipolar_layers is not None else self.epipolar_layers
        active_epipolar_alpha = epipolar_alpha if epipolar_alpha is not None else self.epipolar_alpha
        active_epipolar_temperature = epipolar_temperature if epipolar_temperature is not None else self.epipolar_temperature

        is_self_attn_with_epi = context is None and Cam_K is not None and Cam_Rt is not None
        h, w = height, width

        if additional_tokens is not None:
            n_tokens_to_mask = additional_tokens.shape[1]
            x = torch.cat([additional_tokens, x], dim=1)
            
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q_orig, k_orig, v_orig = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        if version.parse(xformers.__version__) >= version.parse("0.0.21"):
            max_bs = 32768
            N = q_orig.shape[0]
            n_batches = math.ceil(N / max_bs)
            out_orig = []
            for i_batch in range(n_batches):
                batch = slice(i_batch * max_bs, (i_batch + 1) * max_bs)
                out_orig.append(
                    xformers.ops.memory_efficient_attention(
                        q_orig[batch], k_orig[batch], v_orig[batch], attn_bias=None, op=self.attention_op
                    )
                )
            out_orig = torch.cat(out_orig, 0)
        else:
            out_orig = xformers.ops.memory_efficient_attention(
                q_orig, k_orig, v_orig, attn_bias=None, op=self.attention_op
            )

        out_orig = (
            out_orig.unsqueeze(0)
            .reshape(b, self.heads, out_orig.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out_orig.shape[1], self.heads * self.dim_head)
        )

        if not is_self_attn_with_epi:
            if additional_tokens is not None:
                out_orig = out_orig[:, n_tokens_to_mask:]
            return self.to_out(out_orig)

        # --- EPIPOLAR ATTENTION LOGIC ---
        if level in active_epipolar_layers:

            alpha = active_epipolar_alpha
            num_samples_per_line = w // 2
            num_frames = x.shape[0] // 2
            inner_dim = self.heads * self.dim_head
            
            q, k, v = q.view(b, h, w, inner_dim), k.view(b, h, w, inner_dim), v.view(b, h, w, inner_dim)
            
            q_uc, q_c = q.chunk(2)
            k_uc, k_c = k.chunk(2)
            v_uc, v_c = v.chunk(2)
            
            gathered_features_c, vis_attn_c = self._compute_epipolar_features(
                q_c, k_c, v_c, Cam_K, Cam_Rt, h, w, num_frames, num_samples_per_line,
                vis_target_frame_idx=vis_target_frame_idx,
                vis_target_pixel_coords=vis_target_pixel_coords,
                c2w=c2w,   # FIX: was silently dropped, defaulting to False regardless of caller
                temperature=active_epipolar_temperature,  # NEW: forward-time override, was self.X-only
            )
            gathered_features_uc, vis_attn_uc = self._compute_epipolar_features(
                q_uc, k_uc, v_uc, Cam_K, Cam_Rt, h, w, num_frames, num_samples_per_line,
                c2w=c2w,   # FIX: was silently dropped, defaulting to False regardless of caller
                temperature=active_epipolar_temperature,  # NEW: forward-time override, was self.X-only
            )
            
            if vis_attn_c is not None and Attn_Vis is not None:
                Attn_Vis.append(vis_attn_c)
            
            gathered_features = torch.cat([gathered_features_uc, gathered_features_c], dim=0)
            gathered_features = gathered_features.view(b, h * w, inner_dim)

            out_fused = alpha * gathered_features + (1 - alpha) * out_orig

        else:
            out_fused = out_orig
        
        if additional_tokens is not None:
            out_fused = out_fused[:, n_tokens_to_mask:]
        return self.to_out(out_fused)
        
    def _compute_epipolar_features(self, q, k, v, Cam_K, Cam_Rt, h, w, num_frames, num_samples_per_line, vis_target_frame_idx=-1, vis_target_pixel_coords=None, c2w=False, temperature=None):
        """
        Compute epipolar attention features.
        
        Args:
            Cam_K: camera intrinsics (FULL RESOLUTION, batched [B, 3, 3])
            Cam_Rt: camera extrinsics (C2W Poses, batched [B, N, 3, 4])
            h, w: height and width of the LATENT feature map
            ...
        """
        # Cam_Rt contains C2W poses. We need W2C poses.
        # Invert the reference pose.
        Rt_ref_C2W = Cam_Rt[0]
        if c2w:
            Rt_ref_W2C = invert_rt_matrix(Rt_ref_C2W)
        else:
            Rt_ref_W2C = Rt_ref_C2W
        
        k_ref  = k[0] # h, w, inner_dim
        v_ref  = v[0] # h, w, inner_dim

        # K must be scaled from image resolution to latent resolution
        K_full_res = Cam_K[0] # Get the [3, 3] full-res intrinsic tensor

        # Get original H, W from the K matrix itself (c_x = W/2, c_y = H/2)
        W_orig = K_full_res[0, 2] * 2.0
        H_orig = K_full_res[1, 2] * 2.0
        
        # Calculate scaling factors
        scale_x = w / W_orig
        scale_y = h / H_orig

        # Create the scaled K_latent
        K_latent = K_full_res.clone()
        K_latent[0, 0] *= scale_x # f_x
        K_latent[1, 1] *= scale_y # f_y
        K_latent[0, 2] *= scale_x # c_x
        K_latent[1, 2] *= scale_y # c_y

        all_gathered_v = [v[0]] # The first frame's gathered feature is itself

        # Pre-prepare reference features ONCE (outside the loop)
        k_ref_res = k_ref.permute(2, 0, 1).unsqueeze(0) # Shape: 1, C, H, W
        v_ref_res = v_ref.permute(2, 0, 1).unsqueeze(0) # Shape: 1, C, H, W
        
        # Pre-compute normalized x coordinates (reused across frames)
        x_samples = torch.linspace(0, w - 1, num_samples_per_line, device=q.device)
        x_samples_norm = (x_samples / (w - 1)) * 2 - 1
        
        attn_weights_to_return = None # Initialize to None

        for i in range(1, num_frames):
            q_tgt = q[i] # h, w, inner_dim
            
            # --- POSE CONVENTION (FIXED) ---
            Rt_tgt_C2W = Cam_Rt[i]
            # FIX: was `Rt_tgt_W2C = invert_rt_matrix(Rt_tgt_C2W)` unconditionally,
            # regardless of c2w. This must mirror the reference branch above --
            # applying the pose-orientation fix without ALSO fixing the F/F^T
            # orientation in get_epipolar_lines (or vice versa) will silently break
            # cases where the two previously happened to cancel (see
            # epipolar_bug_full_explanation.md Sec. 2 for the full derivation).
            # Both are fixed together in this file.
            if c2w:
                Rt_tgt_W2C = invert_rt_matrix(Rt_tgt_C2W)
            else:
                Rt_tgt_W2C = Rt_tgt_C2W
            
            # 1. Compute epipolar lines for ALL pixels at once using W2C poses and SCALED K_latent
            epipolar_lines = get_epipolar_lines(
                K_latent,   # <-- Use the scaled K matrix
                Rt_ref_W2C, # <-- Use the inverted W2C ref pose
                Rt_tgt_W2C, # <-- Use the inverted W2C tgt pose
                h, w
            )
            epipolar_lines = epipolar_lines.view(h * w, 3)

            # 2. Calculate sample coordinates for ALL pixels
            a, b, c = epipolar_lines[:, 0], epipolar_lines[:, 1], epipolar_lines[:, 2]
            
            y_samples = (-a.unsqueeze(1) * x_samples.unsqueeze(0) - c.unsqueeze(1)) / (b.unsqueeze(1) + 1e-6)
            y_samples_norm = (y_samples / (h - 1)) * 2 - 1

            # FIX (validity masking): samples whose line exits the image vertically
            # (or whose line is near-vertical, causing y_samples to blow up) were
            # silently clamped to the image border by grid_sample's padding_mode,
            # producing confident-looking but meaningless features. Flag them here;
            # they get masked to -inf before the softmax below, and pixels with NO
            # valid samples at all fall back to their own (target-frame) feature
            # rather than an all-invalid softmax row (which would be NaN).
            valid_mask_full = (y_samples_norm.abs() <= 1.0)  # [h*w, num_samples_per_line]

            # Reshape grid to process all samples at once: [h*w*num_samples, 2]
            sample_grid_flat = torch.stack([
                x_samples_norm.expand(h*w, -1).reshape(-1),  # Flatten all x coords
                y_samples_norm.reshape(-1)  # Flatten all y coords
            ], dim=-1)
            
            # Reshape for grid_sample: [1, 1, h*w*num_samples, 2]
            sample_grid_batch = sample_grid_flat.unsqueeze(0).unsqueeze(0)
            
            q_tgt_flat = q_tgt.view(h * w, -1)
            v_tgt_flat = v[i].reshape(h * w, -1)  # FIX: pass-through source for invalid-sample pixels

            # --- VECTORIZED SAMPLING (SINGLE GRID_SAMPLE CALL) ---
            k_sampled_all = nn.functional.grid_sample(
                k_ref_res, sample_grid_batch, 
                mode='bilinear', padding_mode='border', align_corners=True
            )
            v_sampled_all = nn.functional.grid_sample(
                v_ref_res, sample_grid_batch, 
                mode='bilinear', padding_mode='border', align_corners=True
            )
            k_sampled = k_sampled_all.squeeze(0).squeeze(1).t().view(h * w, num_samples_per_line, -1)
            v_sampled = v_sampled_all.squeeze(0).squeeze(1).t().view(h * w, num_samples_per_line, -1)

            chunk_size = 2048
            gathered_v_chunks = []
            
            for j in range(0, h * w, chunk_size):
                chunk_end = min(j + chunk_size, h * w)
                current_chunk_size = chunk_end - j

                q_tgt_chunk = q_tgt_flat[j:chunk_end]
                k_sampled_chunk = k_sampled[j:chunk_end]
                v_sampled_chunk = v_sampled[j:chunk_end]
                valid_chunk = valid_mask_full[j:chunk_end]        # (chunk, num_samples)
                any_valid_chunk = valid_chunk.any(dim=-1)          # (chunk,)

                q_tensor_chunk = q_tgt_chunk.view(current_chunk_size, self.heads, self.dim_head)
                k_tensor_chunk = k_sampled_chunk.view(current_chunk_size, num_samples_per_line, self.heads, self.dim_head).permute(0, 2, 1, 3)
                v_tensor_chunk = v_sampled_chunk.view(current_chunk_size, num_samples_per_line, self.heads, self.dim_head).permute(0, 2, 1, 3)
                
                q_norm = nn.functional.normalize(q_tensor_chunk, dim=-1)
                k_norm = nn.functional.normalize(k_tensor_chunk, dim=-1)
                
                with torch.cuda.amp.autocast(enabled=False):
                    q_f32 = q_norm.float()
                    k_f32 = k_norm.float()
                    v_f32 = v_tensor_chunk.float()
                    
                    # FIX (temperature): was `1.0 / math.sqrt(self.dim_head)` applied
                    # to already-L2-normalized q/k, confining logits to a tiny range
                    # and producing a near-uniform softmax (measured entropy ratio
                    # 1.0000 -- i.e. a plain average of all samples, not correspondence
                    # selection). See __init__ for justification of the new default.
                    # NEW: forward-time override (temperature param), falls back to
                    # self.epipolar_temperature if not provided -- same pattern as
                    # active_epipolar_temperature above.
                    _temp = temperature if temperature is not None else self.epipolar_temperature
                    attn_scores = torch.einsum('bhd,bhsd->bhs', q_f32, k_f32) * _temp

                    # FIX (validity masking): mask samples whose line left the image,
                    # but leave fully-invalid rows unmasked here (to avoid an all -inf
                    # softmax row, which is NaN) -- those rows get overridden with a
                    # pass-through value below instead.
                    mask_chunk = valid_chunk.unsqueeze(1).expand(-1, self.heads, -1)  # (chunk, heads, samples)
                    safe_mask = mask_chunk | (~any_valid_chunk).view(-1, 1, 1)
                    attn_scores = attn_scores.masked_fill(~safe_mask, float('-inf'))

                    attn_weights = attn_scores.softmax(dim=-1)
                    
                    if i == vis_target_frame_idx and vis_target_pixel_coords is not None:
                        target_flat_idx = vis_target_pixel_coords[1] * w + vis_target_pixel_coords[0]
                        if j <= target_flat_idx < chunk_end:
                            idx_in_chunk = target_flat_idx - j
                            attn_weights_to_return = attn_weights[idx_in_chunk].mean(dim=0)
                    
                    gathered_per_head = torch.einsum('bhs,bhsd->bhd', attn_weights, v_f32)
                    gathered_v_chunk = gathered_per_head.reshape(current_chunk_size, -1)

                gathered_v_chunk = gathered_v_chunk.to(q_tensor_chunk.dtype)

                # FIX (validity masking, part 2): pixels whose epipolar line left the
                # image entirely have no real information to gather -- fall back to
                # their own target-frame feature instead of the "unmasked" softmax
                # result computed above purely to avoid NaN.
                gathered_v_chunk = torch.where(
                    any_valid_chunk.unsqueeze(-1),
                    gathered_v_chunk,
                    v_tgt_flat[j:chunk_end].to(gathered_v_chunk.dtype),
                )

                gathered_v_chunks.append(gathered_v_chunk)
            
            gathered_v = torch.cat(gathered_v_chunks, dim=0)
            all_gathered_v.append(gathered_v.view(h, w, -1))

        return torch.stack(all_gathered_v, dim=0), attn_weights_to_return


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,  # ampere
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        attn_mode="softmax",
        sdp_backend=None,
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
            logpy.warn(
                f"Attention mode '{attn_mode}' is not available. Falling "
                f"back to native attention. This is not a problem in "
                f"Pytorch >= 2.0. FYI, you are running with PyTorch "
                f"version {torch.__version__}."
            )
            attn_mode = "softmax"
        elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            logpy.warn(
                "We do not support vanilla attention anymore, as it is too "
                "expensive. Sorry."
            )
            if not XFORMERS_IS_AVAILABLE:
                assert (
                    False
                ), "Please install xformers via e.g. 'pip install xformers==0.0.16'"
            else:
                logpy.info("Falling back to xformers efficient attention.")
                attn_mode = "softmax-xformers"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            backend=sdp_backend,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        if self.checkpoint:
            logpy.debug(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0, **kwargs
    ):
        kwargs   = {"x": x, "Cam_K": kwargs['Cam_K'], "Cam_Rt": kwargs['Cam_Rt'], "Attn_Vis": kwargs['Attn_Vis'], "height": kwargs['height'], "width": kwargs['width'], "level": kwargs['level'], "c2w": kwargs['c2w'],
                     # NEW: optional epipolar overrides, threaded the same way as
                     # Cam_K etc. above but via .get() (not bracket access) so
                     # this stays backward compatible with any caller upstream
                     # that hasn't been updated to pass these yet -- None means
                     # "use MemoryEfficientCrossAttention's construction-time default".
                     "epipolar_layers": kwargs.get('epipolar_layers'),
                     "epipolar_alpha": kwargs.get('epipolar_alpha'),
                     "epipolar_temperature": kwargs.get('epipolar_temperature')}

        if context is not None:
            kwargs.update({"context": context})

        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})

        if n_times_crossframe_attn_in_self:
            kwargs.update(
                {"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self}
            )

        # return mixed_checkpoint(self._forward, kwargs, self.parameters(), self.checkpoint)
        if self.checkpoint:
            # inputs = {"x": x, "context": context}
            return checkpoint(self._forward, x, context, None, 0, kwargs['Cam_K'], kwargs['Cam_Rt'], kwargs['Attn_Vis'], kwargs['height'], kwargs['width'], kwargs['level'], kwargs['c2w'], kwargs['epipolar_layers'], kwargs['epipolar_alpha'], kwargs['epipolar_temperature'])
            # return checkpoint(self._forward, inputs, self.parameters(), self.checkpoint)
        else:
            return self._forward(**kwargs)

    def _forward(
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0, Cam_K=None, Cam_Rt=None, Attn_Vis = None, height=88, width=120, level=0, c2w=False,
        epipolar_layers=None, epipolar_alpha=None, epipolar_temperature=None
    ):
        
        x = (
            self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                additional_tokens=additional_tokens,
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self if not self.disable_self_attn else 0,
                Cam_K  = Cam_K,
                Cam_Rt = Cam_Rt,
                Attn_Vis = Attn_Vis,
                height = height,
                width  = width,
                level  = level,
                c2w    = c2w,
                epipolar_layers = epipolar_layers,
                epipolar_alpha = epipolar_alpha,
                epipolar_temperature = epipolar_temperature,
            )
            + x
        )
        x = (
            self.attn2(
                self.norm2(x), context=context, additional_tokens=additional_tokens, Cam_K  = Cam_K, Cam_Rt = Cam_Rt, Attn_Vis = Attn_Vis, height = height, width = width, level=level, c2w=c2w,
                epipolar_layers=epipolar_layers, epipolar_alpha=epipolar_alpha, epipolar_temperature=epipolar_temperature,
            )
            + x
        )
        x = self.ff(self.norm3(x)) + x
        return x


class BasicTransformerSingleLayerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention  # on the A100s not quite as fast as the above version
        # (todo might depend on head_dim, check, falls back to semi-optimized kernels for dim!=[16,32,64,128])
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        attn_mode="softmax",
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        # inputs = {"x": x, "context": context}
        # return checkpoint(self._forward, inputs, self.parameters(), self.checkpoint)
        return checkpoint(self._forward, x, context)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context) + x
        x = self.ff(self.norm2(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        attn_type="softmax",
        use_checkpoint=True,
        # sdp_backend=SDPBackend.FLASH_ATTENTION
        sdp_backend=None,
    ):
        super().__init__()
        logpy.debug(
            f"constructing {self.__class__.__name__} of depth {depth} w/ "
            f"{in_channels} channels and {n_heads} heads."
        )

        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                logpy.warn(
                    f"{self.__class__.__name__}: Found context dims "
                    f"{context_dim} of depth {len(context_dim)}, which does not "
                    f"match the specified 'depth' of {depth}. Setting context_dim "
                    f"to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            # self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
        checkpoint: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                BasicTransformerBlock(
                    dim,
                    heads,
                    dim_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    attn_mode="softmax-xformers",
                    checkpoint=checkpoint,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, context)
        return x

