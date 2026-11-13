"""Neural building blocks and the two networks that read the obs window.

Generic layers (attention, adaLN, norms, activations), the observation encoder,
and the actor flow model + ensemble chunk-Q critic. Key: FlowModel, Critic,
StateEncoder.
"""

from __future__ import annotations

import contextlib
import math
import os
import random
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ---- from utils.py ----
@lru_cache(maxsize=1)
def cuda_available() -> bool:
    """Report whether CUDA is usable. The result is cached on first call."""
    return torch.cuda.is_available()


@lru_cache(maxsize=1)
def is_rtx_1000_pro_blackwell() -> bool:
    """Detect an NVIDIA RTX PRO 1000 Blackwell GPU."""
    if not cuda_available():
        return False
    try:
        name = torch.cuda.get_device_name(0).lower()
    except (RuntimeError, ValueError):
        return False
    return "rtx pro 1000" in name and "blackwell" in name


@lru_cache(maxsize=1)
def detect_gpu() -> dict | None:
    """Device name, compute capability, and the precision flags they imply."""
    if not cuda_available():
        return None
    try:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
    except (RuntimeError, ValueError, AttributeError):
        return None
    return {
        "device_name": name,
        "compute_capability": [major, minor],
        "supports": {
            "bf16": major >= 8,
            "tf32": major >= 8,
            "fp16_tensor_cores": major >= 7,
        },
    }


def resolve_device() -> torch.device:
    """Execution device from ``MYFPO_DEVICE``."""
    want = os.environ.get("MYFPO_DEVICE", "auto").lower()
    if want == "cpu":
        return torch.device("cpu")
    if want not in {"auto", "cuda"}:
        raise ValueError(
            f"MYFPO_DEVICE={want!r} unknown; expected 'auto', 'cpu', or 'cuda'"
        )
    return torch.device("cuda") if cuda_available() else torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch, including every CUDA device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_storage_dtype(
    device: str | torch.device | None = None,
) -> torch.dtype:
    """Storage dtype for the real execution device."""
    if device is not None and torch.device(device).type == "cpu":
        return torch.float32
    return torch.bfloat16 if is_rtx_1000_pro_blackwell() else torch.float32


def resolve_compute_dtype() -> torch.dtype:
    """Matmul and autocast dtype, from ``MYFPO_DTYPE``. BF16 needs CUDA."""
    want = os.environ.get("MYFPO_DTYPE", "auto").lower()
    if want not in {"auto", "fp32", "bf16"}:
        raise ValueError(
            f"MYFPO_DTYPE={want!r} unknown; expected 'auto', 'fp32', or 'bf16'"
        )
    if want == "fp32" or not cuda_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32


def autocast_context(device=None):
    """A bf16 autocast context on CUDA. A no-op elsewhere."""
    if resolve_compute_dtype() != torch.bfloat16:
        return contextlib.nullcontext()
    dev_type = device.type if isinstance(device, torch.device) else "cuda"
    if dev_type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def rms(x: torch.Tensor) -> torch.Tensor:
    """Root mean square over every element."""
    return torch.sqrt(torch.mean(x * x))


def cudagraph_safe(x: torch.Tensor, enabled: bool = True) -> torch.Tensor:
    """Copy a cudagraph-owned tensor; call before a consumer keeps it.
    ``enabled=False`` returns input unchanged."""
    return x.clone() if enabled else x


def cudagraph_step_begin() -> None:
    """No-op unless cudagraph active; marks step start so memory can recycle."""
    torch.compiler.cudagraph_mark_step_begin()


def to_device_async(x: torch.Tensor, device) -> torch.Tensor:
    """Move a tensor to the device. On CUDA the copy overlaps CPU work."""
    device = torch.device(device)
    if device.type == "cuda" and x.device != device:
        if not x.is_pinned():
            x = x.pin_memory()
        return x.to(device, non_blocking=True)
    return x.to(device)


# A width of 1 gives the norm nothing to normalize, so every hidden width in
# the model has to clear this. One constant keeps the five checks in step.
MIN_WIDTH = 2

class RMSNorm(nn.RMSNorm):
    """RMSNorm. The compute runs in fp32; the stored weight keeps the model dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise the trailing dimensions with the learned scale."""
        input_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = F.rms_norm(
                x.to(dtype=torch.float32),
                self.normalized_shape,
                self.weight.to(dtype=torch.float32),
                self.eps,
            )
        return y.to(dtype=input_dtype)


def make_rms_norm(hidden_dim: int, eps: float = 1e-6) -> nn.RMSNorm:
    """Build a repo ``RMSNorm`` at the given width."""
    return RMSNorm(hidden_dim, eps=eps)


def _check_positive_dims(in_dim: int, hidden_dim: int, out_dim: int, module: str) -> None:
    """One spelling of the positive-width guard for two-layer MLPs."""
    if in_dim < 1 or hidden_dim < 1 or out_dim < 1:
        raise ValueError(f"{module} dimensions must be positive")


def _check_attention_dims(hidden_dim: int, n_heads: int) -> None:
    """One spelling of the attention head-split guard.

    The even-head_dim (RoPE) check stays with ``SelfAttention``; only that
    module uses RoPE.
    """
    if hidden_dim < 1:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
    if n_heads < 1:
        raise ValueError(f"n_heads must be positive, got {n_heads}")
    if hidden_dim % n_heads != 0:
        raise ValueError(f"hidden_dim={hidden_dim} must split over {n_heads} heads")


class MishFFN(nn.Module):
    """``Linear -> LayerNorm -> Mish -> Linear``. The DiT block's feed-forward unit."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        _check_positive_dims(in_dim, hidden_dim, out_dim, "MishFFN")
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to the hidden width, norm, apply Mish, and project to out_dim."""
        return self.out(F.mish(self.norm(self.fc(x))))


class CondMLP(nn.Module):
    """SiLU MLP for conditioning regression."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        _check_positive_dims(in_dim, hidden_dim, out_dim, "CondMLP")
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to the hidden width, apply SiLU, and project to out_dim."""
        return self.fc2(F.silu(self.fc1(x)))


# DiT scales its integer timesteps to [0, 1000]. The flow lives on t in
# [0, 1], so the same scale here gives the code the resolution the MLP needs:
# the top channel completes about 159 cycles over the full path.
TIME_SCALE = 1000.0


class SinusoidalEncoding(nn.Module):
    """Fixed sinusoidal codes. The module holds one freq table as a buffer."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        if dim < MIN_WIDTH:
            raise ValueError(f"SinusoidalEncoding dim must be >= 2, got {dim}")
        if max_period <= 0:
            raise ValueError(f"max_period must be > 0, got {max_period}")
        self.dim = dim
        self.half = dim // 2
        # Log-spaced, decaying. This is the DDPM/DiT table. A linear 2**i table
        # would run to 32768 rad over t in [0, 1] and code noise, not smoothness.
        self.register_buffer(
            "freqs",
            torch.exp(
                -math.log(max_period)
                * torch.arange(self.half, dtype=torch.float32)
                / self.half
            ),
            persistent=False,
        )

    def encode_times(self, t: torch.Tensor) -> torch.Tensor:
        """Embed continuous times with a sinusoid. Accept ``[N]``, return ``[N, dim]``."""
        t = t.to(dtype=torch.float32)
        freqs = self.freqs.to(t.device)
        ang = (t.unsqueeze(-1) * TIME_SCALE) * freqs
        emb = torch.cat([ang.cos(), ang.sin()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN. Compute ``x * (1 + scale) + shift``."""
    return torch.addcmul(shift, x, 1.0 + scale)


# ---- from attention.py ----
class SelfAttention(nn.Module):
    """SDPA self-attention over the action tokens, causally masked.

    Token ``k`` sees only tokens ``0..k``. This is a deliberate deviation from
    ACT and Diffusion Policy, which attend bidirectionally over the chunk. It
    buys prefix consistency: the first ``num_action_steps`` tokens a chunk will
    execute do not depend on the tail that will be discarded.
    """

    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        _check_attention_dims(hidden_dim, n_heads)
        head_dim = hidden_dim // n_heads
        if head_dim % 2 != 0:
            # RoPE rotates (even, odd) pairs. An odd head_dim leaves a trailing
            # element unpaired and the broadcast fails deep inside the forward.
            raise ValueError(f"head_dim={head_dim} must be even for RoPE")
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.register_buffer(
            "inv_freq",
            1.0 / (10000 ** (torch.arange(hidden_dim // (2 * n_heads)).float() / (hidden_dim // (2 * n_heads)))),
            persistent=False,
        )
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention over the ``L`` action tokens. The shape is preserved."""
        b, seq, _ = x.shape
        h = self.n_heads
        d = self.head_dim

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, seq, h, d).transpose(1, 2)
        k = k.view(b, seq, h, d).transpose(1, 2)
        v = v.view(b, seq, h, d).transpose(1, 2)

        # RoPE rotates the (even, odd) pairs of each head.
        pos = torch.arange(seq, device=x.device, dtype=torch.float32)
        freqs = torch.einsum("k,d->kd", pos, self.inv_freq)
        cos, sin = freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]
        q1, q2 = q[..., 0::2], q[..., 1::2]
        k1, k2 = k[..., 0::2], k[..., 1::2]
        q = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
        k = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(b, seq, h * d)
        return self.out(y)




class CrossAttention(nn.Module):
    """SDPA where the query and the key/value come from different streams.

    There is no causal mask. The critic reads a chunk it has already seen, so
    no prefix rule applies. The sequence lengths of the two sides may differ.
    No RoPE: token order already lives in the frame and step features.
    """

    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        _check_attention_dims(hidden_dim, n_heads)
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.kv_proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, query: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """``query`` is ``[B, Lq, D]``, ``context`` is ``[B, Lk, D]``. Returns
        ``[B, Lq, D]``.
        """
        if query.dim() != 3 or context.dim() != 3:
            raise ValueError(
                f"query and context must be 3D, got {tuple(query.shape)} and "
                f"{tuple(context.shape)}"
            )
        if query.shape[0] != context.shape[0]:
            raise ValueError(
                f"query batch {query.shape[0]} != context batch {context.shape[0]}"
            )
        if query.shape[2] != context.shape[2]:
            raise ValueError(
                f"query width {query.shape[2]} != context width {context.shape[2]}"
            )
        b, lq, _ = query.shape
        lk = context.shape[1]
        h, d = self.n_heads, self.head_dim
        q = self.q_proj(query).view(b, lq, h, d).transpose(1, 2)
        k, v = self.kv_proj(context).chunk(2, dim=-1)
        k = k.view(b, lk, h, d).transpose(1, 2)
        v = v.view(b, lk, h, d).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.out(y.transpose(1, 2).reshape(b, lq, h * d))
class AdaLNBlock(nn.Module):
    """Pre-norm adaLN-Zero block: self-attention and one MLP, conditioned on ``c``.

    ``DiTBlock`` and ``ResMLPBlock`` are this block with a different MLP. The
    block lives here once, so a fix to the adaLN path reaches both actors.
    """

    def __init__(self, hidden_dim: int, n_heads: int, mlp: nn.Module):
        super().__init__()
        if hidden_dim < MIN_WIDTH:
            raise ValueError("hidden_dim must be at least 2")
        self.norm1 = make_rms_norm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, n_heads)
        self.norm2 = make_rms_norm(hidden_dim)
        self.mlp = mlp
        self.adaLN = CondMLP(hidden_dim, hidden_dim, 6 * hidden_dim)
        nn.init.constant_(self.adaLN.fc2.weight, 0.0)
        nn.init.constant_(self.adaLN.fc2.bias, 0.0)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        """``[B, K, H]`` action tokens, with adaLN conditioning on ``c``. ``c`` is
        ``[B, H]`` for a whole chunk or ``[B, K, H]`` for per-token time.
        """
        out, _ = self.forward_parts(x, c)
        return out

    def forward_with_stats(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The same block, plus the adaLN gate and scale diagnostics."""
        out, (gate_msa, gate_mlp, scale_msa, scale_mlp) = self.forward_parts(x, c)
        with torch.inference_mode():
            stats = {
                "gate_msa": rms(gate_msa),
                "gate_mlp": rms(gate_mlp),
                "scale_msa": rms(1.0 + scale_msa),
                "scale_mlp": rms(1.0 + scale_mlp),
            }
        return out, stats

    def forward_parts(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """The block itself, with the adaLN parts the diagnostics read."""
        if x.dim() != 3:
            raise ValueError(f"x must have shape [B, K, H], got {tuple(x.shape)}")
        b = x.shape[0]
        if c.shape[0] != b:
            raise ValueError(f"c has batch {c.shape[0]} but x has {b}")
        token_axis = c.dim() == 3
        if token_axis:
            k = x.shape[1]
            if c.shape[1] != k:
                raise ValueError(f"c has {c.shape[1]} conditioning tokens but x has {k}")
        else:
            if c.dim() != 2 or c.shape[1] != self.adaLN.fc1.in_features:
                raise ValueError(f"c must have shape [{b}, {self.adaLN.fc1.in_features}], got {tuple(c.shape)}")
            c = c[:, None, :]

        parts = self.adaLN(c).view(b, c.shape[1], 6, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = parts.unbind(
            dim=2
        )
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(h)
        out = x + gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return out, (gate_msa, gate_mlp, scale_msa, scale_mlp)


class DiTBlock(AdaLNBlock):
    """The DiT block: attention, one MishFFN, adaLN."""

    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float):
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0 or int(hidden_dim * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must produce a positive hidden width")
        super().__init__(
            hidden_dim,
            n_heads,
            MishFFN(hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim),
        )


# ---- networks.py ----
# ACTOR

def effective_rank(features: torch.Tensor) -> torch.Tensor:
    """``exp`` spectral entropy of ``[n, hidden]``. The collapse probe."""
    matrix = features.contiguous().reshape(-1, features.shape[-1]).float()
    sigma = torch.linalg.svdvals(matrix)
    p = sigma / sigma.sum().clamp_min(1e-12)
    # Drop zero components so the log stays finite.
    p = p[p > 0]
    if p.numel() == 0:
        # exp(0) = 1 for an all-zero spectrum.
        return torch.ones((), device=features.device, dtype=torch.float32)
    entropy = -(p * torch.log(p)).sum()
    return torch.exp(entropy)


class FlowActor(nn.Module):
    """The stem both actors share: embed, project, condition, decode."""

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
        time_embed_dim: int,
        num_blocks: int,
        block_cls: type[nn.Module],
        block_kwargs: dict,
    ):
        super().__init__()
        if action_dim < 1 or hidden_dim < MIN_WIDTH:
            raise ValueError("action_dim must be positive and hidden_dim at least 2")
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if time_embed_dim < MIN_WIDTH:
            raise ValueError(f"time_embed_dim={time_embed_dim} must be at least 2")
        if num_blocks < 1:
            raise ValueError(f"an actor needs >= 1 block, got {num_blocks}")
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_horizon = action_horizon
        self.time_embed_dim = time_embed_dim
        self.num_blocks = num_blocks
        self.timestep_embedder = SinusoidalEncoding(self.time_embed_dim)
        self.in_proj = nn.Linear(action_dim, hidden_dim)
        self.time_mlp = CondMLP(time_embed_dim, 4 * hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            block_cls(**block_kwargs) for _ in range(self.num_blocks)
        )
        self.norm_out = make_rms_norm(hidden_dim)
        self.adaLN_out = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.constant_(self.adaLN_out.weight, 0.0)
        nn.init.constant_(self.adaLN_out.bias, 0.0)
        self.out = nn.Linear(hidden_dim, action_dim)
        nn.init.constant_(self.out.weight, 0.0)
        nn.init.constant_(self.out.bias, 0.0)

    @torch.inference_mode()
    def weight_magnitude(self) -> torch.Tensor:
        """Mean absolute weight. The correlate of Dohare. One scalar per module."""
        total = None
        count = 0
        for param in self.parameters():
            s = param.detach().abs().sum(dtype=torch.float32)
            total = s if total is None else total + s
            count += param.numel()
        return total / max(count, 1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | float,
        condition: torch.Tensor,
        *,
        collect_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict the velocity. ``collect_stats=True`` returns ``(velocity, stats)``."""
        if x_t.dim() != 3 or x_t.shape[2] != self.action_dim:
            raise ValueError(f"x_t must have shape [B, K, {self.action_dim}], got {tuple(x_t.shape)}")
        if condition.dim() != 2 or condition.shape[0] != x_t.shape[0] or condition.shape[1] != self.hidden_dim:
            raise ValueError(f"condition must have shape [{x_t.shape[0]}, {self.hidden_dim}], got {tuple(condition.shape)}")
        if x_t.shape[1] > self.action_horizon:
            raise ValueError(f"action token count {x_t.shape[1]} exceeds action_horizon {self.action_horizon}")

        t = torch.as_tensor(t, dtype=torch.float32, device=x_t.device)
        cond = condition.to(device=x_t.device, dtype=x_t.dtype)
        token_axis = t.dim() in (2, 3) and tuple(t.shape[:2]) == tuple(x_t.shape[:2])
        if token_axis:
            if t.dim() == 3:
                if t.shape[-1] != 1:
                    raise ValueError(f"t must end in a singleton action-dimension axis, got {tuple(t.shape)}")
                t = t.squeeze(-1)
            temb = self.timestep_embedder.encode_times(t.reshape(-1))
            temb = temb.reshape(x_t.shape[0], x_t.shape[1], -1).to(
                device=x_t.device, dtype=x_t.dtype
            )
            c = self.time_mlp(temb) + cond[:, None, :]
        elif t.dim() <= 1:
            t = t.reshape(-1)
            if t.numel() not in (1, x_t.shape[0]):
                raise ValueError(f"t must contain 1 or {x_t.shape[0]} values, got {t.numel()}")
            temb = self.timestep_embedder.encode_times(t).to(
                device=x_t.device, dtype=x_t.dtype
            )
            c = self.time_mlp(temb) + cond
        else:
            raise ValueError(f"t must be scalar, [B], or [B, K, 1], got {tuple(t.shape)}")

        h = self.in_proj(x_t)

        def decode(hidden: torch.Tensor) -> torch.Tensor:
            """Modulate and project to the action dimension."""
            shift, scale = self.adaLN_out(c).chunk(2, dim=-1)
            if not token_axis:
                shift, scale = shift[:, None, :], scale[:, None, :]
            return self.out(modulate(self.norm_out(hidden), shift, scale))

        if not collect_stats:
            for block in self.blocks:
                h = block(h, c)
            return decode(h)

        block_stats: list[dict[str, torch.Tensor]] = []
        for block in self.blocks:
            h, s = block.forward_with_stats(h, c)
            block_stats.append(s)
        stats = self.collect_stats(temb, condition, c, h, block_stats)
        return decode(h), stats

    @torch.compiler.disable
    def collect_stats(
        self,
        temb: torch.Tensor,
        condition: torch.Tensor,
        c: torch.Tensor,
        h: torch.Tensor,
        block_stats: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Diagnostics, deliberately outside any compiled graph."""
        with torch.inference_mode():
            return {
                "temb": rms(temb),
                "fused": rms(condition),
                "c": rms(c),
                **{
                    k: torch.stack([s[k] for s in block_stats]).mean()
                    for k in ("gate_msa", "gate_mlp", "scale_msa", "scale_mlp")
                },
                "effective_rank": effective_rank(h),
                "weight_magnitude": self.weight_magnitude(),
            }


class FlowModel(FlowActor):
    """The DiT actor. One block is self-attention, one MishFFN, and adaLN."""

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
        dit_depth: int,
        dit_heads: int,
        dit_mlp_ratio: float,
        time_embed_dim: int,
    ):
        if dit_depth < 1 or dit_heads < 1:
            raise ValueError(
                f"dit_depth={dit_depth} and dit_heads={dit_heads} must be >= 1"
            )
        if (
            not math.isfinite(dit_mlp_ratio)
            or dit_mlp_ratio <= 0
            or int(hidden_dim * dit_mlp_ratio) < 1
        ):
            raise ValueError("dit_mlp_ratio must produce a positive hidden width")
        super().__init__(
            action_dim,
            action_horizon,
            hidden_dim,
            time_embed_dim,
            dit_depth,
            DiTBlock,
            {"hidden_dim": hidden_dim, "n_heads": dit_heads,
             "mlp_ratio": dit_mlp_ratio},
        )


class ResidualMLP(nn.Module):
    """The residual unit of Wang et al. (2025), arXiv:2503.14858, Section 3."""

    def __init__(self, hidden_dim: int, hidden_mult: float, num_units: int):
        super().__init__()
        if hidden_dim < MIN_WIDTH:
            raise ValueError("hidden_dim must be at least 2")
        if num_units < 1:
            raise ValueError(f"num_units={num_units} must be >= 1")
        inner = int(hidden_dim * hidden_mult)
        if inner < 1:
            raise ValueError("resmlp_hidden_mult must produce a positive width")
        self.denses = nn.ModuleList(
            nn.Linear(hidden_dim if i == 0 else inner, inner)
            for i in range(num_units)
        )
        self.norms = nn.ModuleList(make_rms_norm(inner) for _ in range(num_units))
        self.out_norm = make_rms_norm(inner)
        self.out = nn.Linear(inner, hidden_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x + residual(x)``. Zero at init, so the block starts as identity."""
        h = x
        for dense, norm in zip(self.denses, self.norms, strict=True):
            h = norm(dense(h))
            h = F.silu(h)
        return x + self.out(self.out_norm(h))


class ResMLPBlock(AdaLNBlock):
    """The ``resmlp`` block: attention, then the paper's residual MLP."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        hidden_mult: float,
        num_units: int,
    ):
        super().__init__(
            hidden_dim,
            n_heads,
            ResidualMLP(hidden_dim, hidden_mult, num_units),
        )


class FlowResMLP(FlowActor):
    """The ``resmlp`` actor. DiT's attention and adaLN, the paper's MLP unit."""

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
        resmlp_blocks: int,
        resmlp_units: int,
        resmlp_hidden_mult: float,
        n_heads: int,
        time_embed_dim: int,
    ):
        if resmlp_blocks < 1:
            raise ValueError(f"resmlp_blocks={resmlp_blocks} must be >= 1")
        if resmlp_units < 1:
            raise ValueError(f"resmlp_units={resmlp_units} must be >= 1")
        if n_heads < 1:
            raise ValueError(f"n_heads={n_heads} must be >= 1")
        super().__init__(
            action_dim,
            action_horizon,
            hidden_dim,
            time_embed_dim,
            resmlp_blocks,
            ResMLPBlock,
            {"hidden_dim": hidden_dim, "n_heads": n_heads,
             "hidden_mult": resmlp_hidden_mult, "num_units": resmlp_units},
        )

# CRITIC


class ActionAffine(nn.Module):
    """Per-element affine. No mean subtraction, so chunk magnitude survives.
    Keeps the LayerNorm state-dict keys for old checkpoints."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the learned per-element scale and shift."""
        return x * self.weight + self.bias


class StateAffine(nn.Module):
    """Window RMS scale plus a per-element affine, with no mean subtraction.
    The obs is already z-scored, so this adds one window-level scale.
    Keeps the LayerNorm state-dict keys for old checkpoints."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RMS scale, then the per-element affine."""
        scale = torch.mean(x.float() * x.float(), dim=-1, keepdim=True)
        scale = torch.rsqrt(scale + self.eps).to(x.dtype)
        return x * scale * self.weight + self.bias


def make_mlp_layers(
    in_dim: int, hidden_dim: int, num_layers: int, dropout: float
) -> nn.Module:
    """``Linear -> LayerNorm -> Mish`` per layer, with optional dropout on the
    first. One final linear maps to the scalar."""
    layers: list[nn.Module] = []
    for i in range(num_layers):
        layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
        if i == 0 and dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.LayerNorm(hidden_dim), nn.Mish()]
    layers.append(nn.Linear(hidden_dim, 1))
    return nn.Sequential(*layers)

class CriticMember(nn.Module):
    """One Q(s, chunk) head. The state window is one token per frame and the
    action chunk one token per step. Each stream cross-attends to the other,
    then the last state token and the mean action token enter the trunk.
    The chunk self-attention is causal and carries RoPE."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_obs_steps: int,
        action_horizon: int,
        *,
        condition_on_exec_length: bool = True,
        num_layers: int = 2,
        condition_dim: int | None = None,
        dropout: float = 0.0,
        state_hidden: int | None = None,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_obs_steps = num_obs_steps
        self.action_horizon = action_horizon
        self.condition_on_exec_length = condition_on_exec_length
        self.condition_dim = condition_dim
        state_dim = (
            condition_dim
            if condition_dim is not None
            else (state_hidden if state_hidden is not None else hidden_dim)
        )
        # validate_critic requires width % 8, so four even heads fit.
        attn_heads = 4
        self.attn_heads = attn_heads

        if condition_dim is not None and num_obs_steps != 1:
            raise ValueError(
                f"condition_dim is set, so num_obs_steps must be 1 (unused), got "
                f"{num_obs_steps}"
            )
        # Pool is 2 * state_dim. The first MLP layer resizes to hidden_dim.
        self.input_dim = 2 * state_dim + (
            1 if condition_on_exec_length else 0
        )
        # Separate norms: one over the concat would couple unrelated quantities.
        # The state norm keeps the mean.
        self.norm_state = (
            StateAffine(num_obs_steps * obs_dim) if condition_dim is None else None
        )
        # Reuse the actor's window trunk. The old flat first layer was 5.1M of
        # the 6.4M critic on Humanoid.
        self.state_encoder = (
            StateEncoder(obs_dim, state_dim, num_obs_steps)
            if condition_dim is None
            else None
        )
        # An affine, not LayerNorm: LayerNorm subtracts the chunk mean, so all
        # +5 and all 0 would look the same.
        self.norm_action = ActionAffine(action_horizon * action_dim)
        # One token per action step, projected to the stream width.
        self.act_in = nn.Linear(action_dim, state_dim)
        # The chunk is pooled by its mean, so a causal attention pass restores
        # step order. RoPE carries the position.
        self.act_attn = SelfAttention(state_dim, attn_heads)
        self.act_attn_norm = make_rms_norm(state_dim)
        # Both passes read the raw streams; neither sees the other's output.
        self.obs_to_act = CrossAttention(state_dim, attn_heads)
        self.act_to_obs = CrossAttention(state_dim, attn_heads)
        self.cross_norm_obs = make_rms_norm(state_dim)
        self.cross_norm_act = make_rms_norm(state_dim)
        self.mlp = make_mlp_layers(self.input_dim, hidden_dim, num_layers, dropout)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        exec_length_frac: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Accept ``[B, T, obs]`` or ``[B, condition_dim]``, plus ``[B, H,
        action_dim]`` and the scalar exec length. Returns ``[B]``.
        """
        if self.condition_dim is None:
            window = self.norm_state(states.flatten(1)).view(states.shape)
            # Run the output norm on every token so its parameters see a gradient.
            obs_tokens = self.state_encoder.out_norm(
                self.state_encoder.encode_tokens(window)
            )
        else:
            obs_tokens = states.unsqueeze(1)
        chunk_flat = self.norm_action(actions.flatten(1))
        act_tokens = self.act_in(chunk_flat.view(actions.shape))
        act_tokens = act_tokens + self.act_attn(
            self.act_attn_norm(act_tokens)
        )
        obs = obs_tokens + self.obs_to_act(
            self.cross_norm_obs(obs_tokens), act_tokens
        )
        act = act_tokens + self.act_to_obs(
            self.cross_norm_act(act_tokens), obs_tokens
        )
        parts = [obs[:, -1], act.mean(dim=1)]
        if self.condition_on_exec_length:
            if exec_length_frac is None:
                raise ValueError(
                    "exec_length_frac is required when condition_on_exec_length=True; it "
                    "tells the critic how much of the planned chunk was executed"
                )
            if exec_length_frac.dim() != 1 or exec_length_frac.shape[0] != states.shape[0]:
                raise ValueError(
                    f"exec_length_frac must be [B] matching states, got {tuple(exec_length_frac.shape)}"
                )
            parts.append(exec_length_frac.to(dtype=parts[0].dtype).unsqueeze(-1))
        return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)


class Critic(nn.Module):
    """Q(s_t, a_{t:t+h}) over the full window. ``exec_length_frac`` says how
    much of the chunk ran."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        num_obs_steps: int,
        action_horizon: int,
        num_critics: int,
        *,
        condition_on_exec_length: bool = True,
        num_layers: int = 2,
        condition_dim: int | None = None,
        dropout: float = 0.0,
        state_hidden: int | None = None,
    ):
        super().__init__()
        if condition_dim is not None and condition_dim < 1:
            raise ValueError("condition_dim must be >= 1 when set")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_obs_steps = num_obs_steps
        self.action_horizon = action_horizon
        self.num_critics = num_critics
        self.condition_on_exec_length = condition_on_exec_length
        self.condition_dim = condition_dim
        self.dropout = dropout

        self.networks = nn.ModuleList(
            [
                CriticMember(
                    obs_dim,
                    action_dim,
                    hidden_dim,
                    1 if condition_dim is not None else num_obs_steps,
                    action_horizon,
                    condition_on_exec_length=condition_on_exec_length,
                    num_layers=num_layers,
                    condition_dim=condition_dim,
                    dropout=dropout,
                    state_hidden=state_hidden,
                )
                for _ in range(num_critics)
            ]
        )

    @property
    def input_dim(self) -> int:
        """First-layer input width. Shifts by 1 with the exec-length input."""
        return self.networks[0].input_dim

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        exec_length_frac: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Accept ``[B, T, obs]`` or ``[B, condition_dim]``, plus ``[B, H,
        action_dim]`` and the scalar exec length. Returns ``[B]``.
        """
        if self.condition_dim is not None:
            if states.dim() != 2 or states.shape[1] != self.condition_dim:
                raise ValueError(
                    f"condition must have shape [B, {self.condition_dim}], got "
                    f"{tuple(states.shape)}"
                )
        else:
            if states.dim() != 3 or states.shape[2] != self.obs_dim:
                raise ValueError(f"states must have shape [B, {self.num_obs_steps}, {self.obs_dim}], got {tuple(states.shape)}")
            if states.shape[1] != self.num_obs_steps:
                raise ValueError(f"states has {states.shape[1]} time steps but the critic was built for num_obs_steps={self.num_obs_steps}")
        if actions.dim() != 3:
            raise ValueError(f"actions must have shape [B, H, action_dim], got {tuple(actions.shape)}")
        if actions.shape[2] != self.action_dim:
            raise ValueError(f"actions feature dimension must be {self.action_dim}, got {actions.shape[2]}")
        if actions.shape[0] != states.shape[0]:
            raise ValueError(f"actions batch {actions.shape[0]} != states batch {states.shape[0]}")
        if actions.shape[1] != self.action_horizon:
            raise ValueError(f"actions has {actions.shape[1]} steps but the critic was built for action_horizon={self.action_horizon}")
        return torch.stack(
            [net(states, actions, exec_length_frac) for net in self.networks]
        )


# ---- StateEncoder (from processor.py) ----
class StateEncoder(nn.Module):
    """Project frame, causal attention over window, read last, RMSNorm.
    Shared projection (not flat dense); RMS (not LayerNorm) to keep mean."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        max_obs_steps: int,
        n_heads: int = 2,
    ):
        super().__init__()
        if hidden_dim < MIN_WIDTH:
            raise ValueError("hidden_dim must be at least 2")
        if max_obs_steps < 1:
            raise ValueError(f"max_obs_steps must be >= 1, got {max_obs_steps}")
        self.obs_dim = obs_dim
        self.max_obs_steps = max_obs_steps
        self.n_heads = n_heads
        self.frame = CondMLP(obs_dim, hidden_dim, hidden_dim)
        self.attn_norm = make_rms_norm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, n_heads)
        self.out_norm = make_rms_norm(hidden_dim)

    def encode_tokens(self, states: torch.Tensor) -> torch.Tensor:
        """Return the attended window ``[B, T, hidden]``, before the pool.

        The critic's cross-attention reads one token per frame, so it needs the
        sequence. ``forward`` pools this sequence to the last frame.
        """
        if states.dim() != 3 or states.shape[2] != self.obs_dim:
            raise ValueError(f"states must have shape [B, {self.max_obs_steps}, {self.obs_dim}], got {tuple(states.shape)}")
        if states.shape[1] != self.max_obs_steps:
            raise ValueError(f"states has {states.shape[1]} frames, built for max_obs_steps={self.max_obs_steps}")
        h = self.frame(states)
        return h + self.attn(self.attn_norm(h))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Accept the window ``[B, T, obs]``. Return one normalized conditioning
        vector of width ``hidden_dim``.
        """
        return self.out_norm(self.encode_tokens(states)[:, -1])
