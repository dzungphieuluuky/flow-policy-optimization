"""Generic building blocks shared by the network modules."""

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
    buys prefix consistency: the first ``n_action_steps`` tokens a chunk will
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
