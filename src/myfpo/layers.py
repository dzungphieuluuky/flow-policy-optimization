"""Generic building blocks shared by the network modules."""

from __future__ import annotations
import contextlib
import math
import os
from functools import lru_cache
import numpy as np
import torch
from lerobot.processor.converters import to_tensor
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
    return torch.linalg.vector_norm(x) / math.sqrt(x.numel())


def cudagraph_safe(x: torch.Tensor, enabled: bool = True) -> torch.Tensor:
    """A copy of a tensor that a cudagraph may still own.

    A module compiled with ``mode="reduce-overhead"`` runs on a cudagraph, and
    the tensor it returns is memory the cudagraph pool owns: the next
    invocation of that module can overwrite it, and the pool can retire the
    tensor while another compiled region is still holding it. Reading it then
    raises "accessing tensor output of CUDAGraphs that has been overwritten by
    a subsequent run".

    The copy leaves the pool, so the consumer of the tensor reads a stable
    value. Call it on the way out of a compiled module, before the tensor
    crosses into another one. With ``enabled`` false nothing was compiled, so
    the input comes back unchanged.
    """
    return x.clone() if enabled else x


def cudagraph_step_begin() -> None:
    """Tell the cudagraph manager that a new step starts here.

    A module compiled with ``mode="reduce-overhead"`` runs on a cudagraph, and
    the manager holds the memory of the finished step until it is told that
    the step is over. Marking the step lets it recycle that memory, and it
    keeps the manager from warning that it cannot hit the fast path.

    It is a no-op when nothing is compiled, so it can be called
    unconditionally.
    """
    torch.compiler.cudagraph_mark_step_begin()


def to_device_async(x: torch.Tensor, device) -> torch.Tensor:
    """Move a tensor to the device. On CUDA the copy overlaps CPU work."""
    device = torch.device(device)
    if device.type == "cuda" and x.device != device:
        if not x.is_pinned():
            x = x.pin_memory()
        return x.to(device, non_blocking=True)
    return x.to(device)


def to_tensor_1d(
    x: torch.Tensor | np.ndarray | float,
    dtype: torch.dtype | None = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """To a tensor, always at least 1-D. A singleton stays ``[1]``, never ``()``."""
    return torch.atleast_1d(to_tensor(x, dtype=dtype, device=device))


# A width of 1 gives the norm nothing to normalise, so every hidden width in
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


def build_rms_norm(hidden_dim: int, eps: float = 1e-6) -> nn.RMSNorm:
    return RMSNorm(hidden_dim, eps=eps)


class SwiGLU(nn.Module):
    """Compute down(silu(gate(x)) * up(x))."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        if in_dim < 1 or hidden_dim < 1 or out_dim < 1:
            raise ValueError("SwiGLU dimensions must be positive")
        self.gate_up = nn.Linear(in_dim, 2 * hidden_dim)
        self.down = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g).mul_(u))


class CondMLP(nn.Module):
    """SiLU MLP for conditioning regression."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        if in_dim < 1 or hidden_dim < 1 or out_dim < 1:
            raise ValueError("CondMLP dimensions must be positive")
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

class TemporalConv(nn.Module):
    """A learned local filter along a time axis. It computes no statistics.

    The input is ``[B, C, T]`` and the output is ``[B, C, T]``. The width
    never changes, so every shape after the flatten stays the same.

    ``causal=True`` pads on the left only, so the output at position i reads
    positions <= i and never the future.
    """

    def __init__(self, channels: int, kernel_size: int, causal: bool = False):
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd number, got {kernel_size}"
            )
        self.causal = causal
        self.left_pad = kernel_size - 1 if causal else 0
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=0 if causal else kernel_size // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, C, T]`` in, ``[B, C, T]`` out."""
        if self.causal:
            x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


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
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if n_heads < 1:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must split over {n_heads} heads")
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


class AdaLNBlock(nn.Module):
    """Pre-norm adaLN-Zero block: self-attention and one MLP, conditioned on ``c``.

    ``DiTBlock`` and ``ResMLPBlock`` are this block with a different MLP. The
    block lives here once, so a fix to the adaLN path reaches both actors.
    """

    def __init__(self, hidden_dim: int, n_heads: int, mlp: nn.Module):
        super().__init__()
        if hidden_dim < MIN_WIDTH:
            raise ValueError("hidden_dim must be at least 2")
        self.norm1 = build_rms_norm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, n_heads)
        self.norm2 = build_rms_norm(hidden_dim)
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
    """The DiT block: attention, one SwiGLU, adaLN."""

    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float):
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0 or int(hidden_dim * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must produce a positive hidden width")
        super().__init__(
            hidden_dim,
            n_heads,
            SwiGLU(hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim),
        )
