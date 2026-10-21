"""Generic building blocks shared by the network modules."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

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
        ang = t.unsqueeze(-1) * freqs
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

