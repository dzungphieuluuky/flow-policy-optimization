"""Generic building blocks shared by the network modules."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

class _RMSNorm(nn.RMSNorm):
    """RMSNorm. The compute runs in fp32. The stored weight follows the model"""

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
    return _RMSNorm(hidden_dim, eps=eps)


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
        if dim < 2:
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

    def compute_freqs(self, device: torch.device | None = None) -> torch.Tensor:
        """Return the ``[half]`` freq table in fp32 on ``device``."""
        if device is None or torch.device(device) == self.freqs.device:
            return self.freqs
        return self.freqs.to(torch.device(device))

    def encode_times(self, t: torch.Tensor) -> torch.Tensor:
        """Embed continuous times with a sinusoid. Accept ``[N]``, and return"""
        t = t.to(dtype=torch.float32)
        freqs = self.compute_freqs(t.device).to(torch.float32)
        ang = t.unsqueeze(-1) * freqs
        emb = torch.cat([ang.cos(), ang.sin()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN. Compute ``x * (1 + scale) + shift``."""
    return torch.addcmul(shift, x, 1.0 + scale)

