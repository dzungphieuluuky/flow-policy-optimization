"""Generic building blocks shared by the network modules."""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

class _RMSNorm(nn.RMSNorm):
    """RMSNorm whose *compute* runs in fp32; the stored weight follows the model
    dtype. A ``.to(bfloat16)`` on the owner intentionally downcasts the weight;
    the forward-time fp32 cast is a numerical guard, not a precision policy."""

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__(hidden_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Norm trailing dims with the learned scale, reduce in fp32, cast back."""
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
    """down(silu(gate(x)) * up(x)). Gate+up fused into one GEMM; torch weight is
    ``[out, in]`` so fused = ``cat([gate.weight, up.weight], dim=0)``."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        if in_dim < 1 or hidden_dim < 1 or out_dim < 1:
            raise ValueError("SwiGLU dimensions must be positive")
        self.gate_up = nn.Linear(in_dim, 2 * hidden_dim)
        self.down = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


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
    """Fixed sinusoidal codes. Freq tables cached per device."""

    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        if dim < 2:
            raise ValueError(f"SinusoidalEncoding dim must be >= 2, got {dim}")
        self.dim = dim
        self.scale = 1.0 if scale is None else float(scale)
        self.half = dim // 2
        self._freq_cache: OrderedDict[torch.device, torch.Tensor] = OrderedDict()

    def compute_freqs(self, device: torch.device) -> torch.Tensor:
        """Cached ``[half]`` freq table for a device.

        Holds 4 devices max, drops the LRU past that.
        """
        device = torch.device(device)
        cached = self._freq_cache.get(device)
        if cached is not None:
            self._freq_cache.move_to_end(device)
            return cached
        i = torch.arange(self.half, device=device, dtype=torch.float32)
        cached = torch.exp(-math.log(10000.0) * i / max(self.half, 1))
        self._freq_cache[device] = cached
        if len(self._freq_cache) > 4:
            self._freq_cache.popitem(last=False)
        return cached

    def encode_times(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal embed of continuous times. ``[N]`` -> ``[N, 2 * half]``."""
        t = t.to(dtype=torch.float32) * self.scale
        ang = t.unsqueeze(-1) * self.compute_freqs(t.device)
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN: ``x * (1 + scale) + shift``. ``torch.addcmul(shift, x, 1 + scale)``
    is the same value in one kernel instead of three, on the widest tensor in
    the DiT, twelve times per microbatch."""
    return torch.addcmul(shift, x, 1.0 + scale)

