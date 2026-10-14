"""Generic building blocks shared by the network modules."""

from __future__ import annotations

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
        # mul_ into silu's output rather than a second tensor: silu's backward
        # reads its input, not its output, so this is autograd-safe and drops
        # one activation the size of gate_up's output.
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
    """Fixed sinusoidal codes. One freq table, held as a buffer.

    Built in ``__init__``, never in ``forward``. A table made inside a
    ``torch.compile`` region belongs to inductor's cudagraph pool, and the next
    run overwrites it, so reading that same copy back on a later call dies with
    "accessing tensor output of CUDAGraphs that has been overwritten". Under
    ``mode="reduce-overhead"`` -- what :func:`train_sjfpo.build_policy` picks for
    sm_70 and up -- that is every forward after the first, because the Euler
    solver reuses one graph across its steps and hands it a fresh ``t`` each
    time.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim < 2:
            raise ValueError(f"SinusoidalEncoding dim must be >= 2, got {dim}")
        self.dim = dim
        self.half = dim // 2
        # FPO++ (actor_critic.py:_embed_timestep): doubling powers, not the
        # usual 1/10000^i sweep. Numerically exact -- torch's fp32 cos does
        # argument reduction, and against a float64 reference the error is zero
        # at every width. What a wide dim costs is bandwidth the model cannot
        # use: at 64 the top channel runs ~40 cycles per ULP of t. FPO++ ships
        # 8 (isaaclab) and 32 (manipulation); our default is their larger one.
        #
        # Non-persistent: derived from ``dim``, so it stays out of every
        # state_dict and an older checkpoint still loads. ``half`` matches the
        # ``[half]`` table the old per-device cache used to build.
        self.register_buffer(
            "freqs",
            torch.pow(
                torch.tensor(2.0), torch.arange(self.half, dtype=torch.float32)
            ),
            persistent=False,
        )

    def compute_freqs(self, device: torch.device | None = None) -> torch.Tensor:
        """``[half]`` freq table, fp32, on ``device``.

        Same tensor every call, so no caller can hold a copy that a later run
        overwrites. A foreign ``device`` copies, and that copy is not kept.
        """
        if device is None or torch.device(device) == self.freqs.device:
            return self.freqs
        return self.freqs.to(torch.device(device))

    def encode_times(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal embed of continuous times. ``[N]`` -> ``[N, 2 * half]``.

        cos before sin, matching FPO++'s ``_embed_timestep``. The frequencies
        already span the range, so ``t`` is not scaled separately. The freq
        table is a buffer: read, never written, so it needs no clone. The cast
        to fp32 is a no-op unless someone downcast the whole module.
        """
        t = t.to(dtype=torch.float32)
        freqs = self.compute_freqs(t.device).to(torch.float32)
        ang = t.unsqueeze(-1) * freqs
        emb = torch.cat([ang.cos(), ang.sin()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN: ``x * (1 + scale) + shift``. ``torch.addcmul(shift, x, 1 + scale)``
    is the same value in one kernel instead of three, on the widest tensor in
    the DiT, once per block. Reads the raw ``scale``, so callers that bound the
    gain must clamp before calling."""
    return torch.addcmul(shift, x, 1.0 + scale)

