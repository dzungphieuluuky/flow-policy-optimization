"""Generic building blocks shared by the network modules."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

class _RMSNorm(nn.RMSNorm):
    """RMSNorm. The *compute* runs in fp32. The stored weight follows the model
    dtype.

    A ``.to(bfloat16)`` call on the owner downcasts the weight on purpose. The
    fp32 cast in the forward pass is a numerical guard. It is not a precision
    policy.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise the trailing dimensions with the learned scale. Reduce in
        fp32. Cast the result back.
        """
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
    """Compute down(silu(gate(x)) * up(x)).

    The gate and the up projection share one GEMM. The torch weight has the
    shape ``[out, in]``, so the fused weight is
    ``cat([gate.weight, up.weight], dim=0)``.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        if in_dim < 1 or hidden_dim < 1 or out_dim < 1:
            raise ValueError("SwiGLU dimensions must be positive")
        self.gate_up = nn.Linear(in_dim, 2 * hidden_dim)
        self.down = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.gate_up(x).chunk(2, dim=-1)
        # The multiply writes into the output of silu instead of a second
        # tensor. The backward pass of silu reads its input, not its output, so
        # this is safe for autograd. It also drops one activation that has the
        # size of the output of gate_up.
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
    """Fixed sinusoidal codes. The module holds one freq table as a buffer.

    The table is built in ``__init__``. It is never built in ``forward``.

    A table built inside a ``torch.compile`` region belongs to the cudagraph
    pool of inductor. The next run overwrites it. Reading that same copy on a
    later call then dies with the message "accessing tensor output of
    CUDAGraphs that has been overwritten".

    The failure arrives under ``mode="reduce-overhead"``. That is the mode that
    :func:`train_sjfpo.build_policy` picks for sm_70 and above. The failure
    covers every forward pass after the first one. The Euler solver reuses one
    graph across its steps and hands that graph a fresh ``t`` each time.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim < 2:
            raise ValueError(f"SinusoidalEncoding dim must be >= 2, got {dim}")
        self.dim = dim
        self.half = dim // 2
        # FPO++ uses doubling powers (actor_critic.py:_embed_timestep). It does
        # not use the usual 1/10000^i sweep. The result is numerically exact.
        # The fp32 cos of torch does argument reduction. Against a float64
        # reference the error is zero at every width.
        #
        # A wide dim has a cost. The cost is bandwidth that the model cannot
        # use. At dim 64 the top channel runs about 40 cycles per ULP of t. FPO++
        # ships 8 for isaaclab and 32 for manipulation. The default here is
        # their larger value.
        #
        # The buffer is not persistent. The value comes from ``dim``. So it stays
        # out of every state_dict and an older checkpoint still loads. The width
        # ``half`` matches the ``[half]`` table of the old per-device cache.
        self.register_buffer(
            "freqs",
            torch.pow(
                torch.tensor(2.0), torch.arange(self.half, dtype=torch.float32)
            ),
            persistent=False,
        )

    def compute_freqs(self, device: torch.device | None = None) -> torch.Tensor:
        """Return the ``[half]`` freq table in fp32 on ``device``.

        Every call returns the same tensor. So no caller can keep a copy that a
        later run overwrites. A foreign ``device`` makes a copy. The function
        does not keep that copy.
        """
        if device is None or torch.device(device) == self.freqs.device:
            return self.freqs
        return self.freqs.to(torch.device(device))

    def encode_times(self, t: torch.Tensor) -> torch.Tensor:
        """Embed continuous times with a sinusoid. Accept ``[N]``, return
        ``[N, 2 * half]``.

        The function places cos before sin. This matches the ``_embed_timestep``
        of FPO++.

        The freq table already spans the range. So the function does not scale
        ``t`` separately.

        The table is a buffer. The function reads it and never writes it. So the
        function needs no clone. The cast to fp32 does nothing unless a caller
        downcast the whole module.
        """
        t = t.to(dtype=torch.float32)
        freqs = self.compute_freqs(t.device).to(torch.float32)
        ang = t.unsqueeze(-1) * freqs
        emb = torch.cat([ang.cos(), ang.sin()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN. Compute ``x * (1 + scale) + shift``.

    The call ``torch.addcmul(shift, x, 1 + scale)`` gives the same value in one
    kernel instead of three. The tensor is the widest in the DiT. The call runs
    once per block.

    The function reads the raw ``scale``. So a caller that bounds the gain must
    clamp before it calls.
    """
    return torch.addcmul(shift, x, 1.0 + scale)

