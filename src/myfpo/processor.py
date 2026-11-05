"""myfpo pre/post helpers. The pair normalizes the obs and squashes the policy
output into the env's action bounds.
"""

from __future__ import annotations
import warnings
import torch
from torch import nn
from myfpo.layers import MIN_WIDTH, CondMLP, SelfAttention, make_rms_norm

# ---- from encoder.py ----
class RunningMeanStd(nn.Module):
    """Per-dimension running mean and variance over raw obs frames."""

    def __init__(self, dim: int, eps: float = 1e-8, clip: float = 10.0):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim={dim} must be positive")
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))
        self.uninitialized = True
        self.passthrough_warned = False
        self.eps = eps
        self.clip = clip

    @torch.inference_mode()
    def update(self, x: torch.Tensor) -> None:
        """Fold raw frames into the running statistics."""
        if x.shape[-1] != self.mean.numel():
            raise ValueError(f"update got last dim {x.shape[-1]}, expected {self.mean.numel()}")
        batch = (
            x.detach()
            .reshape(-1, x.shape[-1])
            .to(self.mean.device, dtype=torch.float32)
        )
        b = batch.shape[0]
        if b == 0:
            return
        b_var, b_mean = torch.var_mean(batch, dim=0, unbiased=False)
        delta = b_mean - self.mean
        total = self.count + b
        self.mean.add_((delta * b / total).to(self.mean.dtype))
        self.var.copy_(
            (
                self.count * self.var
                + b * b_var
                + delta.square() * (self.count * b / total)
            )
            / total
        )
        self.count.copy_(total)
        self.uninitialized = False

    def _load_from_state_dict(self, *args, **kwargs):
        """Load the state. Derive the latch again."""
        super()._load_from_state_dict(*args, **kwargs)
        self.uninitialized = bool(self.count.item() == 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise with the frozen statistics."""
        if self.uninitialized:
            if not self.passthrough_warned:
                warnings.warn(
                    f"{type(self).__name__}: forward before the first update(); "
                    "returning raw input", stacklevel=2,
                )
                self.passthrough_warned = True
            return x
        norm = (x - self.mean) * torch.rsqrt(self.var + self.eps)
        return norm.clamp(-self.clip, self.clip).to(x.dtype)

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


def env_squash_terms(
    low: torch.Tensor, high: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The scale and bias of the env's action bounds.

    ``scale = (high - low) / 2`` and ``bias = low + scale``. The bias is
    the low end plus the half width, so the two terms come from one
    difference and cannot drift apart.
    """
    scale = 0.5 * (high - low)
    return scale, low + scale


def squash_to_env(
    raw: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Map a raw policy output into the env's action bounds.

    ``bias + scale * tanh(raw)``. The ``tanh`` is the only thing that bounds
    the action, and it is applied exactly once, on the way to the env. The
    flow regresses on z-scored targets (the default); it never trains on the
    env-facing squash of its own output.

    ``scale`` and ``bias`` are the affine terms of the env's bounds, from
    :func:`env_squash_terms`.
    """
    scale = scale.to(device=raw.device, dtype=raw.dtype)
    bias = bias.to(device=raw.device, dtype=raw.dtype)
    return torch.addcmul(bias, scale, torch.tanh(raw))


def squash_to_env_bounds(
    raw: torch.Tensor, low: torch.Tensor, high: torch.Tensor
) -> torch.Tensor:
    """Map a raw policy output into the env's action bounds.

    One place computes the affine terms, so the collector, the training
    diagnostics and the policy cannot drift apart.
    """
    scale, bias = env_squash_terms(low, high)
    return squash_to_env(raw, scale, bias)


def env_action_chunk(
    raw: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """The env-facing action: squashed into the env's bounds.

    Every action payload the env receives, and the env-space action the
    critic reads, goes through this one function, so the bound invariant
    has a single home. ``mask`` zeroes padded steps; it is applied after
    the squash.
    """
    squashed = squash_to_env_bounds(raw, low, high)
    if mask is not None:
        squashed = squashed * mask
    return squashed

