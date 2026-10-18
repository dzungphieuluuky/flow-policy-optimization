"""Obs-window encoder and the running obs normalizer."""

from __future__ import annotations

import warnings

import torch
from torch import nn

from myfpo.layers import CondMLP

class RunningMeanStd(nn.Module):
    """Per-dimension running mean and variance over raw obs frames."""

    def __init__(self, dim: int, eps: float = 1e-8, clip: float = 10.0):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim={dim} must be positive")
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))
        self._uninitialised = True
        self._passthrough_warned = False
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
        self._uninitialised = False

    def _load_from_state_dict(self, *args, **kwargs):
        """Load the state. Derive the latch again."""
        super()._load_from_state_dict(*args, **kwargs)
        self._uninitialised = bool(self.count.item() == 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise with the frozen statistics."""
        if self._uninitialised:
            if not self._passthrough_warned:
                warnings.warn(
                    f"{type(self).__name__}: forward before the first update(); "
                    "returning raw input", stacklevel=2,
                )
                self._passthrough_warned = True
            return x
        norm = (x - self.mean) * torch.rsqrt(self.var + self.eps)
        return norm.clamp(-self.clip, self.clip).to(x.dtype)

class StateEncoder(nn.Module):
    """Flatten the window. Pass it to one SiLU MLP. Return the conditioning"""

    def __init__(self, obs_dim: int, hidden_dim: int, max_obs_steps: int):
        super().__init__()
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        if max_obs_steps < 1:
            raise ValueError(f"max_obs_steps must be >= 1, got {max_obs_steps}")
        self.obs_dim = obs_dim
        self.max_obs_steps = max_obs_steps
        self.net = CondMLP(max_obs_steps * obs_dim, hidden_dim, hidden_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Accept the window ``[B, T, obs]``. Return one conditioning vector of"""
        if states.dim() != 3 or states.shape[2] != self.obs_dim:
            raise ValueError(f"states must have shape [B, {self.max_obs_steps}, {self.obs_dim}], got {tuple(states.shape)}")
        if states.shape[1] != self.max_obs_steps:
            raise ValueError(f"states has {states.shape[1]} frames, built for max_obs_steps={self.max_obs_steps}")
        return self.net(states.flatten(1))

