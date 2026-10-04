"""Obs-window encoder and the running obs normalizer."""

from __future__ import annotations

import warnings

import torch
from torch import nn

from spectral_joint_policy.layers import CondMLP

class RunningMeanStd(nn.Module):
    """Per-dim running mean/var over raw obs frames. fp32 stats, fp64 ``count``
    (fp32 dies at 2^24). The warm-up latch is a plain bool: a buffer would show
    up in ``state_dict``, a tensor read would be a D2H sync."""

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
        """Fold raw frames into the running stats. Chan et al. parallel variance
        merge in fp64, cast back to fp32, so updates do not drift."""
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
        # One pass for both moments; mean() then var() walked the batch twice.
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

    @torch.inference_mode()
    def reset(self) -> None:
        """Clear the latch, reset stats."""
        self.mean.zero_()
        self.var.fill_(1.0)
        self.count.zero_()
        self._uninitialised = True
        self._passthrough_warned = False

    def _load_from_state_dict(self, *args, **kwargs):
        """Load state, re-derive the latch. It is not persisted, so
        ``count == 0`` passes frames through."""
        super()._load_from_state_dict(*args, **kwargs)
        self._uninitialised = bool(self.count.item() == 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Norm with frozen stats. Returns ``x`` unchanged until first update;
        the collector is supposed to call ``update()`` first, so warn once if
        we ever see a call before that."""
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
    """Flattened window -> one SiLU MLP -> the conditioning vector. The flatten
    reads frame order, which a mean-pool cannot; DCT concat was a rotation of
    the same flatten, so it added no subspace and was dropped."""

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
        """Window ``[B, T, obs]`` -> one ``[B, hidden]`` conditioning vector."""
        if states.dim() != 3 or states.shape[2] != self.obs_dim:
            raise ValueError(f"states must have shape [B, {self.max_obs_steps}, {self.obs_dim}], got {tuple(states.shape)}")
        if states.shape[1] != self.max_obs_steps:
            raise ValueError(f"states has {states.shape[1]} frames, built for max_obs_steps={self.max_obs_steps}")
        return self.net(states.flatten(1))

