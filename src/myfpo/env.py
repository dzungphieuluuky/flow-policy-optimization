"""Environment-facing helpers.

The env registry (dims from ``configs/envs.yaml``), the observation normalizer,
and the action-bounds resolution. Leaf module: nothing in the package imports
from here except factories and the collector.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

__all__ = [
    "ENV_REGISTRY_PATH",
    "KNOWN_ENVS",
    "RunningMeanStd",
    "get_observation_action_dim",
    "resolve_action_bounds",
]

# ---- from configuration.py ----
ENV_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "envs.yaml"

with ENV_REGISTRY_PATH.open() as f:
    _ENV_REGISTRY = yaml.safe_load(f) or {}

KNOWN_ENVS = tuple(_ENV_REGISTRY.keys())


def get_observation_action_dim(env_id: str) -> tuple[int, int]:
    """Obs and action dims for ``env_id``, from ``configs/envs.yaml``."""
    if not _ENV_REGISTRY:
        raise KeyError(f"registry empty at {ENV_REGISTRY_PATH}")
    entry = _ENV_REGISTRY.get(env_id)
    if entry is None:
        raise KeyError(f"unknown env {env_id!r}; add it to {ENV_REGISTRY_PATH}")
    return int(entry["obs_dim"]), int(entry["action_dim"])


# ---- from processor.py ----
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


# ---- from rollout.py ----
def resolve_action_bounds(
    low: float | np.ndarray | torch.Tensor,
    high: float | np.ndarray | torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coerce env action bounds into ``dtype`` tensors, strict ``high > low``."""

    def _coerce(x):
        if isinstance(x, torch.Tensor):
            return x
        return np.asarray(x, dtype=np.float32)

    lo = torch.as_tensor(_coerce(low), dtype=dtype).reshape(-1)
    hi = torch.as_tensor(_coerce(high), dtype=dtype).reshape(-1)
    if lo.shape != hi.shape:
        raise ValueError(
            f"action_low shape {tuple(lo.shape)} != action_high shape {tuple(hi.shape)}"
        )
    if not (torch.isfinite(lo).all() and torch.isfinite(hi).all()):
        raise ValueError(
            "action bounds must be finite in every dim; an unbounded Box maps "
            "to an infinite scale and the env would get NaN actions"
        )
    if not bool(((hi - lo) > 0).all()):
        raise ValueError("action bounds must satisfy high > low in every dim")
    return lo, hi
