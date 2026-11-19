"""flow_policy_optimization.config: config, normalization and bounds."""

from __future__ import annotations

import dataclasses
import warnings
from typing import Literal

import numpy as np
import torch
from torch import nn

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

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        batch = x.detach().reshape(-1, x.shape[-1]).to(self.mean.device, dtype=torch.float32)
        b = batch.shape[0]
        if b == 0:
            return
        b_var, b_mean = torch.var_mean(batch, dim=0, unbiased=False)
        delta = b_mean - self.mean
        total = self.count + b
        self.mean.add_((delta * b / total).to(self.mean.dtype))
        self.var.copy_((self.count * self.var + b * b_var + delta.square() * (self.count * b / total)) / total)
        self.count.copy_(total)
        self.uninitialized = False

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.uninitialized = bool(self.count.item() == 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.uninitialized:
            return ((x - self.mean) * torch.rsqrt(self.var + self.eps)).clamp(-self.clip, self.clip).to(x.dtype)
        if not self.passthrough_warned:
            warnings.warn(f"{type(self).__name__}: forward before the first update(); returning raw input", stacklevel=2)
            self.passthrough_warned = True
        return x
def resolve_action_bounds(
    low: float | np.ndarray | torch.Tensor,
    high: float | np.ndarray | torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coerce env action bounds into ``dtype`` tensors with strict ``high > low``."""
    lo = torch.as_tensor(low if isinstance(low, torch.Tensor) else np.asarray(low, dtype=np.float32), dtype=dtype).reshape(-1)
    hi = torch.as_tensor(high if isinstance(high, torch.Tensor) else np.asarray(high, dtype=np.float32), dtype=dtype).reshape(-1)
    if lo.shape != hi.shape:
        raise ValueError(f"action_low shape {tuple(lo.shape)} != action_high shape {tuple(hi.shape)}")
    if not (torch.isfinite(lo).all() and torch.isfinite(hi).all()):
        raise ValueError("action bounds must be finite in every dim")
    if not bool(((hi - lo) > 0).all()):
        raise ValueError("action bounds must satisfy high > low in every dim")
    return lo, hi
@dataclasses.dataclass
class FlowPolicyOptimizationConfig:
    """Hyperparameters for lean flow_policy_optimization."""

    # window / chunk shape
    num_obs_steps: int = 4
    action_horizon: int = 4
    num_action_steps: int | None = None
    hidden_dim: int = 128
    time_embed_dim: int = 8
    critic_hidden_dim: int = 512
    critic_num_layers: int = 2
    # envs
    num_envs: int = 64
    # MC sampling
    num_mc_samples: int = 16
    min_sampled_timestep: float = 0.1
    max_sampled_timestep: float = 0.9
    num_steps_noise_start: int = 10
    # objective
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    value_loss_coef: float = 1.0
    value_loss_beta: float = 10.0
    # A loss-difference ratio moves much faster than a likelihood ratio, so a
    # 5% clip saturates after one step. 0.2 keeps the trust region active.
    clip_param: float = 0.2
    spo_clip_param: float = 0.2
    # ASPO bounds the negative-advantage branch. Sparse rewards may need "ppo".
    trust_region_mode: Literal["ppo", "spo", "aspo"] = "aspo"
    cfm_clamp: float = 20.0
    # Symmetric cap: exp(2) = 7.4, not exp(10) = 22026.
    cfm_diff_clamp_min: float = -2.0
    cfm_diff_clamp_max: float = 2.0
    cfm_loss_reduction: Literal["sum", "mean", "sqrt"] = "mean"
    cfm_loss_use_huber: bool = True
    cfm_loss_huber_delta: float = 0.5
    velocity_clip_value: float | None = None
    # exploration / storage
    action_perturb_std: float = 0.02
    actor_scale: float = 1.0
    # optimizer / batching
    policy_lr: float = 1e-4
    critic_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    adam_betas: tuple[float, float] = (0.9, 0.999)
    max_grad_norm: float = 1.0
    chunks_per_update: int = 2048
    chunks_per_minibatch: int = 512
    num_learning_epochs: int = 16
    # target critic
    target_tau: float = 0.005
    # eval / reporting
    eval_max_episodes: int = 50
    report_to: str = "all"

    def __post_init__(self) -> None:
        if self.num_action_steps is None:
            self.num_action_steps = self.action_horizon
        self.adam_betas = tuple(self.adam_betas)
        if not self.cfm_diff_clamp_min < 0.0 < self.cfm_diff_clamp_max:
            raise ValueError("cfm_diff_clamp_min < 0 < cfm_diff_clamp_max is required")
        if self.trust_region_mode not in ("ppo", "spo", "aspo"):
            raise ValueError(f"trust_region_mode must be one of 'ppo', 'spo', 'aspo', got {self.trust_region_mode!r}")
        if self.cfm_loss_reduction not in ("sum", "mean", "sqrt"):
            raise ValueError(f"cfm_loss_reduction must be one of 'sum', 'mean', 'sqrt', got {self.cfm_loss_reduction!r}")
        if not 1 <= self.num_action_steps <= self.action_horizon:
            raise ValueError(f"num_action_steps must be in [1, {self.action_horizon}], got {self.num_action_steps}")
        if self.num_obs_steps < 1:
            raise ValueError(f"num_obs_steps must be >= 1, got {self.num_obs_steps}")
        if self.chunks_per_minibatch > self.chunks_per_update:
            raise ValueError(f"chunks_per_minibatch ({self.chunks_per_minibatch}) must be <= chunks_per_update ({self.chunks_per_update})")
        if not 0.0 <= self.target_tau <= 1.0:
            raise ValueError(f"target_tau must be in [0, 1], got {self.target_tau}")
        for name in ("gamma", "gae_lambda"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    def asdict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Networks (actor, critic, encoder)
