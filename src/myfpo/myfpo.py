"""myfpo: minimal research implementation (CleanRL / RSL-RL style)."""

from __future__ import annotations

import copy
import csv
import dataclasses
import json
import math
import os
import random
import time
import types
import warnings
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

__all__ = [
    "ENV_REGISTRY_PATH",
    "KNOWN_ENVS",
    "MyfpoConfig",
    "RunningMeanStd",
    "get_observation_action_dim",
    "resolve_action_bounds",
]

TIME_SCALE = 1000.0
CSV_FIELDS = ["episode", "return", "length", "success", "jerk"]
MANISKILL_TASKS = {"MS-PickCube-v1": "PickCube-v1"}
MANISKILL_TASK_OPTIONS = {"MS-PickCube-v1": {"ignore_terminations": True}}
DEFAULT_OBS_MODE = "state"
DEFAULT_CONTROL_MODE = "pd_joint_delta_pos"

# ---------------------------------------------------------------------------
# Env registry, normalization, bounds, and config
# ---------------------------------------------------------------------------

ENV_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "envs.yaml"
with ENV_REGISTRY_PATH.open() as _f:
    _ENV_REGISTRY = yaml.safe_load(_f) or {}
KNOWN_ENVS = tuple(_ENV_REGISTRY.keys())


def get_observation_action_dim(env_id: str) -> tuple[int, int]:
    """Obs and action dims for ``env_id``, from ``configs/envs.yaml``."""
    if not _ENV_REGISTRY:
        raise KeyError(f"registry empty at {ENV_REGISTRY_PATH}")
    entry = _ENV_REGISTRY.get(env_id)
    if entry is None:
        raise KeyError(f"unknown env {env_id!r}; add it to {ENV_REGISTRY_PATH}")
    return int(entry["obs_dim"]), int(entry["action_dim"])


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
        if x.shape[-1] != self.mean.numel():
            raise ValueError(f"update got last dim {x.shape[-1]}, expected {self.mean.numel()}")
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
class MyfpoConfig:
    """Hyperparameters for lean myfpo."""

    # window / chunk shape
    num_obs_steps: int = 4
    action_horizon: int = 4
    num_action_steps: int | None = None
    hidden_dim: int = 128
    time_embed_dim: int = 8
    critic_hidden_dim: int = 512
    critic_num_layers: int = 2
    critic_condition_on_exec_length: bool = True
    # envs
    num_envs: int = 64
    # MC sampling
    num_mc_samples: int = 16
    min_sampled_timestep: float = 0.1
    max_sampled_timestep: float = 0.9
    num_steps_noise_start: int = 10
    num_baseline_samples: int = 4
    # objective
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    value_loss_coef: float = 1.0
    value_loss_beta: float = 10.0
    clip_param: float = 0.05
    spo_clip_param: float = 0.05
    trust_region_mode: str = "ppo"
    cfm_clamp: float = 20.0
    cfm_diff_clamp_max: float = 10.0
    cfm_loss_reduction: str = "sqrt"
    cfm_loss_use_huber: bool = True
    cfm_loss_huber_delta: float = 0.5
    velocity_clip_value: float | None = None
    # exploration / storage
    action_perturb_std: float = 0.02
    storage_action_noise_std: float = 0.0
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
    # eval / reporting
    eval_max_episodes: int = 50
    report_to: str = "all"

    def __post_init__(self) -> None:
        if self.num_action_steps is None:
            self.num_action_steps = self.action_horizon
        self.adam_betas = tuple(self.adam_betas)

    @property
    def num_minibatches(self) -> int:
        return max(1, self.chunks_per_update // self.chunks_per_minibatch)

    def asdict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Networks (actor, critic, encoder)
# ---------------------------------------------------------------------------


def _ortho_linear(in_dim: int, out_dim: int, std: float = math.sqrt(2.0)) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


class RMSNorm(nn.RMSNorm):
    """RMSNorm in fp32 compute while preserving model dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = F.rms_norm(x.float(), self.normalized_shape, self.weight.float(), self.eps)
        return y.to(dtype=x.dtype)


class CondMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        if min(in_dim, hidden_dim, out_dim) < 1:
            raise ValueError("CondMLP dimensions must be positive")
        self.fc1 = _ortho_linear(in_dim, hidden_dim)
        self.fc2 = _ortho_linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class SinusoidalEncoding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        if dim < 2 or max_period <= 0:
            raise ValueError(f"invalid SinusoidalEncoding(dim={dim}, max_period={max_period})")
        self.dim = dim
        self.half = dim // 2
        self.register_buffer(
            "freqs",
            torch.exp(-math.log(max_period) * torch.arange(self.half, dtype=torch.float32) / self.half),
            persistent=False,
        )

    def encode_times(self, t: torch.Tensor) -> torch.Tensor:
        ang = (t.float().unsqueeze(-1) * TIME_SCALE) * self.freqs.to(t.device)
        emb = torch.cat([ang.cos(), ang.sin()], dim=-1)
        return F.pad(emb, (0, 1)) if self.dim % 2 else emb


class SelfAttention(nn.Module):
    """Causal SDPA self-attention with RoPE."""

    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        if hidden_dim < 1 or n_heads < 1 or hidden_dim % n_heads != 0:
            raise ValueError(f"invalid SelfAttention(hidden_dim={hidden_dim}, n_heads={n_heads})")
        head_dim = hidden_dim // n_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim={head_dim} must be even for RoPE")
        self.n_heads = n_heads
        self.head_dim = head_dim
        half = head_dim // 2
        self.register_buffer("inv_freq", 1.0 / (10000.0 ** (torch.arange(half).float() / half)), persistent=False)
        self.qkv = _ortho_linear(hidden_dim, 3 * hidden_dim)
        self.out = _ortho_linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq, _ = x.shape
        h, d = self.n_heads, self.head_dim
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, seq, h, d).transpose(1, 2)
        k = k.view(b, seq, h, d).transpose(1, 2)
        v = v.view(b, seq, h, d).transpose(1, 2)
        freqs = torch.outer(torch.arange(seq, device=x.device, dtype=torch.float32), self.inv_freq)
        cos, sin = freqs.cos()[None, None], freqs.sin()[None, None]
        q = torch.stack([q[..., 0::2] * cos - q[..., 1::2] * sin, q[..., 0::2] * sin + q[..., 1::2] * cos], dim=-1).flatten(-2)
        k = torch.stack([k[..., 0::2] * cos - k[..., 1::2] * sin, k[..., 0::2] * sin + k[..., 1::2] * cos], dim=-1).flatten(-2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, seq, h * d))


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        if hidden_dim < 1 or n_heads < 1 or hidden_dim % n_heads != 0:
            raise ValueError(f"invalid CrossAttention(hidden_dim={hidden_dim}, n_heads={n_heads})")
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.q_proj = _ortho_linear(hidden_dim, hidden_dim)
        self.kv_proj = _ortho_linear(hidden_dim, 2 * hidden_dim)
        self.out = _ortho_linear(hidden_dim, hidden_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if query.dim() != 3 or context.dim() != 3 or query.shape[0] != context.shape[0] or query.shape[2] != context.shape[2]:
            raise ValueError(f"incompatible query {tuple(query.shape)} and context {tuple(context.shape)}")
        b, lq, _ = query.shape
        lk, h, d = context.shape[1], self.n_heads, self.head_dim
        q = self.q_proj(query).view(b, lq, h, d).transpose(1, 2)
        k, v = self.kv_proj(context).chunk(2, dim=-1)
        k = k.view(b, lk, h, d).transpose(1, 2)
        v = v.view(b, lk, h, d).transpose(1, 2)
        return self.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, lq, h * d))


class StateEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, max_obs_steps: int, n_heads: int = 2):
        super().__init__()
        if hidden_dim < 2 or max_obs_steps < 1:
            raise ValueError(f"invalid StateEncoder(hidden_dim={hidden_dim}, max_obs_steps={max_obs_steps})")
        self.obs_dim = obs_dim
        self.max_obs_steps = max_obs_steps
        self.n_heads = n_heads
        self.frame = CondMLP(obs_dim, hidden_dim, hidden_dim)
        self.attn_norm = RMSNorm(hidden_dim, eps=1e-6)
        self.attn = SelfAttention(hidden_dim, n_heads)
        self.out_norm = RMSNorm(hidden_dim, eps=1e-6)

    def encode_tokens(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape != (states.shape[0], self.max_obs_steps, self.obs_dim):
            raise ValueError(f"expected [B, {self.max_obs_steps}, {self.obs_dim}], got {tuple(states.shape)}")
        h = self.frame(states)
        return h + self.attn(self.attn_norm(h))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.encode_tokens(states)[:, -1])


class FlowMLP(nn.Module):
    """Flat MLP flow velocity head over ``[x_flat | t_flat | cond]``."""

    def __init__(self, action_dim: int, action_horizon: int, hidden_dim: int, time_embed_dim: int, num_layers: int = 4):
        super().__init__()
        if action_dim < 1 or action_horizon < 1 or hidden_dim < 2 or time_embed_dim < 2 or num_layers < 1:
            raise ValueError("invalid FlowMLP dimensions")
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_horizon = action_horizon
        self.time_embed_dim = time_embed_dim
        self.num_layers = num_layers
        flat = action_horizon * action_dim
        self.timestep_embedder = SinusoidalEncoding(time_embed_dim)
        self.in_block = CondMLP(flat + action_horizon * time_embed_dim + hidden_dim, 4 * hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(CondMLP(hidden_dim, 4 * hidden_dim, hidden_dim) for _ in range(num_layers))
        self.norm_out = RMSNorm(hidden_dim, eps=1e-6)
        self.out = nn.Linear(hidden_dim, flat)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _embed_times(self, t: torch.Tensor, b: int, k: int) -> torch.Tensor:
        if t.dim() == 3 and t.shape[-1] != 1:
            raise ValueError(f"t must end in singleton axis, got {tuple(t.shape)}")
        if t.dim() in (2, 3) and tuple(t.shape[:2]) == (b, k):
            grid = t.squeeze(-1) if t.dim() == 3 else t
        elif t.dim() <= 1:
            t = t.reshape(-1)
            if t.numel() == 1:
                grid = t.expand(b, k)
            elif t.numel() == b:
                grid = t[:, None].expand(b, k)
            else:
                raise ValueError(f"t must contain 1 or {b} values, got {t.numel()}")
        else:
            raise ValueError(f"t must be scalar, [B], or [B, K, 1], got {tuple(t.shape)}")
        return self.timestep_embedder.encode_times(grid.reshape(-1)).reshape(b, k, -1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor | float, condition: torch.Tensor) -> torch.Tensor:
        b, k, _ = x_t.shape
        if x_t.dim() != 3 or x_t.shape[2] != self.action_dim or k > self.action_horizon:
            raise ValueError(f"invalid x_t shape {tuple(x_t.shape)}")
        if condition.shape != (b, self.hidden_dim):
            raise ValueError(f"expected condition [{b}, {self.hidden_dim}], got {tuple(condition.shape)}")
        t_tensor = torch.as_tensor(t, dtype=torch.float32, device=x_t.device)
        temb = self._embed_times(t_tensor, b, k).reshape(b, -1).to(dtype=x_t.dtype)
        h = self.in_block(torch.cat([x_t.reshape(b, -1), temb, condition.to(device=x_t.device, dtype=x_t.dtype)], dim=-1))
        for block in self.blocks:
            h = block(h)
        return self.out(self.norm_out(h)).reshape_as(x_t)


class ActionAffine(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias


class StateAffine(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(torch.mean(x.float().square(), dim=-1, keepdim=True) + self.eps).to(x.dtype)
        return x * scale * self.weight + self.bias


class Critic(nn.Module):
    """Cross-attention Q(s_t, a_{t:t+h}) critic."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        num_obs_steps: int,
        action_horizon: int,
        *,
        condition_on_exec_length: bool = True,
        num_layers: int = 2,
        state_hidden: int | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_obs_steps = num_obs_steps
        self.action_horizon = action_horizon
        self.condition_on_exec_length = condition_on_exec_length
        state_dim = state_hidden if state_hidden is not None else hidden_dim
        self.input_dim = 2 * state_dim + (1 if condition_on_exec_length else 0)

        self.norm_state = StateAffine(num_obs_steps * obs_dim)
        self.state_encoder = StateEncoder(obs_dim, state_dim, num_obs_steps)
        self.norm_action = ActionAffine(action_horizon * action_dim)
        self.act_in = _ortho_linear(action_dim, state_dim)
        self.act_attn = SelfAttention(state_dim, 4)
        self.act_attn_norm = RMSNorm(state_dim, eps=1e-6)
        self.obs_to_act = CrossAttention(state_dim, 4)
        self.act_to_obs = CrossAttention(state_dim, 4)
        self.cross_norm_obs = RMSNorm(state_dim, eps=1e-6)
        self.cross_norm_act = RMSNorm(state_dim, eps=1e-6)

        layers: list[nn.Module] = []
        for i in range(num_layers):
            layers.extend([_ortho_linear(self.input_dim if i == 0 else hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish()])
        final = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.mlp = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, exec_length_frac: torch.Tensor | None = None) -> torch.Tensor:
        if states.shape != (states.shape[0], self.num_obs_steps, self.obs_dim):
            raise ValueError(f"states must be [B, {self.num_obs_steps}, {self.obs_dim}], got {tuple(states.shape)}")
        if actions.shape != (states.shape[0], self.action_horizon, self.action_dim):
            raise ValueError(f"actions must be [{states.shape[0]}, {self.action_horizon}, {self.action_dim}], got {tuple(actions.shape)}")

        obs_tokens = self.state_encoder.out_norm(
            self.state_encoder.encode_tokens(self.norm_state(states.flatten(1)).view(states.shape))
        )
        act_tokens = self.act_in(self.norm_action(actions.flatten(1)).view(actions.shape))
        act_tokens = act_tokens + self.act_attn(self.act_attn_norm(act_tokens))
        obs = obs_tokens + self.obs_to_act(self.cross_norm_obs(obs_tokens), act_tokens)
        act = act_tokens + self.act_to_obs(self.cross_norm_act(act_tokens), obs_tokens)

        parts = [obs[:, -1], act.mean(dim=1)]
        if self.condition_on_exec_length:
            if exec_length_frac is None or exec_length_frac.shape != (states.shape[0],):
                raise ValueError("exec_length_frac of shape [B] is required when condition_on_exec_length=True")
            parts.append(exec_length_frac.to(dtype=parts[0].dtype).unsqueeze(-1))
        return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Flow & RL objective math
# ---------------------------------------------------------------------------


def expand_mc_samples(actions: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], int]:
    """Interpolate ``x_t = lerp(eps, actions, t)`` over the MC axis -> ``[B*M, ...]``."""
    b, m = eps.shape[0], eps.shape[1]
    flow_shape = tuple(actions.shape[1:])
    if tuple(eps.shape) != (b, m, *flow_shape):
        raise ValueError(f"eps shape {tuple(eps.shape)} != {(b, m, *flow_shape)}")
    t = torch.as_tensor(t, dtype=torch.float32, device=eps.device)
    if tuple(t.shape) != (b, m, flow_shape[0], 1):
        raise ValueError(f"t shape {tuple(t.shape)} != {(b, m, flow_shape[0], 1)}")

    flat_t = t.reshape(b * m, flow_shape[0], 1)
    flat_act = actions.unsqueeze(1).expand(b, m, *flow_shape).reshape(b * m, *flow_shape)
    flat_eps = eps.reshape(b * m, *flow_shape)
    x_t = torch.lerp(flat_eps.float(), flat_act.float(), flat_t.expand_as(flat_act).float())
    return x_t, flat_t.squeeze(-1), flow_shape, m


def compute_cfm_loss(
    v_hat: torch.Tensor,
    v_star: torch.Tensor,
    cfg: MyfpoConfig,
    reduce_mode: str,
    action_is_pad: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if v_hat.shape != v_star.shape:
        raise ValueError(f"v_hat shape {tuple(v_hat.shape)} != v_star shape {tuple(v_star.shape)}")
    diff = v_hat - v_star
    if cfg.cfm_loss_use_huber:
        delta = cfg.cfm_loss_huber_delta
        abs_d = diff.abs()
        sq = torch.where(abs_d <= delta, diff.square(), 2.0 * delta * abs_d - delta * delta)
    else:
        sq = diff.square()

    n_valid = None
    if action_is_pad is not None:
        sq = sq * (~action_is_pad).to(sq.dtype)[:, None, :, None]
        n_valid = ((~action_is_pad).sum(dim=-1, keepdim=True).to(sq.dtype) * sq.shape[-1]).clamp(min=1.0)

    if reduce_mode == "scalar":
        return sq.mean() if n_valid is None else sq.sum() / (n_valid.sum() * sq.shape[1])
    if reduce_mode not in ("sample", "sample_and_token"):
        raise ValueError(f"invalid reduce_mode={reduce_mode!r}")

    flat = sq.flatten(2)
    if cfg.cfm_loss_reduction == "sum":
        sample = flat.sum(dim=-1)
    elif cfg.cfm_loss_reduction == "sqrt":
        sample = flat.sum(dim=-1) / (n_valid if n_valid is not None else math.sqrt(flat.shape[-1]))
    else:
        sample = flat.mean(dim=-1) if n_valid is None else flat.sum(dim=-1) / n_valid
    return sample if reduce_mode == "sample" else (sample, sq.mean(dim=-1))


def compute_policy_ratio(cfm_loss_old: torch.Tensor, cfm_loss_new: torch.Tensor, cfg: MyfpoConfig) -> torch.Tensor:
    delta = cfm_loss_old.clamp(max=cfg.cfm_clamp) - cfm_loss_new.clamp(max=cfg.cfm_clamp)
    safe_delta = delta.clamp(max=cfg.cfm_diff_clamp_max)
    return torch.exp(safe_delta.detach() + delta - delta.detach())


def compute_trust_region_surrogate(ratio: torch.Tensor, advantage: torch.Tensor, cfg: MyfpoConfig) -> torch.Tensor:
    ratio_adv = ratio * advantage
    ppo = torch.min(ratio_adv, ratio.clamp(1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * advantage)
    if cfg.trust_region_mode == "ppo":
        return ppo
    spo = ratio_adv - (advantage.abs() / (2.0 * cfg.spo_clip_param)) * (ratio - 1.0).square()
    if cfg.trust_region_mode == "spo":
        return spo
    return torch.where(advantage > 0, ppo, spo)


def sample_action(predict_velocity, shape, condition, num_steps: int, eps: torch.Tensor, device, *, t_min: float, t_max: float) -> torch.Tensor:
    if device is None or not 0.0 <= t_min <= t_max <= 1.0 or tuple(eps.shape) != tuple(shape):
        raise ValueError("invalid sample_action arguments")
    grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=torch.float32)
    x = eps.to(device)
    with torch.inference_mode():
        for i in range(num_steps):
            t_cur = grid[i].clamp(t_min, t_max).reshape(1)
            x = torch.addcmul(x, predict_velocity(x, t_cur, condition), grid[i + 1] - grid[i])
    return x


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def token_cfm_loss(
    v: torch.Tensor, actions: torch.Tensor, eps: torch.Tensor, mask: torch.Tensor, cfg: MyfpoConfig, *, n: int, num_mc: int, flow_shape
) -> tuple[torch.Tensor, torch.Tensor]:
    return compute_cfm_loss(v.reshape(n, num_mc, *flow_shape).float(), (actions.unsqueeze(1) - eps).float(), cfg, "sample_and_token", mask)


class MyfpoPolicy(nn.Module):
    """Flow actor + state encoder + single cross-attention critic."""

    def __init__(self, config: MyfpoConfig, obs_dim: int, action_dim: int, action_low, action_high):
        super().__init__()
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = FlowMLP(action_dim, config.action_horizon, config.hidden_dim, config.time_embed_dim)
        self.encoder = StateEncoder(obs_dim, config.hidden_dim, config.num_obs_steps)
        critic_width = config.critic_hidden_dim if config.critic_hidden_dim is not None else config.hidden_dim
        self.critic = Critic(
            obs_dim, critic_width, action_dim, config.num_obs_steps, config.action_horizon,
            condition_on_exec_length=config.critic_condition_on_exec_length,
            num_layers=config.critic_num_layers, state_hidden=config.hidden_dim,
        )
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False).eval()
        self.obs_normalizer = RunningMeanStd(obs_dim)
        self.frame_buffer: deque[torch.Tensor] = deque(maxlen=config.num_obs_steps)
        self.action_buffer: deque[torch.Tensor] = deque()
        low, high = resolve_action_bounds(action_low, action_high)
        self.register_buffer("action_low", low.float(), persistent=False)
        self.register_buffer("action_high", high.float(), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.critic_target.eval()
        return self

    def reset(self) -> None:
        self.frame_buffer.clear()
        self.action_buffer.clear()

    def _to_flow_space(self, actions: torch.Tensor) -> torch.Tensor:
        return actions / self.config.actor_scale

    def _from_flow_space(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.config.actor_scale

    def compute_predicted_velocity(self, x_t, t, condition):
        v = self.actor(x_t, t, condition)
        bound = self.config.velocity_clip_value
        return v if bound is None else v.clamp(-bound, bound)

    @torch.inference_mode()
    def encode_states(self, states: torch.Tensor) -> torch.Tensor:
        states = states.to(device=self.obs_normalizer.mean.device, dtype=torch.float32)
        return self.encoder(self.obs_normalizer(states))

    @torch.inference_mode()
    def sample_chunk(self, condition: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        shape = (condition.shape[0], cfg.action_horizon, self.action_dim)
        eps = torch.randn(*shape, device=condition.device)
        chunk = sample_action(
            self.compute_predicted_velocity, shape, condition,
            num_steps=cfg.num_steps_noise_start, eps=eps, device=condition.device,
            t_min=cfg.min_sampled_timestep, t_max=cfg.max_sampled_timestep,
        ).float()
        chunk = self._from_flow_space(chunk)
        if self.training and cfg.action_perturb_std > 0:
            chunk = chunk + cfg.action_perturb_std * torch.randn_like(chunk)
        return chunk.clamp(-1.0, 1.0)

    def predict_action_chunk(self, states: torch.Tensor) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=torch.float32)
        if states.dim() == 2 and states.shape[0] == self.config.num_obs_steps:
            states = states.unsqueeze(0)
        elif states.dim() != 3:
            raise ValueError(f"observation.state must be [B, T, obs] or [T, obs]; got {tuple(states.shape)}")
        return self.sample_chunk(self.encode_states(states))

    def select_action(self, obs) -> torch.Tensor:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.action_low.device)
        if obs.dim() == 1:
            if self.frame_buffer:
                self.frame_buffer.append(obs.detach())
            else:
                for _ in range(self.config.num_obs_steps):
                    self.frame_buffer.append(obs.detach().clone())
            window = torch.stack(tuple(self.frame_buffer), dim=0)
        elif obs.dim() == 2:
            window = obs
        else:
            raise ValueError(f"select_action expects [obs] or [T, obs]; got {tuple(obs.shape)}")
        if not self.action_buffer:
            chunk = self.predict_action_chunk(window)
            n_exec = max(1, min(self.config.num_action_steps, chunk.shape[1]))
            env_chunk = to_env_action(chunk, self.action_low, self.action_high)
            self.action_buffer.extend(env_chunk[0, i] for i in range(n_exec))
        return self.action_buffer.popleft()

    @torch.no_grad()
    def compute_behavior_cfm_losses(
        self, normalized_states: torch.Tensor, actions: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, action_is_pad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self._to_flow_space(actions)
        x_t, flat_t, flow_shape, num_mc = expand_mc_samples(actions, eps, t)
        cond = self.encoder(normalized_states)
        cond_flat = cond.unsqueeze(1).expand(cond.shape[0], num_mc, cond.shape[1]).reshape(-1, cond.shape[1])
        v_old = self.compute_predicted_velocity(x_t, flat_t, cond_flat).float()
        return token_cfm_loss(v_old, actions, eps, action_is_pad, self.config, n=eps.shape[0], num_mc=num_mc, flow_shape=flow_shape)


# ---------------------------------------------------------------------------
# Chunk GAE, RolloutBuffer, and SameStep Collector
# ---------------------------------------------------------------------------


def to_env_action(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return low + (x + 1.0) * 0.5 * (high - low)


def bound_for_storage(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-1.0, 1.0)


def env_squash_terms(low: torch.Tensor, high: torch.Tensor):
    scale = 0.5 * (high - low)
    return scale, low + scale


def to_critic_action(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return x if mask is None else x * mask


@torch.no_grad()
def compute_chunk_advantages(
    step_rewards: torch.Tensor,
    v_current: torch.Tensor,
    v_next: torch.Tensor,
    chunk_starts: torch.Tensor,
    chunk_lengths: torch.Tensor,
    *,
    chunk_terminated: torch.Tensor,
    chunk_size: int,
    gamma: float,
    chunk_truncated: torch.Tensor | None = None,
    chunk_prev: torch.Tensor | None = None,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """V-anchored chunk GAE returning ``(advantages, lambda_returns)``."""
    if not (0.0 <= gamma <= 1.0 and 0.0 <= gae_lambda <= 1.0):
        raise ValueError(f"gamma={gamma}, gae_lambda={gae_lambda} must be in [0, 1]")
    device = v_current.device
    rewards = step_rewards.to(device=device, dtype=torch.float32).reshape(-1)
    v_cur = v_current.to(device=device, dtype=torch.float32).reshape(-1)
    v_nxt = v_next.to(device=device, dtype=torch.float32).reshape(-1)
    starts = chunk_starts.to(device=device, dtype=torch.long).reshape(-1)
    lengths = chunk_lengths.to(device=device, dtype=torch.long).reshape(-1)
    terminated = chunk_terminated.to(device=device, dtype=torch.bool).reshape(-1)
    n = starts.numel()
    if n == 0:
        empty = torch.empty(0, device=device, dtype=torch.float32)
        return empty, empty

    truncated = torch.zeros(n, device=device, dtype=torch.bool) if chunk_truncated is None else chunk_truncated.to(device=device, dtype=torch.bool).reshape(-1)
    if gae_lambda > 0.0 and (chunk_truncated is None or chunk_prev is None):
        raise ValueError("chunk_truncated and chunk_prev are required when gae_lambda > 0")

    offsets = torch.arange(chunk_size, device=device)
    gathered = rewards[(starts.unsqueeze(1) + offsets).clamp(max=rewards.numel() - 1)]
    valid = offsets.unsqueeze(0) < lengths.unsqueeze(1)
    chunk_rew = (gathered * valid * (gamma ** offsets)).sum(dim=1)

    gamma_h = gamma ** lengths.float()
    done = terminated | truncated
    chunk_rew = chunk_rew + gamma_h * (truncated & ~terminated) * v_nxt
    delta = chunk_rew + gamma_h * (~done) * v_nxt - v_cur
    if gae_lambda == 0.0:
        return delta, delta + v_cur

    prev = chunk_prev.to(device=device, dtype=torch.long).reshape(-1)
    linked = prev >= 0
    successor = torch.full((n,), -1, dtype=torch.long, device=prev.device)
    successor[prev[linked]] = torch.arange(n, device=prev.device)[linked]
    succ_cpu = successor.cpu().tolist()
    trace_cpu = (gamma_h * (~done) * gae_lambda).detach().cpu()
    adv_cpu = delta.detach().cpu().clone()
    for i in range(n - 1, -1, -1):
        j = succ_cpu[i]
        if j >= 0:
            adv_cpu[i] = adv_cpu[i] + trace_cpu[i] * adv_cpu[j]
    adv = adv_cpu.to(device=device, dtype=delta.dtype)
    return adv, adv + v_cur


@dataclasses.dataclass
class EnvCarry:
    windows: torch.Tensor


class RolloutBuffer:
    """Fixed-capacity fp32 CPU storage for chunk and executed-step rows."""

    def __init__(self, *, max_chunks: int, max_steps: int, obs_steps: int, obs_dim: int, action_horizon: int, action_dim: int):
        if max_chunks < 1 or max_steps < 1:
            raise ValueError("max_chunks and max_steps must be positive")
        self.max_chunks, self.max_steps = max_chunks, max_steps
        self.obs_steps, self.obs_dim = obs_steps, obs_dim
        self.action_horizon, self.action_dim = action_horizon, action_dim

        self.normalizer_observations = torch.zeros((max_steps, obs_dim), dtype=torch.float32)
        self.step_rewards = torch.zeros((max_steps,), dtype=torch.float32)
        self.chunk_state = torch.zeros((max_chunks, obs_steps, obs_dim), dtype=torch.float32)
        self.chunk_next_state = torch.zeros((max_chunks, obs_steps, obs_dim), dtype=torch.float32)
        self.chunk_actions = torch.zeros((max_chunks, action_horizon, action_dim), dtype=torch.float32)
        self.chunk_starts = torch.zeros((max_chunks,), dtype=torch.int64)
        self.chunk_lengths = torch.zeros((max_chunks,), dtype=torch.int64)
        self.chunk_terminated = torch.zeros((max_chunks,), dtype=torch.bool)
        self.chunk_truncated = torch.zeros((max_chunks,), dtype=torch.bool)
        self.chunk_env_id = torch.zeros((max_chunks,), dtype=torch.int16)
        self.chunk_prev = torch.full((max_chunks,), -1, dtype=torch.int64)
        self._last_chunk: dict[int, int] = {}
        self.n_chunks = 0
        self.n_steps = 0
        self.carry: EnvCarry | None = None

    def reset(self) -> None:
        self.n_chunks = self.n_steps = 0
        self._last_chunk.clear()

    def active_prefix(self) -> dict[str, torch.Tensor]:
        nc, ns = self.n_chunks, self.n_steps
        return {
            "normalizer_observations": self.normalizer_observations[:ns],
            "step_rewards": self.step_rewards[:ns],
            "chunk_state": self.chunk_state[:nc],
            "chunk_next_state": self.chunk_next_state[:nc],
            "chunk_actions": self.chunk_actions[:nc],
            "chunk_starts": self.chunk_starts[:nc],
            "chunk_lengths": self.chunk_lengths[:nc],
            "chunk_terminated": self.chunk_terminated[:nc],
            "chunk_truncated": self.chunk_truncated[:nc],
            "chunk_env_id": self.chunk_env_id[:nc],
            "chunk_prev": self.chunk_prev[:nc],
        }

    def write_chunks(
        self, *, start_window: torch.Tensor, next_window: torch.Tensor, actions: torch.Tensor,
        lengths: torch.Tensor, frames: torch.Tensor, rewards: torch.Tensor,
        terminated: torch.Tensor, truncated: torch.Tensor, env_ids, n_steps: int,
    ) -> None:
        k = int(lengths.numel())
        if k == 0:
            return
        if self.n_chunks + k > self.max_chunks or n_steps < 1 or self.n_steps + n_steps > self.max_steps:
            raise RuntimeError("RolloutBuffer capacity exceeded")
        lo, hi = self.n_chunks, self.n_chunks + k
        slo, shi = self.n_steps, self.n_steps + n_steps
        mask = (torch.arange(self.action_horizon)[None, :] < lengths[:, None]).reshape(-1)

        self.chunk_state[lo:hi] = start_window
        self.chunk_next_state[lo:hi] = next_window
        self.chunk_actions[lo:hi] = actions
        self.chunk_starts[lo:hi] = slo + torch.cumsum(lengths, 0) - lengths
        self.chunk_lengths[lo:hi] = lengths
        self.chunk_terminated[lo:hi] = terminated
        self.chunk_truncated[lo:hi] = truncated
        self.chunk_env_id[lo:hi] = torch.as_tensor(env_ids, dtype=torch.int16)

        prev_list = []
        for j, raw_e in enumerate(torch.as_tensor(env_ids).detach().cpu().tolist()):
            e = int(raw_e)
            prev_list.append(self._last_chunk.get(e, -1))
            self._last_chunk[e] = lo + j
        self.chunk_prev[lo:hi] = torch.tensor(prev_list, dtype=torch.int64)
        self.normalizer_observations[slo:shi] = frames.reshape(-1, self.obs_dim)[mask]
        self.step_rewards[slo:shi] = rewards.reshape(-1)[mask]
        self.n_chunks, self.n_steps = hi, shi


def is_torch_env(envs) -> bool:
    return callable(getattr(envs, "step_torch", None))


def take_final_observations(info, done_np, reference: torch.Tensor, strict: bool) -> torch.Tensor:
    """Replace done rows of ``reference`` with true terminal observations from ``info``."""
    rows = np.flatnonzero(done_np)
    if rows.size == 0:
        return reference
    final = info.get("final_obs") if isinstance(info, dict) else None
    if final is None and isinstance(info, dict):
        final = info.get("final_observation")
    if final is None:
        if strict:
            raise ValueError(f"SAME_STEP env reset but supplied no final observation for rows {rows.tolist()}")
        return reference

    out = reference.clone()
    n_envs = reference.shape[0]
    idx = torch.as_tensor(rows, dtype=torch.long)

    def _select(vals: torch.Tensor) -> torch.Tensor:
        if vals.shape[0] == n_envs:
            return vals[idx]
        if vals.shape[0] == rows.size:
            return vals
        raise ValueError(f"final observation has {vals.shape[0]} rows, expected {n_envs} or {rows.size}")

    if torch.is_tensor(final):
        out[idx] = _select(final.to(device=reference.device, dtype=reference.dtype))
    else:
        arr = np.asarray(final)
        if arr.dtype == object:
            for k, i in enumerate(rows):
                val = arr[i] if arr.shape[0] == n_envs else arr[k]
                if val is None:
                    if strict:
                        raise ValueError(f"final observation for done row {i} is None")
                    continue
                out[i] = torch.as_tensor(np.asarray(val, dtype=np.float32), device=reference.device)
        else:
            out[idx] = _select(torch.as_tensor(arr, device=reference.device, dtype=reference.dtype))
    return out


def collect_chunks_vec(
    envs, policy, total_steps: int, action_low, action_high, *,
    num_obs_steps: int, num_action_steps: int, buffer: RolloutBuffer,
    carry: EnvCarry | None = None, action_noise_std: float = 0.0, seed: int | None = None,
) -> RolloutBuffer:
    """Collect executed chunks from a lockstep SameStep vector env."""
    low, high = resolve_action_bounds(action_low, action_high)
    if not is_torch_env(envs):
        mode = getattr(envs, "autoreset_mode", None)
        mode_name = str(getattr(mode, "name", None) or mode or "").upper().replace("_", "")
        if mode_name != "SAMESTEP":
            raise ValueError(f"collector requires SameStep autoreset; got {mode!r}")
    if num_action_steps > num_obs_steps:
        raise ValueError(f"num_action_steps={num_action_steps} exceeds num_obs_steps={num_obs_steps}")

    n = int(envs.num_envs)
    act_dim = int(np.asarray(envs.single_action_space.shape).prod())
    base, rem = divmod(total_steps, n)
    targets = np.array([base + (i < rem) for i in range(n)], dtype=np.int64)

    if seed is None:
        seed = int(torch.randint(0, 2**31, (1,)).item())
    if carry is None:
        obs0, _ = envs.reset(seed=[seed + i for i in range(n)])
        obs0_t = obs0.detach().cpu().float() if torch.is_tensor(obs0) else torch.from_numpy(np.asarray(obs0, dtype=np.float32))
        windows = obs0_t.clone() if obs0_t.dim() == 3 else obs0_t[:, None, :].expand(n, num_obs_steps, -1).clone()
    else:
        windows = carry.windows.float().clone()

    obs_dim = windows.shape[-1]
    horizon = buffer.action_horizon
    active = np.zeros(n, dtype=bool)
    step_limits = np.zeros(n, dtype=np.int64)
    executed_total = np.zeros(n, dtype=np.int64)
    remaining = int(targets.sum())
    chunk_pos = torch.zeros(n, dtype=torch.long)
    raw_plan = torch.zeros(n, horizon, act_dim)
    exec_plan = torch.zeros(n, horizon, act_dim)
    start_window = torch.zeros(n, num_obs_steps, obs_dim)
    reward_win = torch.zeros(n, horizon)
    obs_win = torch.zeros(n, horizon, obs_dim)
    next_actions = torch.zeros(n, act_dim)
    roll_table = (torch.arange(num_obs_steps)[:, None] + torch.arange(num_obs_steps)[None, :]) % num_obs_steps
    torch_env = is_torch_env(envs)

    while remaining > 0:
        need = np.flatnonzero((~active) & (executed_total < targets))
        if need.size:
            wf = torch.as_tensor(need, dtype=torch.long)
            states = windows[wf]
            raw = policy(states).detach().cpu().float()
            raw_plan[wf] = raw
            exec_plan[wf] = to_env_action(raw, low, high)
            start_window[wf] = states
            chunk_pos[wf] = 0
            step_limits[need] = np.minimum(num_action_steps, targets[need] - executed_total[need])
            active[need] = True

        live = torch.from_numpy(np.flatnonzero(active))
        next_actions.zero_()
        if live.numel():
            next_actions[live] = exec_plan[live, chunk_pos[live].clamp(max=horizon - 1)]

        if torch_env:
            obs_r, rew_r, term_r, trunc_r, info = envs.step_torch(next_actions)
            raw_obs, rew_t = obs_r.detach().cpu().float(), rew_r.detach().cpu().float()
            term_np, trunc_np = term_r.cpu().numpy().astype(bool), trunc_r.cpu().numpy().astype(bool)
        else:
            obs_r, rew_r, term_r, trunc_r, info = envs.step(next_actions.numpy())
            raw_obs = torch.from_numpy(np.asarray(obs_r, dtype=np.float32))
            rew_t = torch.from_numpy(np.asarray(rew_r, dtype=np.float32))
            term_np, trunc_np = np.asarray(term_r, dtype=bool), np.asarray(trunc_r, dtype=bool)

        done_np = term_np | trunc_np
        reset_frames = raw_obs if done_np.any() else None
        obs_t = take_final_observations(info, done_np, raw_obs, strict=True) if done_np.any() else raw_obs

        if live.numel():
            p = chunk_pos[live]
            reward_win[live, p] = rew_t[live]
            obs_win[live, p] = obs_t[live]
            windows[live, p] = obs_t[live]
            chunk_pos[live] = p + 1

        idle = np.flatnonzero(~active)
        if idle.size:
            d_idle = done_np[idle]
            if (~d_idle).any():
                m_rows = torch.as_tensor(idle[~d_idle])
                w = windows[m_rows]
                windows[m_rows] = torch.cat([w[:, 1:], obs_t[m_rows, None]], dim=1)
            if d_idle.any():
                d_rows = torch.as_tensor(idle[d_idle])
                windows[d_rows] = reset_frames[d_rows].unsqueeze(1).expand(-1, num_obs_steps, -1)

        closing = np.flatnonzero(active & (done_np | (chunk_pos.numpy() >= step_limits)))
        if closing.size:
            rows = torch.as_tensor(closing)
            lens_np = chunk_pos.numpy()[closing]
            n_new = int(lens_np.sum())
            lengths = torch.as_tensor(lens_np, dtype=torch.long)
            perm = roll_table[lengths % num_obs_steps]
            next_win = torch.gather(windows[rows], 1, perm[:, :, None].expand(-1, -1, obs_dim))
            stored = raw_plan[rows]
            if action_noise_std > 0:
                stored = bound_for_storage(stored + action_noise_std * torch.randn_like(stored))
            buffer.write_chunks(
                start_window=start_window[rows], next_window=next_win, actions=stored,
                lengths=lengths, frames=obs_win[rows], rewards=reward_win[rows],
                terminated=torch.as_tensor(term_np)[rows], truncated=torch.as_tensor(trunc_np)[rows],
                env_ids=closing, n_steps=n_new,
            )
            windows[rows] = next_win
            executed_total[closing] += lens_np
            remaining -= n_new
            active[closing] = False
            c_done = closing[done_np[closing]]
            if c_done.size:
                cd_rows = torch.as_tensor(c_done)
                windows[cd_rows] = reset_frames[cd_rows].unsqueeze(1).expand(-1, num_obs_steps, -1)

    buffer.carry = EnvCarry(windows=windows)
    return buffer


@dataclasses.dataclass
class RolloutState:
    buffer: RolloutBuffer
    carry: EnvCarry | None = None


def make_rollout_state(obs_dim: int, action_dim: int, cfg: MyfpoConfig, total_steps_hint: int = 1) -> RolloutState:
    return RolloutState(
        buffer=RolloutBuffer(
            max_chunks=max(1, total_steps_hint), max_steps=max(1, total_steps_hint),
            obs_steps=cfg.num_obs_steps, obs_dim=obs_dim,
            action_horizon=cfg.action_horizon, action_dim=action_dim,
        )
    )


@torch.no_grad()
def _estimate_state_values(
    policy: MyfpoPolicy, raw_states: torch.Tensor, norm_states: torch.Tensor,
    exec_frac: torch.Tensor, mask: torch.Tensor, critic: nn.Module,
) -> torch.Tensor:
    cond = policy.encode_states(raw_states)
    qs = [
        critic(norm_states, to_critic_action(policy.sample_chunk(cond), mask=mask), exec_frac).float()
        for _ in range(policy.config.num_baseline_samples)
    ]
    return torch.stack(qs).mean(0)


@torch.no_grad()
def run_rollout(policy: MyfpoPolicy, env, total_steps: int, *, action_low, action_high, state: RolloutState) -> dict:
    """Collect one rollout and compute V-anchored GAE + behavior CFM losses."""
    if total_steps <= 0:
        raise ValueError(f"total_steps={total_steps} must be positive")
    cfg = policy.config
    device = next(policy.parameters()).device
    buffer = state.buffer
    buffer.reset()

    collect_chunks_vec(
        env, policy.predict_action_chunk, total_steps, action_low, action_high,
        num_obs_steps=cfg.num_obs_steps, num_action_steps=cfg.num_action_steps,
        buffer=buffer, carry=state.carry, action_noise_std=cfg.storage_action_noise_std,
    )
    state.carry = buffer.carry
    active = buffer.active_prefix()
    n_chunks = buffer.n_chunks

    states = active["chunk_state"].to(device)
    next_states = active["chunk_next_state"].to(device)
    actions = active["chunk_actions"].to(device)
    lengths = active["chunk_lengths"].to(device)

    all_norm = policy.obs_normalizer(torch.cat([states, next_states], dim=0))
    norm_states, norm_next = all_norm[:n_chunks], all_norm[n_chunks:]

    critic_was_train = policy.critic.training
    policy.critic.eval()
    policy.critic_target.eval()
    try:
        exec_frac = lengths.float() / float(cfg.action_horizon)
        boot_frac = torch.full((n_chunks,), cfg.num_action_steps / float(cfg.action_horizon), device=device)
        len_mask = (torch.arange(cfg.action_horizon, device=device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1).float()
        boot_mask = (torch.arange(cfg.action_horizon, device=device).unsqueeze(0) < cfg.num_action_steps).unsqueeze(-1).float()
        v_cur = _estimate_state_values(policy, states, norm_states, exec_frac, len_mask, policy.critic)
        v_nxt = _estimate_state_values(policy, next_states, norm_next, boot_frac, boot_mask, policy.critic_target)
    finally:
        policy.critic.train(critic_was_train)

    adv, returns = compute_chunk_advantages(
        active["step_rewards"], v_cur, v_nxt,
        active["chunk_starts"].reshape(-1), lengths.reshape(-1),
        chunk_terminated=active["chunk_terminated"].reshape(-1),
        chunk_size=cfg.action_horizon, gamma=cfg.gamma,
        chunk_truncated=active["chunk_truncated"].reshape(-1),
        chunk_prev=active["chunk_prev"], gae_lambda=cfg.gae_lambda,
    )
    if cfg.normalize_advantage and adv.numel() > 0:
        adv = (adv - adv.mean()) / adv.std(correction=0).clamp_min(1e-8)

    action_is_pad = torch.arange(cfg.action_horizon, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
    lo, hi = cfg.min_sampled_timestep, cfg.max_sampled_timestep
    t = torch.rand(n_chunks, cfg.num_mc_samples, cfg.action_horizon, 1, device=device) * (hi - lo) + lo
    eps = torch.randn(n_chunks, cfg.num_mc_samples, cfg.action_horizon, buffer.action_dim, device=device)
    beh_loss, beh_loss_tok = policy.compute_behavior_cfm_losses(norm_states, actions, t, eps, action_is_pad)

    return {
        "observation.state": states,
        "action": actions,
        "action_is_pad": action_is_pad,
        "chunk_truncated": active["chunk_truncated"],
        "normalized_advantages": adv,
        "raw_returns": returns,
        "normalizer_observations": active["normalizer_observations"],
        "boot_observations": norm_next,
        "mc_t": t,
        "mc_eps": eps,
        "behavior_cfm_loss": beh_loss,
        "behavior_cfm_loss_token": beh_loss_tok,
        "step_rewards": active["step_rewards"],
    }


# ---------------------------------------------------------------------------
# Optimizer & Update
# ---------------------------------------------------------------------------


def make_optimizer(policy: MyfpoPolicy) -> torch.optim.Optimizer:
    cfg = policy.config
    decay, no_decay = [], []
    for name, p in policy.actor.named_parameters():
        (no_decay if ("out." in name or (name.endswith(".weight") and "norm" in name.lower())) else decay).append(p)
    for name, p in policy.encoder.named_parameters():
        (no_decay if (name.endswith(".weight") and "norm" in name.lower()) else decay).append(p)
    return torch.optim.Adam(
        [
            {"params": decay, "lr": cfg.policy_lr, "weight_decay": cfg.optimizer_weight_decay},
            {"params": no_decay, "lr": cfg.policy_lr, "weight_decay": 0.0},
            {"params": list(policy.critic.parameters()), "lr": cfg.critic_lr, "weight_decay": 0.0},
        ],
        betas=tuple(cfg.adam_betas),
        eps=1e-8,
    )


@dataclasses.dataclass
class UpdateResult:
    metrics: dict[str, float]
    grad_norm_actor: float
    grad_norm_critic: float


def compute_loss(policy: MyfpoPolicy, batch: dict[str, torch.Tensor], device) -> tuple[torch.Tensor, dict[str, float]]:
    cfg = policy.config
    states = policy.obs_normalizer(torch.as_tensor(batch["observation.state"], dtype=torch.float32, device=device))
    raw_actions = torch.as_tensor(batch["action"], dtype=torch.float32, device=device)
    flow_actions = policy._to_flow_space(raw_actions)
    action_is_pad = torch.as_tensor(batch["action_is_pad"], dtype=torch.bool, device=device)
    advantages = torch.as_tensor(batch["normalized_advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch["raw_returns"], dtype=torch.float32, device=device)
    t = torch.as_tensor(batch["mc_t"], dtype=torch.float32, device=device)
    eps = torch.as_tensor(batch["mc_eps"], dtype=torch.float32, device=device)
    cfm_old_tok = torch.as_tensor(batch["behavior_cfm_loss_token"], dtype=torch.float32, device=device)

    x_t, flat_t, flow_shape, num_mc = expand_mc_samples(flow_actions, eps, t)
    cond = policy.encoder(states)
    cond_flat = cond.unsqueeze(1).expand(states.shape[0], num_mc, cond.shape[1]).reshape(-1, cond.shape[1])
    v_new = policy.compute_predicted_velocity(x_t, flat_t, cond_flat).float()

    cfm_loss_new, cfm_loss_new_tok = token_cfm_loss(
        v_new, flow_actions, eps, action_is_pad, cfg, n=states.shape[0], num_mc=num_mc, flow_shape=flow_shape
    )
    ratio = compute_policy_ratio(cfm_old_tok, cfm_loss_new_tok, cfg)
    surrogate = compute_trust_region_surrogate(ratio, advantages[:, None, None], cfg)
    keep = (~action_is_pad).to(surrogate.dtype)
    policy_loss = -((surrogate * keep.unsqueeze(1)).sum(dim=(1, 2)) / (keep.sum(dim=-1).clamp_min(1.0) * surrogate.shape[1])).mean()

    critic_actions = to_critic_action(raw_actions, mask=keep.unsqueeze(-1))
    exec_frac = keep.sum(dim=-1) / float(cfg.action_horizon)
    value_pred = policy.critic(states, critic_actions, exec_frac).float()
    value_loss = F.smooth_l1_loss(value_pred, returns.reshape(-1), reduction="mean", beta=cfg.value_loss_beta)
    loss = policy_loss + cfg.value_loss_coef * value_loss

    with torch.no_grad():
        r_var = max(returns.var(unbiased=False).item(), 1e-8)
        ratio_fp = ratio.detach()
        metrics = {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "total_loss": float(loss.item()),
            "cfm_loss_new": float(cfm_loss_new.mean().item()),
            "ratio_mean": float(ratio_fp.mean().item()),
            "ratio_std": float(ratio_fp.std(correction=0).item()),
            "clipfrac": float(((ratio_fp - 1.0).abs() > cfg.clip_param).float().mean().item()),
            "explained_var": float(1.0 - (returns - value_pred).var(unbiased=False).item() / r_var),
        }
    return loss, metrics


def update(policy: MyfpoPolicy, optimizer, batch: dict[str, Any], cfg: MyfpoConfig) -> UpdateResult:
    policy.train()
    device = next(policy.actor.parameters()).device
    n = int(batch["raw_returns"].shape[0])
    if n == 0:
        return UpdateResult(metrics={}, grad_norm_actor=0.0, grad_norm_critic=0.0)

    actor_ids = {id(p) for p in (*policy.actor.parameters(), *policy.encoder.parameters())}
    actor_params = [p for p in policy.parameters() if p.requires_grad and id(p) in actor_ids]
    critic_params = [p for p in policy.parameters() if p.requires_grad and id(p) not in actor_ids]

    # Stage training tensors on device once per rollout instead of once per minibatch.
    dev_batch = {
        "observation.state": torch.as_tensor(batch["observation.state"], dtype=torch.float32, device=device),
        "action": torch.as_tensor(batch["action"], dtype=torch.float32, device=device),
        "action_is_pad": torch.as_tensor(batch["action_is_pad"], dtype=torch.bool, device=device),
        "normalized_advantages": torch.as_tensor(batch["normalized_advantages"], dtype=torch.float32, device=device),
        "raw_returns": torch.as_tensor(batch["raw_returns"], dtype=torch.float32, device=device),
        "mc_t": torch.as_tensor(batch["mc_t"], dtype=torch.float32, device=device),
        "mc_eps": torch.as_tensor(batch["mc_eps"], dtype=torch.float32, device=device),
        "behavior_cfm_loss_token": torch.as_tensor(batch["behavior_cfm_loss_token"], dtype=torch.float32, device=device),
    }

    inds = np.arange(n)
    totals: dict[str, float] = {}
    counts = 0
    grad_norm_actor = grad_norm_critic = 0.0
    for _ in range(cfg.num_learning_epochs):
        np.random.shuffle(inds)
        for start in range(0, n, cfg.chunks_per_minibatch):
            mb = torch.as_tensor(inds[start : start + cfg.chunks_per_minibatch], dtype=torch.long, device=device)
            loss, metrics = compute_loss(policy, {k: v[mb] for k, v in dev_batch.items()}, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm_actor = float(torch.nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm))
            grad_norm_critic = float(torch.nn.utils.clip_grad_norm_(critic_params, cfg.max_grad_norm))
            optimizer.step()
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v
            counts += 1
    return UpdateResult(
        metrics={k: v / max(counts, 1) for k, v in totals.items()},
        grad_norm_actor=grad_norm_actor,
        grad_norm_critic=grad_norm_critic,
    )


# ---------------------------------------------------------------------------
# Evaluation & ManiSkill3 wrapper
# ---------------------------------------------------------------------------


def bootstrap_interval(returns: list[float], resamples: int = 10_000, low_q: float = 0.025, high_q: float = 0.975, seed: int = 0) -> tuple[float, float, float]:
    vals = np.asarray([float(r) for r in returns], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(vals.mean())
    if vals.size < 2:
        return mean, float("nan"), float("nan")
    if not 0.0 <= low_q < high_q <= 1.0 or resamples < 1:
        raise ValueError("invalid bootstrap_interval parameters")
    means = vals[np.random.default_rng(seed).integers(0, vals.size, size=(resamples, vals.size))].mean(axis=1)
    return mean, float(np.percentile(means, 100.0 * low_q)), float(np.percentile(means, 100.0 * high_q))


def compute_action_jerk(actions: np.ndarray) -> float | None:
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] < 3:
        return None
    return float(np.square(a[2:] - 2.0 * a[1:-1] + a[:-2]).mean())


def write_eval_csv(results: list[dict[str, Any]], out_dir: str | Path, *, append: bool = False) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "eval_history.csv"
    mode = "a" if append and csv_path.is_file() else "w"
    with csv_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if mode == "w":
            writer.writeheader()
        writer.writerows(results)
    return csv_path


def summarize_results(results: list[dict[str, Any]], seed: int = 0) -> dict[str, float]:
    returns = [float(r["return"]) for r in results if r.get("return") is not None]
    mean, low, high = bootstrap_interval(returns, seed=seed)
    vals = np.asarray(returns, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return {
        "return_mean": mean,
        "return_median": float(np.median(vals)) if vals.size else float("nan"),
        "return_ci_low": low,
        "return_ci_high": high,
        "return_n": float(vals.size),
    }


def compute_eval_summary_metrics(
    mean_return: float, eval_results: list[dict], all_eval_returns: list[float], *,
    r_rand: float | None = None, best_return: float | None = None, steps_to_best: int | None = None,
) -> dict:
    returns = [r["return"] for r in eval_results]
    lengths = [r["length"] for r in eval_results]
    successes = [r["success"] for r in eval_results if r.get("success") is not None]
    jerks = [r["jerk"] for r in eval_results if r.get("jerk") is not None]
    stat = lambda v, fn: float(fn(v)) if v else float("nan")

    data: dict[str, Any] = {
        "eval/return_mean": float(mean_return),
        "eval/return_min": stat(returns, np.min),
        "eval/return_max": stat(returns, np.max),
        "eval/return_std": stat(returns, np.std),
        "eval/mean_length": stat(lengths, np.mean),
        "eval/max_length": stat(lengths, np.max),
        "eval/std_length": stat(lengths, np.std),
        "eval/episode_count": len(returns),
    }
    if best_return is not None:
        data["eval/return_best"] = float(best_return)
    if steps_to_best is not None:
        data["eval/steps_to_best"] = int(steps_to_best)
    if r_rand is not None:
        data["eval/random_policy_return"] = float(r_rand)
    if len(all_eval_returns) >= 2 and all_eval_returns[-1] == mean_return:
        prev = all_eval_returns[-2]
        data["eval/return_improvement"] = float(mean_return - prev)
        data["eval/return_improvement_frac"] = float((mean_return - prev) / max(abs(prev), 1e-8))
    if successes:
        data["eval/success_rate"] = float(np.mean(successes))
    if jerks:
        data["eval/mean_jerk"] = float(np.mean(jerks))
        data["eval/max_jerk"] = float(np.max(jerks))
    return data


@contextmanager
def _pinned_eval_mode(policy: Any):
    module = getattr(policy, "policy", None)
    if not isinstance(module, torch.nn.Module):
        yield
        return
    was = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was)


class _BaseEvaluator:
    def __init__(self, eval_seed: int, max_episodes: int, max_steps: int | None) -> None:
        if max_episodes < 0 or (max_steps is not None and max_steps < 0):
            raise ValueError(f"invalid eval budget max_episodes={max_episodes}, max_steps={max_steps}")
        self.eval_seed, self.max_episodes, self.max_steps = eval_seed, max_episodes, max_steps

    def summarize(self, results: list[dict[str, Any]], seed: int = 0) -> dict[str, float]:
        return summarize_results(results, seed=seed)


class Evaluator(_BaseEvaluator):
    """Deterministic evaluator for flat-obs CPU Gymnasium envs."""

    def __init__(self, env: Any, eval_seed: int = 0, max_episodes: int = 12, max_steps: int | None = None):
        if env is None:
            raise ValueError("Evaluator requires a non-None env")
        if hasattr(env, "observation_space"):
            import gymnasium as gym
            if isinstance(env.observation_space, gym.spaces.Dict):
                raise AssertionError("Evaluator expects flat Box obs")
        self.env = env
        super().__init__(eval_seed, max_episodes, max_steps)

    def rollout_single_episode(self, policy: Callable[[np.ndarray], np.ndarray], seed: int, episode: int, step_budget: int | None) -> tuple[dict[str, Any], int]:
        obs, _ = self.env.reset(seed=seed)
        if hasattr(policy, "reset"):
            policy.reset()
        ep_return, ep_length, ep_actions, ep_success, done = 0.0, 0, [], None, False
        while not done and (step_budget is None or ep_length < step_budget):
            action = policy(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            ep_return += float(reward)
            ep_length += 1
            done = bool(terminated) or bool(truncated)
            ep_actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
            if isinstance(info, dict) and info.get("success") is not None:
                ep_success = float(bool(info["success"]))
        jerk = compute_action_jerk(np.stack(ep_actions)) if ep_actions else None
        return {"episode": episode, "return": ep_return, "length": ep_length, "success": ep_success, "jerk": jerk}, ep_length

    def evaluate(self, policy: Callable[[np.ndarray], np.ndarray], *, out_dir: str | Path | None = None, append_csv: bool = False) -> list[dict[str, Any]]:
        rng = np.random.default_rng(self.eval_seed)
        results, total_steps = [], 0
        with _pinned_eval_mode(policy):
            for ep in range(self.max_episodes):
                if self.max_steps is not None and total_steps >= self.max_steps:
                    break
                budget = None if self.max_steps is None else self.max_steps - total_steps
                res, taken = self.rollout_single_episode(policy, int(rng.integers(0, 2**31)), ep, budget)
                results.append(res)
                total_steps += taken
        if out_dir is not None:
            write_eval_csv(results, out_dir, append=append_csv)
        return results

    def clone(self, max_episodes: int) -> Evaluator:
        return Evaluator(self.env, eval_seed=self.eval_seed, max_episodes=max_episodes, max_steps=self.max_steps)

    def random_policy(self, env: Any, seed: int | None = None) -> Callable:
        if seed is not None:
            env.action_space.seed(int(seed))
        return lambda _obs: env.action_space.sample()


class VectorEvaluator(_BaseEvaluator):
    """Deterministic batched evaluator for ManiSkill GPU torch envs."""

    def __init__(self, env: Any, *, num_obs_steps: int, num_action_steps: int, action_horizon: int, eval_seed: int = 0, max_episodes: int = 12, max_steps: int | None = None):
        if env is None or num_obs_steps < 1 or not (1 <= num_action_steps <= action_horizon):
            raise ValueError("invalid VectorEvaluator configuration")
        self.env = env
        self.num_obs_steps = num_obs_steps
        self.num_action_steps = num_action_steps
        self.action_horizon = action_horizon
        super().__init__(eval_seed, max_episodes, max_steps)
        low, high = resolve_action_bounds(env.single_action_space.low, env.single_action_space.high)
        dev = torch.device(env.device) if hasattr(env, "device") else torch.device("cpu")
        self.action_low, self.action_high = low.to(dev), high.to(dev)

    def evaluate(self, policy: Any, *, out_dir: str | Path | None = None, append_csv: bool = False, video_path: str | Path | None = None, video_fps: int = 30) -> list[dict[str, Any]]:
        rng = np.random.default_rng(self.eval_seed)
        n, device = int(self.env.num_envs), torch.device(self.env.device)
        fork_devices = [device.index or 0] if device.type == "cuda" else []
        results: list[dict[str, Any]] = []
        planner = getattr(policy, "plan", None)

        with torch.random.fork_rng(devices=fork_devices), torch.inference_mode(), _pinned_eval_mode(policy):
            torch.manual_seed(self.eval_seed)
            while len(results) < self.max_episodes:
                obs, _ = self.env.reset(seed=[int(rng.integers(0, 2**31)) for _ in range(n)])
                if hasattr(policy, "reset"):
                    policy.reset()
                window = obs[:, None, :].repeat(1, self.num_obs_steps, 1)
                ep_return = torch.zeros(n, device=device)
                success = torch.zeros(n, dtype=torch.bool, device=device)
                saw_success, frames, steps = False, [], 0
                recording = video_path is not None and not results
                images = [self._grab_frame()] if recording else []
                chunk: torch.Tensor | None = None

                while True:
                    if chunk is None or steps % self.num_action_steps == 0:
                        planned = (planner(window) if planner is not None else policy(window)).to(device)
                        chunk = to_env_action(planned, self.action_low, self.action_high)
                    action = chunk[:, steps % self.num_action_steps]
                    obs, rew, term, trunc, info = self.env.step_torch(action)
                    frames.append(action)
                    if recording:
                        images.append(self._grab_frame())
                    ep_return += rew
                    if isinstance(info, dict) and "success" in info:
                        success |= info["success"]
                        saw_success = True
                    steps += 1
                    window = torch.cat((window[:, 1:], obs[:, None, :]), dim=1)
                    if bool(trunc.all()) or bool(term.all()) or (self.max_steps is not None and steps >= self.max_steps):
                        break

                if recording and images:
                    from mani_skill.utils.visualization.misc import images_to_video
                    p = Path(video_path)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    images_to_video(images, str(p.parent), p.stem, fps=video_fps, verbose=False)

                actions_np = torch.stack(frames, dim=1).cpu().numpy()
                returns_np, succ_np = ep_return.cpu().numpy(), success.cpu().numpy()
                for row in range(n):
                    if len(results) >= self.max_episodes:
                        break
                    results.append({
                        "episode": len(results),
                        "return": float(returns_np[row]),
                        "length": steps,
                        "success": float(succ_np[row]) if saw_success else None,
                        "jerk": compute_action_jerk(actions_np[row]),
                    })
        if out_dir is not None:
            write_eval_csv(results, out_dir, append=append_csv)
        return results

    def _grab_frame(self) -> np.ndarray:
        frames = np.asarray(self.env.render())
        if frames.ndim == 3:
            return frames
        from mani_skill.utils.visualization.misc import tile_images
        return tile_images(frames, nrows=max(1, int(np.sqrt(len(frames)))))

    def clone(self, max_episodes: int) -> VectorEvaluator:
        return VectorEvaluator(
            self.env, num_obs_steps=self.num_obs_steps, num_action_steps=self.num_action_steps,
            action_horizon=self.action_horizon, eval_seed=self.eval_seed, max_episodes=max_episodes, max_steps=self.max_steps,
        )

    def random_policy(self, env: Any, seed: int | None = None) -> Callable:
        return lambda _w: env.random_chunk(int(self.action_horizon))


def is_maniskill_id(env_id: str) -> bool:
    return env_id in MANISKILL_TASKS


class ManiSkillTorchEnv:
    """GPU-parallel ManiSkill 3 env conforming to the SameStep collector contract."""

    def __init__(self, alias: str, num_envs: int, *, ignore_terminations: bool | None = None, obs_mode: str = DEFAULT_OBS_MODE, control_mode: str = DEFAULT_CONTROL_MODE, render_mode: str | None = None):
        if alias not in MANISKILL_TASKS or num_envs < 1:
            raise ValueError(f"invalid ManiSkillTorchEnv({alias!r}, num_envs={num_envs})")
        import gymnasium as gym
        from gymnasium.vector import AutoresetMode
        import mani_skill.envs  # noqa: F401
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

        self.alias, self.task_id = alias, MANISKILL_TASKS[alias]
        self.obs_mode, self.control_mode, self.render_mode = obs_mode, control_mode, render_mode
        if ignore_terminations is None:
            ignore_terminations = bool(MANISKILL_TASK_OPTIONS.get(alias, {}).get("ignore_terminations", False))
        base = gym.make(self.task_id, num_envs=int(num_envs), obs_mode=obs_mode, control_mode=control_mode, render_mode=render_mode)
        self.vec = ManiSkillVectorEnv(base, auto_reset=True, ignore_terminations=ignore_terminations)
        self.num_envs = int(self.vec.num_envs)
        self.device = torch.device(self.vec.device)
        self.single_observation_space = self.observation_space = self.vec.single_observation_space
        self.single_action_space = self.action_space = self.vec.single_action_space
        self.autoreset_mode = AutoresetMode.SAME_STEP
        self.metadata = {"render_modes": []}
        self.spec = types.SimpleNamespace(id=alias)

    def reset(self, *, seed=None, options=None):
        return self.vec.reset(seed=seed, options=options)

    def step_torch(self, actions: torch.Tensor):
        return self.vec.step(actions.to(device=self.device, dtype=torch.float32))

    def step(self, actions):
        obs, rew, term, trunc, info = self.step_torch(torch.as_tensor(actions, dtype=torch.float32, device=self.device))
        return obs.cpu().numpy(), rew.cpu().numpy(), term.cpu().numpy(), trunc.cpu().numpy(), info

    def close(self) -> None:
        self.vec.close()

    def make_eval_env(self, *, num_envs: int | None = None, render_mode: str | None = None) -> ManiSkillTorchEnv:
        return ManiSkillTorchEnv(
            self.alias, self.num_envs if num_envs is None else int(num_envs),
            ignore_terminations=True, obs_mode=self.obs_mode, control_mode=self.control_mode, render_mode=render_mode,
        )

    def render(self) -> np.ndarray:
        if self.render_mode != "rgb_array":
            raise RuntimeError("render() requires render_mode='rgb_array'")
        frames = self.vec.render()
        return np.asarray(frames.detach().cpu().numpy() if torch.is_tensor(frames) else frames)

    def random_chunk(self, horizon: int) -> torch.Tensor:
        return torch.empty((self.num_envs, int(horizon), int(self.action_space.shape[0])), device=self.device).uniform_(-1.0, 1.0)


def make_maniskill_env(alias: str, num_envs: int = 1, **kwargs) -> ManiSkillTorchEnv:
    return ManiSkillTorchEnv(alias, num_envs, **kwargs)


def register_gym_envs() -> None:
    import gymnasium as gym
    for alias, task_id in MANISKILL_TASKS.items():
        def _make(tid=task_id):
            import mani_skill.envs  # noqa: F401
            from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
            return CPUGymWrapper(gym.make(tid, num_envs=1, obs_mode=DEFAULT_OBS_MODE, control_mode=DEFAULT_CONTROL_MODE))
        try:
            gym.register(id=alias, entry_point=_make)
        except gym.error.Error:
            continue


# ---------------------------------------------------------------------------
# IO, Checkpoint, and Training Loop
# ---------------------------------------------------------------------------


def resolve_device() -> torch.device:
    want = os.environ.get("MYFPO_DEVICE", "auto").lower()
    if want == "cpu":
        return torch.device("cpu")
    if want not in {"auto", "cuda"}:
        raise ValueError(f"MYFPO_DEVICE={want!r} unknown; expected 'auto', 'cpu', or 'cuda'")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(out_dir, name: str, policy: MyfpoPolicy, optimizer, global_step: int) -> str:
    ckpt = Path(out_dir) / "checkpoints" / name / "checkpoint.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "global_step": int(global_step),
        "config": policy.config.asdict(),
    }, ckpt)
    return str(ckpt)


def load_checkpoint(checkpoint_dir: str, policy: MyfpoPolicy, optimizer=None) -> int:
    p = Path(checkpoint_dir)
    ckpt = p / "checkpoint.pt" if p.is_dir() else p
    if not ckpt.is_file():
        raise FileNotFoundError(f"no checkpoint.pt at {ckpt}")
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    policy.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("global_step", 0))


def make_train_env(env, env_name: str, num_envs: int, record: dict | None = None):
    if is_torch_env(env):
        if record is not None:
            record["vector_env"] = "maniskill_gpu"
        return env
    import gymnasium as gym
    if num_envs <= 1:
        if record is not None:
            record["vector_env"] = "single"
        return gym.vector.SyncVectorEnv([lambda: env], autoreset_mode="SameStep")

    def _make_one():
        if env_name.startswith("Meta-World/"):
            import metaworld  # noqa: F401
            return gym.make("Meta-World/MT1", env_name=env_name[len("Meta-World/"):])
        if env_name.startswith("AdroitHand"):
            import gymnasium_robotics  # noqa: F401
        return gym.make(env_name)

    env_fns = [_make_one for _ in range(num_envs)]
    if env_name.startswith("Meta-World/"):
        return gym.vector.SyncVectorEnv(env_fns, autoreset_mode="SameStep")
    try:
        return gym.vector.AsyncVectorEnv(env_fns, context="forkserver", autoreset_mode="SameStep")
    except (ValueError, RuntimeError, ImportError):
        return gym.vector.SyncVectorEnv(env_fns, autoreset_mode="SameStep")


class _EvalAdapter:
    def __init__(self, policy: MyfpoPolicy):
        self.policy = policy

    def reset(self) -> None:
        self.policy.reset()

    def __call__(self, obs) -> np.ndarray:
        return np.atleast_1d(self.policy.select_action(obs).detach().cpu().numpy()).astype(np.float32)

    def plan(self, window) -> torch.Tensor:
        return self.policy.predict_action_chunk(window)


def _init_wandb(cfg: MyfpoConfig, *, method: str, env_name: str, seed: int, total_steps: int):
    if cfg.report_to == "none":
        return None
    import wandb
    cfg_dict = cfg.asdict()
    tags = [
        f"h{cfg.action_horizon}", f"hd{cfg.hidden_dim}", f"ep{cfg.num_learning_epochs}",
        f"lr{cfg.policy_lr:g}", f"eps{cfg.clip_param:g}",
    ]
    name = "_".join([f"{method}_{env_name}_s{seed}", *tags, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")])
    wandb.init(
        project=method, group=f"{method}_{env_name}", name=name, tags=["mujoco"],
        config={**cfg_dict, "method": method, "seed": seed, "env": env_name},
        job_type="smoke" if total_steps <= 10000 else "train",
        mode=os.environ.get("WANDB_MODE", "online"), reinit="finish_previous",
    )
    return wandb


def run_myfpo_training(
    env, out_dir, cfg: MyfpoConfig | None = None, *, seed: int = 3407, total_steps: int = 2000,
    eval_steps: int = 1000, save_steps: int = 10**9, obs_dim: int | None = None,
    action_dim: int | None = None, resume_from_checkpoint: str | None = None,
    env_id: str | None = None, runtime: dict | None = None,
) -> dict:
    """Run one myfpo training loop."""
    cfg = cfg or MyfpoConfig()
    obs_dim = obs_dim or int(np.prod(env.observation_space.shape))
    action_dim = action_dim or int(np.prod(env.action_space.shape))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = getattr(getattr(env, "unwrapped", env), "spec", None)
    env_name = env_id or (getattr(spec, "id", None) if spec is not None else "unknown") or "unknown"

    set_seed(seed)
    device = resolve_device()
    train_env = make_train_env(env, env_name, cfg.num_envs, record=runtime)
    if runtime is not None:
        (out_dir / "run_metadata.json").write_text(json.dumps(runtime, indent=2, sort_keys=True), encoding="utf-8")

    low = np.atleast_1d(np.asarray(env.action_space.low, dtype=np.float32))
    high = np.atleast_1d(np.asarray(env.action_space.high, dtype=np.float32))
    policy = MyfpoPolicy(cfg, obs_dim, action_dim, low, high).to(device)
    optimizer = make_optimizer(policy)
    if resume_from_checkpoint is not None:
        step = load_checkpoint(resume_from_checkpoint, policy, optimizer)
        print(f"myfpo: resumed from {resume_from_checkpoint} at step {step}", flush=True)

    eval_seed = 10000 + seed
    if hasattr(train_env, "make_eval_env"):
        eval_env = train_env.make_eval_env()
        evaluator = VectorEvaluator(
            eval_env, num_obs_steps=cfg.num_obs_steps, num_action_steps=cfg.num_action_steps,
            action_horizon=cfg.action_horizon, eval_seed=eval_seed, max_episodes=cfg.eval_max_episodes,
        )
    else:
        eval_env = env
        evaluator = Evaluator(env, eval_seed=eval_seed, max_episodes=cfg.eval_max_episodes)

    rand_results = evaluator.clone(max_episodes=30).evaluate(evaluator.random_policy(evaluator.env), out_dir=None)
    r_rand = float(np.mean([r["return"] for r in rand_results])) if rand_results else 0.0
    wb = _init_wandb(cfg, method="myfpo", env_name=env_name, seed=seed, total_steps=total_steps)
    artifacts: list[str] = []
    params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"myfpo policy: actor=FlowMLP hidden={cfg.hidden_dim} params={params:,}", flush=True)

    rollout_steps = max(1, int(cfg.chunks_per_update) * int(cfg.action_horizon))
    rollout_state = make_rollout_state(obs_dim, action_dim, cfg, total_steps_hint=max(1, min(total_steps, rollout_steps)))
    best_return: float | None = None
    steps_to_best = 0
    all_eval_returns: list[float] = []
    next_eval, next_save = eval_steps, save_steps
    t_start, elapsed, global_step = time.time(), 0.0, 0

    while global_step < total_steps:
        t0 = time.perf_counter()
        batch = run_rollout(policy, train_env, min(rollout_steps, total_steps - global_step), action_low=low, action_high=high, state=rollout_state)
        collect_s = time.perf_counter() - t0
        result = update(policy, optimizer, batch, cfg)
        global_step += int(batch["step_rewards"].numel())
        policy.obs_normalizer.update(batch["normalizer_observations"])
        with torch.no_grad():
            for pt, p in zip(policy.critic_target.parameters(), policy.critic.parameters()):
                pt.copy_(0.005 * p + 0.995 * pt)

        if wb is not None:
            acts = batch["action"]
            wb.log({
                **{f"train/{k}": v for k, v in result.metrics.items()},
                "train/grad_norm_actor_before_clip": result.grad_norm_actor,
                "train/grad_norm_critic_before_clip": result.grad_norm_critic,
                "train/grad_norm_before_clip": math.hypot(result.grad_norm_actor, result.grad_norm_critic),
                "train/collect_s": collect_s,
                "train/policy_lr": optimizer.param_groups[0]["lr"],
                "train/critic_lr": optimizer.param_groups[-1]["lr"],
                "train/executed_action_abs_mean": float(acts.abs().mean().item()) if acts.numel() else 0.0,
                "train/executed_action_abs_max": float(acts.abs().max().item()) if acts.numel() else 0.0,
            }, step=global_step)

        if global_step >= next_eval or global_step >= total_steps:
            eval_results = evaluator.evaluate(_EvalAdapter(policy), out_dir=out_dir, append_csv=True)
            summary = summarize_results(eval_results, seed=eval_seed)
            mean_ret = float(summary["return_mean"])
            all_eval_returns.append(mean_ret)
            if best_return is None or mean_ret > best_return:
                best_return, steps_to_best = mean_ret, global_step
                best_ckpt = save_checkpoint(out_dir, "best", policy, optimizer, global_step)
                if best_ckpt not in artifacts:
                    artifacts.append(best_ckpt)
            if wb is not None:
                wb.log({
                    **compute_eval_summary_metrics(mean_ret, eval_results, all_eval_returns, r_rand=r_rand, best_return=best_return, steps_to_best=steps_to_best),
                    "eval/return_ci_low": summary["return_ci_low"],
                    "eval/return_ci_high": summary["return_ci_high"],
                    "eval/return_ci_width": summary["return_ci_high"] - summary["return_ci_low"],
                    "eval/return_median": summary["return_median"],
                    "eval/return_n": summary["return_n"],
                }, step=global_step)
            next_eval = max(global_step + eval_steps, next_eval + eval_steps)
            sub_envs = getattr(train_env, "envs", None)
            if (any(e is env for e in sub_envs) if sub_envs is not None else train_env is env):
                rollout_state.carry = None

        if global_step >= next_save:
            ckpt = save_checkpoint(out_dir, f"step_{global_step}", policy, optimizer, global_step)
            if ckpt not in artifacts:
                artifacts.append(ckpt)
            next_save = max(global_step + save_steps, next_save + save_steps)
        elapsed = time.time() - t_start

    fps = global_step / elapsed if elapsed > 0.0 else 0.0
    final_ckpt = save_checkpoint(out_dir, "final", policy, optimizer, global_step)
    if final_ckpt not in artifacts:
        artifacts.append(final_ckpt)
    if (out_dir / "eval_history.csv").exists():
        artifacts.append(str(out_dir / "eval_history.csv"))
    if wb is not None:
        wb.log({"timing/fps": fps, "timing/elapsed_s": elapsed}, step=global_step)
        wb.finish(exit_code=0)
    if train_env is not env:
        train_env.close()
    if eval_env is not env and eval_env is not train_env and hasattr(eval_env, "close"):
        eval_env.close()

    return {
        "method": "myfpo",
        "env_name": env_name,
        "seed": seed,
        "reward_mean_last10": float(np.mean(all_eval_returns)) if all_eval_returns else 0.0,
        "reward_median_last10": float(np.median(all_eval_returns)) if all_eval_returns else 0.0,
        "fps": fps,
        "artifacts": artifacts,
        "best_artifact": next((a for a in artifacts if "best" in a), None),
        "best_return": best_return,
        "global_step": global_step,
    }