"""Networks (JAX/Flax): window encoders, conditioner, flow heads, V(s) critic.

Everything is parameterized with ``flax.linen``. Static hyperparameters come
from :class:`SJFPOConfig` (via ``resolve_config_value``), matching the PyTorch
port's module structure 1:1.
"""

from __future__ import annotations

import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from sjfpo_jax.configuration_sjfpo import SJFPOConfig
from sjfpo_jax.dct_utils import dct
from sjfpo_jax.utils import resolve_storage_dtype

STORAGE_DTYPE = resolve_storage_dtype()

_DEFAULTS: SJFPOConfig | None = None


def _defaults() -> SJFPOConfig:
    global _DEFAULTS
    if _DEFAULTS is None:
        _DEFAULTS = SJFPOConfig()
    return _DEFAULTS


def resolve_config_value(name: str, override: Any) -> Any:
    return override if override is not None else getattr(_defaults(), name)


# ---------------------------------------------------------------------------
# Norm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        d = x.shape[-1]
        w = self.param("weight", nn.initializers.ones, (d,), x.dtype)
        var = jnp.mean(x * x, axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(var + self.eps) * w


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        # Fused gate+up GEMM (matches torch port).
        gu = nn.Dense(2 * self.hidden_dim)(x)
        g, u = jnp.split(gu, 2, axis=-1)
        return nn.Dense(self.out_dim)(nn.silu(g) * u)


class CondMLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.silu(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.out_dim)(x)


def _freqs(half: int) -> jnp.ndarray:
    i = jnp.arange(half, dtype=jnp.float32)
    return jnp.exp(-math.log(10000.0) * i / max(half - 1, 1))


class SinusoidalEncoding(nn.Module):
    """Fixed sinusoidal codes (no learnable parameters)."""

    dim: int
    scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        """Add slot codes by position index (unused; kept for API parity)."""
        L = x.shape[-2]
        pos = jnp.arange(L, dtype=jnp.float32)
        return self.add_to(x, pos)

    def add_to(self, x, positions):
        half = self.dim // 2
        ang = (positions.astype(jnp.float32) * self.scale).reshape(
            *positions.shape, 1
        ) * _freqs(half)
        pe = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
        if pe.shape[-1] < self.dim:
            pe = jnp.pad(pe, [(0, 0)] * (pe.ndim - 1) + [(0, self.dim - pe.shape[-1])])
        return x + pe.astype(x.dtype)

    def encode_times(self, t):
        """Sinusoidal embedding for scalar / [B] timesteps."""
        half = self.dim // 2
        t = jnp.asarray(t, dtype=jnp.float32) * self.scale
        ang = t.reshape(*t.shape, 1) * _freqs(half)
        emb = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
        if self.dim % 2:
            emb = jnp.pad(emb, [(0, 0)] * (emb.ndim - 1) + [(0, 1)])
        return emb


class SelfAttention(nn.Module):
    hidden_dim: int
    n_heads: int = 4

    @nn.compact
    def __call__(self, x, key_mask=None):
        B, L, _ = x.shape
        H = self.n_heads
        D = self.hidden_dim // H
        qkv = nn.Dense(3 * self.hidden_dim)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        # [B, L, H, D] -> [B, H, L, D]
        q = q.reshape(B, L, H, D).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, H, D).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, H, D).transpose(0, 2, 1, 3)

        mask = None
        if key_mask is not None:
            keep = (~jnp.asarray(key_mask, dtype=bool))
            # Force at least one keep to avoid NaN rows (matches torch port).
            keep = keep.at[..., 0].set(True)
            # SDPA mask shape [B, 1, 1, L_k] broadcast.
            mask = keep[None, None, None, :]

        y = jax.nn.dot_product_attention(q, k, v, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(B, L, H * D)
        return nn.Dense(self.hidden_dim)(y)


class WindowEncoder(nn.Module):
    in_dim: int
    hidden_dim: int
    n_heads: int = 4
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, key_mask=None):
        h = SinusoidalEncoding(self.hidden_dim)(nn.Dense(self.hidden_dim)(x))
        h = h + SelfAttention(self.hidden_dim, self.n_heads)(RMSNorm()(h), key_mask)
        mlp_hidden = int(self.hidden_dim * self.mlp_ratio)
        h = h + SwiGLU(mlp_hidden, self.hidden_dim)(RMSNorm()(h))
        return h


# ---------------------------------------------------------------------------
# Running stats
# ---------------------------------------------------------------------------


def normalize_with_stats(x: jnp.ndarray, stats: dict, eps: float = 1e-8, clip: float = 10.0) -> jnp.ndarray:
    """Normalize ``x`` with a running-stats dict; identity until first update.

    Pure function (no Flax module) because there are no learnable parameters
    here — the stats live in the params pytree and are threaded explicitly.
    Mirrors ``SJFPOPolicy.obs_normalizer`` from the torch port.
    """
    if stats is None:
        return x
    mean = stats["mean"]
    var = stats["var"]
    norm = (x - mean) * jax.lax.rsqrt(var + eps)
    norm = jnp.clip(norm, -clip, clip).astype(x.dtype)
    return jnp.where(stats["uninitialized"], x, norm)

def running_mean_std_update(stats, x):
    """Pure-functional update; returns the new stats tree."""
    batch = x.reshape(-1, x.shape[-1]).astype(jnp.float32)
    b = batch.shape[0]
    if b == 0:
        return stats
    b_mean = batch.mean(0)
    b_var = batch.var(0)
    delta = b_mean - stats["mean"]
    total = stats["count"] + b
    new_mean = stats["mean"] + (delta * b / total).astype(stats["mean"].dtype)
    new_var = (
        stats["count"] * stats["var"]
        + b * b_var
        + delta ** 2 * (stats["count"] * b / total)
    ) / total
    return {
        "mean": new_mean,
        "var": new_var.astype(stats["var"].dtype),
        "count": total,
        "uninitialized": jnp.asarray(False),
    }


# ---------------------------------------------------------------------------
# DiT
# ---------------------------------------------------------------------------


def modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


class DiTBlock(nn.Module):
    hidden_dim: int
    n_heads: int
    mlp_ratio: float

    @nn.compact
    def __call__(self, x, c, return_stats=False):
        B = c.shape[0]
        parts = CondMLP(self.hidden_dim, 6 * self.hidden_dim)(c).reshape(B, 6, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            parts[:, i] for i in range(6)
        ]

        h = modulate(RMSNorm()(x), shift_msa[:, None], scale_msa[:, None])
        x = x + gate_msa[:, None] * SelfAttention(self.hidden_dim, self.n_heads)(h)

        mlp_hidden = int(self.hidden_dim * self.mlp_ratio)
        out = x + gate_mlp[:, None] * SwiGLU(mlp_hidden, self.hidden_dim)(
            modulate(RMSNorm()(x), shift_mlp[:, None], scale_mlp[:, None])
        )

        if not return_stats:
            return out
        stats = {
            "gate_msa": jnp.sqrt((gate_msa ** 2).mean()),
            "gate_mlp": jnp.sqrt((gate_mlp ** 2).mean()),
            "scale_msa": jnp.sqrt(((1.0 + scale_msa) ** 2).mean()),
            "scale_mlp": jnp.sqrt(((1.0 + scale_mlp) ** 2).mean()),
        }
        return out, stats


class StateConditioner(nn.Module):
    """Fuse mean-pooled per-frame state with DCT-bin features."""

    obs_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, states):
        # Raw path: MLP + mean-pool.
        h = nn.Dense(self.hidden_dim)(states)
        h = SwiGLU(4 * self.hidden_dim, self.hidden_dim)(h)
        raw = h.mean(axis=1)

        # Spectral path: RMSNorm → sinus encode → Linear → mean.
        spec_in = RMSNorm()(states)
        spec_in = SinusoidalEncoding(self.obs_dim)(dct(spec_in, dim=1))
        spec = nn.Dense(self.hidden_dim)(spec_in).mean(axis=1)

        fused = jnp.concatenate([raw, spec], axis=-1)
        return nn.Dense(self.hidden_dim)(fused)


class SJFPOFlowModel(nn.Module):
    """DiT velocity model over noisy-action tokens, adaLN conditioned."""

    action_dim: int
    hidden_dim: int
    dit_depth: int
    dit_heads: int
    dit_mlp_ratio: float
    dit_time_dim: int
    timestep_scale: float

    @nn.compact
    def __call__(self, x_t, t, condition, return_stats=False):
        # [B, Ka, D_action] -> [B, Ka, hidden]
        h = nn.Dense(self.hidden_dim)(x_t)

        # Learned token positional embedding.
        token_pos = self.param(
            "token_pos",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.hidden_dim),
        )
        h = h + token_pos

        # Time embedding -> conditioning.
        temb = SinusoidalEncoding(self.dit_time_dim, self.timestep_scale).encode_times(t)
        c = CondMLP(4 * self.hidden_dim, self.hidden_dim)(temb) + condition

        blocks = [
            DiTBlock(self.hidden_dim, self.dit_heads, self.dit_mlp_ratio)
            for _ in range(self.dit_depth)
        ]

        block_stats = []
        if not return_stats:
            for block in blocks:
                h = block(h, c)
        else:
            for block in blocks:
                h, s = block(h, c, return_stats=True)
                block_stats.append(s)

        shift_out, scale_out = jnp.split(
            nn.Dense(2 * self.hidden_dim)(c), 2, axis=-1
        )
        out = nn.Dense(self.action_dim)(
            modulate(RMSNorm()(h), shift_out[:, None], scale_out[:, None])
        )

        if not return_stats:
            return out
        stats = {
            "temb": jnp.sqrt((temb ** 2).mean()),
            "fused": jnp.sqrt((condition ** 2).mean()),
            "c": jnp.sqrt((c ** 2).mean()),
            **{
                k: jnp.mean(jnp.stack([s[k] for s in block_stats]))
                for k in ("gate_msa", "gate_mlp", "scale_msa", "scale_mlp")
            },
        }
        return out, stats


class Critic(nn.Module):
    """V(s): MLP on the last frame of a state window."""

    obs_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, states):
        return CondMLP(self.hidden_dim, 1)(states[:, -1, :]).squeeze(-1)