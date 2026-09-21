"""NNX mirrors of ``sjfpo.networks`` (same shapes, same math, no torch).

Constructor args match the torch classes (compat ignored args stay accepted).
Defaults are literals mirroring ``SJFPOConfig`` so this module never imports
torch. ``load_torch_weights`` copies a torch state_dict into an NNX model
(Linear kernels are transposed) for forward-parity checks.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

__all__ = [
    "SwiGLU",
    "CondMLP",
    "SinusoidalEncoding",
    "SelfAttention",
    "RMSNorm",
    "WindowEncoder",
    "RunningMeanStd",
    "modulate",
    "DiTBlock",
    "StateConditioner",
    "SJFPOFlowModel",
    "Critic",
    "load_torch_weights",
]


class SwiGLU(nnx.Module):
    """Gated SiLU feedforward: down(silu(gate(x)) * up(x))."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        self.gate = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.up = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.down = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down(jax.nn.silu(self.gate(x)) * self.up(x))


class CondMLP(nnx.Module):
    """SiLU MLP for conditioning regression (DiT canon)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(jax.nn.silu(self.fc1(x)))


class SinusoidalEncoding(nnx.Module):
    """Fixed sinusoidal codes (stateless: dim/scale are static attrs)."""

    def __init__(self, dim: int, scale: float = 1.0):
        if dim <= 0:
            raise ValueError(f"dim={dim} must be positive")
        self.dim = dim
        self.scale = float(scale)

    def frequencies(self, half: int) -> jax.Array:
        return jnp.exp(
            -math.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / max(half - 1, 1)
        )

    def add_to(self, x: jax.Array, positions: jax.Array) -> jax.Array:
        ang = (positions.astype(jnp.float32) * self.scale)[..., None] * self.frequencies(
            self.dim // 2
        )
        pe = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
        if pe.shape[-1] < self.dim:
            pe = jnp.pad(pe, ((0, 0),) * (pe.ndim - 1) + ((0, self.dim - pe.shape[-1]),))
        return x + pe.astype(x.dtype)

    def encode_times(self, t: jax.Array) -> jax.Array:
        t = jnp.asarray(t, dtype=jnp.float32) * self.scale
        half = self.dim // 2
        ang = t[..., None] * self.frequencies(half)
        emb = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
        if self.dim % 2:
            emb = jnp.pad(emb, ((0, 0),) * (emb.ndim - 1) + ((0, 1),))
        return emb

    def positions(self, length: int) -> jax.Array:
        return jnp.arange(length, dtype=jnp.float32)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.add_to(x, self.positions(x.shape[-2]))


class RMSNorm(nnx.Module):
    """x / sqrt(mean(x²) + eps) * weight (matches ``nn.RMSNorm``)."""

    def __init__(self, hidden_dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        self.weight = nnx.Param(jnp.ones((hidden_dim,), dtype=jnp.float32))
        self.eps = eps
        del rngs  # weight is ones; rngs accepted for uniform call sites

    def __call__(self, x: jax.Array) -> jax.Array:
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        return (x * jax.lax.rsqrt(var + self.eps)).astype(x.dtype) * self.weight.value


class SelfAttention(nnx.Module):
    """Multi-head self-attention (softmax math identical to SDPA, no dropout)."""

    def __init__(self, hidden_dim: int, n_heads: int = 4, *, rngs: nnx.Rngs):
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must split over {n_heads} heads")
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qkv = nnx.Linear(hidden_dim, 3 * hidden_dim, rngs=rngs)
        self.out = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)

    def __call__(self, x: jax.Array, key_mask: jax.Array | None = None) -> jax.Array:
        b, l, _ = x.shape
        q, k, v = jnp.split(self.qkv(x), 3, axis=-1)
        q = q.reshape(b, l, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, l, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, l, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(self.head_dim)
        if key_mask is not None:
            # Same convention as torch: True = ignore as attention key.
            ignore = jnp.asarray(key_mask, dtype=bool).reshape(b, 1, 1, l)
            scores = jnp.where(ignore, jnp.finfo(scores.dtype).min, scores)
        y = jax.nn.softmax(scores, axis=-1) @ v
        y = y.transpose(0, 2, 1, 3).reshape(b, l, -1)
        return self.out(y)


class WindowEncoder(nnx.Module):
    """One pre-norm residual block over a time window -> Z[B, L, hidden]."""

    def __init__(
        self,
        in_dim: int,
        n_obs_steps: int | None = None,
        hidden_dim: int | None = None,
        n_heads: int = 4,
        mlp_ratio: float | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        hidden_dim = 128 if hidden_dim is None else hidden_dim
        mlp_ratio = 4.0 if mlp_ratio is None else float(mlp_ratio)
        del n_obs_steps  # accepted for call-site compat
        self.frame_proj = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.pos_encoding = SinusoidalEncoding(hidden_dim)
        self.norm1 = RMSNorm(hidden_dim, eps=1e-6, rngs=rngs)
        self.attn1 = SelfAttention(hidden_dim, n_heads, rngs=rngs)
        self.norm2 = RMSNorm(hidden_dim, eps=1e-6, rngs=rngs)
        self.mlp1 = SwiGLU(hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim, rngs=rngs)

    def __call__(self, x: jax.Array, key_mask: jax.Array | None = None) -> jax.Array:
        h = self.pos_encoding(self.frame_proj(x))
        if key_mask is not None:
            key_mask = jnp.asarray(key_mask, dtype=bool)
        n = self.norm1(h)
        h = h + self.attn1(n, key_mask)
        return h + self.mlp1(self.norm2(h))


class RunningMeanStd(nnx.Module):
    """Per-dim running stats; Welford update, frozen-stat normalize."""

    def __init__(self, dim: int, eps: float = 1e-8, clip: float = 10.0):
        if dim <= 0:
            raise ValueError(f"dim={dim} must be positive")
        self.mean = nnx.Variable(jnp.zeros((dim,)))
        self.var = nnx.Variable(jnp.ones((dim,)))
        self.count = nnx.Variable(jnp.float32(0.0))
        self.eps = eps
        self.clip = clip

    def update(self, x: jax.Array) -> None:
        batch = jnp.asarray(x).reshape(-1, x.shape[-1])
        b = batch.shape[0]
        if b == 0:
            return
        b_mean = batch.mean(axis=0)
        b_var = batch.var(axis=0)
        delta = b_mean - self.mean.value
        total = self.count.value + b
        self.mean.value = self.mean.value + delta * b / total
        self.var.value = (
            self.count.value * self.var.value
            + b * b_var
            + jnp.square(delta) * self.count.value * b / total
        ) / total
        self.count.value = total

    def __call__(self, x: jax.Array) -> jax.Array:
        if float(self.count.value) == 0.0:
            return x
        norm = (x - self.mean.value) * jax.lax.rsqrt(self.var.value + self.eps)
        return jnp.clip(norm, -self.clip, self.clip)


def modulate(x: jax.Array, shift: jax.Array, scale: jax.Array) -> jax.Array:
    """adaLN modulation: x * (1 + scale) + shift."""
    return x * (1.0 + scale) + shift


class DiTBlock(nnx.Module):
    """Pre-norm DiT block with adaLN-Zero conditioning (zero-init modulation)."""

    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float, *, rngs: nnx.Rngs):
        self.norm1 = RMSNorm(hidden_dim, rngs=rngs)
        self.attn = SelfAttention(hidden_dim, n_heads, rngs=rngs)
        self.norm2 = RMSNorm(hidden_dim, rngs=rngs)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = SwiGLU(hidden_dim, mlp_hidden, hidden_dim, rngs=rngs)
        self.adaLN = CondMLP(hidden_dim, hidden_dim, 6 * hidden_dim, rngs=rngs)
        self.adaLN.fc2.kernel.value = jnp.zeros_like(self.adaLN.fc2.kernel.value)
        self.adaLN.fc2.bias.value = jnp.zeros_like(self.adaLN.fc2.bias.value)
        self.last_gate_norms = nnx.data(None)

    def __call__(self, x: jax.Array, c: jax.Array) -> jax.Array:
        parts = jnp.split(self.adaLN(c)[:, None, :], 6, axis=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = parts
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(h)
        out = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        self.last_gate_norms = nnx.data({
            "gate_msa": jnp.sqrt(jnp.mean(jnp.square(gate_msa))),
            "gate_mlp": jnp.sqrt(jnp.mean(jnp.square(gate_mlp))),
            "scale_msa": jnp.sqrt(jnp.mean(jnp.square(1.0 + scale_msa))),
            "scale_mlp": jnp.sqrt(jnp.mean(jnp.square(1.0 + scale_mlp))),
        })
        return out


class StateConditioner(nnx.Module):
    """Fuse raw state window + DCT bins into one [B, D] vector."""

    def __init__(self, obs_dim: int, hidden_dim: int | None = None, n_heads: int = 4, *, rngs: nnx.Rngs):
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        hidden_dim = 128 if hidden_dim is None else int(hidden_dim)
        del n_heads  # accepted for call-site compat; fusion is linear
        self.obs_dim = obs_dim
        self.state_encoder = WindowEncoder(obs_dim, hidden_dim=hidden_dim, rngs=rngs)
        self.bin_norm = RMSNorm(obs_dim, rngs=rngs)
        self.bin_encoding = SinusoidalEncoding(obs_dim)
        self.key_proj = nnx.Linear(obs_dim, hidden_dim, rngs=rngs)
        self.fuse = nnx.Linear(2 * hidden_dim, hidden_dim, rngs=rngs)

    def __call__(self, states: jax.Array) -> jax.Array:
        from sjfpo_jax.dct import dct

        raw = self.state_encoder(states).mean(axis=1)
        spec = self.key_proj(self.bin_encoding(self.bin_norm(dct(states, dim=1)))).mean(axis=1)
        return self.fuse(jnp.concatenate([raw, spec], axis=-1))


class SJFPOFlowModel(nnx.Module):
    """DiT velocity model over noisy-action tokens, conditioned via adaLN."""

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int | None = None,
        dit_depth: int | None = None,
        dit_heads: int | None = None,
        dit_mlp_ratio: float | None = None,
        dit_time_dim: int | None = None,
        timestep_scale: float = 1000.0,
        *,
        rngs: nnx.Rngs,
    ):
        hidden_dim = 128 if hidden_dim is None else int(hidden_dim)
        dit_depth = 4 if dit_depth is None else int(dit_depth)
        dit_heads = 4 if dit_heads is None else int(dit_heads)
        dit_mlp_ratio = 4.0 if dit_mlp_ratio is None else float(dit_mlp_ratio)
        dit_time_dim = 128 if dit_time_dim is None else int(dit_time_dim)
        self.time_dim = dit_time_dim
        self.timestep_embedder = SinusoidalEncoding(self.time_dim, scale=timestep_scale)
        self.in_proj = nnx.Linear(action_dim, hidden_dim, rngs=rngs)
        self.bin_encoding = SinusoidalEncoding(hidden_dim)
        self.time_mlp = CondMLP(dit_time_dim, 4 * hidden_dim, hidden_dim, rngs=rngs)
        self.blocks = nnx.List(
            [DiTBlock(hidden_dim, dit_heads, dit_mlp_ratio, rngs=rngs) for _ in range(dit_depth)]
        )
        self.norm_out = RMSNorm(hidden_dim, rngs=rngs)
        self.adaLN_out = nnx.Linear(hidden_dim, 2 * hidden_dim, rngs=rngs)
        self.adaLN_out.kernel.value = jnp.zeros_like(self.adaLN_out.kernel.value)
        self.adaLN_out.bias.value = jnp.zeros_like(self.adaLN_out.bias.value)
        self.out = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        self.out.kernel.value = jnp.zeros_like(self.out.kernel.value)
        self.out.bias.value = jnp.zeros_like(self.out.bias.value)
        self.last_cond_stats = nnx.data(None)

    def __call__(self, x_t: jax.Array, t: jax.Array, condition: jax.Array) -> jax.Array:
        temb = self.timestep_embedder.encode_times(t).astype(x_t.dtype)
        if temb.ndim == 1:
            temb = jnp.broadcast_to(temb, (x_t.shape[0], self.time_dim))
        c = self.time_mlp(temb) + condition.astype(x_t.dtype)
        h = self.bin_encoding(self.in_proj(x_t))
        for block in self.blocks:
            h = block(h, c)
        self.last_cond_stats = nnx.data({
            "temb": jnp.sqrt(jnp.mean(jnp.square(temb))),
            "fused": jnp.sqrt(jnp.mean(jnp.square(condition))),
            "c": jnp.sqrt(jnp.mean(jnp.square(c))),
            **{
                k: jnp.mean(jnp.stack([b.last_gate_norms[k] for b in self.blocks]))
                for k in ("gate_msa", "gate_mlp", "scale_msa", "scale_mlp")
            },
        })
        shift_out, scale_out = jnp.split(self.adaLN_out(c)[:, None, :], 2, axis=-1)
        return self.out(modulate(self.norm_out(h), shift_out, scale_out))


class Critic(nnx.Module):
    """V(s) MLP on the current frame: [B, T, obs] -> [B]."""

    def __init__(self, obs_dim: int, hidden_dim: int | None = None, n_obs_steps: int | None = None, *, rngs: nnx.Rngs):
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        hidden_dim = 128 if hidden_dim is None else int(hidden_dim)
        del n_obs_steps  # accepted for call-site compat
        self.mlp = CondMLP(obs_dim, hidden_dim, 1, rngs=rngs)

    def __call__(self, states: jax.Array) -> jax.Array:
        return self.mlp(states[:, -1, :]).squeeze(-1)


def load_torch_weights(nnx_model: nnx.Module, torch_state: dict) -> None:
    """Copy a torch state_dict into an NNX model (Linear kernels transposed).

    ``torch_state`` keys use the torch module names (e.g.
    ``blocks.0.attn.qkv.weight``); the NNX tree must mirror that structure.
    """
    import numpy as np

    graphdef, params, rest = nnx.split(nnx_model, nnx.Param, ...)
    flat_torch = {k: np.asarray(v) for k, v in torch_state.items()}
    # Walk Param leaves with their torch counterpart paths.
    leaves = list(nnx.traversals.flatten_mapping(params).items())
    for path, leaf in leaves:
        dotted = ".".join(str(p) for p in path)
        # NNX Linear stores kernel/bias; torch stores weight/bias.
        if dotted.endswith(".kernel"):
            w = flat_torch.get(dotted[: -len(".kernel")] + ".weight")
            if w is not None:
                leaf.value = jnp.asarray(w.T if w.ndim == 2 else w)
        elif dotted.endswith(".bias"):
            b = flat_torch.get(dotted)
            if b is not None:
                leaf.value = jnp.asarray(b)
        else:  # RMSNorm weight / plain vectors share names.
            w = flat_torch.get(dotted)
            if w is not None:
                leaf.value = jnp.asarray(w)
    nnx.update(nnx_model, params)
