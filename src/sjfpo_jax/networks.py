"""NNX mirrors of ``sjfpo.networks`` (same shapes, same math, no torch).

Constructor args match the torch classes (compat ignored args stay accepted).
Defaults are literals mirroring ``SJFPOConfig`` so this module never imports
torch. ``load_torch_weights`` copies a torch state_dict into an NNX model
(Linear kernels are transposed) for forward-parity checks.

Performance notes
-----------------
* Attention uses ``jax.nn.dot_product_attention`` so cuDNN / Pallas FlashAttention
  kernels can dispatch on GPU / TPU.
* Sinusoidal frequency and positional tables are cached per module instance;
  they become compile-time constants once the module is traced.
* SwiGLU gate/up projections are fused into a single GEMM.
* ``DiTBlock`` / ``SJFPOFlowModel`` collect diagnostics only when explicitly
  requested (``return_stats=True``) — no reductions on the training path.
* ``StateConditioner`` builds a cached orthonormal DCT-II matrix and does DCT
  via a single matmul.
* All small math (RMSNorm, modulate) is written to avoid gratuitous casts and
  intermediate allocations.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from sjfpo_jax.dtypes import resolve_param_dtype

try:  # Fused SDPA is available on all recent JAX versions.
    from jax.nn import dot_product_attention as _sdpa
except ImportError:  # pragma: no cover - very old JAX fallback
    _sdpa = None

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


# ---------------------------------------------------------------------------
# Cached constant helpers
# ---------------------------------------------------------------------------

def _sinusoidal_freqs(dim_half: int) -> jax.Array:
    """Standard transformer frequency table, computed once and reused."""
    if dim_half <= 0:
        return jnp.zeros((0,), dtype=jnp.float32)
    i = jnp.arange(dim_half, dtype=jnp.float32)
    return jnp.exp(-math.log(10000.0) * i / max(dim_half - 1, 1))


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class SwiGLU(nnx.Module):
    """Gated SiLU feedforward: down(silu(gate(x)) * up(x)).

    ``gate`` and ``up`` are fused into a single ``gate_up`` projection so the
    forward pass issues one GEMM instead of two.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, rngs: nnx.Rngs,
                 param_dtype: jnp.dtype | None = None):
        param_dtype = resolve_param_dtype() if param_dtype is None else param_dtype
        self.gate_up = nnx.Linear(in_dim, 2 * hidden_dim, rngs=rngs, param_dtype=param_dtype)
        self.down = nnx.Linear(hidden_dim, out_dim, rngs=rngs, param_dtype=param_dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        g, u = jnp.split(self.gate_up(x), 2, axis=-1)
        return self.down(jax.nn.silu(g) * u)


class CondMLP(nnx.Module):
    """SiLU MLP for conditioning regression (DiT canon)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, rngs: nnx.Rngs,
                 param_dtype: jnp.dtype | None = None):
        param_dtype = resolve_param_dtype() if param_dtype is None else param_dtype
        self.fc1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs, param_dtype=param_dtype)
        self.fc2 = nnx.Linear(hidden_dim, out_dim, rngs=rngs, param_dtype=param_dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(jax.nn.silu(self.fc1(x)))


class SinusoidalEncoding(nnx.Module):
    """Fixed sinusoidal codes with cached frequency / positional tables."""

    def __init__(self, dim: int, scale: float = 1.0):
        if dim <= 0:
            raise ValueError(f"dim={dim} must be positive")
        self.dim = dim
        self.scale = float(scale)
        # NOTE: no stored arrays here (freq tables, PE caches): any array
        # attribute breaks nnx.jit flattening once populated, and cache
        # writes fault inside traced code. Tables are rebuilt per call
        # (arange+exp over <=128 elems: negligible next to DiT).

    # -- tables -------------------------------------------------------------

    def frequencies(self, half: int) -> jax.Array:
        return _sinusoidal_freqs(half)

    def _build_pe(self, length: int) -> jax.Array:
        pos = jnp.arange(length, dtype=jnp.float32) * self.scale
        ang = pos[:, None] * self.frequencies(self.dim // 2)
        pe = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
        if pe.shape[-1] < self.dim:
            pe = jnp.pad(pe, ((0, 0),) * (pe.ndim - 1) + ((0, self.dim - pe.shape[-1]),))
        return pe

    # -- API ----------------------------------------------------------------

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
        return x + self._build_pe(x.shape[-2]).astype(x.dtype)


class RMSNorm(nnx.Module):
    """x / sqrt(mean(x²) + eps) * weight (matches ``nn.RMSNorm``)."""

    def __init__(self, hidden_dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs,
                 param_dtype: jnp.dtype | None = None):
        param_dtype = resolve_param_dtype() if param_dtype is None else param_dtype
        self.weight = nnx.Param(jnp.ones((hidden_dim,), dtype=param_dtype))
        self.eps = eps
        del rngs  # weight is ones; rngs accepted for uniform call sites

    def __call__(self, x: jax.Array) -> jax.Array:
        # Compute statistics in fp32 for stability; cast once at the end.
        x32 = x.astype(jnp.float32)
        inv = jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + self.eps)
        y = x32 * inv
        return (y.astype(x.dtype) * self.weight[...]) if x.dtype != jnp.float32 else (y * self.weight[...])


class SelfAttention(nnx.Module):
    """Multi-head self-attention using fused SDPA when available."""

    def __init__(self, hidden_dim: int, n_heads: int = 4, *, rngs: nnx.Rngs,
                 param_dtype: jnp.dtype | None = None):
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must split over {n_heads} heads")
        param_dtype = resolve_param_dtype() if param_dtype is None else param_dtype
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qkv = nnx.Linear(hidden_dim, 3 * hidden_dim, rngs=rngs, param_dtype=param_dtype)
        self.out = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=param_dtype)

    def _manual_attention(self, q, k, v, scores_mask):
        scores = q @ k.swapaxes(-1, -2) / math.sqrt(self.head_dim)
        if scores_mask is not None:
            scores = jnp.where(scores_mask, scores, jnp.finfo(scores.dtype).min)
        return jax.nn.softmax(scores, axis=-1) @ v

    def __call__(self, x: jax.Array, key_mask: jax.Array | None = None) -> jax.Array:
        b, l, _ = x.shape
        q, k, v = jnp.split(self.qkv(x), 3, axis=-1)
        # [B, L, H, D] layout — the layout both fused and manual paths like.
        q = q.reshape(b, l, self.n_heads, self.head_dim)
        k = k.reshape(b, l, self.n_heads, self.head_dim)
        v = v.reshape(b, l, self.n_heads, self.head_dim)

        if _sdpa is not None:
            mask = None
            if key_mask is not None:
                # Convention: True = ignore. SDPA wants True = keep, shape [B, ..., T, S].
                ignore = jnp.asarray(key_mask, dtype=bool).reshape(b, 1, 1, l)
                mask = ~ignore
            y = _sdpa(q, k, v, mask=mask)
        else:  # pragma: no cover - legacy fallback
            # Transpose to [B, H, L, D] for the manual path.
            qh = q.transpose(0, 2, 1, 3)
            kh = k.transpose(0, 2, 1, 3)
            vh = v.transpose(0, 2, 1, 3)
            smask = None
            if key_mask is not None:
                ignore = jnp.asarray(key_mask, dtype=bool).reshape(b, 1, 1, l)
                smask = ~ignore
            y = self._manual_attention(qh, kh, vh, smask).transpose(0, 2, 1, 3)

        return self.out(y.reshape(b, l, -1))


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
        param_dtype: jnp.dtype | None = None,
    ):
        hidden_dim = 128 if hidden_dim is None else hidden_dim
        mlp_ratio = 4.0 if mlp_ratio is None else float(mlp_ratio)
        param_dtype = resolve_param_dtype() if param_dtype is None else param_dtype
        del n_obs_steps  # accepted for call-site compat
        self.frame_proj = nnx.Linear(in_dim, hidden_dim, rngs=rngs, param_dtype=param_dtype)
        self.pos_encoding = SinusoidalEncoding(hidden_dim)
        self.norm1 = RMSNorm(hidden_dim, eps=1e-6, rngs=rngs, param_dtype=param_dtype)
        self.attn1 = SelfAttention(hidden_dim, n_heads, rngs=rngs, param_dtype=param_dtype)
        self.norm2 = RMSNorm(hidden_dim, eps=1e-6, rngs=rngs, param_dtype=param_dtype)
        self.mlp1 = SwiGLU(hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim, rngs=rngs,
                            param_dtype=param_dtype)

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
        self.mean = nnx.Variable(jnp.zeros((dim,), dtype=jnp.float32))
        self.var = nnx.Variable(jnp.ones((dim,), dtype=jnp.float32))
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
        delta = b_mean - self.mean[...]
        total = self.count[...] + b
        self.mean[...] = self.mean[...] + delta * b / total
        self.var[...] = (
            self.count[...] * self.var[...]
            + b * b_var
            + jnp.square(delta) * self.count[...] * b / total
        ) / total
        self.count[...] = total

    def __call__(self, x: jax.Array) -> jax.Array:
        norm = (x - self.mean[...]) * jax.lax.rsqrt(self.var[...] + self.eps)
        norm = jnp.clip(norm, -self.clip, self.clip)
        return jnp.where(self.count[...] == 0.0, x, norm)


def modulate(x: jax.Array, shift: jax.Array, scale: jax.Array) -> jax.Array:
    """adaLN modulation: x * (1 + scale) + shift."""
    return x * (1.0 + scale) + shift


class DiTBlock(nnx.Module):
    """Pre-norm DiT block with adaLN-Zero conditioning.

    ``return_stats=True`` yields (out, stats); the default path returns only
    the output tensor so no diagnostic reductions run during training.
    """

    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float, *, rngs: nnx.Rngs,
                 param_dtype: jnp.dtype | None = None):
        param_dtype = resolve_param_dtype() if param_dtype is None else param_dtype
        self.norm1 = RMSNorm(hidden_dim, rngs=rngs, param_dtype=param_dtype)
        self.attn = SelfAttention(hidden_dim, n_heads, rngs=rngs, param_dtype=param_dtype)
        self.norm2 = RMSNorm(hidden_dim, rngs=rngs, param_dtype=param_dtype)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = SwiGLU(hidden_dim, mlp_hidden, hidden_dim, rngs=rngs, param_dtype=param_dtype)
        self.adaLN = CondMLP(hidden_dim, hidden_dim, 6 * hidden_dim, rngs=rngs,
                              param_dtype=param_dtype)
        # Zero-init the modulation head: block starts as identity.
        self.adaLN.fc2.kernel[...] = jnp.zeros_like(self.adaLN.fc2.kernel[...])
        self.adaLN.fc2.bias[...] = jnp.zeros_like(self.adaLN.fc2.bias[...])

    def _split_modulation(self, c: jax.Array):
        # Reshape to [B, 6, H] then split on the 6 axis (all views, no copy).
        parts = self.adaLN(c).reshape(c.shape[0], 6, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            parts, 6, axis=1
        )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def __call__(self, x: jax.Array, c: jax.Array, return_stats: bool = False):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self._split_modulation(c)
        )
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(h)
        out = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        if not return_stats:
            return out
        stats = {
            "gate_msa": jnp.sqrt(jnp.mean(jnp.square(gate_msa))),
            "gate_mlp": jnp.sqrt(jnp.mean(jnp.square(gate_mlp))),
            "scale_msa": jnp.sqrt(jnp.mean(jnp.square(1.0 + scale_msa))),
            "scale_mlp": jnp.sqrt(jnp.mean(jnp.square(1.0 + scale_mlp))),
        }
        return out, stats


class StateConditioner(nnx.Module):
    """Fuse raw state window + DCT bins into one [B, D] vector."""

    def __init__(self, obs_dim: int, hidden_dim: int | None = None, n_heads: int = 4, *,
                 rngs: nnx.Rngs, param_dtype: jnp.dtype | None = None):
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        hidden_dim = 128 if hidden_dim is None else int(hidden_dim)
        param_dtype = resolve_param_dtype() if param_dtype is None else param_dtype
        del n_heads  # accepted for call-site compat; fusion is linear
        self.obs_dim = obs_dim
        self.state_encoder = WindowEncoder(obs_dim, hidden_dim=hidden_dim, rngs=rngs,
                                           param_dtype=param_dtype)
        self.bin_norm = RMSNorm(obs_dim, rngs=rngs, param_dtype=param_dtype)
        self.bin_encoding = SinusoidalEncoding(obs_dim)
        self.key_proj = nnx.Linear(obs_dim, hidden_dim, rngs=rngs, param_dtype=param_dtype)
        self.fuse = nnx.Linear(2 * hidden_dim, hidden_dim, rngs=rngs, param_dtype=param_dtype)

    def __call__(self, states: jax.Array) -> jax.Array:
        from sjfpo_jax.dct import dct

        raw = self.state_encoder(states).mean(axis=1)
        # Time-axis DCT-II via the cached basis in dct.py (identical math to
        # torch; a local matmul here contracted the wrong axes for obs != T).
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
        param_dtype: jnp.dtype | None = None,
    ):
        hidden_dim = 128 if hidden_dim is None else int(hidden_dim)
        dit_depth = 4 if dit_depth is None else int(dit_depth)
        dit_heads = 4 if dit_heads is None else int(dit_heads)
        dit_mlp_ratio = 4.0 if dit_mlp_ratio is None else float(dit_mlp_ratio)
        dit_time_dim = 128 if dit_time_dim is None else int(dit_time_dim)
        param_dtype = resolve_param_dtype() if param_dtype is None else param_dtype
        self.time_dim = dit_time_dim
        self.timestep_embedder = SinusoidalEncoding(self.time_dim, scale=timestep_scale)
        self.in_proj = nnx.Linear(action_dim, hidden_dim, rngs=rngs, param_dtype=param_dtype)
        self.bin_encoding = SinusoidalEncoding(hidden_dim)
        self.time_mlp = CondMLP(dit_time_dim, 4 * hidden_dim, hidden_dim, rngs=rngs,
                                param_dtype=param_dtype)
        _List = getattr(nnx, "List", list)
        self.blocks = _List(
            [DiTBlock(hidden_dim, dit_heads, dit_mlp_ratio, rngs=rngs, param_dtype=param_dtype)
             for _ in range(dit_depth)]
        )
        self.norm_out = RMSNorm(hidden_dim, rngs=rngs, param_dtype=param_dtype)
        self.adaLN_out = nnx.Linear(hidden_dim, 2 * hidden_dim, rngs=rngs, param_dtype=param_dtype)
        self.adaLN_out.kernel[...] = jnp.zeros_like(self.adaLN_out.kernel[...])
        self.adaLN_out.bias[...] = jnp.zeros_like(self.adaLN_out.bias[...])
        self.out = nnx.Linear(hidden_dim, action_dim, rngs=rngs, param_dtype=param_dtype)
        self.out.kernel[...] = jnp.zeros_like(self.out.kernel[...])
        self.out.bias[...] = jnp.zeros_like(self.out.bias[...])

    def __call__(
        self, x_t: jax.Array, t: jax.Array, condition: jax.Array, return_stats: bool = False
    ):
        temb = self.timestep_embedder.encode_times(t).astype(x_t.dtype)
        if temb.ndim == 1:
            temb = jnp.broadcast_to(temb, (x_t.shape[0], self.time_dim))
        c = self.time_mlp(temb) + condition.astype(x_t.dtype)

        h = self.bin_encoding(self.in_proj(x_t))

        # Fast path: no stats — blocks return just the tensor.
        if not return_stats:
            for block in self.blocks:
                h = block(h, c, return_stats=False)
            shift_out, scale_out = jnp.split(self.adaLN_out(c)[:, None, :], 2, axis=-1)
            return self.out(modulate(self.norm_out(h), shift_out, scale_out))

        # Diagnostic path.
        gate_stats: dict[str, list] = {
            k: [] for k in ("gate_msa", "gate_mlp", "scale_msa", "scale_mlp")
        }
        for block in self.blocks:
            h, b_stats = block(h, c, return_stats=True)
            for k in gate_stats:
                gate_stats[k].append(b_stats[k])
        stats = {
            "temb": jnp.sqrt(jnp.mean(jnp.square(temb))),
            "fused": jnp.sqrt(jnp.mean(jnp.square(condition))),
            "c": jnp.sqrt(jnp.mean(jnp.square(c))),
            **{k: jnp.mean(jnp.stack(v)) for k, v in gate_stats.items()},
        }
        shift_out, scale_out = jnp.split(self.adaLN_out(c)[:, None, :], 2, axis=-1)
        v = self.out(modulate(self.norm_out(h), shift_out, scale_out))
        return v, stats


class Critic(nnx.Module):
    """V(s) MLP on the current frame: [B, T, obs] -> [B]."""

    def __init__(self, obs_dim: int, hidden_dim: int | None = None, n_obs_steps: int | None = None, *,
                 rngs: nnx.Rngs, param_dtype: jnp.dtype | None = None):
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        hidden_dim = 128 if hidden_dim is None else int(hidden_dim)
        param_dtype = resolve_param_dtype() if param_dtype is None else param_dtype
        del n_obs_steps  # accepted for call-site compat
        self.mlp = CondMLP(obs_dim, hidden_dim, 1, rngs=rngs, param_dtype=param_dtype)

    def __call__(self, states: jax.Array) -> jax.Array:
        return self.mlp(states[:, -1, :]).squeeze(-1)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_torch_weights(nnx_model: nnx.Module, torch_state: dict) -> None:
    """Copy a torch state_dict into an NNX model.

    Handles the fused ``gate_up`` projection in ``SwiGLU`` by concatenating
    the torch ``gate`` and ``up`` weights along the output dimension. Linear
    kernels are transposed (torch ``[out, in]`` -> NNX ``[in, out]``).
    """
    import numpy as np

    graphdef, params, rest = nnx.split(nnx_model, nnx.Param, ...)
    flat_torch = {k: np.asarray(v) for k, v in torch_state.items()}
    leaves = list(nnx.traversals.flatten_mapping(params).items())

    for path, leaf in leaves:
        target_dtype = jnp.asarray(leaf[...]).dtype
        dotted = ".".join(str(p) for p in path)

        # Fused SwiGLU: NNX ``X.gate_up`` <- torch ``X.gate`` + ``X.up``.
        if dotted.endswith(".gate_up.kernel"):
            prefix = dotted[: -len(".gate_up.kernel")]
            gw = flat_torch.get(prefix + ".gate.weight")
            uw = flat_torch.get(prefix + ".up.weight")
            if gw is not None and uw is not None:
                fused = np.concatenate([gw, uw], axis=0).T  # [in, 2*hidden]
                leaf[...] = jnp.asarray(fused, dtype=target_dtype)
            continue
        if dotted.endswith(".gate_up.bias"):
            prefix = dotted[: -len(".gate_up.bias")]
            gb = flat_torch.get(prefix + ".gate.bias")
            ub = flat_torch.get(prefix + ".up.bias")
            if gb is not None and ub is not None:
                fused = np.concatenate([gb, ub], axis=0)
                leaf[...] = jnp.asarray(fused, dtype=target_dtype)
            continue

        # Regular Linear: kernel <- transposed weight.
        if dotted.endswith(".kernel"):
            w = flat_torch.get(dotted[: -len(".kernel")] + ".weight")
            if w is not None:
                leaf[...] = jnp.asarray(w.T if w.ndim == 2 else w, dtype=target_dtype)
        elif dotted.endswith(".bias"):
            b = flat_torch.get(dotted)
            if b is not None:
                leaf[...] = jnp.asarray(b, dtype=target_dtype)
        else:  # RMSNorm weight / plain vectors share names.
            w = flat_torch.get(dotted)
            if w is not None:
                leaf[...] = jnp.asarray(w, dtype=target_dtype)

    nnx.update(nnx_model, params)