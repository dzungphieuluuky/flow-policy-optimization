"""Networks in Flax NNX: window encoders, conditioner, flow head, critic.

Port of ``spectral_joint_policy/networks.py``. Same modules, same
initialisation (zero-init adaLN, ``std=0.02`` token positions), same three
attention/adaLN layout branches, same ``return_stats`` contract.

Two structural differences, both required by JAX rather than chosen:

* **Parameters are NNX ``Param`` arrays, not ``nnx.Linear`` submodules.** The
  critic vmaps over a leading member axis, and vmap needs the members stacked on
  one axis with the state reachable as a plain dict. Every module's forward is a
  module-level function of ``(params, x)``, and ``__call__`` extracts its own
  state and calls it, so the actor path and the vmapped critic path share one
  implementation.
* **The critic is one stacked param set, not a ``ModuleList``**. A Python loop
  over members inside the jitted loss is the compile-time blow-up jax discussion
  #23200 is about.

Kernels are stored ``[in, out]`` where torch stores ``[out, in]``, so the
checkpoint converter transposes every Linear. ``SinusoidalEncoding``'s frequency
table and the DCT basis are module-level constants: one device, so no per-device
cache.

The ``DiTBlock`` 4-D coordinate-axis branch of the torch original has no caller
in this tree -- ``SJFPOFlowModel`` always passes ``[B, K, H]`` -- so it is not
ported.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from sjfpo_jax.dct_utils import extract_dct_coeffs

RMS_EPS = 1e-6


def rms_norm(x, scale, eps: float = RMS_EPS):
    """RMSNorm: ``x * rsqrt(mean(x^2) + eps) * scale``, fp32 throughout.

    ``jax.nn.rms_norm`` does not exist in jax 0.11.2 -- the JAX_MIGRATION.md §3
    "verify this" resolves to no. Flax ships RMSNorm only as a Module, and a
    Module cannot hold the critic's ``[num_critics, hidden]`` stacked weight, so
    this is the whole of it: one mean, one reciprocal square root, one scale.
    """
    return x * (1.0 / jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)) * scale


def linear(x, kernel, bias):
    """``x @ kernel + bias`` for an NNX ``[in, out]`` kernel.

    The torch weights are ``[out, in]``, so the checkpoint converter transposes;
    here the contraction is already in the right order.
    """
    return jnp.einsum("...i,io->...o", x, kernel) + bias


def modulate(x, shift, scale):
    """adaLN modulation: ``x * (1 + scale) + shift``."""
    return shift + x * (1.0 + scale)


def vmapped_members(module, fn, x):
    """Apply ``fn(params, x)`` once per member of a stacked param set.

    ``x`` is closed over, not mapped: every critic member sees the same window
    batch, so the only mapped axis is the member axis on the params pytree.
    """
    return jax.vmap(lambda params: fn(params, x))(params_of(module))


def linear_init(rngs, *shape):
    """Variance-scaling ``[in, out]`` kernel; ``in_axis=-2`` also fits a stack."""
    return nnx.initializers.variance_scaling(
        1.0, "fan_in", "truncated_normal", in_axis=-2
    )(rngs.params(), shape)


def params_of(module):
    """This module's parameters as a plain nested dict of arrays."""
    return nnx.to_pure_dict(nnx.state(module, nnx.Param))


# --------------------------------------------------------------------------- blocks


def swiglu_forward(p, x):
    """Gated SiLU feedforward: ``down(silu(gate(x)) * up(x))``."""
    gate, up = jnp.split(linear(x, p["gate_up"]["kernel"], p["gate_up"]["bias"]), 2, axis=-1)
    return linear(jax.nn.silu(gate) * up, p["down"]["kernel"], p["down"]["bias"])


class SwiGLU(nnx.Module):
    """Gated SiLU feedforward. ``[..., in_dim]`` -> ``[..., out_dim]``.

    Gate and up stay fused in one GEMM, as in torch.
    """

    gate_up: dict = nnx.data()
    down: dict = nnx.data()

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, rngs: nnx.Rngs):
        if in_dim < 1 or hidden_dim < 1 or out_dim < 1:
            raise ValueError("SwiGLU dimensions must be positive")
        self.gate_up = {
            "kernel": nnx.Param(linear_init(rngs, in_dim, 2 * hidden_dim)),
            "bias": nnx.Param(jnp.zeros((2 * hidden_dim,), dtype=jnp.float32)),
        }
        self.down = {
            "kernel": nnx.Param(linear_init(rngs, hidden_dim, out_dim)),
            "bias": nnx.Param(jnp.zeros((out_dim,), dtype=jnp.float32)),
        }

    def __call__(self, x):
        return swiglu_forward(params_of(self), x)


def cond_mlp_forward(p, x):
    """Two-layer SiLU-gated MLP. ``[..., in_dim]`` -> ``[..., out_dim]``."""
    return linear(
        jax.nn.silu(linear(x, p["fc1"]["kernel"], p["fc1"]["bias"])),
        p["fc2"]["kernel"],
        p["fc2"]["bias"],
    )


class CondMLP(nnx.Module):
    """SiLU MLP for conditioning regression."""

    fc1: dict = nnx.data()
    fc2: dict = nnx.data()

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, rngs: nnx.Rngs):
        if in_dim < 1 or hidden_dim < 1 or out_dim < 1:
            raise ValueError("CondMLP dimensions must be positive")
        self.fc1 = {
            "kernel": nnx.Param(linear_init(rngs, in_dim, hidden_dim)),
            "bias": nnx.Param(jnp.zeros((hidden_dim,), dtype=jnp.float32)),
        }
        self.fc2 = {
            "kernel": nnx.Param(linear_init(rngs, hidden_dim, out_dim)),
            "bias": nnx.Param(jnp.zeros((out_dim,), dtype=jnp.float32)),
        }

    def __call__(self, x):
        return cond_mlp_forward(params_of(self), x)


def sinusoid_freqs(half: int) -> jax.Array:
    """``[half]`` geometric frequency table.

    Deliberately **not** memoised. A cache keyed on a Python int looks safe and is
    not: whichever trace happens to populate it first stores a *tracer*, and the
    next trace that reads it is handed a value from a closed-over jaxpr -- which
    JAX rejects as an unexpected tracer. Built fresh every call it is a constant
    fold inside ``jit`` and nothing at all outside one.
    """
    i = jnp.arange(half, dtype=jnp.float32)
    return jnp.exp(-math.log(10000.0) * i / max(half - 1, 1))


class SinusoidalEncoding(nnx.Module):
    """Fixed sinusoidal codes, no parameters."""

    def __init__(self, dim: int, scale: float = 1.0):
        if dim < 2:
            raise ValueError(f"SinusoidalEncoding dim must be >= 2, got {dim}")
        self.dim = dim
        self.scale = 1.0 if scale is None else float(scale)
        self.half = dim // 2

    def add_to(self, x, positions):
        """Add sinusoidal codes at ``positions``, growing ``[..., L, D]``."""
        ang = jnp.reshape(positions.astype(jnp.float32) * self.scale, positions.shape + (1,))
        ang = ang * sinusoid_freqs(self.half)
        pe = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
        if pe.shape[-1] < self.dim:
            pe = jnp.pad(pe, [(0, 0)] * (pe.ndim - 1) + [(0, self.dim - pe.shape[-1])])
        return x + pe.astype(x.dtype)

    def encode_times(self, t):
        """Sinusoidal embed of continuous times. ``[N]`` -> ``[N, 2 * half]``."""
        t = jnp.asarray(t, dtype=jnp.float32) * self.scale
        ang = t[..., None] * sinusoid_freqs(self.half)
        emb = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
        if self.dim % 2:
            emb = jnp.pad(emb, [(0, 0)] * (emb.ndim - 1) + [(0, 1)])
        return emb

    def __call__(self, x):
        """Add slot-indexed position codes to ``[..., L, D]``."""
        return self.add_to(x, jnp.arange(x.shape[-2], dtype=jnp.float32))


def self_attention_forward(p, x, keep=None):
    """Multi-head self attention with an optional ``[B, L]`` key-padding mask.

    ``keep`` is 1/True to keep a key, 0/False to ignore it. SDPA's bool mask is
    the opposite of JAX's additive bias, so it inverts here; an all-masked row
    would emit NaN, so key 0 is forced to keep.
    """
    b, l, _ = x.shape
    h, d = p["n_heads"], p["head_dim"]
    qkv = linear(x, p["qkv"]["kernel"], p["qkv"]["bias"])
    q, k, v = jnp.split(qkv, 3, axis=-1)
    # Fused [B, L, H*D] projection, one reshape. jax.nn.dot_product_attention
    # takes (batch, seq, heads, depth); torch's SDPA takes (batch, heads, seq,
    # depth), so there is no transpose to undo on either side.
    q, k, v = (t.reshape(b, l, h, d) for t in (q, k, v))

    bias = None
    if keep is not None:
        if keep.shape != (b, l):
            raise ValueError(
                f"attention_mask must have shape [B, L] = {(b, l)}, got {keep.shape}"
            )
        keep = keep.astype(jnp.bool_).at[:, 0].set(True)
        bias = jnp.where(keep[:, None, None, :], 0.0, -jnp.inf)
    y = jax.nn.dot_product_attention(q, k, v, bias=bias)
    return linear(y.reshape(b, l, h * d), p["out"]["kernel"], p["out"]["bias"])


class SelfAttention(nnx.Module):
    """Multi-head self attention. Arbitrary query/key masks are not supported."""

    qkv: dict = nnx.data()
    out: dict = nnx.data()
    n_heads: int = nnx.Static
    head_dim: int = nnx.Static

    def __init__(self, hidden_dim: int, n_heads: int, rngs: nnx.Rngs):
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if n_heads < 1:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must split over {n_heads} heads")
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qkv = {
            "kernel": nnx.Param(linear_init(rngs, hidden_dim, 3 * hidden_dim)),
            "bias": nnx.Param(jnp.zeros((3 * hidden_dim,), dtype=jnp.float32)),
        }
        self.out = {
            "kernel": nnx.Param(linear_init(rngs, hidden_dim, hidden_dim)),
            "bias": nnx.Param(jnp.zeros((hidden_dim,), dtype=jnp.float32)),
        }

    def __call__(self, x, attention_mask=None):
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, L, D], got {x.shape}")
        if x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError(
                f"x must have positive batch and sequence sizes, got {x.shape[:2]}"
            )
        p = params_of(self) | {"n_heads": self.n_heads, "head_dim": self.head_dim}
        return self_attention_forward(p, x, attention_mask)


class RunningMeanStd(nnx.Module):
    """Per-dim running mean/var over raw obs frames.

    ``mean``/``var`` fp32. ``count`` fp64, else it dies at 2^24 frames. The
    forward keeps the input dtype and is identity until the first update.

    ``uninitialised`` is a plain Python bool, as in torch: it is non-persistent,
    so it never appears in a checkpoint and is re-derived on load. As an array it
    would cost a full-size ``where`` on every forward plus a device sync to read.
    """

    mean: jax.Array = nnx.data()
    var: jax.Array = nnx.data()
    count: jax.Array = nnx.data()
    uninitialised: bool = nnx.Static
    eps: float = nnx.Static
    clip: float = nnx.Static

    def __init__(self, dim: int, eps: float = 1e-8, clip: float = 10.0):
        if dim <= 0:
            raise ValueError(f"dim={dim} must be positive")
        self.mean = nnx.data(jnp.zeros((dim,), dtype=jnp.float32))
        self.var = nnx.data(jnp.ones((dim,), dtype=jnp.float32))
        self.count = nnx.data(jnp.zeros((), dtype=jnp.float64))
        self.uninitialised = True
        self.eps = eps
        self.clip = clip

    def update(self, x):
        """Fold raw frames into the running stats, in fp64, cast back to fp32.

        Chan et al. parallel variance merge, so incremental updates do not drift.
        """
        batch = x.reshape(-1, x.shape[-1]).astype(jnp.float32)
        b = batch.shape[0]
        if b == 0:
            return
        batch_var, batch_mean = jnp.var(batch, axis=0, ddof=0), jnp.mean(batch, axis=0)
        delta = batch_mean - self.mean
        total = self.count + b
        # fp64 for the merge: fp32 drifts over millions of incremental updates.
        self.mean = self.mean + (delta * b / total).astype(jnp.float32)
        self.var = (
            (
                self.count * self.var
                + b * batch_var
                + jnp.square(delta) * (self.count * b / total)
            )
            / total
        ).astype(jnp.float32)
        self.count = total
        self.uninitialised = False

    def reset(self):
        """Clear the latch and reset stats."""
        self.mean = jnp.zeros_like(self.mean)
        self.var = jnp.ones_like(self.var)
        self.count = jnp.zeros_like(self.count)
        self.uninitialised = True

    def __call__(self, x):
        """Norm with frozen stats. Returns ``x`` unchanged until first update."""
        if self.uninitialised:
            return x
        norm = (x - self.mean) * 1.0 / jnp.sqrt(self.var + self.eps)
        return jnp.clip(norm, -self.clip, self.clip).astype(x.dtype)


# --------------------------------------------------------------------------- the DiT


def dit_block_forward(p, x, c, return_stats=False):
    """One pre-norm DiT block, adaLN-Zero conditioned.

    Layout ``[B, K, H]``: attention runs over the K action tokens. ``c`` is
    ``[B, D]`` (broadcast over tokens) or ``[B, K, D]``. The modulation head is
    zero-init, so the stack starts as the identity.
    """
    b, k, _ = x.shape
    if c.ndim == 3:
        parts = cond_mlp_forward(p["adaLN"], c).reshape(b, k, 6, -1)
    else:
        parts = cond_mlp_forward(p["adaLN"], c).reshape(b, 1, 6, -1)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        jnp.moveaxis(parts, 2, 0)
    )

    normed = rms_norm(x, p["norm1"]["scale"])
    h = modulate(normed, shift_msa, scale_msa)
    h = h + gate_msa * self_attention_forward(p["attn"], h)
    normed = rms_norm(h, p["norm2"]["scale"])
    out = h + gate_mlp * swiglu_forward(p["mlp"], modulate(normed, shift_mlp, scale_mlp))

    if not return_stats:
        return out
    stats = {
        "gate_msa": rms(gate_msa),
        "gate_mlp": rms(gate_mlp),
        "scale_msa": rms(1.0 + scale_msa),
        "scale_mlp": rms(1.0 + scale_mlp),
    }
    return out, stats


class DiTBlock(nnx.Module):
    """Pre-norm DiT block, adaLN-Zero conditioning.

    ``return_stats=True`` gives ``(out, stats)``. The default path runs no
    diagnostic reductions, so training pays nothing.
    """

    norm1: dict = nnx.data()
    attn: SelfAttention = nnx.data()
    norm2: dict = nnx.data()
    mlp: SwiGLU = nnx.data()
    adaLN: CondMLP = nnx.data()

    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float, rngs: nnx.Rngs):
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0 or int(hidden_dim * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must produce a positive hidden width")
        self.norm1 = {"scale": nnx.Param(jnp.ones((hidden_dim,), dtype=jnp.float32))}
        self.attn = SelfAttention(hidden_dim, n_heads, rngs)
        self.norm2 = {"scale": nnx.Param(jnp.ones((hidden_dim,), dtype=jnp.float32))}
        self.mlp = SwiGLU(hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim, rngs)
        self.adaLN = CondMLP(hidden_dim, hidden_dim, 6 * hidden_dim, rngs)
        # adaLN-Zero: the stack starts as the identity, a no-op.
        self.adaLN.fc2["kernel"] = nnx.Param(jnp.zeros_like(self.adaLN.fc2["kernel"]))
        self.adaLN.fc2["bias"] = nnx.Param(jnp.zeros_like(self.adaLN.fc2["bias"]))

    def __call__(self, x, c, return_stats: bool = False):
        return dit_block_forward(params_of(self), x, c, return_stats)


def rms(x):
    """Root mean square over every element, one reduction.

    ``jnp.linalg.norm`` rather than ``square().mean().sqrt()``, which walks the
    tensor three times.
    """
    return jnp.linalg.norm(x) / math.sqrt(x.size)


def rms_norm_forward(p, x):
    """RMSNorm over the last axis with a learned fp32 scale."""
    return rms_norm(x, p["scale"])


def state_conditioner_forward(p, states):
    """Window ``[B, T, obs]`` -> one ``[B, D]`` conditioning vector.

    The raw path is a per-frame MLP plus mean-pool, not attention: the spectral
    branch already encodes time and the caller mean-pools anyway.
    """
    raw = swiglu_forward(
        p["frame_mlp"],
        rms_norm_forward(p["frame_norm"], linear(states, p["frame_proj"]["kernel"], p["frame_proj"]["bias"])),
    )
    raw = jnp.mean(raw, axis=1)
    # num_modes == T keeps every frequency, the whole orthonormal transform.
    coeffs = extract_dct_coeffs(states, num_modes=states.shape[1])
    # The encoder has no parameters, so the pure form rebuilds it from obs_dim
    # rather than carrying a module through the vmap.
    spec = linear(
        SinusoidalEncoding(coeffs.shape[-1])(rms_norm_forward(p["freq_norm"], coeffs)),
        p["key_proj"]["kernel"],
        p["key_proj"]["bias"],
    )
    spec = jnp.mean(spec, axis=1)
    return linear(jnp.concatenate([raw, spec], axis=-1), p["fuse"]["kernel"], p["fuse"]["bias"])


class StateConditioner(nnx.Module):
    """Fuse a mean-pooled per-frame state with DCT-frequency features.

    Known limit, carried over from torch: ``freq_encoding`` runs on the feature
    axis and then mean-pools over frequency, so the bins get no identity. See
    REPORT.md item 11.
    """

    frame_proj: dict = nnx.data()
    frame_norm: dict = nnx.data()
    frame_mlp: SwiGLU = nnx.data()
    freq_norm: dict = nnx.data()
    freq_encoding: SinusoidalEncoding = nnx.data()
    key_proj: dict = nnx.data()
    fuse: dict = nnx.data()
    obs_dim: int = nnx.Static

    def __init__(self, obs_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        if obs_dim < 2:
            raise ValueError("StateConditioner obs_dim must be at least 2")
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        self.obs_dim = obs_dim
        self.frame_proj = {
            "kernel": nnx.Param(linear_init(rngs, obs_dim, hidden_dim)),
            "bias": nnx.Param(jnp.zeros((hidden_dim,), dtype=jnp.float32)),
        }
        self.frame_norm = {"scale": nnx.Param(jnp.ones((hidden_dim,), dtype=jnp.float32))}
        self.frame_mlp = SwiGLU(hidden_dim, 4 * hidden_dim, hidden_dim, rngs)
        self.freq_norm = {"scale": nnx.Param(jnp.ones((obs_dim,), dtype=jnp.float32))}
        self.freq_encoding = SinusoidalEncoding(obs_dim)
        self.key_proj = {
            "kernel": nnx.Param(linear_init(rngs, obs_dim, hidden_dim)),
            "bias": nnx.Param(jnp.zeros((hidden_dim,), dtype=jnp.float32)),
        }
        self.fuse = {
            "kernel": nnx.Param(linear_init(rngs, 2 * hidden_dim, hidden_dim)),
            "bias": nnx.Param(jnp.zeros((hidden_dim,), dtype=jnp.float32)),
        }

    def apply(self, params, states):
        """Pure form, so the critic can vmap it over its member axis."""
        return state_conditioner_forward(params, states)

    def __call__(self, states):
        if states.ndim != 3:
            raise ValueError(
                f"states must have shape [B, T, obs], got {states.shape}"
            )
        if states.shape[0] < 1 or states.shape[1] < 1:
            raise ValueError("states must have positive batch and time dimensions")
        if states.shape[2] != self.obs_dim:
            raise ValueError(
                f"states feature dimension must be {self.obs_dim}, got {states.shape[2]}"
            )
        return self.apply(params_of(self), states)


def flow_model_forward(p, x_t, t, condition, token_horizon, return_stats=False):
    """Predict velocity from token actions and per-token times.

    ``t``: scalar, ``[B]``, or ``[B, K, 1]``. One time per token, shared across
    that token's action dims. The last layer is zero-init, so an untrained model
    predicts near-zero velocity.
    """
    x_t = jnp.asarray(x_t, dtype=jnp.float32)
    t = jnp.asarray(t, dtype=jnp.float32)
    if x_t.ndim != 3:
        raise ValueError(f"x_t must have shape [B, K, D], got {x_t.shape}")
    if x_t.shape[0] < 1 or x_t.shape[1] < 1:
        raise ValueError("x_t must have positive batch and token dimensions")
    if x_t.shape[1] > token_horizon:
        raise ValueError(
            f"action token count {x_t.shape[1]} exceeds action_horizon {token_horizon}"
        )
    if condition.ndim != 2 or condition.shape[0] != x_t.shape[0]:
        raise ValueError(
            f"condition must have shape [{x_t.shape[0]}, {p['hidden_dim']}], got {condition.shape}"
        )
    if condition.shape[1] != p["hidden_dim"]:
        raise ValueError(
            f"condition feature dimension must be {p['hidden_dim']}, got {condition.shape[1]}"
        )

    token_axis = t.ndim in (2, 3) and t.shape[:2] == x_t.shape[:2]
    if token_axis:
        if t.ndim == 3:
            if t.shape[-1] != 1:
                raise ValueError(
                    f"t must end in a singleton action-dimension axis, got {t.shape}"
                )
            t = t[..., 0]
        temb = p["timestep_embedder"].encode_times(t.reshape(-1)).reshape(
            x_t.shape[0], x_t.shape[1], -1
        )
        c = cond_mlp_forward(p["time_mlp"], temb) + condition[:, None, :]
        shift_out, scale_out = jnp.split(linear(c, p["adaLN_out"]["kernel"], p["adaLN_out"]["bias"]), 2, axis=-1)
    else:
        if t.ndim > 1:
            # torch raises here too: [B, 1] is neither scalar, [B], nor [B, K, 1].
            raise ValueError(
                f"t must be scalar, [B], or [B, K, 1], got {t.shape}"
            )
        t = t.reshape(-1)
        if t.size not in (1, x_t.shape[0]):
            raise ValueError(
                f"t must contain 1 or {x_t.shape[0]} values, got {t.size}"
            )
        temb = p["timestep_embedder"].encode_times(t)
        c = cond_mlp_forward(p["time_mlp"], temb) + condition
        shift_out, scale_out = jnp.split(
            linear(c[:, None, :], p["adaLN_out"]["kernel"], p["adaLN_out"]["bias"]), 2, axis=-1
        )

    token_pos = p["token_pos"][:, : x_t.shape[1], :]
    h = linear(x_t, p["in_proj"]["kernel"], p["in_proj"]["bias"]) + token_pos

    block_stats = []
    for i in range(p["dit_depth"]):
        block = p["blocks"][i]
        h = dit_block_forward(block, h, c, return_stats)
        if return_stats:
            h, stats = h
            block_stats.append(stats)
    h = rms_norm(h, p["norm_out"]["scale"])
    out = linear(modulate(h, shift_out, scale_out), p["out"]["kernel"], p["out"]["bias"])
    if not return_stats:
        return out
    stats = {
        "temb": rms(temb),
        "fused": rms(condition),
        "c": rms(c),
        **{
            k: jnp.mean(jnp.stack([s[k] for s in block_stats]))
            for k in ("gate_msa", "gate_mlp", "scale_msa", "scale_mlp")
        },
    }
    return out, stats


class SJFPOFlowModel(nnx.Module):
    """DiT velocity model over noisy action tokens, adaLN conditioned."""

    in_proj: dict = nnx.data()
    token_pos: jax.Array = nnx.data()
    time_mlp: CondMLP = nnx.data()
    blocks: list = nnx.data()
    norm_out: dict = nnx.data()
    adaLN_out: dict = nnx.data()
    out: dict = nnx.data()
    timestep_embedder: SinusoidalEncoding = nnx.data()
    action_dim: int = nnx.Static
    hidden_dim: int = nnx.Static
    action_horizon: int = nnx.Static

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
        dit_depth: int,
        dit_heads: int,
        dit_mlp_ratio: float,
        time_embed_dim: int,
        timestep_scale: float,
        rngs: nnx.Rngs,
    ):
        if action_dim < 1 or hidden_dim < 2:
            raise ValueError("action_dim must be positive and hidden_dim at least 2")
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if dit_depth < 1 or dit_heads < 1 or time_embed_dim < 2:
            raise ValueError("dit_depth, dit_heads, and time_embed_dim must be positive")
        if not math.isfinite(dit_mlp_ratio) or dit_mlp_ratio <= 0 or int(hidden_dim * dit_mlp_ratio) < 1:
            raise ValueError("dit_mlp_ratio must produce a positive hidden width")
        if not math.isfinite(timestep_scale) or timestep_scale <= 0:
            raise ValueError("timestep_scale must be finite and positive")
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_horizon = action_horizon
        self.timestep_embedder = SinusoidalEncoding(time_embed_dim, scale=timestep_scale)
        self.in_proj = {
            "kernel": nnx.Param(linear_init(rngs, action_dim, hidden_dim)),
            "bias": nnx.Param(jnp.zeros((hidden_dim,), dtype=jnp.float32)),
        }
        self.token_pos = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (1, action_horizon, hidden_dim))
        )
        self.time_mlp = CondMLP(time_embed_dim, 4 * hidden_dim, hidden_dim, rngs)
        self.blocks = [
            DiTBlock(hidden_dim, dit_heads, dit_mlp_ratio, rngs) for _ in range(dit_depth)
        ]
        self.norm_out = {"scale": nnx.Param(jnp.ones((hidden_dim,), dtype=jnp.float32))}
        self.adaLN_out = {
            "kernel": nnx.Param(jnp.zeros((hidden_dim, 2 * hidden_dim), dtype=jnp.float32)),
            "bias": nnx.Param(jnp.zeros((2 * hidden_dim,), dtype=jnp.float32)),
        }
        self.out = {
            "kernel": nnx.Param(jnp.zeros((hidden_dim, action_dim), dtype=jnp.float32)),
            "bias": nnx.Param(jnp.zeros((action_dim,), dtype=jnp.float32)),
        }

    def __call__(self, x_t, t, condition, return_stats: bool = False):
        # The block stack is a list of Modules, so its params come back nested;
        # flatten them once here and hand dit_block_forward plain dicts.
        blocks = [
            params_of(block)
            | {"attn": params_of(block.attn) | {"n_heads": block.attn.n_heads, "head_dim": block.attn.head_dim}}
            for block in self.blocks
        ]
        p = params_of(self) | {
            "blocks": blocks,
            "timestep_embedder": self.timestep_embedder,
            "dit_depth": len(blocks),
            "hidden_dim": self.hidden_dim,
            "action_dim": self.action_dim,
        }
        return flow_model_forward(p, x_t, t, condition, self.action_horizon, return_stats)


def critic_member_forward(p, states):
    """One critic member: a conditioner over the window, then a linear head."""
    return linear(
        state_conditioner_forward(p["conditioners"], states),
        p["head"]["kernel"],
        p["head"]["bias"],
    ).squeeze(-1)


class Critic(nnx.Module):
    """V(s) over the full window, as ``num_critics`` independent members.

    Each member is a ``StateConditioner`` plus a linear head, so the critic sees
    the window the way the actor does. Members stay independent or the ensemble
    is worthless: a shared trunk leaves only a linear map differing, and ``min``
    over that means nothing.

    Returns ``[num_critics, B]``, unaggregated. The caller picks ``min`` for the
    GAE backup and ``mean`` for the value loss.
    """

    conditioners: dict = nnx.data()
    head: dict = nnx.data()
    obs_dim: int = nnx.Static
    n_obs_steps: int = nnx.Static
    num_critics: int = nnx.Static

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        n_obs_steps: int = 1,
        num_critics: int = 1,
        rngs: nnx.Rngs = None,
    ):
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if n_obs_steps < 1:
            raise ValueError(f"n_obs_steps must be >= 1, got {n_obs_steps}")
        if num_critics < 1:
            raise ValueError(f"num_critics must be >= 1, got {num_critics}")
        self.obs_dim = obs_dim
        self.n_obs_steps = n_obs_steps
        self.num_critics = num_critics
        # One member axis, leading: [num_critics, in, out] and
        # [num_critics, out]. in_axis=-2 is still the fan_in axis.
        self.conditioners = {
            "frame_proj": {
                "kernel": nnx.Param(linear_init(rngs, num_critics, obs_dim, hidden_dim)),
                "bias": nnx.Param(jnp.zeros((num_critics, hidden_dim), dtype=jnp.float32)),
            },
            "frame_norm": {"scale": nnx.Param(jnp.ones((num_critics, hidden_dim), dtype=jnp.float32))},
            "frame_mlp": {
                "gate_up": {
                    "kernel": nnx.Param(
                        linear_init(rngs, num_critics, hidden_dim, 2 * 4 * hidden_dim)
                    ),
                    "bias": nnx.Param(jnp.zeros((num_critics, 2 * 4 * hidden_dim), dtype=jnp.float32)),
                },
                "down": {
                    "kernel": nnx.Param(
                        linear_init(rngs, num_critics, 4 * hidden_dim, hidden_dim)
                    ),
                    "bias": nnx.Param(jnp.zeros((num_critics, hidden_dim), dtype=jnp.float32)),
                },
            },
            "freq_norm": {"scale": nnx.Param(jnp.ones((num_critics, obs_dim), dtype=jnp.float32))},
            "key_proj": {
                "kernel": nnx.Param(linear_init(rngs, num_critics, obs_dim, hidden_dim)),
                "bias": nnx.Param(jnp.zeros((num_critics, hidden_dim), dtype=jnp.float32)),
            },
            "fuse": {
                "kernel": nnx.Param(linear_init(rngs, num_critics, 2 * hidden_dim, hidden_dim)),
                "bias": nnx.Param(jnp.zeros((num_critics, hidden_dim), dtype=jnp.float32)),
            },
        }
        self.head = {
            "kernel": nnx.Param(linear_init(rngs, num_critics, hidden_dim, 1)),
            "bias": nnx.Param(jnp.zeros((num_critics, 1), dtype=jnp.float32)),
        }

    def __call__(self, states):
        if states.ndim != 3:
            raise ValueError(f"states must have shape [B, T, obs], got {states.shape}")
        if states.shape[2] != self.obs_dim:
            raise ValueError(
                f"states feature dimension must be {self.obs_dim}, got {states.shape[2]}"
            )
        if states.shape[1] != self.n_obs_steps:
            raise ValueError(
                f"states has {states.shape[1]} time steps but the critic was built "
                f"for n_obs_steps={self.n_obs_steps}; the collector builds windows "
                "to the declared length"
            )
        return vmapped_members(self, critic_member_forward, states)
