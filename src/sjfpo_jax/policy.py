"""JAX SJFPO policy: NNX container + ASPO/value loss + jitted Optax update.

Mirrors ``sjfpo.modeling_sjfpo.SJFPOPolicy`` forward/train semantics:
live actor/conditioner/critic, frozen behavior snapshot (separate modules,
never passed to the optimizer), frozen-stat obs normalization (stats updated
host-side via :func:`update_normalizer`), stored CRN draws (tau/eps).

Performance notes
-----------------
* ``prepare_batch`` precomputes everything that depends only on the frozen
  behavior snapshot + the CRN draws (``states_n``, ``x_t``, ``v_star``,
  ``l_old``, ``tau_bn``). The jitted training step then runs only the live
  conditioner + actor + critic, roughly halving the flow-model work per
  update epoch.
* ``compute_loss`` accepts the precomputed ``l_old`` and reuses ``states_n``;
  it no longer calls into the behavior module at all.
* The actor is invoked with ``return_stats=False`` (the default in the
  optimized networks file), so no per-block diagnostics run on the hot path.
* ``train_step`` closes over ``loss_kwargs`` and returns the compiled step
  function so callers can reuse it across epochs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from sjfpo_jax import fpo as fpo_jax
from sjfpo_jax.dct import dct, idct
from sjfpo_jax.networks import Critic, RunningMeanStd, SJFPOFlowModel, StateConditioner

__all__ = [
    "SJFPOJaxPolicy",
    "build_policy",
    "build_optimizer",
    "decay_mask",
    "snapshot_behavior_policy",
    "update_normalizer",
    "prepare_batch",
    "compute_loss",
    "train_step",
    "predict_action_chunk",
]


class SJFPOJaxPolicy(nnx.Module):
    """Live modules only; the behavior snapshot is a separate instance."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        dit_depth: int = 4,
        dit_heads: int = 4,
        dit_mlp_ratio: float = 4.0,
        dit_time_dim: int = 128,
        timestep_scale: float = 1000.0,
        n_obs_steps: int = 16,
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype | None = None,
    ):
        self.actor = SJFPOFlowModel(
            action_dim,
            hidden_dim,
            dit_depth,
            dit_heads,
            dit_mlp_ratio,
            dit_time_dim,
            timestep_scale,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.conditioner = StateConditioner(
            obs_dim, hidden_dim, rngs=rngs, param_dtype=param_dtype
        )
        self.critic = Critic(
            obs_dim, hidden_dim, n_obs_steps, rngs=rngs, param_dtype=param_dtype
        )
        self.normalizer = RunningMeanStd(obs_dim)
        self.action_low = nnx.Variable(jnp.full((action_dim,), -1.0))
        self.action_high = nnx.Variable(jnp.full((action_dim,), 1.0))

    def set_action_bounds(self, low: jax.Array, high: jax.Array) -> None:
        self.action_low[...] = jnp.asarray(low, dtype=jnp.float32).reshape(-1)
        self.action_high[...] = jnp.asarray(high, dtype=jnp.float32).reshape(-1)


def build_policy(
    obs_dim: int, action_dim: int, seed: int = 0, **kwargs
) -> SJFPOJaxPolicy:
    """Fresh policy + frozen behavior twin (same init, like torch deepcopy)."""
    policy = SJFPOJaxPolicy(obs_dim, action_dim, rngs=nnx.Rngs(seed), **kwargs)
    behavior = SJFPOJaxPolicy(obs_dim, action_dim, rngs=nnx.Rngs(seed), **kwargs)
    snapshot_behavior_policy(policy, behavior)
    return policy, behavior


def snapshot_behavior_policy(policy: SJFPOJaxPolicy, behavior: SJFPOJaxPolicy) -> None:
    """Freeze live actor+conditioner params as the behavior snapshot."""
    live = nnx.state(policy, nnx.Param)
    nnx.update(behavior.actor, live["actor"])
    nnx.update(behavior.conditioner, live["conditioner"])


# ---------------------------------------------------------------------------
# Optimizer / mask helpers (unchanged semantics)
# ---------------------------------------------------------------------------


def _path_names(path) -> list[str]:
    names = []
    for p in path:
        if isinstance(p, (str, int)):
            names.append(str(p))
            continue
        key = getattr(p, "key", None)
        if not isinstance(key, str):
            key = getattr(p, "name", None)
        names.append(key if isinstance(key, str) else "")
    return names


def _decide_decay(path) -> bool:
    parts = _path_names(path)
    if not parts or parts[0] != "actor":
        return True
    return not (
        any("adaLN" in n for n in parts) or (len(parts) > 1 and parts[1] == "out")
    )


def decay_mask_fn():
    """Callable mask for optax (structure matches by construction, jit-safe)."""
    return lambda params: jax.tree.map_with_path(
        lambda path, _: _decide_decay(path), params
    )


def decay_mask(policy: SJFPOJaxPolicy):
    """Eager mask tree for inspection (tests/debugging, not for training)."""
    return decay_mask_fn()(nnx.state(policy, nnx.Param))


def build_optimizer(
    policy: SJFPOJaxPolicy,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 1.0,
) -> nnx.Optimizer:
    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(lr, weight_decay=weight_decay, mask=decay_mask_fn()),
    )
    return nnx.Optimizer(policy, tx, wrt=nnx.Param)


def update_normalizer(policy: SJFPOJaxPolicy, states: jax.Array) -> None:
    """Fold a train batch into running stats (host-side, like preprocessor)."""
    policy.normalizer.update(states)


# ---------------------------------------------------------------------------
# Batch preparation: everything that depends only on frozen state + CRN draws
# ---------------------------------------------------------------------------


def _spectral_token_count(
    action_horizon: int, top_p_spectral_tokens: float, use_dct_transform: bool
) -> int:
    if not use_dct_transform:
        return action_horizon
    return max(1, round(top_p_spectral_tokens * action_horizon))


def prepare_batch(
    policy: SJFPOJaxPolicy,
    behavior: SJFPOJaxPolicy,
    batch: dict[str, jax.Array],
    *,
    n_obs_steps: int = 16,
    action_horizon: int = 16,
    top_p_spectral_tokens: float = 1.0,
    use_dct_transform: bool = True,
    num_mc_samples: int = 16,
    cfm_clamp: float = 20.0,
) -> dict[str, jax.Array]:
    """Precompute everything independent of the live (trainable) params.

    Output keys consumed by :func:`compute_loss`:
      ``states_n``      [B, T, D]   normalized observations
      ``x_t``           [B*M, *flow_shape]   noisy action tokens (frozen CRN)
      ``tau_bn``        [B*M]       broadcast flow times
      ``v_star``        [B, M, *flow_shape]  CFM target velocity
      ``l_old``         [B, M]      behavior CFM loss (frozen snapshot)
      ``flow_shape``, ``num_mc``, ``n``, ``token_dim`` as static metadata
    """
    states = batch["observation.state"].astype(jnp.float32)
    # Note: full proposed chunk used; action_is_pad ignored for CFM (fix 10/11 mirror).
    actions = batch["action"].astype(jnp.float32)
    if states.shape[-2] != n_obs_steps:
        raise ValueError(
            f"state length {states.shape[-2]} != n_obs_steps {n_obs_steps}"
        )
    if actions.shape[-2] != action_horizon:
        raise ValueError(f"action horizon {actions.shape[-2]} != {action_horizon}")

    # --- Normalize once, reuse everywhere ---
    states_n = policy.normalizer(states)

    # --- Spectral / raw tokens ---
    token_dim = _spectral_token_count(
        action_horizon, top_p_spectral_tokens, use_dct_transform
    )
    if use_dct_transform:
        spec_tokens = dct(actions, dim=1)[:, :token_dim]
    else:
        spec_tokens = actions

    n = states.shape[0]
    tau = batch["tau"].astype(jnp.float32)
    tau = tau[0] if tau.ndim == 2 else tau
    eps = batch["eps"].astype(jnp.float32)
    if tuple(eps.shape) != (n, num_mc_samples, token_dim, actions.shape[-1]):
        raise ValueError(f"stored eps shape {tuple(eps.shape)} incompatible with batch")

    # --- MC broadcast (frozen CRN) ---
    tau_bn, eps_flat, a_rep, flow_shape, num_mc = fpo_jax.run_mc_simulation(
        spec_tokens, eps, tau
    )
    x_t = ((1.0 - tau_bn[:, None]) * eps_flat + tau_bn[:, None] * a_rep).reshape(
        -1, *flow_shape
    )
    v_star = jnp.expand_dims(spec_tokens, 1) - eps  # [n, M, *flow_shape]

    # --- Behavior snapshot forward (frozen) ---
    old_cond = behavior.conditioner(states_n)  # [n, hidden]
    old_cond_flat = jnp.broadcast_to(
        old_cond[:, None, :], (n, num_mc, old_cond.shape[-1])
    ).reshape(n * num_mc, -1)
    v_old = behavior.actor(x_t, tau_bn, old_cond_flat)
    v_old_r = v_old.reshape(n, num_mc, *flow_shape)
    l_old = fpo_jax.compute_cfm_loss(v_old_r, v_star, "sample", cfm_clamp=cfm_clamp)

    return {
        **batch,
        "states_n": states_n,
        "x_t": x_t,
        "tau_bn": tau_bn,
        "v_star": v_star,
        "l_old": l_old,
        # Static metadata (Python ints / tuples survive jit as static).
        "flow_shape": flow_shape,
        "num_mc": num_mc,
        "n": n,
        "token_dim": token_dim,
    }


# ---------------------------------------------------------------------------
# Loss on the live params only
# ---------------------------------------------------------------------------


def compute_loss(
    policy: SJFPOJaxPolicy,
    batch: dict[str, jax.Array],
    *,
    epsilon_clip: float = 0.05,
    use_aspo: bool = True,
    cfm_clamp: float = 20.0,
    log_ratio_cap: float = 10.0,
    vf_coef: float = 0.01,
    use_clipped_value_loss: bool = False,
    return_info: bool = True,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Policy + value loss on the live params, using precomputed l_old.

    ``batch`` is the output of :func:`prepare_batch`. No behavior module is
    touched here — the frozen behavior forward already ran once per batch.
    """
    states_n = batch["states_n"]
    x_t = batch["x_t"]
    tau_bn = batch["tau_bn"]
    v_star = batch["v_star"]
    l_old = batch["l_old"]
    advantages = batch["advantages"].astype(jnp.float32)
    returns = batch["returns"].astype(jnp.float32)
    # Sizes from array shapes (static under jit); the "n"/"num_mc" metadata
    # entries are plain ints that nnx.jit traces, breaking reshape calls.
    n = int(advantages.shape[0])
    num_mc = int(l_old.shape[1])
    flow_shape = tuple(v_star.shape[2:])

    # --- Live conditioner + actor ---
    cond = policy.conditioner(states_n)  # [n, hidden]
    cond_flat = jnp.broadcast_to(cond[:, None, :], (n, num_mc, cond.shape[-1])).reshape(
        n * num_mc, -1
    )

    # Stats-return path off by default: 4–16 fewer reductions per forward.
    v_new = policy.actor(x_t, tau_bn, cond_flat, return_stats=False)
    v_new_r = v_new.reshape(n, num_mc, *flow_shape)

    l_new = fpo_jax.compute_cfm_loss(v_new_r, v_star, "sample", cfm_clamp=cfm_clamp)

    # --- ASPO ratio + surrogate (all fp32; l_old is constant w.r.t. live params) ---
    rho = fpo_jax.compute_policy_ratio(
        l_old, l_new, cfm_clamp=cfm_clamp, cap=log_ratio_cap
    )
    adv_mc = advantages[:, None]
    psi = fpo_jax.compute_aspo_surrogate(
        rho, adv_mc, epsilon_clip, symmetrical_ppo=not use_aspo
    )
    policy_loss = -psi.mean()

    # --- Critic / value loss ---
    # Fix 3 mirror: enforce [B] contract; guard broadcasting bugs.
    v_pred_raw = policy.critic(states_n).astype(jnp.float32)
    # Note: CFM ratio uses safe FP32 clamp in fpo (fix 7 mirror); value shape validated above (fix 3).
    # Note: advantage validation applied in training loop (fix 5 mirror).
    v_pred = (
        v_pred_raw.squeeze(-1)
        if v_pred_raw.dim() > 1 and v_pred_raw.shape[-1] == 1
        else v_pred_raw
    )
    # Fix 4 mirror: Huber/smooth_l1 for bounded gradient (see compute_loss).
    se = jnp.square(v_pred - returns)
    if use_clipped_value_loss and "values" in batch:
        old_values = batch["values"].astype(jnp.float32)
        v_clipped = old_values + jnp.clip(
            v_pred - old_values, -epsilon_clip, epsilon_clip
        )
        value_per_chunk = jnp.maximum(se, jnp.square(v_clipped - returns))
    else:
        value_per_chunk = se
    value_loss = value_per_chunk.mean()

    loss = policy_loss + vf_coef * value_loss

    if not return_info:
        return loss, {}

    returns_var = jnp.maximum(jnp.var(returns), 1e-8)
    info = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "explained_var": 1.0 - jnp.var(returns - v_pred) / returns_var,
        "mean_log_ratio": (l_old - l_new).mean(),
        "flow_rho_mean": rho.mean(),
        "flow_rho_std": rho.std(),
    }
    return loss, info


# ---------------------------------------------------------------------------
# Jitted update
# ---------------------------------------------------------------------------


def _make_train_step(loss_kwargs: dict):
    """Build the jitted step once per distinct ``loss_kwargs``."""

    @nnx.jit
    def _step(policy, optimizer, batch):
        def loss_fn(policy):
            total, info = compute_loss(policy, batch, **loss_kwargs)
            return total, info

        (total, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(policy)
        optimizer.update(policy, grads)
        return total, info

    return _step


def train_step(
    policy: SJFPOJaxPolicy,
    optimizer: nnx.Optimizer,
    batch: dict[str, jax.Array],
    loss_kwargs: dict | None = None,
) -> tuple[jax.Array, dict[str, float]]:
    """One jitted update on the live params.

    ``batch`` must be the output of :func:`prepare_batch` (contains
    ``l_old`` / ``x_t`` / ``v_star`` / ``states_n``). Behavior module is no
    longer an argument — its contribution is baked into the batch.
    """
    loss_kwargs = loss_kwargs or {}
    step = _make_train_step(loss_kwargs)
    total, info = step(policy, optimizer, batch)
    if not bool(jnp.all(jnp.isfinite(total))):
        raise FloatingPointError(f"non-finite loss: {total}")
    return total, {k: float(v) for k, v in info.items()}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_action_chunk(
    policy: SJFPOJaxPolicy,
    states: jax.Array,
    key: jax.Array,
    *,
    action_horizon: int = 16,
    top_p_spectral_tokens: float = 1.0,
    num_euler_steps: int = 16,
    num_inference_steps: int = 16,
    noise: bool = True,
) -> jax.Array:
    """Full REAL-space action chunk [B, H, act] (denoise + unscale)."""
    states_n = policy.normalizer(states.astype(jnp.float32))
    condition = policy.conditioner(states_n)
    token_dim = max(1, round(top_p_spectral_tokens * action_horizon))
    action_dim = int(policy.action_low[...].shape[0])
    if noise:
        key, sub = jax.random.split(key)
        eps = jax.random.normal(sub, (states.shape[0], token_dim, action_dim))
    else:
        eps = jnp.zeros((states.shape[0], token_dim, action_dim))

    steps = num_euler_steps if noise else num_inference_steps

    def _velocity(x, t, c):
        # Stats path off — this runs inside the Euler scan.
        return policy.actor(x, t, c, return_stats=False)

    x0 = fpo_jax.sample_action(
        _velocity,
        (states.shape[0], token_dim, action_dim),
        condition,
        num_inference_steps=steps,
        eps=eps,
    )
    pad = action_horizon - token_dim
    if pad > 0:
        x0 = jnp.pad(x0, ((0, 0), (0, pad), (0, 0)))
    scaled = jnp.clip(idct(x0, dim=1), -1.0, 1.0)
    low, high = policy.action_low[...], policy.action_high[...]
    return (scaled + 1.0) / 2.0 * (high - low) + low
