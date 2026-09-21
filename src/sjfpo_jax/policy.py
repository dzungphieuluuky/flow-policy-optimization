"""JAX SJFPO policy: NNX container + ASPO/value loss + jitted Optax update.

Mirrors ``sjfpo.modeling_sjfpo.SJFPOPolicy`` forward/train semantics:
live actor/conditioner/critic, frozen behavior snapshot (separate modules,
never passed to the optimizer), frozen-stat obs normalization (stats updated
host-side via :func:`update_normalizer`), stored CRN draws (tau/eps).
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
    ):
        self.actor = SJFPOFlowModel(
            action_dim, hidden_dim, dit_depth, dit_heads,
            dit_mlp_ratio, dit_time_dim, timestep_scale, rngs=rngs,
        )
        self.conditioner = StateConditioner(obs_dim, hidden_dim, rngs=rngs)
        self.critic = Critic(obs_dim, hidden_dim, n_obs_steps, rngs=rngs)
        self.normalizer = RunningMeanStd(obs_dim)
        self.action_low = nnx.Variable(jnp.full((action_dim,), -1.0))
        self.action_high = nnx.Variable(jnp.full((action_dim,), 1.0))

    def set_action_bounds(self, low: jax.Array, high: jax.Array) -> None:
        self.action_low[...] = jnp.asarray(low, dtype=jnp.float32).reshape(-1)
        self.action_high[...] = jnp.asarray(high, dtype=jnp.float32).reshape(-1)


def build_policy(obs_dim: int, action_dim: int, seed: int = 0, **kwargs) -> SJFPOJaxPolicy:
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


def _path_names(path) -> list[str]:
    """Key names as strings (sequence indices -> "").

    Exact match: substring would wrongly catch norm_out (torch rule is
    startswith("out.")).
    """
    names = []
    for p in path:
        key = getattr(p, "key", None)
        if not isinstance(key, str):
            key = getattr(p, "name", None)
        names.append(key if isinstance(key, str) else "")
    return names


def _decide_decay(path) -> bool:
    # Exact torch mirror (get_optim_params loops actor params only):
    # no-decay iff "adaLN" anywhere in the actor path or the top-level
    # actor "out" head. Conditioner/critic (incl. attn out projections)
    # keep the preset weight decay.
    parts = _path_names(path)
    if not parts or parts[0] != "actor":
        return True
    return not (
        any("adaLN" in n for n in parts)
        or (len(parts) > 1 and parts[1] == "out")
    )


def decay_mask_fn():
    """Callable mask for optax (structure matches by construction, jit-safe).

    A prebuilt mask tree breaks optax pairing under jit; computing it from
    the updates tree inside update_fn does not.
    """
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
    """AdamW with per-param decay mask + global-norm clip (verified API)."""
    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(lr, weight_decay=weight_decay, mask=decay_mask_fn()),
    )
    return nnx.Optimizer(policy, tx, wrt=nnx.Param)


def update_normalizer(policy: SJFPOJaxPolicy, states: jax.Array) -> None:
    """Fold a train batch into running stats (host-side, like preprocessor)."""
    policy.normalizer.update(states)


def compute_loss(
    policy: SJFPOJaxPolicy,
    behavior: SJFPOJaxPolicy,
    batch: dict[str, jax.Array],
    *,
    n_obs_steps: int = 16,
    action_horizon: int = 16,
    top_p_spectral_tokens: float = 1.0,
    use_dct_transform: bool = True,
    num_mc_samples: int = 16,
    min_timestep: float = 0.005,
    max_timestep: float = 0.995,
    epsilon_clip: float = 0.05,
    use_aspo: bool = True,
    cfm_clamp: float = 20.0,
    log_ratio_cap: float = 10.0,
    vf_coef: float = 0.01,
    use_clipped_value_loss: bool = False,
) -> tuple[jax.Array, dict[str, float]]:
    """Combined ASPO policy + value loss (train mode, frozen stats/CRN)."""
    states = batch["observation.state"].astype(jnp.float32)
    actions = batch["action"].astype(jnp.float32)
    advantages = batch["advantages"].astype(jnp.float32)
    returns = batch["returns"].astype(jnp.float32)
    if states.shape[-2] != n_obs_steps:
        raise ValueError(f"state length {states.shape[-2]} != n_obs_steps {n_obs_steps}")
    if actions.shape[-2] != action_horizon:
        raise ValueError(f"action horizon {actions.shape[-2]} != {action_horizon}")
    states_n = policy.normalizer(states)
    token_dim = max(1, round(top_p_spectral_tokens * action_horizon)) if use_dct_transform else action_horizon
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
    tau_bn, eps_flat, a_rep, flow_shape, num_mc = fpo_jax.run_mc_simulation(
        spec_tokens, eps, tau
    )
    x_t = ((1.0 - tau_bn[:, None]) * eps_flat + tau_bn[:, None] * a_rep).reshape(
        -1, *flow_shape
    )
    cond = policy.conditioner(states_n)
    cond_flat = jnp.broadcast_to(cond[:, None, :], (n, num_mc, cond.shape[-1])).reshape(
        n * num_mc, -1
    )
    old_cond_flat = jnp.broadcast_to(
        behavior.conditioner(policy.normalizer(states))[:, None, :],
        (n, num_mc, cond.shape[-1]),
    ).reshape(n * num_mc, -1)
    v_old, _ = behavior.actor(x_t, tau_bn, old_cond_flat)
    v_new, cond_stats = policy.actor(x_t, tau_bn, cond_flat)
    v_star = jnp.expand_dims(spec_tokens, 1) - eps
    v_old_r = v_old.reshape(n, num_mc, *flow_shape)
    v_new_r = v_new.reshape(n, num_mc, *flow_shape)
    adv_mc = advantages[:, None]
    l_old = fpo_jax.compute_cfm_loss(v_old_r, v_star, "sample", cfm_clamp=cfm_clamp)
    l_new = fpo_jax.compute_cfm_loss(v_new_r, v_star, "sample", cfm_clamp=cfm_clamp)
    rho = fpo_jax.compute_policy_ratio(l_old, l_new, cfm_clamp=cfm_clamp, cap=log_ratio_cap)
    psi = fpo_jax.compute_aspo_surrogate(
        rho, adv_mc, epsilon_clip, symmetrical_ppo=not use_aspo
    )
    policy_per_chunk = -psi.mean(axis=1)
    v_pred = policy.critic(states_n).astype(jnp.float32)
    se = (v_pred - returns) ** 2
    if use_clipped_value_loss and "values" in batch:
        old_values = batch["values"].astype(jnp.float32)
        v_clipped = old_values + jnp.clip(v_pred - old_values, -epsilon_clip, epsilon_clip)
        value_per_chunk = jnp.maximum(se, (v_clipped - returns) ** 2)
    else:
        value_per_chunk = se
    policy_loss = policy_per_chunk.mean()
    value_loss = value_per_chunk.mean()
    loss = policy_loss + vf_coef * value_loss
    # Arrays-out (jit-safe): the host wrapper converts to floats and checks
    # finiteness after the compiled step returns concrete values.
    returns_var = jnp.var(returns).clip(min=1e-8)
    info = {
        "policy_loss": policy_per_chunk.mean(),
        "value_loss": value_per_chunk.mean(),
        "explained_var": 1.0 - jnp.var(returns - v_pred) / returns_var,
        "mean_log_ratio": (l_old - l_new).mean(),
        "flow_rho_mean": rho.mean(),
        "flow_rho_std": rho.std(),
        **{f"cond_{k}": v for k, v in cond_stats.items()},
    }
    return loss, info


def train_step(
    policy: SJFPOJaxPolicy,
    behavior: SJFPOJaxPolicy,
    optimizer: nnx.Optimizer,
    batch: dict[str, jax.Array],
    loss_kwargs: dict | None = None,
) -> tuple[jax.Array, dict[str, float]]:
    """One jitted update: CRN-fixed loss, grads to live params only."""
    loss_kwargs = loss_kwargs or {}

    @nnx.jit
    def _step(policy, behavior, optimizer, batch):
        def loss_fn(policy):
            total, info = compute_loss(policy, behavior, batch, **loss_kwargs)
            return total, info

        (total, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(policy)
        optimizer.update(policy, grads)
        return total, info

    total, info = _step(policy, behavior, optimizer, batch)
    if not bool(jnp.all(jnp.isfinite(total))):
        raise FloatingPointError(f"non-finite loss: {total}")
    return total, {k: float(v) for k, v in info.items()}


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
    # Torch parity: noisy rollouts integrate with num_euler_steps.
    steps = num_euler_steps if noise else num_inference_steps
    def _velocity(x, t, c):
        v, _ = policy.actor(x, t, c)
        return v

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
