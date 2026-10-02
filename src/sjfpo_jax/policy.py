"""SJFPO loss path: the pure helpers, the behaviour CFM loss, and ``forward_loss``.

Port of the loss half of ``SJFPOPolicy`` (``modeling_sjfpo.py``). Free functions
over ``SJFPOState`` rather than methods on a module, so ``jax.jit`` and
``jax.grad`` work on them directly and the parity test hands both frameworks the
same batch dict.

Two contracts carried over from torch that are easy to lose:

* ``l_old`` / ``l_old_token`` are computed **once per rollout** against the
  frozen snapshot and arrive in the batch. ``forward_loss`` raises rather than
  recomputing them, because recomputing per minibatch compares two different
  policy snapshots and the trust region silently bounds nothing.
* The execution mask and the velocity layout are chosen **together**
  (``prepare_cfm_loss_inputs``). They are only consistent with each other and a
  mismatch fails silently.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.struct import dataclass

from sjfpo_jax.dct_utils import (
    compute_spectral_energy_retained,
    extract_dct_coeffs,
    velocity_to_action_steps,
)
from sjfpo_jax.fpo_core import (
    compute_aspo_surrogate,
    compute_cfm_loss,
    compute_policy_ratio,
    run_monte_carlo_sampling,
    sample_action,
    sample_noise,
    sample_timesteps,
)
from sjfpo_jax.gae import compute_chunk_advantages, compute_gae_diagnostics
from sjfpo_jax.state import SJFPOState, build_optimizer, load_pair, optimizer_params


def module(pair):
    """Rebuild an NNX module from its ``(graphdef, state)`` pair, at trace time."""
    return nnx.merge(*pair)


def rms(x):
    """Root mean square over every element, one reduction."""
    return jnp.linalg.norm(x) / jnp.sqrt(x.size)


# --------------------------------------------------------------- the rollout


def generate_rollout(
    state: SJFPOState,
    env,
    total_steps: int,
    action_low,
    action_high,
    cfg,
    key: jax.Array,
    *,
    noisy: bool = True,
    seed: int | None = None,
) -> dict:
    """Roll out open-loop chunks and build one training batch.

    Mirror of ``SJFPOPolicy.generate_rollout`` (``modeling_sjfpo.py:686``). A
    chunk can execute one step before terminating, so the count can hit one per
    requested step. Each chunk is one macro-transition: the backup sums its
    executed rewards, discounts by ``gamma ** h``, and the recursion runs between
    consecutive chunks of the same env.

    The batch carries one raw post-step frame per executed step, separate from
    the overlapping obs windows so the normalizer does not double count, plus
    the per-chunk/MC CFM draws the policy loss reuses for old and new. Those two
    are drawn here and once only: they must be the *same* draws for ``l_old`` and
    ``l_new`` or the ratio is meaningless.

    Returns ``(batch, state)``. The state comes back because the normaliser was
    not touched -- it is the caller's ``update_observation_stats`` job -- but the
    buffer is consumed, so the trainer needs the step delta to advance
    ``global_step``.
    ``noisy=False`` integrates from eps=0 with no perturbation, which is the
    deterministic mode the parity test drives.

    ``seed`` pins the env reset. torch's ``generate_rollout`` has no such
    parameter and derives one from the global torch generator, so the parity test
    patches ``modeling_sjfpo.collect_chunks_vec`` to pass one instead; the seam
    lives here because without it the two rollouts cannot be compared at all.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps={total_steps} must be positive")

    from sjfpo_jax.collector import RolloutBuffer, collect_chunks_vec

    # obs_dim off the normaliser rather than cfg: the torch config has no such
    # field (torch reads it off PolicyFeature) and the buffer must hold exactly
    # the width the normaliser will be fed.
    obs_dim = int(state.obs_normalizer.mean.shape[0])
    action_dim = int(state.action_low.shape[0])
    max_chunks = max(1, total_steps)
    buffer = RolloutBuffer(
        max_chunks=max_chunks,
        max_steps=total_steps,
        obs_steps=cfg.n_obs_steps,
        obs_dim=obs_dim,
        action_horizon=cfg.action_horizon,
        action_dim=action_dim,
        # One chunk per requested step, which is the true worst case: a chunk cut
        # by a terminal can execute a single step. The buffer's own default is
        # ceil(total_steps / action_horizon) + 8, and torch sidesteps the
        # shortfall by growing its arrays after the fact (ensure_chunk_capacity).
        # This buffer is fixed-capacity, so it has to be sized right up front.
        chunk_capacity=max_chunks,
    )

    propose_key, sample_key = jax.random.split(key)
    wavefront = {"n": 0}

    def propose(states):
        # One noise draw per wavefront. A fixed key would hand every chunk the
        # same eps, which correlates the exploration noise across the whole
        # rollout -- and with noisy=False the key is unused, so this only costs
        # anything on the noisy path. The counter is safe because the collector
        # is a Python loop and this function is never traced.
        k = jax.random.fold_in(propose_key, wavefront["n"])
        wavefront["n"] += 1
        # The collector holds a host-side env, so the chunk crosses to the host
        # and comes back: one transfer per wavefront, which is what
        # collect_chunks_vec batches for.
        chunk = predict_action_chunk(
            state,
            jnp.asarray(states, dtype=jnp.float32),
            cfg,
            zero_sampling=not noisy,
            key=k,
            perturb=noisy,
        )
        return np.asarray(jax.device_get(chunk))

    buffer = collect_chunks_vec(
        env,
        propose,
        total_steps,
        action_low,
        action_high,
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        chunk_size=cfg.action_horizon,
        buffer=buffer,
        seed=seed,
    )

    active = buffer.active()
    n_chunks = buffer.n_chunks
    num_tokens = cfg.num_flow_tokens

    lengths = active["chunk_lengths"]
    chunk_starts = active["chunk_starts"]
    all_rewards = active["step_rewards"]
    actions = active["chunk_actions"]
    states = active["chunk_state"]

    # Only the chunk's first and post-chunk states are read by the backup, so the
    # critic and the normaliser see 2 * n_chunks windows, not 2 * n_steps. The
    # normaliser updates separately from raw frames via normalizer_observations.
    # The chunk's own half is normalised again inside compute_behavior_cfm_loss,
    # from the raw window, which is the same function of the same input.
    normalised_all = state.obs_normalizer(
        jnp.concatenate([states, active["chunk_next_state"]], axis=0)
    )

    all_values = module(state.critic)(normalised_all).astype(jnp.float32)
    # min over the ensemble, backup only. The loss trains every member separately:
    # min has no case inside a V loss, no max to counteract, so it would only
    # bias V low.
    chunk_value = jnp.min(all_values[:, :n_chunks], axis=0)
    if cfg.use_target_critic:
        boot_values = module(state.target_critic)(normalised_all[n_chunks:]).astype(jnp.float32)
        chunk_next_value = jnp.min(boot_values, axis=0)
    else:
        chunk_next_value = jnp.min(all_values[:, n_chunks:], axis=0)

    raw_advantages, raw_returns = compute_chunk_advantages(
        all_rewards,
        chunk_value,
        chunk_next_value,
        chunk_starts,
        lengths,
        chunk_terminated=active["chunk_terminated"],
        chunk_truncated=active["chunk_truncated"],
        chunk_env_id=active["chunk_env_id"],
        chunk_size=cfg.action_horizon,
        gamma=cfg.gamma,
        lam=cfg.gae_lambda,
    )
    raw_values = chunk_value
    normalised_advantages = raw_advantages
    # Population std, matching returns_var. The unbiased default is NaN at a
    # single chunk, which a short final rollout produces, and inflates by
    # sqrt(n/(n-1)) elsewhere.
    if cfg.normalize_advantage and raw_advantages.size > 0:
        normalised_advantages = (raw_advantages - raw_advantages.mean()) / jnp.clip(
            jnp.std(raw_advantages, ddof=0), min=1e-8
        )

    gae_diagnostics = compute_gae_diagnostics(
        active["chunk_env_id"][:n_chunks],
        active["chunk_terminated"][:n_chunks],
        active["chunk_truncated"][:n_chunks],
        raw_advantages,
        raw_returns,
        active.get("chunk_budget_truncated", None)[:n_chunks]
        if "chunk_budget_truncated" in active
        else None,
    )

    action_is_pad = (
        jnp.arange(cfg.action_horizon, dtype=jnp.int32)[None, :] >= lengths[:, None]
    )

    tau = sample_timesteps(
        (n_chunks, cfg.num_mc_samples, num_tokens), cfg, sample_key
    )[..., None]
    eps = sample_noise(
        (n_chunks, cfg.num_mc_samples, num_tokens, action_dim), cfg, sample_key
    )
    expected_eps_shape = (n_chunks, cfg.num_mc_samples, num_tokens, action_dim)
    if eps.shape != expected_eps_shape:
        raise ValueError(
            f"MC noise shape {eps.shape} != expected {expected_eps_shape}"
        )

    # 1.0 at num_kept_tokens == action_horizon (the transform is then a rotation
    # and this is a no-op), falling as the spectrum truncates. None when the DCT
    # is off, so the tracker can skip the key.
    spectral_energy = (
        float(compute_spectral_energy_retained(actions, cfg.num_flow_tokens))
        if cfg.use_dct_transform
        else None
    )

    batch = {
        "observation.state": states,
        "action": actions,
        "action_is_pad": action_is_pad,
        "raw_advantages": raw_advantages,
        "normalised_advantages": normalised_advantages,
        "raw_returns": raw_returns,
        "raw_values": raw_values,
        "normalizer_observations": active["step_frames"],
        # How far the live critic moved from the bootstrap value the targets were
        # built with. With a target critic that gap is the price of the slow
        # teacher; without one it is the target's drift.
        "boot_observations": normalised_all[n_chunks:],
        "boot_values_start": chunk_next_value[:, None],
        "mc_tau": tau,
        "mc_eps": eps,
        "chunk_clip_counts": active["chunk_clip_counts"][:n_chunks],
        "gae_diagnostics": gae_diagnostics,
        "compute_spectral_energy_retained": spectral_energy,
    }
    # l_old is drawn here rather than inside the loss: the ratio compares the
    # snapshot and the live policy on the *same* CFM samples.
    loss, loss_token = compute_behavior_cfm_loss(state, batch, cfg)
    batch["behavior_cfm_loss"] = loss
    batch["behavior_cfm_loss_token"] = loss_token
    return batch


# ------------------------------------------------------------------ the update


def _state_from_params(state: SJFPOState, grouped: dict) -> SJFPOState:
    """Rebuild the state from ``optimizer_params``' regrouping.

    Differentiating w.r.t. ``grouped`` rather than the whole ``state`` is what
    makes the gradient tree line up with the optimizer's: a ``(graphdef, state)``
    pair has a graphdef with no leaves, so ``jax.grad`` over the state returns a
    ``None`` for it and every regrouping after that has to re-derive the shape.
    """
    return state.replace(
        actor=(state.actor[0], grouped["actor"]["actor"]),
        conditioner=(state.conditioner[0], grouped["actor"]["conditioner"]),
        critic=(state.critic[0], grouped["critic"]["critic"]),
    )


def build_update_step(cfg, params):
    """Returns ``update(state, opt_state, batch) -> (state, opt_state, info)``.

    One optimizer step, in this order, and the order is the contract:

    1. ``value_and_grad`` of :func:`forward_loss` w.r.t. the grouped params,
       taking :func:`compute_info` from the *same* ``terms`` -- a second forward
       would be a second traced program, and ``cond_effective_rank`` is only
       meaningful on the row population the loss used.
    2. ``chain.update``: per-role pre-clip norm, masked AdamW moments, masked
       decay, per-leaf lr. In that order, so the decay term is scaled by the
       learning rate like every other term.
    3. ``update_target_critic``. Per **optimizer** step, not per rollout, which
       is what makes it a slow teacher.

    ``update_ema`` is deliberately **not** here: torch calls it once per rollout
    (``train_sjfpo.py:519``), and ``global_step`` is a static field on the state,
    so folding it in would retrace the whole step once per rollout.

    The chain, the decay mask and the per-leaf lr tree are built once, here,
    rather than per call: they depend only on ``cfg`` and the parameter
    structure, and ``build_optimizer`` walks every leaf three times.
    """
    chain, _ = build_optimizer(cfg, params)

    def loss_and_info(grouped, state, batch):
        loss, terms = forward_loss(_state_from_params(state, grouped), batch, cfg)
        return loss, compute_info(terms, cfg)

    @jax.jit
    def update(state: SJFPOState, opt_state, batch):
        grouped = optimizer_params(state)
        (_, info), grads = jax.value_and_grad(
            loss_and_info, argnums=0, has_aux=True
        )(grouped, state, batch)
        updates, opt_state = chain.update(grads, opt_state, grouped)
        # optax returns updates, not parameters: p + u, not u.
        stepped = jax.tree_util.tree_map(lambda u, p: p + u, updates, grouped)
        state = _state_from_params(state, stepped)
        state = update_target_critic(state, cfg)
        return state, opt_state, info

    return update


# --------------------------------------------------------------- pure helpers


def get_token_count_for_cfm_loss(cfg) -> int:
    """How many tokens the CFM loss reduces over.

    The action-step grid when the loss runs on rows, the frequencies when it does
    not. ``token_mean`` inherits this length, so the two must agree or the ratio
    compares different-width vectors.
    """
    on_action_grid = not cfg.use_dct_transform or cfg.cfm_mask_unexecuted
    return cfg.action_horizon if on_action_grid else cfg.num_flow_tokens


def prepare_cfm_loss_inputs(cfg, action_is_pad, *velocities):
    """Execution mask plus the velocities that mask can index.

    ``action_is_pad`` marks steps a chunk never ran, and a mask only indexes a
    velocity laid out over the ``action_horizon`` steps:

    - DCT off: already that layout
    - DCT on with ``cfm_mask_unexecuted``: ``num_flow_tokens`` coefficients where
      every step maps to every coeff, so expand back onto the action steps first
    - the mask drops the tail, so a short chunk gets no gradient from unrun steps
      instead of a wrong-signed one

    The expansion is exact: the inverse DCT is linear, so at
    ``num_kept_tokens == action_horizon`` the basis is orthonormal, the transform
    is a rotation, and the loss is unchanged.

    Both ``l_old`` and ``l_new`` must come through here, else the ratio compares
    means over different positions.
    """
    if cfg.use_dct_transform and cfg.cfm_mask_unexecuted:
        steps = get_token_count_for_cfm_loss(cfg)
        return action_is_pad, tuple(velocity_to_action_steps(v, steps) for v in velocities)
    # DCT off: the velocities already are the action steps. DCT on with the mask
    # off, the mask would index the frequency axis, so drop it.
    mask = None if cfg.use_dct_transform else action_is_pad
    return mask, velocities


def compute_target_velocity(a, eps):
    """Flow-matching target velocity ``v* = a - eps``, over the MC draws."""
    return a[:, None] - eps


def compute_mc_xt(flatten_mc_timestep, flatten_mc_noise, flatten_mc_action, flow_shape):
    """Interpolate noise to action to form the flow point. ``[B*M, *flow_shape]``.

    ``run_monte_carlo_sampling`` returns flattened ``[B*M, K*D]`` rows, so put
    the token axis back before blending: the timestep is ``[B*M, K, 1]`` and
    broadcasts across the action dim, exactly as torch's ``expand`` does.

    ``jnp`` has no ``lerp``, so the blend is written out: ``noise + t*(a - noise)``.
    """
    rows = flatten_mc_noise.shape[0]
    noise = flatten_mc_noise.reshape(rows, *flow_shape)
    action = flatten_mc_action.reshape(rows, *flow_shape)
    t = jnp.broadcast_to(flatten_mc_timestep, (rows, flow_shape[0], flow_shape[-1]))
    return noise + t * (action - noise)


def compute_value_huber_loss(pred, target, delta):
    """Elementwise Huber, broadcasting ``target`` over the member axis.

    ``delta`` is a 0-dim array so the threshold can track the batch return RMS.
    ``jnp`` has no Huber, so this is spelled out, and the linear branch is scaled
    by ``delta``:

    - ``|x| <= delta`` -> ``0.5 * x^2``
    - ``|x| > delta``  -> ``delta * (|x| - 0.5 * delta)``

    That second line is not the textbook ``|x| - 0.5*delta``; it is what
    ``F.huber_loss`` computes and what ``tests/test_value_loss.py`` pins (delta=2,
    residual 3 -> ``2*(3-1) = 4``, not ``2.0``). Both branches agree only when
    ``delta == 1``, so a probe at delta=1 cannot tell them apart -- and at the
    running config delta is the batch return RMS, so getting it wrong scales the
    value loss by ~9 and reads as a critic that cannot fit.
    """
    residual = pred - target
    absolute = jnp.abs(residual)
    return jnp.where(
        absolute <= delta,
        0.5 * jnp.square(residual),
        delta * (absolute - 0.5 * delta),
    )


# --------------------------------------------------------- behaviour CFM loss


def compute_behavior_cfm_loss(state: SJFPOState, batch, cfg):
    """The frozen snapshot's CFM loss for one batch: ``(l_old, l_old_token)``.

    Behavior weights are the snapshot taken at rollout start, so this is the
    ``l_old`` term of the ratio. Both reductions are masked exactly like the
    current-policy losses they are compared against, so the ratio never mixes
    means over different positions.

    Observations arrive raw and are normalised here, so the batch stays raw
    everywhere; torch normalises once in ``generate_rollout`` instead, which is
    the same function of the same inputs.
    """
    states = state.obs_normalizer(batch["observation.state"])
    actions = batch["action"]
    tau, eps = batch["mc_tau"], batch["mc_eps"]
    action_is_pad = batch["action_is_pad"]

    token_count = eps.shape[2]
    if cfg.use_dct_transform:
        spectral_action = extract_dct_coeffs(actions, num_modes=token_count)
    else:
        spectral_action = actions

    flow_shape = spectral_action.shape[1:]
    num_mc = eps.shape[1]
    flatten_timestep, flatten_noise, flatten_action = run_monte_carlo_sampling(
        spectral_action, eps, tau
    )
    x_t = compute_mc_xt(flatten_timestep, flatten_noise, flatten_action, flow_shape)
    actor_tau = flatten_timestep.squeeze(-1)

    condition = module(state.old_conditioner)(states)
    condition_flat = jnp.broadcast_to(condition[:, None, :], (condition.shape[0], num_mc, condition.shape[-1]))
    condition_flat = condition_flat.reshape(num_mc * condition.shape[0], condition.shape[-1])
    v_old = module(state.old_actor)(x_t, actor_tau, condition_flat)

    v_star = compute_target_velocity(spectral_action, eps)
    v_old = v_old.reshape(eps.shape[0], num_mc, *flow_shape)
    loss_mask, (v_old, v_star) = prepare_cfm_loss_inputs(cfg, action_is_pad, v_old, v_star)
    return compute_cfm_loss(v_old, v_star, cfg, "both", loss_mask)


# ------------------------------------------------------------------ the loss


def forward_loss(state: SJFPOState, batch, cfg, *, reduction: str = "mean"):
    """ASPO policy loss and Huber value loss for one minibatch.

    One chunk is one macro-action; only executed steps count, because
    ``action_is_pad`` masks the rest out of the CFM loss.

    Returns ``(loss, terms)``. ``terms`` carries the per-chunk vectors and the
    intermediates :func:`compute_info` reduces -- the trainer logs them by name,
    and a second forward pass for diagnostics would double the work. Under
    ``reduction="none"`` the loss is per chunk instead of a scalar.

    ``KeyError`` when the batch carries no ``behavior_cfm_loss``: it comes from
    ``generate_rollout`` and must not be recomputed here.
    """
    if "behavior_cfm_loss" not in batch or "behavior_cfm_loss_token" not in batch:
        raise KeyError(
            "forward_loss requires 'behavior_cfm_loss' and 'behavior_cfm_loss_token' "
            "from generate_rollout. They are computed once per rollout against the "
            "frozen snapshot and must not be recomputed here."
        )
    l_old = batch["behavior_cfm_loss"].astype(jnp.float32)
    l_old_token = batch["behavior_cfm_loss_token"].astype(jnp.float32)

    states = state.obs_normalizer(batch["observation.state"])
    actions = batch["action"]
    action_is_pad = batch["action_is_pad"]
    normalised_advantages = batch["normalised_advantages"]
    raw_returns = batch["raw_returns"]

    if states.shape[-2] != cfg.n_obs_steps:
        raise ValueError(
            f"state length {states.shape[-2]} != n_obs_steps {cfg.n_obs_steps}"
        )
    if actions.shape[-2] != cfg.action_horizon:
        raise ValueError(
            f"action horizon {actions.shape[-2]} != action_horizon {cfg.action_horizon}"
        )
    if action_is_pad.shape != (states.shape[0], cfg.action_horizon):
        raise ValueError(
            f"action_is_pad shape {action_is_pad.shape} != "
            f"{(states.shape[0], cfg.action_horizon)}"
        )

    tokens = get_token_count_for_cfm_loss(cfg)

    # Validation. An invalid batch that silently reshapes is worse than one that
    # raises, and every condition here mirrors torch's (modeling_sjfpo.py 1067,
    # 1071, 1076, 1097, 1103, 1109, 1147, 1154, 1193).
    if ("mc_tau" in batch) != ("mc_eps" in batch):
        raise ValueError("mc_tau and mc_eps must be provided together")
    n = states.shape[0]
    num_mc = cfg.num_mc_samples
    action_dim = actions.shape[-1]
    expected_tau = (n, num_mc, cfg.num_flow_tokens, 1)
    if batch["mc_tau"].shape != expected_tau:
        raise ValueError(f"mc_tau shape {batch['mc_tau'].shape} != {expected_tau}")
    expected_eps = (n, num_mc, cfg.num_flow_tokens, action_dim)
    if batch["mc_eps"].shape != expected_eps:
        raise ValueError(
            f"stored MC noise shape {batch['mc_eps'].shape} != {expected_eps}"
        )
    if l_old.shape != (n, num_mc):
        raise ValueError(f"behavior_cfm_loss shape {l_old.shape} != {(n, num_mc)}")
    expected_tokens = (n, num_mc, tokens)
    if l_old_token.shape != expected_tokens:
        raise ValueError(
            f"behavior_cfm_loss_token shape {l_old_token.shape} != {expected_tokens}"
        )

    dct_action_tokens = (
        extract_dct_coeffs(actions, num_modes=cfg.num_flow_tokens)
        if cfg.use_dct_transform
        else actions
    )
    tau, eps = batch["mc_tau"], batch["mc_eps"]
    flow_shape = dct_action_tokens.shape[1:]
    flatten_timestep, flatten_noise, flatten_action = run_monte_carlo_sampling(
        dct_action_tokens, eps, tau
    )
    x_t_flat = compute_mc_xt(flatten_timestep, flatten_noise, flatten_action, flow_shape)
    actor_tau = flatten_timestep.squeeze(-1)

    condition = module(state.conditioner)(states)
    cond_flat = jnp.broadcast_to(condition[:, None, :], (n, num_mc, condition.shape[-1]))
    cond_flat = cond_flat.reshape(n * num_mc, condition.shape[-1])
    v_new = module(state.actor)(x_t_flat, actor_tau, cond_flat)

    v_star = compute_target_velocity(dct_action_tokens, eps)
    v_new_r = v_new.reshape(n, num_mc, *flow_shape)
    loss_mask, (v_new_r, v_star) = prepare_cfm_loss_inputs(
        cfg, action_is_pad, v_new_r, v_star
    )

    adv_mc = normalised_advantages.astype(jnp.float32)[:, None]
    l_new, l_new_token = compute_cfm_loss(v_new_r, v_star, cfg, "both", loss_mask)
    ratio_token = compute_policy_ratio(l_old_token, l_new_token, cfg)
    ratio = compute_policy_ratio(l_old, l_new, cfg, advantage=adv_mc)
    surrogate = compute_aspo_surrogate(ratio, adv_mc, cfg)
    policy_per_chunk = -jnp.mean(surrogate, axis=1)

    # --- the value half -----------------------------------------------------
    raw_value_pred = module(state.critic)(states).astype(jnp.float32)
    raw_returns_flat = raw_returns.reshape(-1)
    if raw_value_pred.shape[-1] != raw_returns_flat.shape[0]:
        raise ValueError(
            f"value/return shape mismatch: raw_value_pred="
            f"{raw_value_pred.shape}, raw_returns={raw_returns_flat.shape}"
        )
    returns_rms = rms(raw_returns_flat)
    huber_delta = cfg.value_huber_delta * jnp.maximum(returns_rms, 1e-6)
    value_loss_per_chunk = jnp.mean(
        compute_value_huber_loss(
            raw_value_pred, raw_returns_flat[None, :], huber_delta
        ),
        axis=0,
    )
    if cfg.use_clipped_value_loss and "raw_values" in batch:
        rollout_values = batch["raw_values"].astype(jnp.float32)[None, :]
        clipped_value_pred = rollout_values + jnp.clip(
            raw_value_pred - rollout_values, -cfg.clip_param, cfg.clip_param
        )
        value_loss_per_chunk = jnp.maximum(
            value_loss_per_chunk,
            jnp.mean(
                compute_value_huber_loss(clipped_value_pred, raw_returns_flat[None, :], huber_delta),
                axis=0,
            ),
        )

    policy_loss = jnp.mean(policy_per_chunk)
    value_loss = jnp.mean(value_loss_per_chunk)
    if reduction == "mean":
        loss = policy_loss + cfg.value_loss_coef * value_loss
    elif reduction == "none":
        loss = policy_per_chunk + cfg.value_loss_coef * value_loss_per_chunk
    else:
        raise ValueError(f"reduction={reduction!r} must be 'mean' or 'none'")

    # One extra actor row block with statistics on, for compute_info's cond_* keys.
    # It has to happen here: a second forward from the trainer would be a second
    # traced program, and these reductions are part of the same one. Eight rows,
    # not one: 8 chunks x 16 tokens = 128 rows against a 128-wide layer, so the
    # effective rank is measured on a square-ish matrix. One row would give 16
    # rows, and the rank could not exceed 16.
    _, cond_stats = module(state.actor)(
        x_t_flat[:8], actor_tau[:8], cond_flat[:8], return_stats=True
    )

    terms = {
        "policy_per_chunk": policy_per_chunk,
        "value_loss_per_chunk": value_loss_per_chunk,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "l_old": l_old,
        "l_new": l_new,
        "ratio": ratio,
        "ratio_token": ratio_token,
        "action_is_pad": action_is_pad,
        "v_star": v_star,
        "raw_value_pred": raw_value_pred,
        "raw_returns": raw_returns_flat,
        "normalised_advantages": normalised_advantages,
        "returns_rms": returns_rms,
        "cond_stats": cond_stats,
        "num_critics": raw_value_pred.shape[0],
        "tokens": tokens,
        "action_dim": action_dim,
    }
    return loss, terms


def compute_info(terms, cfg):
    """Every diagnostic reduction, as 0-dim arrays still on device.

    The trainer does one ``jax.device_get`` on the whole dict per update, which is
    JAX's version of torch's single packed ``.tolist()``. 28 named scalars plus
    ``cond_*``.

    The keys are an interface, not a convenience: the trainer logs them by name
    (``train_sjfpo.py:570-600``), so a renamed key silently stops a metric
    appearing on W&B.
    """
    l_old, l_new = terms["l_old"], terms["l_new"]
    ratio, ratio_token = terms["ratio"], terms["ratio_token"]
    action_is_pad = terms["action_is_pad"]
    raw_value_pred, raw_returns = terms["raw_value_pred"], terms["raw_returns"]

    # The log ratio is read off the raw delta, never the clamped one, so the
    # figure is never a clamp artefact.
    delta = l_old - l_new
    log_ratio_abs = jnp.abs(delta)
    def outside(r):
        """1.0 where the ratio sits outside the trust region."""
        return ((r < 1.0 - cfg.clip_param) | (r > 1.0 + cfg.clip_param)).astype(jnp.float32)

    ratio_token_mean = jnp.mean(ratio_token, axis=(0, 1))
    num_tokens = ratio_token_mean.shape[0]
    split = max(1, num_tokens // 2)
    low_slice = ratio_token[..., :split]
    low_outside = jnp.mean(outside(low_slice))
    # A single-token axis has no high half; torch reuses the low figure there.
    high_outside = jnp.mean(outside(ratio_token[..., split:])) if num_tokens > 1 else low_outside

    # explained_var scores the member mean, not the min the backup uses: min is
    # biased low, so scoring it would flatter the critic. Not comparable in
    # absolute terms to a single-critic run, but the trend is.
    mean_value_pred = jnp.mean(raw_value_pred, axis=0)
    returns_var = jnp.clip(jnp.var(raw_returns, ddof=0), min=1e-8)
    explained_var = 1.0 - jnp.var(raw_returns - mean_value_pred, ddof=0) / returns_var

    l_old_std = jnp.std(l_old, ddof=0)
    l_new_std = jnp.std(l_new, ddof=0)
    corr_denom = l_old_std * l_new_std + 1e-8
    l_old_new_corr = jnp.mean((l_old - jnp.mean(l_old)) * (l_new - jnp.mean(l_new))) / corr_denom

    info = {
        "policy_loss": jnp.mean(terms["policy_per_chunk"]),
        "value_loss": jnp.mean(terms["value_loss_per_chunk"]),
        "explained_var": explained_var,
        "mean_log_ratio": jnp.mean(delta),
        "log_ratio_max": jnp.max(log_ratio_abs),
        "log_ratio_clip_frac": jnp.mean((log_ratio_abs > cfg.log_ratio_cap).astype(jnp.float32)),
        # Does the trust region ever bind? Near zero means clip_param constrains
        # nothing, whatever its value.
        "clipfrac": jnp.mean(outside(ratio)),
        "value_huber_max": jnp.max(jnp.abs(mean_value_pred - raw_returns)),
        "value_pred_max": jnp.max(jnp.abs(mean_value_pred)),
        "flow_ratio_mean": jnp.mean(ratio),
        "flow_ratio_std": jnp.std(ratio, ddof=0),
        "flow_ratio_token_range": jnp.max(ratio_token_mean) - jnp.min(ratio_token_mean),
        "flow_ratio_token_lowfreq": jnp.mean(ratio_token_mean[:split]),
        "flow_ratio_token_highfreq": (
            jnp.mean(ratio_token_mean[split:]) if num_tokens > 1 else jnp.mean(ratio_token_mean)
        ),
        "flow_ratio_token_outside_frac": jnp.mean(outside(ratio_token)),
        "flow_ratio_token_low_outside_frac": low_outside,
        "flow_ratio_token_high_outside_frac": high_outside,
        "flow_v_star_rms": rms(terms["v_star"]),
        "l_old_mean": jnp.mean(l_old),
        "l_new_mean": jnp.mean(l_new),
        "l_old_std": l_old_std,
        "l_new_std": l_new_std,
        # Samples above cfm_clamp contribute no gradient at all, so a run that
        # stays high is training the critic only.
        "cfm_above_clamp_frac": jnp.mean((l_new > cfg.cfm_clamp).astype(jnp.float32)),
        "l_old_new_corr": l_old_new_corr,
        # Chunks with any never-executed action row. Those rows still enter the
        # DCT target, so this bounds the contamination.
        "truncated_chunk_frac": jnp.mean(jnp.any(action_is_pad, axis=1).astype(jnp.float32)),
        "executed_action_step_frac": jnp.mean((~action_is_pad).astype(jnp.float32)),
        "value_target_std": jnp.std(raw_returns, ddof=0),
        "value_pred_std": jnp.std(raw_value_pred, ddof=0),
    }
    info |= {f"cond_{k}": v for k, v in terms["cond_stats"].items()}
    return info


def predict_action_chunk(
    state: SJFPOState,
    states: jax.Array,
    cfg,
    *,
    zero_sampling: bool = True,
    key: jax.Array | None = None,
    perturb: bool = False,
) -> jax.Array:
    """Predict a full action chunk. ``[B, T, obs]`` -> ``[B, H, action_dim]``.

    Action perturbation runs **before** the unscale, matching FPO++, so the noise
    lands in normalised action space rather than scaling with the env's action
    range. ``zero_sampling=True`` integrates from ``eps=0`` over
    ``num_inference_steps``, which is what FPO++ uses for eval and for the
    reported metrics; ``False`` integrates from sampled noise over
    ``num_euler_steps``.

    ``perturb`` stands in for torch's ``self.training`` flag, which a free
    function cannot read. The caller decides.
    """
    states = state.obs_normalizer(states)
    condition = module(state.conditioner)(states)
    shape = (states.shape[0], cfg.num_flow_tokens, state.action_low.shape[0])

    # zero_sampling starts the flow at eps=0; the other path draws through
    # sample_noise so noise_scaling_mode applies. The old code called torch.randn
    # here and skipped it, so freq_weighted trained pink and rolled out white.
    if zero_sampling:
        eps = jnp.zeros(shape, dtype=jnp.float32)
        num_steps = cfg.num_inference_steps
    else:
        if key is None:
            raise ValueError("zero_sampling=False needs a key to draw eps with")
        eps = sample_noise(shape, cfg, key)
        num_steps = cfg.num_euler_steps

    x0 = sample_action(module(state.actor), eps, condition, num_steps)

    if cfg.use_dct_transform:
        # velocity_to_action_steps reads the coeffs back on the half-open grid
        # t_j = j*H/H. linspace(0, 1, K) instead queries j = 0, 1.067, ..., K,
        # which time-warps every executed action by K/(K-1).
        scaled = velocity_to_action_steps(x0, cfg.action_horizon)
    else:
        scaled = x0[:, : cfg.action_horizon]
    scaled = jnp.clip(scaled, -1.0, 1.0)

    if perturb and cfg.action_perturb_std > 0:
        noise = jax.random.normal(key, scaled.shape, dtype=jnp.float32)
        scaled = scaled + cfg.action_perturb_std * noise
    # Clip to [-1, 1] first, then rescale. The other order makes the trainer's
    # action_abs_max > 1.05 guard trip on a perfectly healthy policy.
    low, high = state.action_low, state.action_high
    return 0.5 * (high + low) + (0.5 * (high - low)) * scaled


# ------------------------------------------------- the receding eval chunk


@dataclass(frozen=True)
class ActionBuffer:
    """The receding-chunk eval state: an obs ring and an action queue.

    torch keeps two ``deque``s on the policy (``modeling_sjfpo.py:240-241``) and
    clears them in ``reset()``. Fixed-size arrays here instead, so a step is a
    pure function of ``(state, buffer, obs)`` and a future eval can be vmapped
    over episodes.

    ``frames`` is a ring of the last ``n_obs_steps`` observations; ``frame_pos``
    is the next write slot and ``frame_len`` how many are valid. ``actions`` holds
    at most one chunk's worth of rows; ``head`` is the next row to execute and
    ``length`` how many are left.
    """

    frames: jax.Array
    frame_pos: jax.Array
    frame_len: jax.Array
    actions: jax.Array
    head: jax.Array
    length: jax.Array


def make_action_buffer(obs_dim: int, n_obs_steps: int, action_dim: int) -> ActionBuffer:
    """An empty buffer. This is ``SJFPOPolicy.reset()``."""
    return ActionBuffer(
        frames=jnp.zeros((n_obs_steps, obs_dim), dtype=jnp.float32),
        frame_pos=jnp.zeros((), dtype=jnp.int32),
        frame_len=jnp.zeros((), dtype=jnp.int32),
        actions=jnp.zeros((n_obs_steps, action_dim), dtype=jnp.float32),
        head=jnp.zeros((), dtype=jnp.int32),
        length=jnp.zeros((), dtype=jnp.int32),
    )


def _chronological_frames(buffer: ActionBuffer, n_obs_steps: int) -> jax.Array:
    """``[n_obs_steps, obs_dim]`` oldest-first, padded the way torch pads.

    While the ring is still filling, torch's ``deque(maxlen=n)`` still holds the
    frames it has seen plus the *first* frame repeated to fill the window -- the
    evicted element is the oldest, which is that repeat. So a partly filled ring
    is ``frames[0]`` repeated ``n - len`` times followed by ``frames[:len]``, not
    zero padding.

    Both candidate windows are built with static-shaped gathers and then selected,
    because a traced cursor cannot be used as a slice bound: ``frames[pos:]``
    raises ``IndexError`` under ``jit``. ``jnp.roll`` takes the cursor, and the
    filling case is a gather whose index is ``clamp(i - (n - len), 0, n - 1)``.
    """
    frames, length, pos = buffer.frames, buffer.frame_len, buffer.frame_pos
    if not isinstance(length, jax.core.Tracer):
        if int(length) >= n_obs_steps:
            return jnp.roll(frames, -int(pos), axis=0)
        return frames[jnp.clip(jnp.arange(n_obs_steps) - (n_obs_steps - int(length)), 0, n_obs_steps - 1)]
    rows = jnp.arange(n_obs_steps)
    filling = frames[jnp.clip(rows - (n_obs_steps - length), 0, n_obs_steps - 1)]
    return jnp.where(length >= n_obs_steps, jnp.roll(frames, -pos, axis=0), filling)


def select_action(
    state: SJFPOState,
    buffer: ActionBuffer,
    obs: jax.Array,
    cfg,
    *,
    zero_sampling: bool = True,
    key: jax.Array | None = None,
) -> tuple[jax.Array, ActionBuffer]:
    """Next action, receding through a cached chunk. Returns ``(action, buffer)``.

    Mirror of ``SJFPOPolicy.select_action`` (``modeling_sjfpo.py:584``).
    ``obs`` is one frame ``[obs]``, which is what the evaluator passes, or a full
    window ``[T, obs]``, which replaces the ring outright the way torch's
    two-dimensional branch does.

    The executed order is ``chunk[0, n_action_steps - 1]`` **first** and
    ``chunk[0, 0]`` last: torch refills its deque back-to-front and pops from the
    front, so the chunk plays in reverse. That is what the reference does and
    this matches it.

    The queue is only written when it is *empty*, so the refill always starts at
    slot zero and no wrap-around arithmetic is needed: the ring is logically empty
    at that point. Sizing it to ``n_obs_steps`` therefore has to cover
    ``n_action_steps``, which it does because both come from the same config and
    ``n_action_steps <= action_horizon``.
    """
    obs = jnp.asarray(obs, dtype=jnp.float32)
    if obs.ndim == 1:
        pos = buffer.frame_pos
        buffer = ActionBuffer(
            frames=buffer.frames.at[pos].set(obs),
            frame_pos=(pos + 1) % buffer.frames.shape[0],
            frame_len=jnp.minimum(buffer.frame_len + 1, buffer.frames.shape[0]),
            actions=buffer.actions,
            head=buffer.head,
            length=buffer.length,
        )
        window = _chronological_frames(buffer, buffer.frames.shape[0])[None]
    elif obs.ndim == 2:
        if obs.shape[0] != cfg.n_obs_steps:
            raise ValueError(
                f"select_action window has {obs.shape[0]} frames, "
                f"expected n_obs_steps={cfg.n_obs_steps}"
            )
        # torch's standardize_obs_tensor turns [T, obs] into [1, T, obs] here.
        window = obs[None]
    else:
        raise ValueError(
            f"select_action expects [obs] or [T, obs]; got {tuple(obs.shape)}"
        )

    empty = buffer.length == 0
    if not isinstance(empty, jax.core.Tracer) and bool(empty):
        chunk = predict_action_chunk(
            state, window, cfg, zero_sampling=zero_sampling, key=key
        )
        n_execute = max(1, min(cfg.n_action_steps, chunk.shape[1]))
        # torch appends chunk[0, i] for i from n_execute-1 down to 0 and pops
        # from the front, so the executed order is chunk[0, n_execute-1] **first**
        # and chunk[0, 0] last -- the chunk plays in reverse. Reversing on the way
        # in reproduces that, and the read is then plain forward-from-zero.
        buffer = ActionBuffer(
            frames=buffer.frames,
            frame_pos=buffer.frame_pos,
            frame_len=buffer.frame_len,
            actions=buffer.actions.at[:n_execute].set(chunk[0, :n_execute][::-1]),
            head=jnp.zeros((), dtype=jnp.int32),
            length=jnp.full((), n_execute, dtype=jnp.int32),
        )
    elif isinstance(empty, jax.core.Tracer):
        # Traced: take the chunk unconditionally and select, so one program covers
        # both the refill and the cached steps. The window is [1, T, obs] either
        # way, and the cached branch simply ignores the result.
        chunk = predict_action_chunk(
            state, window, cfg, zero_sampling=zero_sampling, key=key
        )
        n_execute = max(1, min(cfg.n_action_steps, chunk.shape[1]))
        refilled = ActionBuffer(
            frames=buffer.frames,
            frame_pos=buffer.frame_pos,
            frame_len=buffer.frame_len,
            actions=buffer.actions.at[:n_execute].set(chunk[0, :n_execute][::-1]),
            head=jnp.zeros((), dtype=jnp.int32),
            length=jnp.full((), n_execute, dtype=jnp.int32),
        )
        buffer = jax.tree_util.tree_map(
            lambda cached, fresh: jnp.where(empty, fresh, cached), buffer, refilled
        )

    action = buffer.actions[buffer.head]
    buffer = buffer.replace(
        head=buffer.head + 1,
        length=jnp.maximum(buffer.length - 1, 0),
    )
    return action, buffer


# ------------------------------------------------- snapshots, Polyak, EMA


def snapshot_behavior_policy(state: SJFPOState) -> SJFPOState:
    """Refresh the frozen pi_old snapshot. Call once per rollout, before collection.

    Shares the live arrays rather than copying them, which is safe because JAX
    arrays are immutable: the next optimizer step rebinds ``state.actor`` to new
    arrays and ``old_actor`` keeps the values it was given. ``copy.deepcopy`` and
    ``load_state_dict`` both exist in torch only to get this same effect.
    """
    return state.replace(old_actor=state.actor, old_conditioner=state.conditioner)


def update_target_critic(state: SJFPOState, cfg) -> SJFPOState:
    """Polyak the target critic toward the live one, per optimizer step.

    First call hard-copies. ``gamma ** action_horizon`` puts ~0.85 of every value
    target on the live critic's own output, and targets then sit for a whole
    rollout while the critic trains -- the same slow teacher qc (``tau=0.005``) and
    TD-JEPA use.
    """
    if not state.target_critic_initialised:
        return state.replace(target_critic=state.critic, target_critic_initialised=True)
    decay = cfg.critic_ema_decay
    # An NNX State flattens *through* Param to the raw array, so the map takes
    # arrays and must return arrays. Wrapping the result in nnx.Param puts a
    # Param where the next map expects an array, and the second optimizer step
    # fails on a pytree structure mismatch.
    blended = jax.tree_util.tree_map(
        lambda target, live: decay * target + (1.0 - decay) * live,
        state.target_critic[1],
        state.critic[1],
    )
    return state.replace(target_critic=(state.critic[0], blended))


def update_ema(state: SJFPOState, cfg, global_step: int) -> SJFPOState:
    """Update the FPO++ actor EMA. Only warmed weights enter the average.

    Seeds the shadow on the first call, **hard-copies at warmup end**, and decays
    only after. The gate is ``>=`` behind a one-shot latch, not ``==``: global_step
    moves by a whole rollout, so ``== 500`` was never hit and the average stayed
    permanently unready. Without the latch every later call would re-hard-copy and
    throw the running average away.
    """
    decay = cfg.ema_decay
    if decay <= 0:
        return state
    actor_params = state.actor[1]
    if state.ema_shadow is None:
        state = state.replace(ema_shadow=actor_params)
    if not state.ema_initialised and global_step >= cfg.ema_warmup_steps:
        return state.replace(ema_shadow=actor_params, ema_initialised=True)
    if global_step > cfg.ema_warmup_steps:
        shadow = jax.tree_util.tree_map(
            lambda shadow_p, live_p: decay * shadow_p + (1.0 - decay) * live_p,
            state.ema_shadow,
            actor_params,
        )
        return state.replace(ema_shadow=shadow)
    return state


def ema_ready(state: SJFPOState, cfg, global_step: int) -> bool:
    """Can the EMA shadow be used for evaluation yet? Strictly after warmup."""
    return (
        cfg.ema_decay > 0
        and state.ema_shadow is not None
        and state.ema_initialised
        and global_step > cfg.ema_warmup_steps
    )


def swap_in_ema(state: SJFPOState, cfg, global_step: int):
    """Put the average in the actor. Returns ``(state, displaced)``.

    The caller holds ``displaced`` across evaluation and hands it back to
    :func:`swap_out_ema`, so training never runs on the average. ``displaced`` is
    ``None`` when the shadow is not ready -- a no-op, not an error.
    """
    if not ema_ready(state, cfg, global_step):
        return state, None
    swapped = jax.tree_util.tree_map(
        lambda _, shadow: shadow,
        state.actor[1],
        state.ema_shadow,
    )
    return state.replace(actor=(state.actor[0], swapped)), state.actor[1]


def swap_out_ema(state: SJFPOState, displaced) -> SJFPOState:
    """Put the weights :func:`swap_in_ema` displaced back into the actor."""
    if displaced is None:
        return state
    return state.replace(actor=(state.actor[0], displaced))


def ema_state_dict(state: SJFPOState, cfg) -> dict:
    """Export the shadow for the checkpoint. Decay, latch and weights."""
    return {
        "decay": cfg.ema_decay,
        "initialised": state.ema_initialised,
        "shadow_params": nnx.to_pure_dict(state.ema_shadow) if state.ema_shadow is not None else {},
    }


def load_ema_state_dict(state: SJFPOState, saved: dict) -> SJFPOState:
    """Restore a shadow from :func:`ema_state_dict`.

    The saved shadow is a plain nested dict of arrays, and it cannot simply be
    merged into the actor's graphdef: ``nnx.merge(graphdef, tree)`` reports
    "Incorrect number of leaves, expected 40, but got 13" on a tree whose leaf
    paths match the graphdef exactly, and ``nnx.State(tree)`` fails the same way.
    The shadow is the actor's own parameters, so it is loaded through the same
    :func:`state.load_pair` walk every other converted tree uses, and the second
    half of the re-split pair is the shadow.
    """
    if not saved.get("shadow_params"):
        return state.replace(ema_shadow=None, ema_initialised=bool(saved.get("initialised", False)))
    # A State, because every other consumer tree_maps the shadow against one.
    shadow = load_pair(state.actor, saved["shadow_params"])[1]
    return state.replace(ema_shadow=shadow, ema_initialised=bool(saved.get("initialised", True)))
