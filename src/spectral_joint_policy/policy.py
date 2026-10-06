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
from flax import nnx

from sjfpo_jax.dct_utils import extract_dct_coeffs, velocity_to_action_steps
from sjfpo_jax.fpo_core import (
    compute_aspo_surrogate,
    compute_cfm_loss,
    compute_policy_ratio,
    run_monte_carlo_sampling,
    sample_action,
    sample_noise,
)
from sjfpo_jax.state import SJFPOState


def module(pair):
    """Rebuild an NNX module from its ``(graphdef, state)`` pair, at trace time."""
    return nnx.merge(*pair)


def rms(x):
    """Root mean square over every element, one reduction."""
    return jnp.linalg.norm(x) / jnp.sqrt(x.size)


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

    # One extra actor row with statistics on, for compute_info's cond_* keys. It
    # has to happen here: a second forward from the trainer would be a second
    # traced program, and these reductions are part of the same one.
    _, cond_stats = module(state.actor)(
        x_t_flat[:1], actor_tau[:1], cond_flat[:1], return_stats=True
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

    ``nnx.merge`` takes a plain nested dict of arrays as readily as a ``State``,
    which is the conversion the export needs and ``tree_map`` cannot do: a
    ``State`` and a plain dict are different pytree types.
    """
    if not saved.get("shadow_params"):
        return state.replace(ema_shadow=None, ema_initialised=bool(saved.get("initialised", False)))
    # nnx.merge returns a *module*; the shadow has to be a State, because every
    # other consumer tree_maps it against one.
    shadow = nnx.split(nnx.merge(state.actor[0], jax.tree_util.tree_map(jnp.asarray, saved["shadow_params"])), nnx.Param)[1]
    return state.replace(ema_shadow=shadow, ema_initialised=bool(saved.get("initialised", True)))
