"""Trainer-side helpers for the JAX loop: minibatching and the two diagnostics.

Mirror of the same-named functions in ``spectral_joint_policy/train_sjfpo.py``,
same contract, same docstring mathematics. ``spectral_joint_policy.evaluator`` and
``spectral_joint_policy.tracking`` are numpy-only and are imported as they are;
everything that needs a torch tensor is here instead.

``train_sjfpo_run`` arrives in milestone 4.5. What is here now is the part the
loss consumes: which chunks each minibatch sees, and the scalars the trainer logs
once per update.
"""
from __future__ import annotations

import dataclasses
import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from sjfpo_jax.checkpointing import load_checkpoint, save_checkpoint
from sjfpo_jax.gae import GAE_DIAGNOSTIC_KEYS
from sjfpo_jax.policy import (
    ActionBuffer,
    build_update_step,
    generate_rollout,
    make_action_buffer,
    module,
    select_action,
    snapshot_behavior_policy,
    swap_in_ema,
    swap_out_ema,
    update_ema,
)
from sjfpo_jax.runtime import configure_device, describe
from sjfpo_jax.state import (
    build_optimizer,
    build_state,
    optimizer_params,
    pre_clip_norms,
    update_normalizer,
)

# evaluator and tracking are numpy-only (verified: importing either leaves torch
# and lerobot out of sys.modules), so they are reused as they are. Everything else
# torch-shaped is ported below.
from spectral_joint_policy.evaluator import (
    Evaluator,
    check_divergence,
    check_regression,
    compute_mean_chunk_jerk,
)
from spectral_joint_policy.tracking import build_tracker, compute_eval_summary_metrics

# Reused verbatim from the torch loop. Not imported from lerobot, which drags
# torch into this process; see the import-time test.
CHECKPOINTS_DIR = "checkpoints"
ABOVE_CLAMP_ABORT_UPDATES = 5
_OBS_WARMUP_STEPS = 128


class SJFPOAborted(RuntimeError):
    """A guard tripped and the run cannot continue."""


def set_seed(seed: int) -> jax.Array:
    """Seed python and numpy, and return the jax key.

    There is no global JAX RNG to set, so the key is returned and threaded:
    ``generate_rollout`` splits it into the propose and MC streams. torch's
    equivalent comes from ``lerobot.utils.random_utils`` and therefore imports
    torch, which is the whole thing this module avoids.
    """
    random.seed(seed)
    np.random.seed(seed)
    return jax.random.key(seed)

# Batch entries that are not per-chunk, so a chunk index must not slice them.
# ``normalizer_observations`` is one row per executed env step and
# ``gae_diagnostics`` / ``compute_spectral_energy_retained`` are scalars and dicts.
PASSTHROUGH_KEYS = ("normalizer_observations", "gae_diagnostics",
                    "compute_spectral_energy_retained")


def generate_mini_batch_indices(
    n: int,
    num_mini_batches: int,
    key: jax.Array,
    num_learning_epochs: int = 1,
):
    """Shuffled minibatch indices, one array per epoch.

    Every epoch reshuffles; splits are proportional ``(i * n) // n_mb`` so a
    remainder spreads over the frequencies instead of being dropped.

    The torch original draws from the global torch generator and names no seed;
    this takes a ``jax`` key so the whole loop stays on one explicit stream.
    """
    if n < 1:
        raise ValueError(f"n={n} must be >= 1")
    n_mb = max(1, min(int(num_mini_batches), n))
    for epoch in range(max(1, int(num_learning_epochs))):
        perm = jax.random.permutation(jax.random.fold_in(key, epoch), n)
        for i in range(n_mb):
            start = (i * n) // n_mb
            stop = ((i + 1) * n) // n_mb
            yield perm[start:stop]


def sample_minibatch(batch: dict, indices: jax.Array, out: dict | None = None) -> dict:
    """Slice the leading dim of every per-chunk batch value.

    They are all ``[B, ...]``, so one index tensor keeps them consistent.
    ``out`` is reused across minibatches to avoid reallocating every slice.

    ``PASSTHROUGH_KEYS`` ride along whole: one row per executed env step rather
    than per chunk, or a scalar.
    """
    if out is None:
        out = {}
    for key, value in batch.items():
        if key in PASSTHROUGH_KEYS:
            out[key] = value
        else:
            out[key] = value[indices]
    return out


def compute_rollout_diagnostics(batch: dict) -> dict[str, float]:
    """Post-rollout scalars, one device_get for the whole dict.

    Separate reads cost one transfer each, and ``action_abs_max`` blocks on one.
    """
    action = batch["action"].astype(jnp.float32)
    scalars = [
        (~batch["action_is_pad"]).sum(),
        jnp.abs(action).max(),
        jnp.abs(action).mean(),
    ]
    keys = ["step_delta", "action_abs_max", "action_magnitude"]
    clip_counts = batch.get("chunk_clip_counts")
    if clip_counts is not None:
        scalars.append(clip_counts.astype(jnp.float32).mean())
        keys.append("chunk_clip_counts")
    return dict(zip(keys, _one_transfer(scalars)))


def gae_diagnostics_as_scalars(diagnostics: dict[str, jax.Array]) -> dict[str, float]:
    """The GAE block in :data:`GAE_DIAGNOSTIC_KEYS` order, as Python floats.

    ``generate_rollout`` leaves these as 0-dim device arrays so the batch stays a
    pure pytree; this is where the trainer pays for one transfer.
    """
    return dict(zip(GAE_DIAGNOSTIC_KEYS, _one_transfer([diagnostics[k] for k in GAE_DIAGNOSTIC_KEYS])))


def measure_value_drift(state, batch, module) -> tuple[jax.Array, jax.Array]:
    """Both critic-drift metrics off one critic forward.

    ``boot_value_delta``: relative RMS move of the live critic from the bootstrap
    value in use. Small means staleness is not what limits the critic.
    ``value_target_drift``: the same move in value-loss units, a fraction of
    target RMS. Large against V can still be negligible against the return.

    ``module`` is ``policy.module``, passed in rather than imported so this file
    does not depend on the policy module.
    """
    boot_obs = batch["boot_observations"].astype(jnp.float32)
    start = batch["boot_values_start"].astype(jnp.float32).reshape(-1)
    delta = jnp.min(module(state.critic)(boot_obs).astype(jnp.float32), axis=0) - start
    drift_rms = _rms(delta)
    return (
        drift_rms / (_rms(start) + 1e-8),
        drift_rms / (_rms(batch["raw_returns"].astype(jnp.float32)) + 1e-8),
    )


def make_vec_env(env_name: str, num_envs: int):
    """Vectorised env, async workers preferred. Falls back to ``SyncVectorEnv``
    when async is unavailable (no ``/dev/shm``, env-factory pickling failure).

    ``autoreset_mode`` must survive vectorisation in either branch; the collector
    validates NEXT_STEP, see :func:`collector.validate_autoreset_mode`.
    """
    env_fns = [lambda: gym.make(env_name) for _ in range(num_envs)]
    try:
        return gym.vector.AsyncVectorEnv(env_fns)
    except Exception:
        return gym.vector.SyncVectorEnv(env_fns)


def warm_observation_stats(state, env, *, obs_dim: int, seed: int):
    """Fold random-policy frames into the running obs stats.

    The normalizer is identity until its first update and updates only *after* a
    rollout, so without this the whole first rollout runs on raw obs. Env is reset
    after.
    """
    env.action_space.seed(seed)
    obs, _ = env.reset()
    frames = []
    for _ in range(_OBS_WARMUP_STEPS):
        obs, _, terminated, truncated, _ = env.step(
            np.asarray(env.action_space.sample(), dtype=np.float32)
        )
        frames.append(np.asarray(obs, dtype=np.float32).reshape(-1, obs_dim))
        if bool(np.any(terminated)) or bool(np.any(truncated)):
            obs, _ = env.reset()
    stacked = np.concatenate(frames, axis=0)
    state = update_normalizer(state, jnp.asarray(stacked))
    env.reset()
    return state, int(stacked.shape[0])


def sample_random_baseline(evaluator, env, max_episodes: int = 30) -> float:
    """Random-policy return, the divergence reference.

    ``0.0`` when no episode evaluated. Callers gate on ``baseline > 0``, so that
    disables divergence detection instead of falsely flagging every later run.
    """
    baseline_evaluator = type(evaluator)(
        env,
        eval_seed=evaluator.eval_seed,
        max_episodes=max_episodes,
        max_steps=evaluator.max_steps,
    )
    results = baseline_evaluator.evaluate(lambda obs: env.action_space.sample(), out_dir=None)
    if not results:
        return 0.0
    return float(np.mean([r["return"] for r in results]))


class EvalPolicy:
    """Single-action eval adapter over :func:`policy.select_action`.

    ``zero_sampling`` picks the flow's start point: ``True`` from ``eps=0``, what
    eval and the reported metrics use; ``False`` from sampled noise, what the
    training rollouts use. Running both over the same seeds is the only way to see
    the two distributions diverge.

    torch's ``_EvalPolicy`` is a class because the buffers live on the policy and
    ``reset()`` clears them in place. Here the buffer is threaded functionally,
    but ``Evaluator`` calls ``policy.reset()`` and discards the return value, so
    it still has to be an attribute.
    """

    def __init__(self, state, cfg, *, zero_sampling: bool = True, key=None):
        self.state = state
        self.cfg = cfg
        self.zero_sampling = zero_sampling
        # The key is only read on the sampled-eps path; fold_in per call so the
        # noise differs between the two eval passes.
        self.key = key
        self.call = 0
        self.reset()

    def reset(self) -> None:
        """Drop the cached chunk and obs window before a new episode."""
        # obs_dim off the normaliser, not cfg: the torch config has no such field
        # and the buffer must hold exactly the width the normaliser is fed.
        self.buffer: ActionBuffer = make_action_buffer(
            int(self.state.obs_normalizer.mean.shape[0]),
            self.cfg.n_obs_steps,
            int(self.state.action_low.shape[0]),
        )
        self.call = 0

    def __call__(self, obs) -> np.ndarray:
        key = None
        if self.key is not None:
            key = jax.random.fold_in(self.key, self.call)
        self.call += 1
        action, self.buffer = select_action(
            self.state,
            self.buffer,
            jnp.asarray(obs, dtype=jnp.float32),
            self.cfg,
            zero_sampling=self.zero_sampling,
            key=key,
        )
        return np.asarray(action, dtype=np.float32).reshape(-1)


def _rms(x):
    """Root mean square over every element, one reduction."""
    return jnp.linalg.vector_norm(x) / math.sqrt(x.size)


def _one_transfer(scalars):
    """``[s0, s1, ...]`` -> Python floats, in one device_get.

    ``np.asarray`` on a stacked array is the JAX version of torch's single packed
    ``.tolist()``; a per-scalar read would be a transfer each.
    """
    stacked = jnp.stack([jnp.asarray(s, dtype=jnp.float64) for s in scalars])
    return np.asarray(jax.device_get(stacked)).tolist()


# ------------------------------------------------------------------ the run


def train_sjfpo_run(
    env,
    out_dir: str | Path,
    cfg=None,
    *,
    seed: int = 3407,
    total_steps: int = 2000,
    eval_steps: int = 1000,
    save_steps: int = 1000,
    obs_dim: int | None = None,
    action_dim: int | None = None,
    resume_from_checkpoint: str | Path | None = None,
) -> dict:
    """One SJFPO training run in JAX. Returns the same dict torch's does.

    Env builds before any device initialisation, so forked workers do not inherit
    a parent's accelerator context.

    The loop is Python, not a fused ``lax.scan`` over the whole run: one step is
    one ``update_step`` call, and each of those is a compiled program. Fusing the
    rollouts into the step needs a GPU to be worth the debugging.
    """
    from sjfpo_jax.configuration_sjfpo import SJFPOConfig

    cfg = cfg or SJFPOConfig()
    if cfg.grad_accum_steps != 1:
        # update_step is one clip-and-AdamW, so there is no group boundary to
        # accumulate towards. Refusing beats silently ignoring the knob.
        raise ValueError(
            f"grad_accum_steps={cfg.grad_accum_steps} is not supported: "
            "update_step is one optimizer step, so there is no accumulation "
            "boundary. Use 1."
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = getattr(env, "unwrapped", env)
    spec = getattr(unwrapped, "spec", None)
    env_name = getattr(spec, "id", "unknown") if spec is not None else "unknown"
    if obs_dim is None or action_dim is None:
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
    cfg = dataclasses.replace(cfg, obs_dim=int(obs_dim), action_dim=int(action_dim))

    key = set_seed(seed)
    device = configure_device()

    action_low = np.atleast_1d(np.asarray(env.action_space.low, dtype=np.float32))
    action_high = np.atleast_1d(np.asarray(env.action_space.high, dtype=np.float32))

    if resume_from_checkpoint is not None:
        state, opt_state, cfg, counters = load_checkpoint(resume_from_checkpoint)
        global_step = int(counters["global_step"])
        (out_dir / "resume_info.json").write_text(
            json.dumps(
                {
                    "resume_from_checkpoint": str(resume_from_checkpoint),
                    "restored_global_step": global_step,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        state = build_state(cfg, obs_dim, action_dim, key)
        state = state.replace(
            action_low=jnp.asarray(action_low, dtype=jnp.float32),
            action_high=jnp.asarray(action_high, dtype=jnp.float32),
        )
        _, opt_state = build_optimizer(cfg, optimizer_params(state))
        global_step = 0

    train_env = env
    if cfg.num_envs > 1 and env_name != "unknown":
        train_env = make_vec_env(env_name, cfg.num_envs)

    state, warmed = warm_observation_stats(
        state, train_env, obs_dim=int(obs_dim), seed=seed
    )
    print(f"obs normalizer warmed on {warmed} frames", file=sys.stderr)
    print(
        f"SJFPO jax policy on {describe(device)}: actor_depth={cfg.dit_depth} "
        f"hidden={cfg.hidden_dim}",
        file=sys.stderr,
    )

    evaluator = Evaluator(env, eval_seed=10000 + seed, max_episodes=cfg.eval_max_episodes)
    r_rand = sample_random_baseline(evaluator, env)
    tracker = build_tracker(
        cfg,
        out_dir=out_dir,
        method="sjfpo_jax",
        env_name=env_name,
        seed=seed,
        total_steps=total_steps,
    )

    next_eval, next_save = eval_steps, save_steps
    eval_count = 0
    consecutive_below = 0
    consecutive_below_best = 0
    lr_decays = 0
    above_clamp_updates = 0
    diverged = False
    regressed = False
    best_return: float | None = None
    steps_to_best: int | None = None
    recent_eval_returns: deque[float] = deque(maxlen=10)
    all_eval_returns: list[float] = []
    artifacts: list[str] = []
    best_artifact: str | None = None
    elapsed = 0.0
    fps = 0.0
    t_start = time.time()

    # Built once. An lr decay rebuilds it, at most max_lr_decays times per run.
    update = build_update_step(cfg, optimizer_params(state))
    minibatch: dict = {}

    try:
        while global_step < total_steps:
            collect_steps = min(cfg.chunks_per_update * cfg.action_horizon, total_steps - global_step)
            state = snapshot_behavior_policy(state)
            batch = generate_rollout(
                state,
                train_env,
                collect_steps,
                action_low,
                action_high,
                cfg,
                jax.random.fold_in(key, global_step),
                noisy=True,
            )
            rollout = compute_rollout_diagnostics(batch)
            step_delta = int(rollout["step_delta"])
            if step_delta <= 0:
                raise RuntimeError(
                    "rollout returned zero executed steps; aborting to avoid "
                    "an infinite training loop."
                )
            global_step += step_delta
            # The state has to carry it: checkpoint_tree reads
            # state.global_step, so without this every checkpoint records 0 and a
            # resume restarts the run from the beginning. global_step is a static
            # field (pytree_node=False), so replacing it changes no array.
            state = state.replace(global_step=global_step)

            rollout_jerk = None
            if "action" in batch and "action_is_pad" in batch:
                rollout_jerk = compute_mean_chunk_jerk(
                    np.asarray(batch["action"]),
                    np.asarray(batch["action_is_pad"]),
                )
            if rollout["action_abs_max"] > 1.05:
                raise ValueError(
                    f"Unnormalised actions detected: max_abs={rollout['action_abs_max']:.3f} "
                    "(expected scaled [-1, 1]; check rollout/action bounds)"
                )

            infos: list[dict[str, float]] = []
            pre_norms: list[dict[str, float]] = []
            n_chunks = batch["raw_advantages"].shape[0]
            splits = list(
                generate_mini_batch_indices(
                    n_chunks,
                    cfg.num_mini_batches,
                    jax.random.fold_in(jax.random.fold_in(key, 1), global_step),
                    num_learning_epochs=cfg.num_learning_epochs,
                )
            )
            for indices in splits:
                minibatch = sample_minibatch(batch, indices, minibatch)
                state, opt_state, info = update(state, opt_state, minibatch)
                infos.append({k: float(v) for k, v in jax.device_get(info).items()})
                pre_norms.append(pre_clip_norms(opt_state))

            if not infos:
                raise RuntimeError("no minibatches were produced; aborting")
            info = {k: sum(d[k] for d in infos) / len(infos) for k in infos[0]}
            actor_pre = float(np.mean([d["actor"] for d in pre_norms]))
            critic_pre = float(np.mean([d["critic"] for d in pre_norms]))
            above_clamp_frac = info["cfm_above_clamp_frac"]

            # Once per update, not per minibatch: a full critic pass over
            # 2 * n_chunks windows, and the number only grows with steps taken.
            drift = None
            if cfg.log_drift_diagnostics and tracker.enabled:
                drift = measure_value_drift(state, batch, module)

            state = update_ema(state, cfg, global_step)
            state = update_normalizer(state, batch["normalizer_observations"])

            gae_diag = gae_diagnostics_as_scalars(batch["gae_diagnostics"])
            if tracker.enabled:
                data = {f"train/{k}": v for k, v in info.items()}
                data["train/actor_grad_norm_preclip"] = actor_pre
                data["train/critic_grad_norm_preclip"] = critic_pre
                if drift is not None:
                    data["train/boot_value_delta"] = float(drift[0])
                    data["train/value_target_drift"] = float(drift[1])
                if rollout_jerk is not None:
                    data["train/rollout_jerk"] = rollout_jerk
                data["train/action_magnitude"] = rollout["action_magnitude"]
                if "chunk_clip_counts" in rollout:
                    data["train/chunk_clip_counts"] = rollout["chunk_clip_counts"]
                if batch.get("compute_spectral_energy_retained") is not None:
                    data["train/compute_spectral_energy_retained"] = float(
                        batch["compute_spectral_energy_retained"]
                    )
                data["train/gae_linked_frac"] = gae_diag["linked_frac"]
                data["train/gae_advantage_std"] = gae_diag["advantage_std"]
                data["train/returns_rms"] = gae_diag["return_rms"]
                data["train/advantage_std_fraction"] = gae_diag["advantage_std_fraction"]
                data["train/budget_truncated_chunk_frac"] = gae_diag[
                    "budget_truncated_chunk_frac"
                ]
                data["train/total_loss"] = (
                    info["policy_loss"] + cfg.value_loss_coef * info["value_loss"]
                )
                tracker.log(data, step=global_step)

            if above_clamp_frac > 0.5:
                above_clamp_updates += 1
                if above_clamp_updates >= ABOVE_CLAMP_ABORT_UPDATES:
                    raise SJFPOAborted(
                        f"{ABOVE_CLAMP_ABORT_UPDATES} consecutive updates with >50% "
                        f"of CFM samples above cfm_clamp={cfg.cfm_clamp} "
                        f"(frac={above_clamp_frac:.2f}, "
                        f"cfm_loss_reduction={cfg.cfm_loss_reduction!r}). The policy "
                        "ratio is pinned at 1 and gets no gradient. Raise "
                        "cfm_clamp, or use a reduction whose scale fits under it."
                    )
            else:
                above_clamp_updates = 0

            diverged_now = regressed_now = False
            diverge_msg = regress_msg = ""
            if global_step >= next_eval or global_step >= total_steps:
                # The EMA actor once it is warm: the live weights go back right
                # after, so training never runs on it. In a finally, or a raise
                # inside eval leaves the live actor holding the average.
                # swap_in_ema returns (state_with_average, displaced_or_None).
                eval_state, displaced = swap_in_ema(state, cfg, global_step)
                try:
                    eval_results = evaluator.evaluate(
                        EvalPolicy(eval_state, cfg, zero_sampling=True),
                        out_dir=out_dir,
                        append_csv=True,
                    )
                    # Same seeds, same evaluator, so the episode pairs line up.
                    stochastic_results = evaluator.evaluate(
                        EvalPolicy(
                            eval_state,
                            cfg,
                            zero_sampling=False,
                            key=jax.random.fold_in(key, 2),
                        )
                    )
                finally:
                    state = swap_out_ema(state, displaced)

                mean_return = float(np.mean([r["return"] for r in eval_results]))
                stochastic_mean_return = (
                    float(np.mean([r["return"] for r in stochastic_results]))
                    if stochastic_results
                    else None
                )
                all_eval_returns.append(mean_return)
                recent_eval_returns.append(mean_return)
                eval_count += 1
                improved = best_return is None or mean_return > best_return
                if improved:
                    best_return = mean_return
                    steps_to_best = global_step
                    best_dir = out_dir / CHECKPOINTS_DIR / "best"
                    save_checkpoint(best_dir, state, opt_state, cfg, global_step)
                    best_artifact = str(best_dir)
                if tracker.enabled:
                    tracker.log(
                        compute_eval_summary_metrics(
                            mean_return,
                            eval_results,
                            all_eval_returns,
                            eval_count,
                            r_rand=r_rand,
                            consecutive_below=consecutive_below,
                            best_return=best_return,
                            steps_to_best=steps_to_best,
                        ),
                        step=global_step,
                    )
                    tracker.log(
                        {
                            "eval/ema_active": float(displaced is not None),
                            "train/policy_lr": cfg.policy_lr,
                            "train/critic_lr": cfg.critic_lr,
                            "train/lr_decays": lr_decays,
                            **(
                                {"train/stochastic_mean_return": stochastic_mean_return}
                                if stochastic_mean_return is not None
                                else {}
                            ),
                        },
                        step=global_step,
                    )
                    for backend in tracker.backends:
                        getattr(backend, "log_eval_table", lambda *_: None)(
                            eval_results, global_step
                        )
                consecutive_below, diverged_now = check_divergence(
                    mean_return, r_rand, consecutive_below
                )
                if diverged_now:
                    diverge_msg = (
                        f"DIVERGED at step {global_step}: mean_return={mean_return:.2f} "
                        f"< 0.10*R_rand={0.10 * r_rand:.2f}"
                    )
                consecutive_below_best, plateaued = check_regression(
                    mean_return,
                    best_return,
                    consecutive_below_best,
                    return_frac=cfg.stop_return_frac,
                    patience=cfg.stop_patience,
                )
                if plateaued:
                    if lr_decays < cfg.max_lr_decays:
                        lr_decays += 1
                        cfg = dataclasses.replace(
                            cfg,
                            policy_lr=cfg.policy_lr * cfg.lr_decay,
                            critic_lr=cfg.critic_lr * cfg.lr_decay,
                        )
                        # Re-bakes lr_tree and re-traces. At most
                        # max_lr_decays times per run.
                        update = build_update_step(cfg, optimizer_params(state))
                        print(
                            f"plateau at step {global_step}: mean_return="
                            f"{mean_return:.2f} < {cfg.stop_return_frac:g}*best="
                            f"{best_return:.2f} for {cfg.stop_patience} evals, "
                            f"lr x{cfg.lr_decay:g} ({lr_decays}/{cfg.max_lr_decays})",
                            file=sys.stderr,
                        )
                    else:
                        regressed_now = True
                        regress_msg = (
                            f"REGRESSED at step {global_step}: mean_return="
                            f"{mean_return:.2f} < {cfg.stop_return_frac:g}*best="
                            f"{best_return:.2f} after {lr_decays} lr decays; "
                            "the best checkpoint is the artifact"
                        )
                next_eval = max(global_step + eval_steps, next_eval + eval_steps)

            if diverged_now or regressed_now:
                diverged, regressed = diverged_now, regressed_now
                if tracker.enabled:
                    tracker.log(
                        {
                            "eval/diverged": float(diverged_now),
                            "eval/diverge_step": global_step if diverged_now else 0,
                            "eval/regressed": float(regressed_now),
                            "eval/regression_step": global_step if regressed_now else 0,
                        },
                        step=global_step,
                    )
                if diverged_now:
                    print(diverge_msg)
                if regressed_now:
                    print(regress_msg, file=sys.stderr)
                break

            if global_step >= next_save or global_step >= total_steps:
                ckpt_dir = out_dir / CHECKPOINTS_DIR / f"step_{global_step}"
                save_checkpoint(ckpt_dir, state, opt_state, cfg, global_step)
                artifacts.append(str(ckpt_dir))
                next_save = max(global_step + save_steps, next_save + save_steps)

        elapsed = time.time() - t_start
        fps = global_step / max(elapsed, 1e-6)
        if tracker.enabled:
            tracker.log(
                {"timing/fps": fps, "timing/elapsed_s": elapsed}, step=global_step
            )
    finally:
        if train_env is not env:
            train_env.close()
        tracker.finish(exit_code=1 if diverged else 0)

    final_dir = out_dir / CHECKPOINTS_DIR / "final"
    save_checkpoint(final_dir, state, opt_state, cfg, global_step)
    artifacts.append(str(final_dir))
    if best_artifact is not None and best_artifact not in artifacts:
        artifacts.append(best_artifact)
    csv_path = out_dir / "eval_history.csv"
    if csv_path.exists():
        artifacts.append(str(csv_path))

    return {
        "method": "sjfpo_jax",
        "env_name": env_name,
        "seed": seed,
        "reward_mean_last10": (
            float(np.mean(recent_eval_returns)) if recent_eval_returns else 0.0
        ),
        "fps": fps,
        "artifacts": artifacts,
        "best_artifact": best_artifact,
        "best_return": best_return,
        "diverged": diverged,
        "regressed": regressed,
        "lr_decays": lr_decays,
        "global_step": global_step,
    }
