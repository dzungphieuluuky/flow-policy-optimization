"""SJFPO online training entry (LeRobot-native, MuJoCo-only).

``generate_rollout`` gathers a batch, ``forward`` computes the loss. This loop
only steps the optimizer, evaluates, and checkpoints.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.random_utils import set_seed

from spectral_joint_policy.checkpointing import load_policy_checkpoint, save_policy_checkpoint
from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
from spectral_joint_policy.evaluator import (
    Evaluator,
    check_divergence,
    check_regression,
    compute_mean_chunk_jerk,
)
from spectral_joint_policy.modeling_sjfpo import SJFPOAborted, SJFPOPolicy, build_optimizer
from lerobot.utils.constants import CHECKPOINTS_DIR
from spectral_joint_policy.utils import resolve_device, rms, to_numpy_1d
from spectral_joint_policy.tracking import build_tracker, compute_eval_summary_metrics


# Consecutive above-clamp updates before aborting. clamp() has zero gradient
# above its max, so the policy stops learning while the critic keeps training.
ABOVE_CLAMP_ABORT_UPDATES = 5


def generate_mini_batch_indices(
    n: int,
    num_mini_batches: int,
    device: torch.device | str,
    num_learning_epochs: int = 1,
):
    """Shuffled minibatch indices, one tensor per epoch. Every epoch reshuffles;
    splits are proportional ``(i * n) // n_mb`` so a remainder spreads over the
    frequencies instead of being dropped."""
    if n < 1:
        raise ValueError(f"n={n} must be >= 1")
    n_mb = max(1, min(int(num_mini_batches), n))
    device = torch.device(device)
    for _ in range(max(1, int(num_learning_epochs))):
        perm = torch.randperm(n, device=device)
        for i in range(n_mb):
            start = (i * n) // n_mb
            stop = ((i + 1) * n) // n_mb
            yield perm[start:stop]

class _EvalPolicy:
    """Single-action eval adapter over :meth:`SJFPOPolicy.select_action`.

    ``zero_sampling`` picks the flow's start point: ``True`` from ``eps=0``, what
    eval and the reported metrics use; ``False`` from sampled noise, what the
    training rollouts use. Running both over the same seeds is the only way to
    see the two distributions diverge.
    """

    def __init__(self, policy: SJFPOPolicy, zero_sampling: bool = True):
        self.policy = policy
        self.zero_sampling = zero_sampling

    def reset(self) -> None:
        """Drop the policy's cached chunk state before a new episode."""
        self.policy.reset()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        was_training = self.policy.training
        self.policy.eval()
        try:
            action = self.policy.select_action(
                {"observation.state": obs}, zero_sampling=self.zero_sampling
            )
        finally:
            self.policy.train(was_training)
        return to_numpy_1d(action)


def build_policy(
    obs_dim: int,
    action_dim: int,
    cfg: SJFPOConfig,
    device=None,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
    policy_cls: type[SJFPOPolicy] = SJFPOPolicy,
) -> SJFPOPolicy:
    """Attach runtime features, then build the policy.

    ``policy_cls`` lets a swapped actor in without touching the loop.
    """
    device = torch.device(device) if device is not None else resolve_device()
    policy_cfg = dataclasses.replace(
        cfg,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (obs_dim,))
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (action_dim,))},
    )
    policy = policy_cls(policy_cfg).to(device)
    if action_low is not None and action_high is not None:
        policy.set_action_bounds(action_low, action_high)
    return policy


def sample_minibatch(
    batch: dict[str, torch.Tensor],
    indices: torch.Tensor,
    out: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Slice the leading dim of every per-chunk batch value; they are all
    ``[B, ...]``, so one index tensor keeps them consistent.

    ``normalizer_observations`` and ``gae_diagnostics`` pass through: one row
    per executed env step, not per chunk.
    """
    if out is None:
        out = {}
    passthrough = ("normalizer_observations", "gae_diagnostics",
                   "compute_spectral_energy_retained")
    for k, v in batch.items():
        if k in passthrough:
            out[k] = v
        else:
            out[k] = v[indices]
    return out


_OBS_WARMUP_STEPS = 128


def warm_observation_stats(policy, env, *, obs_dim: int, seed: int) -> int:
    """Fold random-policy frames into the running obs stats. The normalizer is
    identity until its first update and updates only after a rollout, so without
    this the whole first rollout runs on raw obs. Env is reset after."""
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
    policy.update_observation_stats(torch.from_numpy(stacked))
    env.reset()
    return int(stacked.shape[0])


@torch.no_grad()
def compute_rollout_diagnostics(batch: dict[str, torch.Tensor]) -> dict[str, float]:
    """Post-rollout scalars in one device->host transfer. Separate ``float()``
    calls cost one CUDA sync each, and ``action_abs_max`` blocks on one."""
    action = batch["action"].float()
    scalars = [
        (~batch["action_is_pad"]).sum(),
        action.abs().max(),
        action.abs().mean(),
    ]
    keys = ["step_delta", "action_abs_max", "action_magnitude"]
    clip_counts = batch.get("chunk_clip_counts")
    if clip_counts is not None:
        scalars.append(clip_counts.float().mean())
        keys.append("chunk_clip_counts")
    return dict(zip(keys, torch.stack(scalars).tolist()))


@torch.no_grad()
def measure_value_drift(policy, batch, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Both critic-drift metrics off one critic forward.

    ``boot_value_delta``: relative RMS move of the live critic from the
    bootstrap value in use. Small means staleness is not what limits the critic.
    ``value_target_drift``: the same move in value-loss units, a fraction of
    target RMS. Large against V can still be negligible against the return.
    """
    # boot_observations is stored in the buffer's bf16; the critic's weights are
    # fp32 and this runs outside autocast, so cast up.
    boot_obs = batch["boot_observations"].to(device=device, dtype=torch.float32)
    start = batch["boot_values_start"].to(device).reshape(-1)
    delta = policy.critic(boot_obs).float().amin(dim=0) - start
    drift_rms = rms(delta)
    return (
        drift_rms / (rms(start) + 1e-8),
        drift_rms / (rms(batch["raw_returns"].to(device)) + 1e-8),
    )


def sample_random_baseline(
    evaluator: Evaluator, env: Any, max_episodes: int = 30
) -> float:
    """Random-policy return, the divergence reference.

    ``0.0`` when no episode evaluated. Callers gate on ``baseline > 0``, so that
    disables divergence detection instead of falsely flagging every later run.
    """
    baseline_evaluator = Evaluator(
        env,
        eval_seed=evaluator.eval_seed,
        max_episodes=max_episodes,
        max_steps=evaluator.max_steps,
    )
    results = baseline_evaluator.evaluate(
        lambda obs: env.action_space.sample(), out_dir=None
    )
    if not results:
        return 0.0
    return float(np.mean([r["return"] for r in results]))

def make_vec_env(env_name: str, num_envs: int):
    """Vectorised env, async workers preferred. Falls back to ``SyncVectorEnv``
    when async is unavailable (no ``/dev/shm``, env-factory pickling failure)."""
    import gymnasium as gym

    env_fns = [lambda: gym.make(env_name) for _ in range(num_envs)]
    # autoreset_mode must survive vectorisation in either branch; the collector
    # validates NEXT_STEP, see validate_autoreset_mode.
    try:
        return gym.vector.AsyncVectorEnv(env_fns)
    except Exception:
        return gym.vector.SyncVectorEnv(env_fns)


def train_sjfpo_run(
    env: Any,
    out_dir: str | Path,
    cfg: SJFPOConfig | None = None,
    *,
    seed: int = 3407,
    total_steps: int = 2000,
    eval_steps: int = 1000,
    save_steps: int = 1000,
    obs_dim: int | None = None,
    action_dim: int | None = None,
    resume_from_checkpoint: str | Path | None = None,
    policy_cls: type[SJFPOPolicy] = SJFPOPolicy,
) -> dict[str, Any]:
    """One SJFPO training run. Env builds before any CUDA init, so forked
    workers do not inherit the parent's CUDA context (cleanrl / SB3 ordering).

    Accelerate contract: backward every microbatch, sync and step only at the
    group boundary, always flush a final partial group. Gradients clip per
    optimizer, not jointly, since the critic's norm owns a shared budget.
    """
    cfg = cfg or SJFPOConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = getattr(env, "unwrapped", env)
    spec = getattr(unwrapped, "spec", None)
    env_name = getattr(spec, "id", "unknown") if spec is not None else "unknown"
    if obs_dim is None or action_dim is None:
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))

    set_seed(seed)
    device = resolve_device()

    train_env = env
    if cfg.num_envs > 1 and env_name != "unknown":
        train_env = make_vec_env(env_name, cfg.num_envs)

    action_low = np.atleast_1d(np.asarray(env.action_space.low, dtype=np.float32))
    action_high = np.atleast_1d(np.asarray(env.action_space.high, dtype=np.float32))

    global_step = 0
    if resume_from_checkpoint is not None:
        if policy_cls is not SJFPOPolicy:
            # load_policy_checkpoint rebuilds a hardcoded SJFPOPolicy, so a
            # subclass comes back with the wrong actor architecture.
            raise ValueError(
                f"resume is not supported for policy_cls={policy_cls.__name__}; "
                "actor checkpoints are not interchangeable between architectures"
            )
        policy, optimizer, counters = load_policy_checkpoint(
            resume_from_checkpoint,
            device=device,
            action_low=action_low,
            action_high=action_high,
        )
        global_step = int(counters["global_step"])
        policy.train()
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
        policy = build_policy(
            obs_dim,
            action_dim,
            cfg,
            device=device,
            action_low=action_low,
            action_high=action_high,
            policy_cls=policy_cls,
        )
        optimizer = build_optimizer(policy)

    warmed = warm_observation_stats(
        policy,
        train_env,
        obs_dim=obs_dim,
        seed=seed,
    )
    print(
        f"obs normalizer warmed on {warmed} frames", file=sys.stderr
    )
    print(
        f"SJFPO policy: actor_depth={cfg.dit_depth} hidden={cfg.hidden_dim} "
        f"params={sum(p.numel() for p in policy.parameters()):,}",
        file=sys.stderr,
    )
    evaluator = Evaluator(
        env, eval_seed=10000 + seed, max_episodes=cfg.eval_max_episodes
    )
    eval_policy = _EvalPolicy(policy)
    # Same adapter, opposite start point. The rollouts that build the critic's
    # targets and the advantages come from sampled noise, so the eps=0 return
    # above is measuring a different distribution than the one being trained.
    stochastic_eval_policy = _EvalPolicy(policy, zero_sampling=False)
    r_rand = sample_random_baseline(evaluator, env)
    tracker = build_tracker(
        cfg,
        out_dir=out_dir,
        method="sjfpo",
        env_name=env_name,
        seed=seed,
        total_steps=total_steps,
    )

    next_eval = eval_steps
    next_save = save_steps

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

    actor_params = [
        p
        for group in policy.get_actor_optim_params()["params"]
        for p in group["params"]
    ]
    critic_params = [
        p
        for group in policy.get_critic_optim_params()["params"]
        for p in group["params"]
    ]

    try:
        while global_step < total_steps:
            steps_to_go = total_steps - global_step
            collect_steps = min(cfg.chunks_per_update * cfg.action_horizon, steps_to_go)
            policy.snapshot_behavior_policy()
            batch = policy.generate_rollout(
                train_env,
                collect_steps,
                action_low=action_low,
                action_high=action_high,
            )
            rollout = compute_rollout_diagnostics(batch)
            step_delta = int(rollout["step_delta"])
            if step_delta <= 0:
                raise RuntimeError(
                    "rollout returned zero executed steps; aborting to avoid "
                    "an infinite training loop."
                )
            global_step += step_delta

            rollout_jerk = None
            if "action" in batch and "action_is_pad" in batch:
                rollout_jerk = compute_mean_chunk_jerk(
                    batch["action"].detach().cpu().numpy(),
                    batch["action_is_pad"].detach().cpu().numpy(),
                )
            action_abs_max = rollout["action_abs_max"]
            if action_abs_max > 1.05:
                raise ValueError(
                    f"Unnormalised actions detected: max_abs={action_abs_max:.3f} "
                    "(expected scaled [-1, 1]; check rollout/action bounds)"
                )

            policy.train()

            infos: list[dict[str, float]] = []
            actor_pre_norms: list[torch.Tensor] = []
            critic_pre_norms: list[torch.Tensor] = []
            above_clamp_fracs: list[torch.Tensor] = []
            n_chunks = batch["raw_advantages"].shape[0]
            microbatches = list(
                generate_mini_batch_indices(
                    n_chunks,
                    cfg.num_mini_batches,
                    batch["raw_advantages"].device,
                    num_learning_epochs=cfg.num_learning_epochs,
                )
            )
            accumulation_steps = cfg.grad_accum_steps
            optimizer.zero_grad(set_to_none=True)
            minibatch: dict[str, torch.Tensor] = {}
            finite_error = torch.zeros(
                (), device=batch["raw_advantages"].device, dtype=torch.bool
            )
            for micro_step, mb_idx in enumerate(microbatches):
                group_start = micro_step - (micro_step % accumulation_steps)
                group_size = min(accumulation_steps, len(microbatches) - group_start)
                minibatch = sample_minibatch(batch, mb_idx, minibatch)
                loss, epoch_info = policy(
                    minibatch,
                    compute_info=tracker.enabled,
                    check_finite=False,
                )
                finite_error |= ~torch.isfinite(loss)
                if loss.dim() != 0:
                    raise ValueError(
                        f"Loss must be scalar before backward; got {tuple(loss.shape)}."
                    )
                (loss / group_size).backward()
                is_boundary = (
                    (micro_step + 1) % accumulation_steps == 0
                    or micro_step + 1 == len(microbatches)
                )
                if is_boundary:
                    pre_clip_actor = torch.nn.utils.clip_grad_norm_(
                        actor_params, cfg.max_grad_norm
                    )
                    actor_pre_norms.append(pre_clip_actor.detach())
                    if cfg.critic_max_grad_norm is None:
                        # Same total norm clip_grad_norm_ computes, without the
                        # per-param Python loop and its stack.
                        pre_clip_critic = torch.linalg.vector_norm(
                            torch.stack(
                                torch._foreach_norm(
                                    [
                                        p.grad
                                        for p in critic_params
                                        if p.grad is not None
                                    ]
                                )
                            )
                        )
                    else:
                        pre_clip_critic = torch.nn.utils.clip_grad_norm_(
                            critic_params, cfg.critic_max_grad_norm
                        )
                    critic_pre_norms.append(pre_clip_critic.detach())
                    above_clamp_fracs.append(policy.cfm_above_clamp_frac)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    policy.step_count += 1
                    policy.update_target_critic()
                if epoch_info is not None:
                    infos.append(epoch_info)

            # Checked once per update, not per boundary: bool() on a device
            # tensor is a sync, and NaN gradients make the clip a no-op anyway,
            # so the run is already lost by the time this raises.
            if bool(finite_error):
                raise SJFPOAborted(
                    "Non-finite loss detected during gradient accumulation"
                )

            # Once per rollout, not per boundary: a full critic pass over
            # 2 * n_chunks windows, and the number only grows with steps taken.
            drift = None
            if cfg.log_drift_diagnostics and tracker.enabled:
                drift = measure_value_drift(policy, batch, device)

            policy.update_ema(global_step)

            policy.update_observation_stats(batch["normalizer_observations"])

            if infos:
                info = {k: sum(d[k] for d in infos) / len(infos) for k in infos[0]}
            else:
                info = None
            scalars = [
                torch.stack(actor_pre_norms).mean(),
                torch.stack(critic_pre_norms).mean(),
                torch.stack(above_clamp_fracs).mean(),
            ]
            if drift is not None:
                scalars.extend(drift)
            (
                actor_grad_norm_mean_pre,
                critic_grad_norm_mean_pre,
                above_clamp_frac,
                *drift_means,
            ) = torch.stack(scalars).tolist()

            if tracker.enabled and info is not None:
                data = {f"train/{k}": v for k, v in info.items()}
                data["train/actor_grad_norm_preclip"] = actor_grad_norm_mean_pre
                data["train/critic_grad_norm_preclip"] = critic_grad_norm_mean_pre
                if cfg.log_drift_diagnostics and drift_means:
                    data["train/boot_value_delta"] = drift_means[0]
                    data["train/value_target_drift"] = drift_means[1]
                if rollout_jerk is not None:
                    data["train/rollout_jerk"] = rollout_jerk
                if "chunk_clip_counts" in rollout:
                    data["train/chunk_clip_counts"] = rollout["chunk_clip_counts"]
                data["train/action_magnitude"] = rollout["action_magnitude"]
                if batch.get("compute_spectral_energy_retained") is not None:
                    data["train/compute_spectral_energy_retained"] = batch[
                        "compute_spectral_energy_retained"
                    ]
                gae_diag = batch.get("gae_diagnostics")
                if gae_diag:
                    data["train/gae_linked_frac"] = gae_diag["linked_frac"]
                    data["train/gae_advantage_std"] = gae_diag["advantage_std"]
                    data["train/returns_rms"] = gae_diag["return_rms"]
                    data["train/advantage_std_fraction"] = gae_diag[
                        "advantage_std_fraction"
                    ]
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

            diverged_now = False
            regressed_now = False
            diverge_msg = ""
            if global_step >= next_eval or global_step >= total_steps:
                # The EMA actor once it is warm: the same weights averaged over ~20
                # rollouts, which turns a curve this noisy into a readable one. The
                # live weights go back right after, so training never runs on it.
                displaced = policy.swap_in_ema(global_step)
                try:
                    eval_results = evaluator.evaluate(
                        eval_policy, out_dir=out_dir, append_csv=True
                    )
                    # Same seeds, same evaluator, so the episode pairs line up.
                    # No out_dir: this is a diagnostic, not a history entry.
                    stochastic_results = evaluator.evaluate(
                        stochastic_eval_policy
                    )
                finally:
                    policy.swap_out_ema(displaced)
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
                    # The peak has to be on disk when it happens: with a
                    # save_steps cadence the nearest periodic checkpoint can be
                    # 100k steps away, and the run always ends on the last one.
                    best_ckpt_dir = out_dir / CHECKPOINTS_DIR / "best"
                    save_policy_checkpoint(
                        best_ckpt_dir,
                        policy=policy,
                        optimizer=optimizer,
                        global_step=global_step,
                        ema_state=policy.ema_state_dict(),
                    )
                    best_artifact = str(best_ckpt_dir / "model.safetensors")
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
                            "eval/ema_active": float(bool(displaced)),
                            "train/policy_lr": optimizer.param_groups[0]["lr"],
                            "train/critic_lr": optimizer.param_groups[-1]["lr"],
                            "train/lr_decays": lr_decays,
                            **(
                                {"train/stochastic_mean_return": stochastic_mean_return}
                                if stochastic_mean_return is not None
                                else {}
                            ),
                        },
                        step=global_step,
                    )
                    for b in tracker.backends:
                        getattr(b, "log_eval_table", lambda *_: None)(eval_results, global_step)
                consecutive_below, diverged_now = check_divergence(
                    mean_return, r_rand, consecutive_below
                )
                if diverged_now:
                    diverge_msg = (
                        f"DIVERGED at step {global_step}: "
                        f"mean_return={mean_return:.2f} "
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
                        for group in optimizer.param_groups:
                            group["lr"] *= cfg.lr_decay
                        plateau_msg = (
                            f"plateau at step {global_step}: mean_return="
                            f"{mean_return:.2f} < {cfg.stop_return_frac:g}*best="
                            f"{best_return:.2f} for {cfg.stop_patience} evals, "
                            f"lr x{cfg.lr_decay:g} ({lr_decays}/"
                            f"{cfg.max_lr_decays})"
                        )
                        print(plateau_msg, file=sys.stderr)
                    else:
                        regressed_now = True
                        regress_msg = (
                            f"REGRESSED at step {global_step}: mean_return="
                            f"{mean_return:.2f} < {cfg.stop_return_frac:g}*best="
                            f"{best_return:.2f} after {lr_decays} lr decays; "
                            f"the best checkpoint is the artifact"
                        )
                next_eval = max(global_step + eval_steps, next_eval + eval_steps)

            if diverged_now or regressed_now:
                diverged = diverged_now
                regressed = regressed_now
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
                save_policy_checkpoint(
                    ckpt_dir,
                    policy=policy,
                    optimizer=optimizer,
                    global_step=global_step,
                    ema_state=policy.ema_state_dict(),
                )
                artifacts.append(str(ckpt_dir / "model.safetensors"))
                next_save = max(global_step + save_steps, next_save + save_steps)

        elapsed = time.time() - t_start
        fps = global_step / max(elapsed, 1e-6)
        if tracker.enabled:
            tracker.log(
                {"timing/fps": fps, "timing/elapsed_s": elapsed},
                step=global_step,
            )
    finally:
        if train_env is not env:
            train_env.close()
        tracker.finish(exit_code=1 if diverged else 0)

    ckpt_dir = out_dir / CHECKPOINTS_DIR / "final"
    save_policy_checkpoint(
        ckpt_dir,
        policy=policy,
        optimizer=optimizer,
        global_step=global_step,
        ema_state=policy.ema_state_dict(),
    )
    artifacts.append(str(ckpt_dir / "model.safetensors"))
    if best_artifact is not None and best_artifact not in artifacts:
        artifacts.append(best_artifact)

    csv_path = out_dir / "eval_history.csv"
    if csv_path.exists():
        artifacts.append(str(csv_path))

    return {
        "method": "sjfpo",
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