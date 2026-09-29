"""SJFPO online training entry (LeRobot-native, MuJoCo-only).

``generate_rollout`` gathers a batch, ``forward`` computes the loss; this loop
only steps the optimizer, evaluates, and checkpoints.

Performance notes
-----------------
* Vectorized training envs use ``AsyncVectorEnv`` (parallel worker processes)
  when available; falls back to ``SyncVectorEnv`` on pickling failures.
* Grad-norm-after-clip is derived analytically (``min(pre, clip_val)``);
  pre-clip norms are accumulated on device and converted to a Python float
  once per rollout, not once per minibatch.
* ``zero_grad(set_to_none=True)`` frees grad buffers between minibatches.
* The ``forward`` diagnostic pass runs only when a tracker will consume it.
* Eval and checkpoint are triggered by explicit step counters, so no eval or
  save is skipped when ``collect_steps`` overshoots the boundary.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from collections import deque
from tqdm import tqdm
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD, TRUNCATED
from lerobot.utils.random_utils import set_seed

from sjfpo.checkpointing import load_policy_checkpoint, save_policy_checkpoint
from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.evaluator import Evaluator, check_divergence, mean_chunk_jerk
from sjfpo.modeling_sjfpo import SJFPOAborted, SJFPOPolicy, build_optimizer
from lerobot.utils.constants import CHECKPOINTS_DIR
from sjfpo.utils import resolve_device, to_numpy_1d
from sjfpo.tracking import build_tracker, eval_summary_metrics


# Consecutive updates with most CFM samples above ``cfm_clamp`` before
# aborting. clamp() has zero gradient above its max, so those samples
# contribute nothing: the ratio stays exactly 1 and the policy receives no gradient
# while the critic keeps training. Abort instead of burning a run that can only
# plateau.
ABOVE_CLAMP_ABORT_UPDATES = 5


def mini_batch_indices(
    n: int,
    num_mini_batches: int,
    device: torch.device | str,
    num_learning_epochs: int = 1,
):
    """Yield shuffled minibatch index tensors, one per epoch.

    Every epoch reshuffles, so repeated epochs do not see the same order.
    Splits are proportional (``(i * n) // n_mb``), so a remainder is spread
    over the frequencies rather than dropped; the count is clamped to ``[1, n]``.
    """
    if n < 1:
        raise ValueError(f"n={n} must be >= 1")
    n_mb = max(1, min(int(num_mini_batches), n))
    device = torch.device(device)
    for _ in range(max(1, int(num_learning_epochs))):
        perm = torch.randperm(n, device=device)          # <- move inside
        for i in range(n_mb):
            start = (i * n) // n_mb
            stop = ((i + 1) * n) // n_mb
            yield perm[start:stop]

class _EvalPolicy:
    """Single-action eval adapter over ``SJFPOPolicy.select_action``."""

    def __init__(self, policy: SJFPOPolicy):
        self.policy = policy

    def reset(self) -> None:
        """Clear the policy's cached action-chunk state before a new episode."""
        self.policy.reset()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        was_training = self.policy.training
        self.policy.eval()
        try:
            action = self.policy.select_action({"observation.state": obs})
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

    ``policy_cls`` lets a swapped actor enter without touching the loop.
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
    """Slice the leading dimension of every per-chunk batch value.

    Every per-chunk batch value is ``[B, ...]``, so one index tensor slices
    them all consistently. ``mc_tau`` ``[B,M,K,1]`` and ``mc_eps`` ``[B,M,K,d]``
    are stored per chunk/MC draw and are sliced with the minibatch.

    ``normalizer_observations`` is passed through unsliced: it holds one row
    per executed env step rather than per chunk, and the normalizer is updated
    once from the complete rollout batch. ``gae_diagnostics`` is a plain dict
    of scalars, passed through for the same reason.
    """
    if out is None:
        out = {}
    for k, v in batch.items():
        if k == "normalizer_observations" or k == "gae_diagnostics":
            out[k] = v
        else:
            out[k] = v[indices]
    return out


_OBS_WARMUP_STEPS = 128


def warm_observation_stats(policy, env, *, obs_dim: int, seed: int) -> int:
    """Fold random-policy frames into the running obs stats before training.

    The stats normalizer is identity until its first update, and updates only
    after a rollout, so without this the entire first rollout runs on raw
    observations. baselines and SB3 pay one step of that; here it would be a
    whole rollout. Env is reset afterwards so training starts on a clean episode.
    """
    rng = np.random.default_rng(seed)
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
def measure_boot_value_drift(policy, batch, device) -> torch.Tensor:
    """Relative RMS move of the live critic from the bootstrap value in use.

    Both branches store the ensemble min that GAE bootstrapped with, so this
    reads as target staleness without a target critic and as the slow teacher's
    lag with one. Small values mean staleness is not what limits the critic.
    """
    now = policy.critic(batch["boot_observations"].to(device)).float().amin(dim=0)
    start = batch["boot_values_start"].to(device).reshape(-1)
    return (now - start).pow(2).mean().sqrt() / (start.pow(2).mean().sqrt() + 1e-8)


@torch.no_grad()
def measure_value_target_drift(policy, batch, device) -> torch.Tensor:
    """Same move, in the units the value loss sees: a fraction of target RMS.

    A drift can be large relative to V yet negligible next to the return
    magnitude. This is the number that decides whether the stale bootstrap
    term matters for the regression.
    """
    delta = (
        policy.critic(batch["boot_observations"].to(device)).float().amin(dim=0)
        - batch["boot_values_start"].to(device).reshape(-1)
    )
    return delta.pow(2).mean().sqrt() / (
        batch["raw_returns"].to(device).pow(2).mean().sqrt() + 1e-8
    )


def random_baseline(
    evaluator: Evaluator, env: Any, max_episodes: int = 30
) -> float:
    """Measure the random-policy return used as the divergence reference.

    Returns ``0.0`` when no episode could be evaluated. Callers gate on
    ``baseline > 0``, so that disables divergence detection rather than
    falsely flagging every later run as diverged.
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

def _make_vec_env(env_name: str, num_envs: int):
    """Build a vectorised training environment, preferring async workers.

    ``AsyncVectorEnv`` parallelizes env stepping across worker processes —
    the single biggest rollout-throughput win for ``num_envs > 1``. Falls
    back to ``SyncVectorEnv`` when the async path is unavailable in the
    current runtime (no ``/dev/shm``, env-factory pickling failure).
    """
    import gymnasium as gym

    env_fns = [lambda: gym.make(env_name) for _ in range(num_envs)]
    # autoreset_mode must survive vectorisation in either branch; the
    # collector validates NEXT_STEP (see _validate_autoreset_mode).
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
    """Run a single SJFPO training run.

    - vectorised env is built before any CUDA init
    - so forked workers do not inherit the parent's CUDA context
    - CleanRL / SB3 ordering

    Accelerate accumulation contract:

    - backward every microbatch
    - sync and step only at the boundary
    - always flush a final partial group

    Gradients are clipped per optimizer, not jointly.
    The critic's norm is orders of magnitude larger,
    so a shared clip would spend the whole budget on it.
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
        train_env = _make_vec_env(env_name, cfg.num_envs)

    action_low = np.atleast_1d(np.asarray(env.action_space.low, dtype=np.float32))
    action_high = np.atleast_1d(np.asarray(env.action_space.high, dtype=np.float32))

    global_step = 0
    if resume_from_checkpoint is not None:
        if policy_cls is not SJFPOPolicy:
            # load_policy_checkpoint rebuilds a hardcoded SJFPOPolicy, so a
            # subclass would come back with the wrong actor architecture.
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
    evaluator = Evaluator(env, eval_seed=10000 + seed, max_episodes=12)
    eval_policy = _EvalPolicy(policy)
    r_rand = random_baseline(evaluator, env)
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
    above_clamp_updates = 0
    diverged = False
    best_return: float | None = None
    steps_to_best: int | None = None
    recent_eval_returns: deque[float] = deque(maxlen=10)
    all_eval_returns: list[float] = []
    artifacts: list[str] = []
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
            step_delta = int((~batch["action_is_pad"]).sum())
            if step_delta <= 0:
                raise RuntimeError(
                    "rollout returned zero executed steps; aborting to avoid "
                    "an infinite training loop."
                )
            global_step += step_delta

            rollout_jerk = None
            if "action" in batch and "action_is_pad" in batch:
                rollout_jerk = mean_chunk_jerk(
                    batch["action"].detach().cpu().numpy(),
                    batch["action_is_pad"].detach().cpu().numpy(),
                )
            action_abs_max = float(batch["action"].abs().max())
            action_magnitude = float(batch["action"].abs().mean())
            if action_abs_max > 1.05:
                raise ValueError(
                    f"Unnormalized actions detected: max_abs={action_abs_max:.3f} "
                    "(expected scaled [-1, 1]; check rollout/action bounds)"
                )

            policy.train()

            infos: list[dict[str, float]] = []
            actor_pre_norms: list[torch.Tensor] = []
            critic_pre_norms: list[torch.Tensor] = []
            above_clamp_fracs: list[torch.Tensor] = []
            boot_deltas: list[torch.Tensor] = []
            target_drifts: list[torch.Tensor] = []
            n_chunks = batch["raw_advantages"].shape[0]
            microbatches = list(
                mini_batch_indices(
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
                    if bool(finite_error):
                        raise SJFPOAborted(
                            "Non-finite loss detected during gradient accumulation"
                        )
                    finite_error.zero_()
                    pre_clip_actor = torch.nn.utils.clip_grad_norm_(
                        actor_params, cfg.max_grad_norm
                    )
                    actor_pre_norms.append(pre_clip_actor.detach())
                    if cfg.critic_max_grad_norm is None:
                        pre_clip_critic = torch.linalg.vector_norm(
                            torch.stack(
                                [
                                    p.grad.norm()
                                    for p in critic_params
                                    if p.grad is not None
                                ]
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
                    boot_deltas.append(
                        measure_boot_value_drift(policy, batch, device)
                    )
                    target_drifts.append(
                        measure_value_target_drift(policy, batch, device)
                    )
                if epoch_info is not None:
                    infos.append(epoch_info)

            policy.update_ema(global_step)

            policy.update_observation_stats(batch["normalizer_observations"])

            if infos:
                info = {k: sum(d[k] for d in infos) / len(infos) for k in infos[0]}
            else:
                info = None
            actor_grad_norm_mean_pre = (
                float(torch.stack(actor_pre_norms).mean()) if actor_pre_norms else 0.0
            )
            critic_grad_norm_mean_pre = (
                float(torch.stack(critic_pre_norms).mean()) if critic_pre_norms else 0.0
            )
            boot_drift_mean = (
                float(torch.stack(boot_deltas).mean()) if boot_deltas else 0.0
            )
            target_drift_mean = (
                float(torch.stack(target_drifts).mean()) if target_drifts else 0.0
            )

            if tracker.enabled and info is not None:
                data = {f"train/{k}": v for k, v in info.items()}
                data["train/actor_grad_norm_preclip"] = actor_grad_norm_mean_pre
                data["train/critic_grad_norm_preclip"] = critic_grad_norm_mean_pre
                data["train/boot_value_delta"] = boot_drift_mean
                data["train/value_target_drift"] = target_drift_mean
                if rollout_jerk is not None:
                    data["train/rollout_jerk"] = rollout_jerk
                clip_counts = batch.get("chunk_clip_counts")
                if clip_counts is not None:
                    data["train/chunk_clip_counts"] = float(clip_counts.float().mean())
                data["train/action_magnitude"] = action_magnitude
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

            above_clamp_frac = (
                float(torch.stack(above_clamp_fracs).mean())
                if above_clamp_fracs
                else 0.0
            )
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
            diverge_msg = ""
            if global_step >= next_eval or global_step >= total_steps:
                eval_results = evaluator.evaluate(
                    eval_policy, out_dir=out_dir, append_csv=True
                )
                mean_return = float(np.mean([r["return"] for r in eval_results]))
                all_eval_returns.append(mean_return)
                recent_eval_returns.append(mean_return)
                eval_count += 1
                if best_return is None or mean_return > best_return:
                    best_return = mean_return
                    steps_to_best = global_step
                if tracker.enabled:
                    tracker.log(
                        eval_summary_metrics(
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
                next_eval = max(global_step + eval_steps, next_eval + eval_steps)

            if diverged_now:
                diverged = True
                if tracker.enabled:
                    tracker.log(
                        {"eval/diverged": 1, "eval/diverge_step": global_step},
                        step=global_step,
                    )
                print(diverge_msg)
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
        "diverged": diverged,
        "global_step": global_step,
    }