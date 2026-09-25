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
* The ``forward`` diagnostic pass is requested only when wandb will consume it.
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
from sjfpo.modeling_sjfpo import SJFPOAborted, SJFPOPolicy
from lerobot.utils.constants import CHECKPOINTS_DIR
from sjfpo.utils import maybe_compile, resolve_device, to_numpy_1d
from sjfpo.wandb_logging import (
    finish_run,
    init_wandb_run_for_method,
    log_eval_summary,
    log_metrics,
)


def mini_batch_indices(
    n: int,
    num_minibatches: int,
    device: torch.device | str,
    num_epochs: int = 1,
):
    """Yield shuffled minibatch indices for each epoch."""
    if n < 1:
        raise ValueError(f"n={n} must be >= 1")
    n_mb = max(1, min(int(num_minibatches), n))
    device = torch.device(device)
    perm = torch.randperm(n, device=device)
    for _ in range(max(1, int(num_epochs))):
        for i in range(n_mb):
            start = (i * n) // n_mb
            stop = ((i + 1) * n) // n_mb
            yield perm[start:stop]


class _EvalPolicy:
    """Single-action eval adapter over ``SJFPOPolicy.select_action``."""

    def __init__(self, policy: SJFPOPolicy):
        self.policy = policy

    def reset(self) -> None:
        self.policy.reset()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        was_training = self.policy.training
        self.policy.eval()
        try:
            action = self.policy.select_action({"observation.state": obs})
        finally:
            self.policy.train(was_training)
        return to_numpy_1d(action)  # atleast_1d: 1-D envs must stay [1]


def _compile_in_place(policy: SJFPOPolicy) -> None:
    """Apply ``maybe_compile`` to CUDA submodules the training path touches."""
    device = next(policy.parameters()).device
    policy.actor = maybe_compile(policy.actor, device=device)
    policy.old_actor = maybe_compile(policy.old_actor, device=device)
    policy.critic = maybe_compile(policy.critic, device=device)


def build_policy(
    obs_dim: int,
    action_dim: int,
    cfg: SJFPOConfig,
    device=None,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
) -> SJFPOPolicy:
    """Attach runtime observation/action features, then build the policy."""
    device = torch.device(device) if device is not None else resolve_device()
    policy_cfg = dataclasses.replace(
        cfg,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (obs_dim,))
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (action_dim,))},
    )
    policy = SJFPOPolicy(policy_cfg).to(device)
    _compile_in_place(policy)
    if action_low is not None and action_high is not None:
        policy.set_action_bounds(action_low, action_high)
    return policy


def _build_optimizer(
    policy: SJFPOPolicy, cfg: SJFPOConfig, params: list[dict], lr: float
):
    kwargs = dataclasses.asdict(cfg.get_optimizer_preset(lr=lr))
    kwargs.pop("grad_clip_norm", None)  # applied manually after backward
    if next(policy.parameters()).is_cuda:
        kwargs["fused"] = True
    return torch.optim.AdamW(params, **kwargs)


def build_optimizers(
    policy: SJFPOPolicy, cfg: SJFPOConfig
) -> tuple[torch.optim.Optimizer, None]:
    """Compatibility wrapper returning the single FPO++ optimizer."""
    return build_optimizer(policy, cfg), None


def build_optimizer(policy: SJFPOPolicy, cfg: SJFPOConfig):
    """Backward-compatible combined optimizer for legacy callers."""
    return _build_optimizer(
        policy, cfg, policy.get_optim_params()["params"], cfg.policy_lr
    )


def sample_minibatch(
    batch: dict[str, torch.Tensor],
    indices: torch.Tensor,
    out: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Slice dim-0 rows; every per-chunk batch value is ``[B, ...]``.

    ``mc_tau`` ``[B,M,K,1]`` and ``mc_eps`` ``[B,M,K,d]`` are stored per
    chunk/MC draw and are sliced with the minibatch. ``normalizer_observations``
    is passed through because
    it is one row per executed environment step, not per chunk.
    """
    if out is None:
        out = {}
    for k, v in batch.items():
        if k == "normalizer_observations":
            # The normalizer is updated once from the complete rollout batch.
            out[k] = v
        else:
            out[k] = v[indices]
    return out


def pad_minibatch_indices(
    indices: torch.Tensor, size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a minibatch by repeating its last row and return a validity mask."""
    valid = int(indices.numel())
    if size < valid:
        raise ValueError(f"compile batch size {size} is smaller than minibatch {valid}")
    if valid == size:
        mask = torch.ones(size, dtype=torch.bool, device=indices.device)
        return indices, mask
    padding = indices[-1:].expand(size - valid)
    padded = torch.cat([indices, padding])
    mask = torch.arange(size, device=indices.device) < valid
    return padded, mask


def random_baseline(
    evaluator: Evaluator, env: Any, max_episodes: int = 30
) -> float:
    """Mean random-policy return over 30 episodes for divergence gating.

    Returns ``0.0`` unchanged — callers gate on ``baseline > 0`` before
    treating the value as meaningful (see ``check_divergence``).
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
        return 1.0
    return float(np.mean([r["return"] for r in results]))


def _make_vec_env(env_name: str, num_envs: int):
    """Build a vectorized training env, preferring async workers.

    ``AsyncVectorEnv`` parallelizes env stepping across worker processes —
    the single biggest rollout-throughput win for ``num_envs > 1``. Falls
    back to ``SyncVectorEnv`` if the async path is unavailable in the
    current runtime (e.g. no /dev/shm, pickling of the env factory fails).
    """
    import gymnasium as gym

    env_fns = [lambda: gym.make(env_name) for _ in range(num_envs)]
    try:
        return gym.vector.AsyncVectorEnv(env_fns)
    except Exception:
        return gym.vector.SyncVectorEnv(env_fns)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


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
    wandb_enabled: bool = True,
) -> dict[str, Any]:
    """Run a single SJFPO training run, returning a summary with artifacts."""
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

    # Build vectorized training env *before* CUDA init so forked workers don't
    # inherit the parent's CUDA context (CleanRL / SB3 ordering).
    train_env = env
    if cfg.num_envs > 1 and env_name != "unknown":
        train_env = _make_vec_env(env_name, cfg.num_envs)

    action_low = np.atleast_1d(np.asarray(env.action_space.low, dtype=np.float32))
    action_high = np.atleast_1d(np.asarray(env.action_space.high, dtype=np.float32))

    global_step = 0
    if resume_from_checkpoint is not None:
        loaded = load_policy_checkpoint(
            resume_from_checkpoint,
            device=device,
            action_low=action_low,
            action_high=action_high,
        )
        if len(loaded) == 4:
            policy, _actor_optimizer, _critic_optimizer, counters = loaded
            optimizer = build_optimizer(policy, policy.config)
        else:
            policy, optimizer, counters = loaded
        global_step = int(counters["global_step"])
        policy.train()
        # Re-apply compilation — from_pretrained builds an eager policy.
        _compile_in_place(policy)
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
        )
        optimizer = build_optimizer(policy, cfg)

    print(
        f"SJFPO policy: actor_depth={cfg.dit_depth} hidden={cfg.hidden_dim} "
        f"params={sum(p.numel() for p in policy.parameters()):,}",
        file=sys.stderr,
    )
    # -- Eval / checkpoint infrastructure -----------------------------------
    evaluator = Evaluator(env, eval_seed=10000 + seed, max_episodes=12)
    eval_policy = _EvalPolicy(policy)
    r_rand = random_baseline(evaluator, env)
    wandb_enabled = init_wandb_run_for_method(
        method="sjfpo",
        env_name=env_name,
        seed=seed,
        total_steps=total_steps,
        cfg_dict=dataclasses.asdict(cfg),
        wandb_enabled=wandb_enabled,
    )

    # Next step at which each periodic action fires.
    next_eval = eval_steps
    next_save = save_steps

    eval_count = 0
    consecutive_below = 0
    diverged = False
    best_return: float | None = None
    steps_to_best: int | None = None
    recent_eval_returns: deque[float] = deque(maxlen=10)
    all_eval_returns: list[float] = []
    artifacts: list[str] = []
    elapsed = 0.0
    fps = 0.0
    t_start = time.time()

    # Hoisted once — indexing the optimizer's param groups every minibatch is
    # pure Python overhead on the critical path.
    optim_params = [p for group in optimizer.param_groups for p in group["params"]]

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
            # Count executed env steps via executed chunk prefixes.
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
            n_chunks = batch["advantages"].shape[0]
            n_minibatches = min(cfg.num_minibatches, n_chunks)
            expected_chunks = (
                cfg.chunks_per_update * cfg.action_horizon + cfg.n_action_steps - 1
            ) // cfg.n_action_steps
            if cfg.compile_batch_size > 0:
                compile_batch_size = cfg.compile_batch_size
            else:
                compile_batch_size = max(
                    1,
                    (expected_chunks + n_minibatches - 1) // n_minibatches,
                )
            compile_batch_size = max(
                compile_batch_size,
                (n_chunks + n_minibatches - 1) // n_minibatches,
            )
            # Match Accelerate's accumulation contract: backward on every
            # microbatch, synchronize/step only at the accumulation boundary,
            # and always flush a final partial accumulation group.
            microbatches = list(
                mini_batch_indices(
                    n_chunks,
                    cfg.num_minibatches,
                    batch["advantages"].device,
                    num_epochs=cfg.num_epochs,
                )
            )
            accumulation_steps = cfg.grad_accum_steps
            optimizer.zero_grad(set_to_none=True)
            minibatch: dict[str, torch.Tensor] = {}
            finite_error = torch.zeros(
                (), device=batch["advantages"].device, dtype=torch.bool
            )
            for micro_step, mb_idx in enumerate(microbatches):
                group_start = micro_step - (micro_step % accumulation_steps)
                group_size = min(accumulation_steps, len(microbatches) - group_start)
                padded_indices, sample_mask = pad_minibatch_indices(
                    mb_idx, compile_batch_size
                )
                padded = int(padded_indices.numel()) != int(mb_idx.numel())
                minibatch = sample_minibatch(batch, padded_indices, minibatch)
                minibatch["sample_mask"] = sample_mask
                loss, epoch_info = policy(
                    minibatch,
                    compute_info=wandb_enabled and not padded,
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
                    pre_clip = torch.nn.utils.clip_grad_norm_(
                        optim_params, cfg.grad_clip_norm
                    )
                    actor_pre_norms.append(pre_clip.detach())
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    policy.step_count += 1
                if epoch_info is not None:
                    infos.append(epoch_info)

            policy.update_ema(global_step)

            # Update once per collected rollout, after the policy/value update.
            # These are raw post-step frames, not overlapping warmup windows.
            policy.update_observation_stats(batch["normalizer_observations"])

            # Aggregate across epochs/minibatches.
            if infos:
                info = {k: sum(d[k] for d in infos) / len(infos) for k in infos[0]}
            else:
                info = None
            all_pre_norms = actor_pre_norms
            grad_norm_mean_pre = (
                float(torch.stack(all_pre_norms).mean()) if all_pre_norms else 0.0
            )
            grad_norm_mean_post = min(grad_norm_mean_pre, cfg.grad_clip_norm)

            if wandb_enabled and info is not None:
                data = {f"train/{k}": v for k, v in info.items()}
                data["train/grad_norm_preclip"] = grad_norm_mean_pre
                data["train/grad_norm_after_clip"] = grad_norm_mean_post
                if rollout_jerk is not None:
                    data["train/rollout_jerk"] = rollout_jerk
                data["train/action_magnitude"] = action_magnitude
                data["train/total_loss"] = (
                    info["policy_loss"] + cfg.vf_coef * info["value_loss"]
                )
                data.update(
                    {
                        "gae/advantages_mean": float(batch["advantages"].mean()),
                        "gae/advantages_std": float(batch["advantages"].std()),
                        "gae/returns_mean": float(batch["returns"].mean()),
                        "gae/values_mean": float(batch["values"].mean()),
                    }
                )
                log_metrics(data, step=global_step)

            # -- Periodic eval ------------------------------------------
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
                if wandb_enabled:
                    log_eval_summary(
                        mean_return,
                        eval_results,
                        all_eval_returns,
                        eval_count,
                        global_step,
                        r_rand=r_rand,
                        consecutive_below=consecutive_below,
                        best_return=best_return,
                        steps_to_best=steps_to_best,
                    )
                consecutive_below, diverged_now = check_divergence(
                    mean_return, r_rand, consecutive_below
                )
                if diverged_now:
                    diverge_msg = (
                        f"DIVERGED at step {global_step}: "
                        f"mean_return={mean_return:.2f} "
                        f"< 0.10*R_rand={0.10 * r_rand:.2f}"
                    )
                # Advance the trigger past the current step.
                next_eval = max(global_step + eval_steps, next_eval + eval_steps)

            if diverged_now:
                diverged = True
                if wandb_enabled:
                    log_metrics(
                        {"eval/diverged": 1, "eval/diverge_step": global_step},
                        step=global_step,
                    )
                print(diverge_msg)
                break

            # -- Periodic checkpoint ------------------------------------
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
        if wandb_enabled:
            log_metrics(
                {"timing/fps": fps, "timing/elapsed_s": elapsed},
                step=global_step,
            )
    finally:
        if train_env is not env:
            train_env.close()
        if wandb_enabled:
            finish_run(exit_code=1 if diverged else 0)

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