"""SJFPO online training entry (LeRobot-native, MuJoCo-only).

``generate_rollout`` gathers a batch, ``forward`` computes the loss; this loop
only steps the optimizer, evaluates, and checkpoints.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.random_utils import set_seed

from sjfpo.checkpointing import load_policy_checkpoint, save_policy_checkpoint
from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.evaluator import Evaluator, check_divergence, mean_chunk_jerk, render_episode_frames
from sjfpo.modeling_sjfpo import SJFPOPolicy
from sjfpo.rollout_storage import mini_batch_indices
from sjfpo.utils import (
    maybe_compile,
    resolve_device,
    to_numpy_1d,
)
from sjfpo.wandb_logging import (
    finish_run,
    init_wandb_run_for_method,
    log_eval_summary,
    log_eval_video,
    log_metrics,
)


class _EvalPolicy:
    """Single-action eval adapter over ``SJFPOPolicy.select_action``."""

    def __init__(self, policy: SJFPOPolicy):
        self.policy = policy

    def reset(self) -> None:
        self.policy.reset()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        action = self.policy.select_action({"observation.state": obs})
        return to_numpy_1d(action)  # atleast_1d: 1-D envs must stay [1], not scalar


def build_policy(
    obs_dim: int, action_dim: int, cfg: SJFPOConfig, device=None,
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
        n_action_steps=max(1, round(cfg.executed_vs_predicted_ratio * cfg.action_horizon)),
    )
    policy = SJFPOPolicy(policy_cfg).to(device)
    # Spectral is parameter-free, normalizer is buffers-only; the snapshot
    # deep-copies the compiled actor so l_old stays compiled too.
    policy.actor = maybe_compile(policy.actor)
    policy.old_actor = maybe_compile(policy.old_actor)
    policy.critic = maybe_compile(policy.critic)
    if action_low is not None and action_high is not None:
        policy.set_action_bounds(action_low, action_high)
    return policy


def build_optimizer(policy: SJFPOPolicy, cfg: SJFPOConfig):
    """Lerobot preset AdamW; bitsandbytes 8-bit AdamW when opted in."""
    params = policy.get_optim_params()["params"]
    if cfg.optimizer_8bit:
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise RuntimeError(
                "optimizer_8bit=True needs bitsandbytes (pip install bitsandbytes)"
            ) from e
        return bnb.optim.AdamW8bit(
            params, lr=cfg.optimizer_lr, weight_decay=cfg.optimizer_weight_decay
        )
    kwargs = dataclasses.asdict(cfg.get_optimizer_preset())
    kwargs.pop("grad_clip_norm", None)  # applied manually after backward
    if next(policy.parameters()).is_cuda:
        kwargs["fused"] = True  # single-kernel AdamW step; CPU falls back
    return torch.optim.AdamW(params, **kwargs)


def sample_minibatch(
    batch: dict[str, torch.Tensor], indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Slice dim-0 rows; every batch value is a [B, ...] tensor."""
    return {k: v[indices] for k, v in batch.items()}


def random_baseline(evaluator: Evaluator, env: Any) -> float:
    """Mean random-policy return for divergence gating (1.0 if empty/zero)."""
    results = evaluator.evaluate(lambda obs: env.action_space.sample(), out_dir=None)
    r_rand = float(np.mean([r["return"] for r in results])) if results else 1.0
    return r_rand or 1.0


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
) -> dict[str, Any]:
    """Run a single SJFPO training run, returning summary with artifacts.

    Plain single-process PyTorch (``PYTHONPATH=src python scripts/train.py
    ...``). Mixed precision auto-detects: bf16 autocast matmuls when the CUDA
    device supports bf16 (see ``torch.cuda.is_bf16_supported``) or Blackwell (bf16 storage + fp8 matmuls), fp32 otherwise.

    ``resume_from_checkpoint`` points at a ``save_policy_checkpoint`` directory
    (``checkpoint_<step>`` or ``checkpoint_final``): policy, optimizer, and
    counters continue from there. ``total_steps`` stays an absolute env-step
    budget (the run stops when ``global_step`` reaches it). The checkpoint's
    own config shapes the policy; CLI hyperparameter overrides do not apply
    to a resumed run (scripts/train.py warns about them).
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
    device = resolve_device()  # SJFPO_DEVICE=cpu|auto honors the CPU override
    train_env = env
    if cfg.num_envs > 1 and env_name != "unknown":
        # Build BEFORE any CUDA init: fork-based workers inherit the parent's
        # CUDA context otherwise (CleanRL/SB3 order the same way). Keep
        # evaluation single-env; fixtures without a registered id stay local.
        import gymnasium as gym

        train_env = gym.vector.SyncVectorEnv(
            [lambda: gym.make(env_name) for _ in range(cfg.num_envs)]
        )
    # Real env bounds (not ±1): InvertedPendulum ±3, Pusher ±2, Humanoid ±0.4.
    action_low = np.atleast_1d(np.asarray(env.action_space.low, dtype=np.float32))
    action_high = np.atleast_1d(np.asarray(env.action_space.high, dtype=np.float32))
    global_step = 0
    if resume_from_checkpoint is not None:
        # Exact continuation: weights, optimizer state, and counters come
        # from the checkpoint dir (raises ResumeError when incomplete).
        policy, optimizer, counters = load_policy_checkpoint(
            resume_from_checkpoint, device=device
        )
        global_step = int(counters["global_step"])
        policy.train()  # loader leaves eval mode; rollouts train like fresh
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
        policy = build_policy(obs_dim, action_dim, cfg, device=device,
                               action_low=action_low, action_high=action_high)
        optimizer = build_optimizer(policy, cfg)
    print(policy, file=sys.stderr)
    evaluator = Evaluator(env, eval_seed=10000 + seed, max_episodes=12)
    eval_policy = _EvalPolicy(policy)
    r_rand = random_baseline(evaluator, env)
    wandb_enabled = init_wandb_run_for_method(
        method="sjfpo",
        env_name=env_name,
        seed=seed,
        total_steps=total_steps,
        cfg_dict=dataclasses.asdict(cfg),
    )

    eval_count = 0
    consecutive_below = 0
    diverged = False
    best_return: float | None = None
    steps_to_best: int | None = None
    artifacts: list = []
    all_eval_returns: list = []
    t_start = time.time()
    try:
        while global_step < total_steps:
            steps_to_go = total_steps - global_step
            collect_steps = min(cfg.chunks_per_update * cfg.action_horizon, steps_to_go)
            batch = policy.generate_rollout(train_env, collect_steps,
                                              action_low=action_low, action_high=action_high)
            global_step += int((~batch["action_is_pad"]).sum())
            # Exploration smoothness: mean jerk over executed chunk prefixes.
            rollout_jerk = None
            if "action" in batch and "action_is_pad" in batch:
                rollout_jerk = mean_chunk_jerk(
                    batch["action"].detach().cpu().numpy(),
                    batch["action_is_pad"].detach().cpu().numpy(),
                )
            policy.train()
            infos = []
            grad_norms_before = []  # pre-clip norms (clip_grad_norm_ returns before-clip)
            grad_norms_after = []  # post-clip norms
            # PPO-style: freeze the behavior snapshot once per collection round
            # (OFF semantics: fixed pi_old + stored eps/t for all epochs), then
            # sweep shuffled minibatches with reshuffled partitions per epoch.
            policy.snapshot_behavior_policy()
            n_chunks = batch["advantages"].shape[0]
            # One GPU randperm per epoch, contiguous splits (no dropped tails).
            for mb_idx in mini_batch_indices(
                n_chunks, cfg.num_minibatches, batch["advantages"].device,
                num_epochs=cfg.num_epochs,
            ):
                optimizer.zero_grad()
                loss, epoch_info = policy(sample_minibatch(batch, mb_idx))
                loss.backward()
                all_params = [
                    p for group in optimizer.param_groups for p in group["params"]
                ]
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    all_params, cfg.grad_clip_norm
                )
                grad_norm_after = 0.0
                for p in all_params:
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        grad_norm_after += param_norm.item() ** 2
                grad_norm_after = grad_norm_after ** 0.5
                grad_norms_before.append(float(grad_norm))
                grad_norms_after.append(float(grad_norm_after))
                optimizer.step()
                policy.step_count += 1
                infos.append(epoch_info)
            info = {k: sum(d[k] for d in infos) / len(infos) for k in infos[0]}
            grad_norm_mean_before = sum(grad_norms_before) / len(grad_norms_before) if grad_norms_before else 0.0
            grad_norm_mean_after = sum(grad_norms_after) / len(grad_norms_after) if grad_norms_after else 0.0
            if wandb_enabled:
                data = {f"train/{k}": v for k, v in info.items()}
                data["train/grad_norm_preclip"] = grad_norm_mean_before
                data["train/grad_norm_after_clip"] = grad_norm_mean_after
                if rollout_jerk is not None:
                    data["train/rollout_jerk"] = rollout_jerk
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
            diverged_now = False
            diverge_msg = ""
            if global_step % eval_steps < cfg.action_horizon or global_step >= total_steps:
                eval_results = evaluator.evaluate(eval_policy, out_dir=out_dir)
                mean_return = float(np.mean([r["return"] for r in eval_results]))
                all_eval_returns.append(mean_return)
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
                    try:
                        pass  # Video rendering disabled to avoid GLFW/libc++ crash in headless
                    except Exception:
                        pass
                consecutive_below, diverged_now = check_divergence(
                    mean_return, r_rand, consecutive_below
                )
                if diverged_now:
                    diverge_msg = (
                        f"DIVERGED at step {global_step}: mean_return={mean_return:.2f} "
                        f"< 0.10*R_rand={0.10 * r_rand:.2f}"
                    )
            if diverged_now:
                diverged = True
                if wandb_enabled:
                    log_metrics(
                        {"eval/diverged": 1, "eval/diverge_step": global_step},
                        step=global_step,
                    )
                print(diverge_msg)
                break
            if global_step % save_steps < cfg.action_horizon or global_step >= total_steps:
                ckpt_dir = out_dir / f"checkpoint_{global_step}"
                save_policy_checkpoint(
                    ckpt_dir,
                    policy=policy,
                    optimizer=optimizer,
                    global_step=global_step,
                )
                artifacts.append(str(ckpt_dir / "model.safetensors"))
        elapsed = time.time() - t_start
        fps = global_step / max(elapsed, 1e-6)
        if wandb_enabled:
            log_metrics(
                {"timing/fps": fps, "timing/elapsed_s": elapsed}, step=global_step
            )
    finally:
        if train_env is not env:
            train_env.close()
        if wandb_enabled:
            finish_run(exit_code=1 if diverged else 0)

    ckpt_dir = out_dir / "checkpoint_final"
    save_policy_checkpoint(
        ckpt_dir,
        policy=policy,
        optimizer=optimizer,
        global_step=global_step,
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
            float(np.mean(all_eval_returns[-10:])) if all_eval_returns else 0.0
        ),
        "fps": fps,
        "artifacts": artifacts,
        "diverged": diverged,
        "global_step": global_step,
    }
