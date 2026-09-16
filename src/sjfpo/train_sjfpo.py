"""SJFPO online training entry (LeRobot-native, MuJoCo-only).

``generate_rollout`` gathers a batch, ``forward`` computes the loss; this loop
only steps the optimizer, evaluates, and checkpoints.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.random_utils import set_seed

from sjfpo.checkpointing import save_policy_checkpoint
from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.evaluator import Evaluator, check_divergence, render_episode_frames
from sjfpo.modeling_sjfpo import SJFPOPolicy
from sjfpo.utils import (
    maybe_compile,
    resolve_device,
    resolve_mixed_precision,
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
    return cfg.get_optimizer_preset().build(params)


def sample_minibatch(
    batch: dict[str, torch.Tensor], indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Slice dim-0 rows; every batch value is a [B, ...] tensor."""
    return {k: v[indices] for k, v in batch.items()}


def random_baseline(evaluator: Evaluator, env: Any, seed: int) -> float:
    """Mean random-policy return for divergence gating (1.0 if empty/zero)."""
    results = evaluator.evaluate(lambda obs: env.action_space.sample(), out_dir=None)
    r_rand = float(np.mean([r["return"] for r in results])) if results else 1.0
    return r_rand or 1.0


def train_sjfpo_run(
    env: Any,
    out_dir: str | Path,
    cfg: SJFPOConfig | None = None,
    *,
    seed: int = 0,
    total_steps: int = 2000,
    eval_steps: int = 1000,
    save_steps: int = 1000,
    obs_dim: int | None = None,
    action_dim: int | None = None,
) -> dict[str, Any]:
    """Run a single SJFPO training run, returning summary with artifacts.

    Single device: ``PYTHONPATH=src python scripts/train.py ...``. Multi-GPU:
    ``PYTHONPATH=src accelerate launch scripts/train.py ...`` — one process
    per GPU; ranks collect seed-offset rollouts and average gradients via
    DDP while eval, checkpointing, and wandb stay on rank 0. Mixed precision
    auto-detects (bf16 when supported, fp32 fallback) unless
    ``ACCELERATE_MIXED_PRECISION`` is set explicitly.
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
    set_seed(seed)  # identical init on every rank (DDP also broadcasts rank 0)
    accelerator = Accelerator(
        mixed_precision=resolve_mixed_precision(),
        cpu=os.environ.get("SJFPO_DEVICE", "auto").lower() == "cpu",
    )
    is_main = accelerator.is_main_process
    device = accelerator.device  # per-process GPU under `accelerate launch`
    # Real env bounds (not ±1): InvertedPendulum ±3, Pusher ±2, Humanoid ±0.4.
    action_low = np.atleast_1d(np.asarray(env.action_space.low, dtype=np.float32))
    action_high = np.atleast_1d(np.asarray(env.action_space.high, dtype=np.float32))
    policy = build_policy(obs_dim, action_dim, cfg, device=device,
                           action_low=action_low, action_high=action_high)
    optimizer = build_optimizer(policy, cfg)
    policy, optimizer = accelerator.prepare(policy, optimizer)
    # Inner module for attribute writes: DDP forwards method calls but
    # `policy.step_count += 1` would set the counter on the wrapper, not the
    # module. Params are shared (no copy), so in-place snapshot/EMA updates
    # via either reference stay in sync.
    raw_policy = accelerator.unwrap_model(policy)
    if accelerator.num_processes > 1:
        # Diverge RNG streams post-build so ranks collect different rollouts
        # (noisy proposals), MC draws (tau/eps), and minibatch orders.
        set_seed(seed + accelerator.process_index)
    if is_main:
        print(policy, file=sys.stderr)
    evaluator = Evaluator(env, eval_seed=10000 + seed, max_episodes=12)
    eval_policy = _EvalPolicy(policy)
    r_rand = random_baseline(evaluator, env, seed) if is_main else 1.0
    wandb_enabled = is_main and init_wandb_run_for_method(
        method="sjfpo",
        env_name=env_name,
        seed=seed,
        total_steps=total_steps,
        cfg_dict=dataclasses.asdict(cfg),
    )

    global_step = 0
    eval_count = 0
    consecutive_below = 0
    diverged = False
    artifacts: list = []
    all_eval_returns: list = []
    t_start = time.time()
    try:
        while global_step < total_steps:
            steps_to_go = total_steps - global_step
            collect_steps = min(cfg.chunks_per_update * cfg.action_horizon, steps_to_go)
            batch = policy.generate_rollout(env, collect_steps,
                                              action_low=action_low, action_high=action_high)
            global_step += int((~batch["action_is_pad"]).sum())
            policy.train()
            infos = []
            # PPO-style: fixed epochs, each epoch sweeps shuffled minibatches
            # (not the whole buffer per update).
            n_chunks = batch["advantages"].shape[0]
            n_mb = max(1, min(cfg.num_minibatches, n_chunks))
            for _ in range(cfg.num_epochs):
                for mb_idx in torch.randperm(n_chunks).split(max(1, n_chunks // n_mb)):
                    optimizer.zero_grad()
                    mb_idx = mb_idx.to(batch["advantages"].device)
                    # Rolling one-step delay: compare live weights to the previous
                    # step's snapshot, refresh the snapshot pre-step (post-step
                    # would compare weights to themselves: ratio 0).
                    loss, epoch_info = policy(sample_minibatch(batch, mb_idx))
                    accelerator.backward(loss)
                    all_params = [
                        p for group in optimizer.param_groups for p in group["params"]
                    ]
                    torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip_norm)
                    optimizer.step()
                    policy.snapshot_behavior_policy()
                    policy.update_ema()
                    raw_policy.step_count += 1
                    infos.append(epoch_info)
            info = {k: sum(d[k] for d in infos) / len(infos) for k in infos[0]}
            if accelerator.num_processes > 1:
                # Mean update stats across ranks (collective: every rank calls).
                keys = list(info)
                averaged = accelerator.reduce(
                    torch.tensor([info[k] for k in keys], device=device),
                    reduction="mean",
                )
                info = dict(zip(keys, averaged.tolist()))
            if wandb_enabled:
                data = {f"train/{k}": v for k, v in info.items()}
                data["train/total_loss"] = (
                    info["policy_loss"]
                    + cfg.vf_coef * info["value_loss"]
                    + cfg.kl_coef * info["kl_loss"]
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
            if is_main and (global_step % eval_steps < cfg.action_horizon or global_step >= total_steps):
                eval_results = evaluator.evaluate(eval_policy, out_dir=out_dir)
                mean_return = float(np.mean([r["return"] for r in eval_results]))
                all_eval_returns.append(mean_return)
                eval_count += 1
                if wandb_enabled:
                    log_eval_summary(
                        mean_return,
                        eval_results,
                        all_eval_returns,
                        eval_count,
                        global_step,
                        r_rand=r_rand,
                        consecutive_below=consecutive_below,
                    )
                    try:
                        frames = render_episode_frames(env_name, eval_policy, seed=10000 + seed)
                        log_eval_video(frames, global_step)
                    except Exception:
                        pass  # headless/GL failures must never break training
                consecutive_below, diverged_now = check_divergence(
                    mean_return, r_rand, consecutive_below
                )
                if diverged_now:
                    diverge_msg = (
                        f"DIVERGED at step {global_step}: mean_return={mean_return:.2f} "
                        f"< 0.10*R_rand={0.10 * r_rand:.2f}"
                    )
            # Broadcast the stop decision: a main-only break would hang the
            # other ranks inside the next DDP backward (collective op).
            stop = accelerator.reduce(
                torch.tensor(float(diverged_now), device=device), reduction="max"
            )
            if bool(stop.item()):
                diverged = True
                if is_main:
                    if wandb_enabled:
                        log_metrics(
                            {"eval/diverged": 1, "eval/diverge_step": global_step},
                            step=global_step,
                        )
                    print(diverge_msg)
                break
            if global_step % save_steps < cfg.action_horizon or global_step >= total_steps:
                accelerator.wait_for_everyone()
                if is_main:
                    ckpt_dir = out_dir / f"checkpoint_{global_step}"
                    save_policy_checkpoint(
                        ckpt_dir,
                        policy=accelerator.unwrap_model(policy),
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
        if wandb_enabled:
            finish_run(exit_code=1 if diverged else 0)

    accelerator.wait_for_everyone()
    if is_main:
        ckpt_dir = out_dir / "checkpoint_final"
        save_policy_checkpoint(
            ckpt_dir,
            policy=accelerator.unwrap_model(policy),
            optimizer=optimizer,
            global_step=global_step,
        )
        artifacts.append(str(ckpt_dir / "model.safetensors"))
    csv_path = out_dir / "eval_history.csv"
    if is_main and csv_path.exists():
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
