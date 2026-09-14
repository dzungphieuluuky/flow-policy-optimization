"""SJFPO online training entry (LeRobot-native, MuJoCo-only).

``generate_rollout`` gathers a batch, ``forward`` computes the loss; this loop
only steps the optimizer, evaluates, and checkpoints.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature

from sjfpo.checkpointing import save_policy_checkpoint
from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.evaluator import Evaluator, check_divergence
from sjfpo.modeling_sjfpo import SJFPOPolicy
from sjfpo.utils import maybe_compile, resolve_device
from sjfpo.wandb_logging import (
    finish_run,
    init_wandb_run_for_method,
    log_eval_summary,
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
        return action.detach().cpu().numpy()


def build_policy(
    obs_dim: int, act_dim: int, cfg: SJFPOConfig, device=None
) -> SJFPOPolicy:
    """Attach runtime observation/action features, then build the policy."""
    device = torch.device(device) if device is not None else resolve_device()
    policy_cfg = dataclasses.replace(
        cfg,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (obs_dim,))
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (act_dim,))},
        n_action_steps=max(1, round(cfg.executed_vs_predicted_ratio * cfg.horizon)),
    )
    policy = SJFPOPolicy(policy_cfg).to(device)
    # Spectral is parameter-free, normalizer is buffers-only; the snapshot
    # deep-copies the compiled actor so l_old stays compiled too.
    policy.actor = maybe_compile(policy.actor)
    policy.old_actor = maybe_compile(policy.old_actor)
    policy.critic = maybe_compile(policy.critic)
    return policy


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
    eval_interval: int = 1000,
    ckpt_interval: int = 1000,
    obs_dim: int | None = None,
    act_dim: int | None = None,
) -> dict[str, Any]:
    """Run a single SJFPO training run, returning summary with artifacts."""
    cfg = cfg or SJFPOConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = getattr(env, "unwrapped", env)
    spec = getattr(unwrapped, "spec", None)
    env_name = getattr(spec, "id", "unknown") if spec is not None else "unknown"
    if obs_dim is None or act_dim is None:
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device()
    policy = build_policy(obs_dim, act_dim, cfg, device=device)
    print(policy, file=sys.stderr)
    optimizer = torch.optim.AdamW(
        policy.get_optim_params()["params"],
        lr=cfg.optimizer_lr,
        weight_decay=cfg.optimizer_weight_decay,
        foreach=True,  # one fused kernel per step instead of per-tensor launches
    )
    evaluator = Evaluator(env, eval_seed=10000 + seed, max_episodes=12)
    eval_policy = _EvalPolicy(policy)
    r_rand = random_baseline(evaluator, env, seed)
    wandb_enabled = init_wandb_run_for_method(
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
            collect_steps = min(cfg.chunks_per_update * cfg.horizon, steps_to_go)
            batch = policy.generate_rollout(env, collect_steps)
            global_step += int((~batch["action_is_pad"]).sum())
            policy.train()
            infos = []
            for _ in range(cfg.num_epochs):
                # Rolling one-step delay: compare live weights to the previous
                # step's snapshot, refresh the snapshot pre-step (post-step
                # would compare weights to themselves: ratio 0).
                loss, epoch_info = policy(batch)
                optimizer.zero_grad()
                loss.backward()
                all_params = [
                    p for group in optimizer.param_groups for p in group["params"]
                ]
                torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip_norm)
                policy.snapshot_behavior_policy()
                optimizer.step()
                policy.step_count += 1
                infos.append(epoch_info)
            info = {k: sum(d[k] for d in infos) / len(infos) for k in infos[0]}
            if wandb_enabled:
                data = {f"train/{k}": v for k, v in info.items()}
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
            if global_step % eval_interval < cfg.horizon or global_step >= total_steps:
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
                consecutive_below, diverged_now = check_divergence(
                    mean_return, r_rand, consecutive_below
                )
                if diverged_now:
                    diverged = True
                    if wandb_enabled:
                        log_metrics(
                            {"eval/diverged": 1, "eval/diverge_step": global_step},
                            step=global_step,
                        )
                    print(
                        f"DIVERGED at step {global_step}: mean_return={mean_return:.2f} "
                        f"< 0.10*R_rand={0.10 * r_rand:.2f}"
                    )
                    break
            if global_step % ckpt_interval < cfg.horizon or global_step >= total_steps:
                ckpt_dir = out_dir / f"checkpoint_{global_step}"
                save_policy_checkpoint(
                    ckpt_dir, policy=policy, optimizer=optimizer, global_step=global_step
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

    ckpt_dir = out_dir / "checkpoint_final"
    save_policy_checkpoint(
        ckpt_dir, policy=policy, optimizer=optimizer, global_step=global_step
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
