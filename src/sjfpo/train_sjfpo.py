"""SJFPO online training entry (LeRobot-native, MuJoCo-only).

Single-round loop reusing the available pieces, no custom rollout/update
machinery of its own: ``SJFPOPolicy.generate_rollout`` gathers a
forward-ready batch, ``SJFPOPolicy.forward`` computes the loss, and the
loop below only steps the optimizer, evaluates, and checkpoints.

Call convention mirrors the old ``run_single`` driver so
``scripts/train.py`` dispatches here with its existing CLI args::

    train_sjfpo_run(env, out_dir, cfg, seed=..., total_steps=...,
                    eval_interval=..., ckpt_interval=...,
                    obs_dim=..., act_dim=...)
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
from sjfpo.config import SJFPOConfig
from sjfpo.configuration_sjfpo import SJFPOConfig as SJFPOPolicyConfig
from sjfpo.evaluator import Evaluator, check_divergence
from sjfpo.modeling_sjfpo import SJFPOPolicy
from sjfpo.utils import resolve_device
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
    """Translate the legacy config into a LeRobot policy on device."""
    device = torch.device(device) if device is not None else resolve_device()
    policy_cfg = SJFPOPolicyConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (obs_dim,))
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (act_dim,))},
        n_obs_steps=cfg.n_obs_steps,
        horizon=cfg.chunk_size,
        n_action_steps=max(
            1, round(cfg.executed_vs_predicted_ratio * cfg.chunk_size)
        ),
        hidden_dim=cfg.hidden_size,
        num_mc_samples=cfg.num_mc_samples,
        euler_train=cfg.euler_train,
        euler_eval=cfg.euler_eval,
        chunks_per_update=cfg.chunks_per_update,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        clip_range_vf=cfg.clip_range_vf,
        cfm_clamp=cfg.cfm_clamp,
        log_ratio_cap=cfg.log_ratio_cap,
        tau_range=cfg.tau_range,
        dit_depth=cfg.dit_depth,
        dit_heads=cfg.dit_heads,
        dit_mlp_ratio=cfg.dit_mlp_ratio,
        dit_time_dim=cfg.dit_time_dim,
        vf_coef=cfg.vf_coef,
        grad_clip_norm=cfg.grad_clip,
        top_p_spectral_tokens=cfg.top_p_spectral_tokens,
        executed_vs_predicted_ratio=cfg.executed_vs_predicted_ratio,
        optimizer_lr=cfg.lr,
        optimizer_weight_decay=cfg.wd,
    )
    return SJFPOPolicy(policy_cfg).to(device)


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
        policy.get_optim_params()["params"], lr=cfg.lr, weight_decay=cfg.wd
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
            collect_steps = min(cfg.chunks_per_update * cfg.chunk_size, steps_to_go)
            batch = policy.generate_rollout(env, collect_steps)
            global_step += int((~batch["action_is_pad"]).sum())
            policy.snapshot_behavior_policy()
            policy.train()
            loss, info = policy(batch)
            optimizer.zero_grad()
            loss.backward()
            all_params = [
                p for group in optimizer.param_groups for p in group["params"]
            ]
            torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
            optimizer.step()
            policy.step_count += 1
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
            if global_step % eval_interval < cfg.chunk_size or global_step >= total_steps:
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
            if global_step % ckpt_interval < cfg.chunk_size or global_step >= total_steps:
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
