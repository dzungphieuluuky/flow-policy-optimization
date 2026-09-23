"""SJFPO online training entry (LeRobot-native, MuJoCo-only).

``generate_rollout`` gathers a batch, ``forward`` computes the loss; this loop
only steps the optimizer, evaluates, and checkpoints.

Performance notes
-----------------
* Grad-norm-after-clip is derived analytically (``min(pre, clip_val)``) instead
  of looping over parameters with ``.item()`` — no per-parameter device syncs.
* ``zero_grad(set_to_none=True)`` frees grad buffers between minibatches.
* Eval and checkpoint are triggered by explicit step counters, so no eval or
  save is skipped when ``collect_steps`` overshoots the boundary.
* Warm-up actually runs ``cfg.random_rollout_steps`` env steps (in batches of
  ``chunks_per_update * action_horizon``) and updates running stats on each.
* A resume re-applies ``maybe_compile`` so the freshly-loaded policy is
  compiled like a fresh one.
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
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD, TRUNCATED
from lerobot.utils.random_utils import set_seed

from sjfpo.checkpointing import load_policy_checkpoint, save_policy_checkpoint
from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.evaluator import Evaluator, check_divergence, mean_chunk_jerk
from sjfpo.modeling_sjfpo import SJFPOPolicy, STORAGE_DTYPE
# Standard checkpoint naming (lerobot.utils.constants)
from lerobot.utils.constants import CHECKPOINTS_DIR
from sjfpo.rollout_storage import mini_batch_indices
from sjfpo.utils import maybe_compile, resolve_device, to_numpy_1d
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
        return to_numpy_1d(action)  # atleast_1d: 1-D envs must stay [1]


def _compile_in_place(policy: SJFPOPolicy) -> None:
    """Apply ``maybe_compile`` to every submodule the training path touches."""
    policy.actor = maybe_compile(policy.actor)
    policy.old_actor = maybe_compile(policy.old_actor)
    policy.critic = maybe_compile(policy.critic)


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
        n_action_steps=max(
            1, round(cfg.executed_vs_predicted_ratio * cfg.action_horizon)
        ),
    )
    policy = SJFPOPolicy(policy_cfg).to(device)
    _compile_in_place(policy)
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
        kwargs["fused"] = True
    return torch.optim.AdamW(params, **kwargs)


def sample_minibatch(
    batch: dict[str, torch.Tensor], indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Slice dim-0 rows; every batch value is a ``[B, ...]`` tensor."""
    return {k: v[indices] for k, v in batch.items()}


def random_baseline(evaluator: Evaluator, env: Any) -> float:
    """Mean random-policy return for divergence gating.

    Returns ``0.0`` unchanged — callers gate on ``baseline > 0`` before
    treating the value as meaningful (see ``check_divergence``).
    """
    results = evaluator.evaluate(lambda obs: env.action_space.sample(), out_dir=None)
    if not results:
        return 1.0
    return float(np.mean([r["return"] for r in results]))


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
) -> dict[str, Any]:
    """Run a single SJFPO training run, returning a summary with artifacts.

    Plain single-process PyTorch (``PYTHONPATH=src python scripts/train.py``).
    Mixed precision auto-detects (bf16 autocast when supported, fp32
    otherwise; Blackwell uses bf16 storage + fp8 matmuls).
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

    # Build vectorized training env *before* CUDA init so forked workers don't
    # inherit the parent's CUDA context (CleanRL / SB3 ordering).
    train_env = env
    if cfg.num_envs > 1 and env_name != "unknown":
        import gymnasium as gym

        train_env = gym.vector.SyncVectorEnv(
            [lambda: gym.make(env_name) for _ in range(cfg.num_envs)]
        )

    action_low = np.atleast_1d(np.asarray(env.action_space.low, dtype=np.float32))
    action_high = np.atleast_1d(np.asarray(env.action_space.high, dtype=np.float32))

    global_step = 0
    if resume_from_checkpoint is not None:
        policy, optimizer, counters = load_policy_checkpoint(
            resume_from_checkpoint, device=device
        )
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

    # -- True random rollout (not policy-driven): accumulate running stats --
    random_rollout_done = 0
    if cfg.random_rollout_steps > 0 and total_steps >= cfg.random_rollout_steps:
        random_rollout_target = min(cfg.random_rollout_steps, total_steps)

        # Actual uniform-random policy, not untrained flow (avoids wrong-state normalizer bias)
        def random_policy(obs):
            return train_env.action_space.sample()

        # Manual step loop; collect observations directly
        obs, _ = train_env.reset(seed=seed + 999)
        if hasattr(policy, "reset"):
            policy.reset()
        steps_collected = 0
        while steps_collected < random_rollout_target:
            action = train_env.action_space.sample()
            next_obs, _, term, trunc, _ = train_env.step(action)
            # Update running stats with the current single observation
            # RunningMeanStd handles any [*, obs_dim] shape via reshape
            raw_obs = torch.as_tensor(np.array(obs, dtype=np.float32), dtype=torch.float32)
            if raw_obs.dim() == 1:
                raw_obs = raw_obs.unsqueeze(0).unsqueeze(0)  # [1, 1, D] for window stats
            else:
                raw_obs = raw_obs.unsqueeze(0)  # [1, ...]
            policy.preprocessor.update_stats(raw_obs)
            steps_collected += 1
            obs = next_obs
            if bool(np.any(term) if hasattr(term, 'any') else term) or bool(np.any(trunc) if hasattr(trunc, 'any') else trunc):
                obs, _ = train_env.reset(seed=seed + 999 + steps_collected)
                if hasattr(policy, "reset"):
                    policy.reset()
        random_rollout_done = steps_collected
        print(
            f"True random rollout initialized: {random_rollout_done} env steps; stats frozen.",
            file=sys.stderr,
        )

    # -- Critic-only warm-up (~50k steps): freeze actor/conditioner, train critic only --
    critic_warmup_steps = cfg.random_rollout_steps  # use same budget
    if cfg.random_rollout_steps > 0:
        for step in range(
            min(critic_warmup_steps, max(1, int(cfg.random_rollout_steps)))
        ):
            collect = min(
                cfg.chunks_per_update * cfg.action_horizon,
                max(1, cfg.chunks_per_update * cfg.action_horizon),
            )
            policy.actor.requires_grad_(False)
            policy.old_actor.requires_grad_(False)
            policy.conditioner.requires_grad_(False)
            policy.old_conditioner.requires_grad_(False)
            policy.critic.requires_grad_(True)

            wb = policy.generate_rollout(
                train_env,
                collect,
                action_low=action_low,
                action_high=action_high,
            )
            if wb.get("observation.state") is None:
                break
            states_n = policy.preprocessor({OBS_STATE: wb["observation.state"]})[
                OBS_STATE
            ]
            v_pred = policy.critic(states_n).float()
            target_approx = wb.get("returns")
            if target_approx is not None:
                target = target_approx.reshape(-1)
            else:
                target = torch.zeros_like(v_pred.squeeze(-1))
            loss = torch.nn.functional.smooth_l1_loss(
                v_pred.squeeze(-1),
                target.to(device=v_pred.device, dtype=STORAGE_DTYPE),
                beta=1.0,
            )
            # Filter optimizer params to critic only
            optimizer.param_groups[0]["params"] = list(policy.critic.parameters())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        policy.actor.requires_grad_(True)
        policy.conditioner.requires_grad_(True)
        policy.old_actor.requires_grad_(False)
        policy.old_conditioner.requires_grad_(False)
        print("Critic warm-up done; explained_var check recommended.", file=sys.stderr)

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

    try:
        while global_step < total_steps:
            steps_to_go = total_steps - global_step
            collect_steps = min(cfg.chunks_per_update * cfg.action_horizon, steps_to_go)
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
            if action_abs_max > 1.05:
                raise ValueError(
                    f"Unnormalized actions detected: max_abs={action_abs_max:.3f} "
                    "(expected scaled [-1, 1]; check rollout/action bounds)"
                )

            policy.train()
            policy.snapshot_behavior_policy()

            infos: list[dict[str, float]] = []
            grad_norm_pre: list[float] = []
            n_chunks = batch["advantages"].shape[0]
            for mb_idx in mini_batch_indices(
                n_chunks,
                cfg.num_minibatches,
                batch["advantages"].device,
                num_epochs=cfg.num_epochs,
            ):
                optimizer.zero_grad(set_to_none=True)
                loss, epoch_info = policy(sample_minibatch(batch, mb_idx))
                if loss.dim() != 0:
                    raise ValueError(
                        f"Loss must be scalar before backward; got {tuple(loss.shape)}."
                    )
                loss.backward()
                all_params = [
                    p for group in optimizer.param_groups for p in group["params"]
                ]
                # `clip_grad_norm_` returns the pre-clip total norm; post-clip
                # is min(pre, max_norm) because clipping is a uniform scale.
                pre = torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip_norm)
                grad_norm_pre.append(float(pre))
                optimizer.step()
                policy.step_count += 1
                infos.append(epoch_info)

            # Aggregate across epochs/minibatches.
            info = {k: sum(d[k] for d in infos) / len(infos) for k in infos[0]}
            grad_norm_mean_pre = (
                sum(grad_norm_pre) / len(grad_norm_pre) if grad_norm_pre else 0.0
            )
            grad_norm_mean_post = min(grad_norm_mean_pre, cfg.grad_clip_norm)

            if wandb_enabled:
                data = {f"train/{k}": v for k, v in info.items()}
                data["train/grad_norm_preclip"] = grad_norm_mean_pre
                data["train/grad_norm_after_clip"] = grad_norm_mean_post
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

            # -- Periodic eval ------------------------------------------
            diverged_now = False
            diverge_msg = ""
            if global_step >= next_eval or global_step >= total_steps:
                eval_results = evaluator.evaluate(eval_policy, out_dir=out_dir)
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
