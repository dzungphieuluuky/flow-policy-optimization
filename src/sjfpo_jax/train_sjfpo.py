"""SJFPO online training entry (JAX, MuJoCo-only).

Mirrors the torch loop: ``generate_rollout`` gathers a batch, ``forward``
computes the loss, and this file steps an Optax optimizer + does eval/save.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.random_utils import set_seed

from sjfpo_jax.checkpointing import load_policy_checkpoint, save_policy_checkpoint
from sjfpo_jax.configuration_sjfpo import SJFPOConfig
from sjfpo_jax.evaluator import Evaluator, check_divergence, mean_chunk_jerk
from sjfpo_jax.modeling_sjfpo import SJFPOPolicy
from sjfpo_jax.rollout_storage import mini_batch_indices
from sjfpo_jax.utils import resolve_device
from sjfpo_jax.wandb_logging import (
    finish_run, init_wandb_run_for_method, log_eval_summary, log_metrics,
)
from sjfpo_jax.networks import running_mean_std_update
from sjfpo_jax.modeling_sjfpo import _trainable_subtree, _merge_subtree

class _EvalPolicy:
    """Single-action eval adapter (uses ``select_action``)."""

    def __init__(self, policy: SJFPOPolicy):
        self.policy = policy
        self.key = jax.random.PRNGKey(0)

    def reset(self) -> None:
        self.policy.reset()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        self.key, sub = jax.random.split(self.key)
        a = self.policy.select_action({OBS_STATE: jnp.asarray(obs)}, key=sub)
        return np.atleast_1d(np.asarray(a))


OBS_STATE = "observation.state"
ACTION = "action"
DONE = "next.done"
TRUNCATED = "next.truncated"
REWARD = "next.reward"


def build_policy(
    obs_dim: int,
    action_dim: int,
    cfg: SJFPOConfig,
    *,
    key,
    action_low=None,
    action_high=None,
) -> SJFPOPolicy:
    policy_cfg = dataclasses.replace(
        cfg,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (obs_dim,))
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (action_dim,))},
        n_action_steps=max(1, max(1, cfg.action_horizon // 2)),
    )
    policy = SJFPOPolicy.create(policy_cfg, obs_dim, action_dim, key)
    if action_low is not None and action_high is not None:
        policy.set_action_bounds(action_low, action_high)
    return policy


def build_optimizer(policy: SJFPOPolicy, cfg: SJFPOConfig):
    """Optax AdamW over the trainable sub-tree, with a separate critic LR.

    Two label classes:
    - ``policy`` — actor + conditioner, LR = ``cfg.policy_lr``
    - ``critic`` — critic, LR = ``3 * cfg.policy_lr``

    Frozen fields (``old_actor`` / ``old_conditioner`` / ``obs_stats``) are
    never handed to the optimizer, so no third branch is needed.
    """
    policy_lr = cfg.policy_lr
    critic_lr = policy_lr * 3.0
    wd = cfg.optimizer_weight_decay
    grad_clip = cfg.grad_clip_norm

    def _key_name(k) -> str | None:
        return getattr(k, "name", None) or getattr(k, "key", None)

    def labels(subtree):
        def visit(path, _leaf):
            names = [_key_name(k) for k in path]
            if names and names[0] == "critic":
                return "critic"
            return "policy"
        return jax.tree_util.tree_map_with_path(visit, subtree)

    policy_tx = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(policy_lr, weight_decay=wd),
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(critic_lr, weight_decay=wd),
    )
    return optax.multi_transform({"policy": policy_tx, "critic": critic_tx}, labels)

def sample_minibatch(batch, indices):
    out = {}
    for k, v in batch.items():
        if k == "tau" and v.ndim == 1:
            out[k] = v
        else:
            out[k] = v[indices]
    return out


def _make_vec_env(env_name, num_envs):
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
) -> dict[str, Any]:
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
    key = jax.random.PRNGKey(seed)

    train_env = env
    if cfg.num_envs > 1 and env_name != "unknown":
        train_env = _make_vec_env(env_name, cfg.num_envs)

    action_low = np.atleast_1d(np.asarray(env.action_space.low, dtype=np.float32))
    action_high = np.atleast_1d(np.asarray(env.action_space.high, dtype=np.float32))

    global_step = 0
    if resume_from_checkpoint is not None:
        policy, global_step = load_policy_checkpoint(resume_from_checkpoint)
    else:
        policy = build_policy(
            obs_dim, action_dim, cfg, key=key,
            action_low=action_low, action_high=action_high,
        )

    optimizer = build_optimizer(policy, cfg)
    opt_state = optimizer.init(_trainable_subtree(policy.params))

    print(
        f"SJFPO policy (JAX): depth={cfg.dit_depth} hidden={cfg.hidden_dim}",
        file=sys.stderr,
    )

    evaluator = Evaluator(env, eval_seed=10000 + seed, max_episodes=12)
    eval_policy = _EvalPolicy(policy)
    wandb_enabled = init_wandb_run_for_method(
        method="sjfpo", env_name=env_name, seed=seed,
        total_steps=total_steps, cfg_dict=dataclasses.asdict(cfg),
    )

    next_eval, next_save = eval_steps, save_steps
    eval_count = 0
    consecutive_below = 0
    diverged = False
    recent_eval_returns: deque[float] = deque(maxlen=10)
    all_eval_returns: list[float] = []
    artifacts: list[str] = []
    t_start = time.time()

    # JIT the loss + grad over the batch pytree.
    @jax.jit
    def loss_and_grad(trainable, params, batch):
        """Grad over the trainable subtree only; ``params`` closes the tree.

        ``params`` is passed as a jit argument (not closed over) so it can
        change per iteration without recompiling. Its bool / int leaves are
        fine because they are not differentiated — ``argnums=0`` restricts
        grad to the float-only ``trainable`` dict.
        """
        def loss_fn(subtree):
            merged = _merge_subtree(params, subtree)
            policy_local = dataclasses.replace(policy, params=merged)
            loss, _ = policy_local.forward(batch, compute_info=False)
            return loss
        return jax.value_and_grad(loss_fn)(trainable)
    
    try:
        while global_step < total_steps:
            steps_to_go = total_steps - global_step
            collect_steps = min(cfg.chunks_per_update * cfg.action_horizon, steps_to_go)
            key, k_roll = jax.random.split(key)
            batch = policy.generate_rollout(
                train_env, collect_steps,
                action_low=action_low, action_high=action_high,
                noisy=True, key=k_roll,
            )
            # Fold raw rollout states into the running stats (train-only).
            policy.params = dataclasses.replace(
                policy.params,
                obs_stats=running_mean_std_update(
                    policy.params.obs_stats, batch[OBS_STATE]
                ),
            )
            step_delta = int((~batch["action_is_pad"]).sum())
            if step_delta <= 0:
                raise RuntimeError("rollout returned zero executed steps")
            global_step += step_delta

            policy.snapshot_behavior_policy()
            n_chunks = batch["advantages"].shape[0]
            key, k_perm = jax.random.split(key)
            for mb_idx in mini_batch_indices(
                n_chunks, cfg.num_minibatches, k_perm, num_epochs=cfg.num_epochs
            ):
                mb = sample_minibatch(batch, mb_idx)
                trainable = _trainable_subtree(policy.params)
                loss, grads = loss_and_grad(trainable, policy.params, mb)
                updates, opt_state = optimizer.update(grads, opt_state, trainable)
                new_trainable = optax.apply_updates(trainable, updates)
                policy.params = _merge_subtree(policy.params, new_trainable)
                policy.step_count += 1
                
            if wandb_enabled:
                log_metrics({"train/loss": float(loss)}, step=global_step)

            # Eval
            if global_step >= next_eval or global_step >= total_steps:
                eval_results = evaluator.evaluate(eval_policy, out_dir=out_dir)
                mean_return = float(np.mean([r["return"] for r in eval_results]))
                all_eval_returns.append(mean_return)
                recent_eval_returns.append(mean_return)
                eval_count += 1
                if wandb_enabled:
                    log_eval_summary(
                        mean_return, eval_results, all_eval_returns,
                        eval_count, global_step,
                    )
                consecutive_below, diverged_now = check_divergence(
                    mean_return, 0.0, consecutive_below
                )
                next_eval = max(global_step + eval_steps, next_eval + eval_steps)
                if diverged_now:
                    diverged = True
                    if wandb_enabled:
                        log_metrics({"eval/diverged": 1}, step=global_step)
                    break

            if global_step >= next_save or global_step >= total_steps:
                ckpt_dir = out_dir / "checkpoints" / f"step_{global_step}"
                save_policy_checkpoint(
                    ckpt_dir, policy=policy,
                    optimizer_state=opt_state, global_step=global_step,
                )
                artifacts.append(str(ckpt_dir))
                next_save = max(global_step + save_steps, next_save + save_steps)

        elapsed = time.time() - t_start
        fps = global_step / max(elapsed, 1e-6)
        if wandb_enabled:
            log_metrics({"timing/fps": fps, "timing/elapsed_s": elapsed}, step=global_step)
    finally:
        if train_env is not env:
            train_env.close()
        if wandb_enabled:
            finish_run(exit_code=1 if diverged else 0)

    ckpt_dir = out_dir / "checkpoints" / "final"
    save_policy_checkpoint(
        ckpt_dir, policy=policy,
        optimizer_state=opt_state, global_step=global_step,
    )
    artifacts.append(str(ckpt_dir))

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