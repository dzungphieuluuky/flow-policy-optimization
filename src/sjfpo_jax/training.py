"""JAX training entry, Phase 5 (no torch deletion; GPU gates pending).

Mirrors ``train_sjfpo_run`` round structure: MJX chunk rollout ->
snapshot -> shuffled minibatch epochs -> eval/divergence/checkpoint.
Eval runs on a plain gym env (speed-insensitive); rollouts need MJX+GPU
unless a ``vec_factory`` stub is injected (tests). Reuses the
backend-agnostic ``Evaluator``/divergence/wandb helpers from ``sjfpo``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.profiler
import numpy as np
from flax import nnx

from sjfpo.evaluator import Evaluator, check_divergence
from sjfpo.wandb_logging import (
    finish_run,
    init_wandb_run_for_method as init_wandb_run,
    log_eval_summary,
    log_metrics,
)
from sjfpo_jax import policy as jpol
from sjfpo_jax.checkpointing import save_checkpoint
from sjfpo_jax.envs import get_spec
from sjfpo_jax.rollout import generate_batch

__all__ = ["spec_dims", "train_jax_run", "profile_dir", "JAX_DEFAULT_NUM_ENVS"]

JAX_DEFAULT_NUM_ENVS = 128


def profile_dir() -> str | None:
    """Trace output dir from ``SJFPO_PROFILE_DIR`` (unset = no tracing)."""
    return os.environ.get("SJFPO_PROFILE_DIR") or None


# Prepared-batch keys that are Python statics (never sliced, never jitted).
_PREPARED_STATICS = ("flow_shape", "num_mc", "n", "token_dim")


def _slice_prepared(prep: dict, idx: jax.Array, num_mc: int) -> dict:
    """Minibatch slice of a prepared batch (flat MC rows need index expansion)."""
    flat_idx = (idx[:, None] * num_mc + jnp.arange(num_mc)[None, :]).reshape(-1)
    mb = {}
    for k, v in prep.items():
        if k in _PREPARED_STATICS:
            continue  # compute_loss derives sizes from array shapes
        if k in ("x_t", "tau_bn"):
            mb[k] = v[flat_idx]
        else:
            mb[k] = v[idx]
    return mb


def spec_dims(env_name: str) -> tuple[int, int]:
    """(obs_dim, action_dim) from the model XML (needs mujoco, not MJX)."""
    import mujoco

    import gymnasium as gym

    from sjfpo_jax.envs import obs_dim as _obs_dim_fn

    spec = get_spec(env_name)
    xml = (
        Path(gym.__file__).resolve().parent
        / "envs"
        / "mujoco"
        / "assets"
        / spec["xml"]
    )
    model = mujoco.MjModel.from_xml_path(str(xml))
    return int(_obs_dim_fn(spec, model.nq, model.nv, model.nbody)), int(model.nu)


class _ChunkEvalPolicy:
    """Single-action eval adapter: replan a chunk, drain it open-loop."""

    def __init__(self, policy, key, n_obs_steps=16, n_action_steps=8,
                 num_inference_steps=16):
        self.policy = policy
        self.key = key
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps
        self._frames: list = []
        self._buffer: list = []

    def reset(self) -> None:
        self._frames = []
        self._buffer = []

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).ravel()
        self._frames.append(obs)
        if len(self._frames) > self.n_obs_steps:
            self._frames.pop(0)
        while len(self._frames) < self.n_obs_steps:
            self._frames.insert(0, self._frames[0])
        if not self._buffer:
            window = jnp.asarray(np.stack(self._frames)[None])
            self.key, sub = jax.random.split(self.key)
            chunk = jpol.predict_action_chunk(
                self.policy, window, sub, noise=False,
                num_inference_steps=self.num_inference_steps,
            )
            n_execute = max(1, min(self.n_action_steps, int(chunk.shape[1])))
            self._buffer = [np.asarray(chunk[0, i]) for i in range(n_execute - 1, -1, -1)]
        return self._buffer.pop()


def train_jax_run(
    *,
    env_name: str,
    eval_env: Any,
    out_dir: str | Path,
    seed: int = 0,
    total_steps: int = 2000,
    eval_steps: int = 1000,
    save_steps: int = 1000,
    # None = JAX GPU default (128); the CLI passes an int only when the user
    # overrode --sjfpo.num_envs, so torch's default of 2 never leaks in.
    num_envs: int | None = None,
    n_obs_steps: int = 16,
    action_horizon: int = 16,
    hidden_dim: int = 128,
    dit_depth: int = 4,
    dit_heads: int = 4,
    dit_mlp_ratio: float = 4.0,
    dit_time_dim: int = 128,
    optimizer_lr: float = 3e-4,
    optimizer_weight_decay: float = 1e-4,
    grad_clip_norm: float = 1.0,
    gamma: float = 0.999,
    gae_lambda: float = 0.995,
    epsilon_clip: float = 0.05,
    vf_coef: float = 0.01,
    num_epochs: int = 16,
    num_minibatches: int = 4,
    chunks_per_update: int = 128,
    num_mc_samples: int = 16,
    num_euler_steps: int = 16,
    num_inference_steps: int = 16,
    top_p_spectral_tokens: float = 1.0,
    executed_vs_predicted_ratio: float = 0.5,
    eval_episodes: int = 12,
    vec_factory=None,
    wandb_enabled: bool = True,
) -> dict[str, Any]:
    """Run a single JAX training run, returning summary with artifacts."""
    _ = get_spec(env_name)  # fail fast on unknown ids.
    if num_envs is None:
        num_envs = JAX_DEFAULT_NUM_ENVS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_dim, action_dim = spec_dims(env_name)
    key = jax.random.PRNGKey(int(seed))
    policy, behavior = jpol.build_policy(
        obs_dim, action_dim, seed=int(seed), hidden_dim=hidden_dim,
        dit_depth=dit_depth, dit_heads=dit_heads, dit_mlp_ratio=dit_mlp_ratio,
        dit_time_dim=dit_time_dim, n_obs_steps=n_obs_steps,
    )
    action_low = np.atleast_1d(np.asarray(eval_env.action_space.low, dtype=np.float32))
    action_high = np.atleast_1d(np.asarray(eval_env.action_space.high, dtype=np.float32))
    policy.set_action_bounds(jnp.asarray(action_low), jnp.asarray(action_high))
    behavior.set_action_bounds(jnp.asarray(action_low), jnp.asarray(action_high))
    optimizer = jpol.build_optimizer(
        policy, lr=optimizer_lr, weight_decay=optimizer_weight_decay,
        grad_clip_norm=grad_clip_norm,
    )
    n_action_steps = max(1, round(executed_vs_predicted_ratio * action_horizon))
    evaluator = Evaluator(eval_env, eval_seed=10000 + seed, max_episodes=eval_episodes)
    eval_policy = _ChunkEvalPolicy(
        policy, key, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps,
        num_inference_steps=num_inference_steps,
    )
    baseline_evaluator = Evaluator(
        eval_env, eval_seed=10000 + seed, max_episodes=30
    )
    results = baseline_evaluator.evaluate(
        lambda obs: eval_env.action_space.sample(), out_dir=None
    )
    r_rand = float(np.mean([r["return"] for r in results])) if results else 1.0
    r_rand = r_rand or 1.0
    wandb_enabled = init_wandb_run(
        method="sjfpo_jax", env_name=env_name, seed=seed, total_steps=total_steps,
        cfg_dict={"hidden_dim": hidden_dim, "optimizer_lr": optimizer_lr},
        wandb_enabled=wandb_enabled,
    )
    prepare_kwargs = {
        "n_obs_steps": n_obs_steps, "action_horizon": action_horizon,
        "top_p_spectral_tokens": top_p_spectral_tokens,
        "num_mc_samples": num_mc_samples,
    }
    loss_kwargs = {"epsilon_clip": epsilon_clip, "vf_coef": vf_coef}
    global_step = 0
    eval_count = 0
    consecutive_below = 0
    diverged = False
    best_return: float | None = None
    steps_to_best: int | None = None
    artifacts: list = []
    all_eval_returns: list = []
    trace_dir = profile_dir()
    if trace_dir is not None:
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(trace_dir)
    t_start = time.time()
    try:
        while global_step < total_steps:
            steps_to_go = total_steps - global_step
            collect_steps = min(chunks_per_update * action_horizon, steps_to_go)
            key, sub = jax.random.split(key)
            # Profiler annotations are no-ops unless SJFPO_PROFILE_DIR started
            # a trace; device ranges then show per-phase in TensorBoard.
            with jax.profiler.StepTraceAnnotation("train", step_num=global_step):
                with jax.profiler.TraceAnnotation("collect"):
                    batch = generate_batch(
                        policy, env_name, collect_steps, jnp.asarray(action_low),
                        jnp.asarray(action_high), sub, n_obs_steps=n_obs_steps,
                        action_horizon=action_horizon, num_envs=num_envs, seed=int(sub[0]),
                        num_mc_samples=num_mc_samples,
                        top_p_spectral_tokens=top_p_spectral_tokens,
                        num_euler_steps=num_euler_steps, gamma=gamma, gae_lambda=gae_lambda,
                        vec_factory=vec_factory,
                    )
                jax.block_until_ready(batch["advantages"])
                global_step += int((~batch["action_is_pad"]).sum())
                jpol.update_normalizer(policy, batch["observation.state"])
                jpol.snapshot_behavior_policy(policy, behavior)
                prep = jpol.prepare_batch(policy, behavior, batch, **prepare_kwargs)
                n_chunks = int(batch["advantages"].shape[0])
                n_mb = max(1, min(num_minibatches, n_chunks))
                infos = []
                with jax.profiler.TraceAnnotation("update"):
                    for _ in range(max(1, int(num_epochs))):
                        key, sub = jax.random.split(key)
                        perm = jax.random.permutation(sub, n_chunks)
                        for i in range(n_mb):
                            idx = perm[(i * n_chunks) // n_mb:((i + 1) * n_chunks) // n_mb]
                            mb = _slice_prepared(prep, idx, num_mc_samples)
                            _, info = jpol.train_step(policy, optimizer, mb, loss_kwargs)
                            infos.append(info)
                jax.block_until_ready(jax.tree.leaves(nnx.state(policy, nnx.Param))[0])
            info = {k: sum(d[k] for d in infos) / len(infos) for k in infos[0]}
            if wandb_enabled:
                log_metrics({f"train/{k}": v for k, v in info.items()}, step=global_step)
            diverged_now = False
            if global_step % eval_steps < action_horizon or global_step >= total_steps:
                eval_policy.reset()
                eval_results = evaluator.evaluate(eval_policy, out_dir=out_dir)
                mean_return = float(np.mean([r["return"] for r in eval_results]))
                all_eval_returns.append(mean_return)
                eval_count += 1
                if best_return is None or mean_return > best_return:
                    best_return = mean_return
                    steps_to_best = global_step
                if wandb_enabled:
                    log_eval_summary(mean_return, eval_results, all_eval_returns,
                                     eval_count, global_step)
                consecutive_below, diverged_now = check_divergence(
                    mean_return, r_rand, consecutive_below
                )
            if diverged_now:
                diverged = True
                break
            if global_step % save_steps < action_horizon or global_step >= total_steps:
                ckpt = save_checkpoint(
                    out_dir / f"checkpoint_{global_step}", policy=policy,
                    optimizer=optimizer, global_step=global_step, env_name=env_name,
                    behavior=behavior,
                )
                artifacts.append(str(Path(ckpt["dir"]) / "arrays"))
                artifacts.append(ckpt["model"])
        elapsed = time.time() - t_start
        fps = global_step / max(elapsed, 1e-6)
        if wandb_enabled:
            log_metrics({"timing/fps": fps, "timing/elapsed_s": elapsed}, step=global_step)
    finally:
        if trace_dir is not None:
            jax.profiler.stop_trace()
        if wandb_enabled:
            finish_run(exit_code=1 if diverged else 0)
    ckpt = save_checkpoint(
        out_dir / "checkpoint_final", policy=policy, optimizer=optimizer,
        global_step=global_step, env_name=env_name, behavior=behavior,
    )
    artifacts.append(str(Path(ckpt["dir"]) / "arrays"))
    artifacts.append(ckpt["model"])
    csv_path = out_dir / "eval_history.csv"
    if csv_path.exists():
        artifacts.append(str(csv_path))
    result = {
        "method": "sjfpo_jax",
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
    if trace_dir is not None:
        result["profile_dir"] = trace_dir
    return result
