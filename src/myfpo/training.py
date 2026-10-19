"""myfpo online training entry (LeRobot-native, MuJoCo-only)."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import sys
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.random_utils import set_seed

from myfpo.checkpointing import load_policy_checkpoint, save_policy_checkpoint
from myfpo.configuration import MyfpoConfig
from myfpo.evaluator import (
    Evaluator,
    compute_mean_chunk_jerk,
    compute_seam_jerk,
)
from myfpo.modeling import MyfpoAborted, MyfpoPolicy, build_optimizer
from myfpo.metrics import subsampled_min
from lerobot.utils.constants import CHECKPOINTS_DIR
from myfpo.utils import resolve_device, rms, to_numpy_1d
from myfpo.tracking import build_tracker, compute_eval_summary_metrics
from myfpo.runner import build_train_metrics, summarise_window


class _EvalPolicy:
    def __init__(self, policy, start_from_zero=True):
        self.policy = policy
        self.start = start_from_zero

    @classmethod
    def for_sampling(cls, policy, sampling):
        return cls(policy, start_from_zero=(sampling == "zero_start"))

    def reset(self):
        self.policy.reset()

    def __call__(self, obs):
        was = self.policy.training
        self.policy.eval()
        try:
            r = self.policy.select_action({"observation.state": obs}, start_from_zero=self.start)
        finally:
            self.policy.train(was)
        return to_numpy_1d(r)

ABOVE_CLAMP_ABORT_UPDATES = 5

BOTH_CLAMPED_ABORT_FRAC = 0.5

def generate_mini_batch_indices(
    n: int,
    num_mini_batches: int,
    device: torch.device | str,
    num_learning_epochs: int = 1,
):
    """Yield shuffled minibatch indices, one tensor per epoch."""
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



def build_policy(
    obs_dim: int,
    action_dim: int,
    cfg: MyfpoConfig,
    device=None,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
) -> MyfpoPolicy:
    """Attach runtime features, then build the policy."""
    device = torch.device(device) if device is not None else resolve_device()
    policy_cfg = dataclasses.replace(
        cfg,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (obs_dim,))
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (action_dim,))},
    )
    policy = MyfpoPolicy(policy_cfg).to(device)
    if action_low is not None and action_high is not None:
        policy.set_action_bounds(action_low, action_high)
    if getattr(cfg, "compile", False):
        mode = "default"
        with contextlib.suppress(Exception):
            if device.type == "cuda":
                idx = int(device.index or 0)
                props = torch.cuda.get_device_properties(idx)
                if (props.major, props.minor) >= (8, 9):
                    mode = "max-autotune"
                elif props.major >= 7:
                    mode = "reduce-overhead"
        policy.actor = torch.compile(policy.actor, dynamic=True, mode=mode, fullgraph=True)
        policy.encoder = torch.compile(policy.encoder, dynamic=True, mode=mode, fullgraph=True)
        policy.critic = torch.compile(policy.critic, dynamic=True, mode=mode, fullgraph=True)
    return policy


def sample_minibatch(
    batch: dict[str, torch.Tensor],
    indices: torch.Tensor,
    out: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Slice the leading dimension of every per-chunk batch value."""
    if out is None:
        out = {}
    passthrough = ("normalizer_observations", "gae_diagnostics")
    for k, v in batch.items():
        if k in passthrough:
            out[k] = v
        else:
            out[k] = v[indices]
    return out


_OBS_WARMUP_STEPS = 128


def apply_lr_schedule(optimizer, step_count: int) -> None:
    """Apply the linear warmup, starting from the bases."""
    for group in optimizer.param_groups:
        base = group["base_lr"]
        warmup = group.get("lr_warmup_steps", 0)
        ramp = 1.0 if warmup <= 0 else min(1.0, step_count / warmup)
        group["lr"] = base * ramp


def split_grad_params(policy) -> dict[str, list[torch.nn.Parameter]]:
    """Split the trainable weights into the groups that clip independently.

    The actor's CFM surrogate and the critic's MSE have different gradient
    scales, so one shared clip lets whichever dominates shrink the other.
    """
    return {
        role: [p for g in params["params"] for p in g["params"]]
        for role, params in (
            ("actor", policy.get_actor_optim_params()),
            ("critic", policy.get_critic_optim_params()),
        )
    }


def warm_observation_stats(policy, env, *, obs_dim: int, seed: int) -> int:
    """Fold random-policy frames into the running obs stats."""
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
    """Read the post-rollout scalars with one device-to-host transfer."""
    action = batch["action"].float()
    is_pad = batch["action_is_pad"]
    executed = (~is_pad)
    exec_action = action[executed]
    scalars = [
        executed.sum(),
        action.abs().max(),
        action.abs().mean(),
        exec_action.abs().mean(),
        exec_action.abs().max(),
    ]
    keys = [
        "executed_action_steps",
        "action_abs_max",
        "action_abs_mean",
        "action_abs_mean_executed",
        "action_abs_max_executed",
    ]
    return dict(zip(keys, torch.stack(scalars).tolist()))




@torch.no_grad()
def measure_value_drift(policy, batch, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Both critic-drift metrics off one critic forward."""
    boot_obs = batch["boot_observations"].to(device=device, dtype=torch.float32)
    boot_actions = batch["boot_actions"].to(device=device, dtype=torch.float32)
    start = batch["boot_values_start"].to(device).reshape(-1)
    boot_exec_frac = policy.exec_length_frac(
        torch.full(
            (boot_obs.shape[0],),
            policy.config.n_action_steps,
            device=boot_obs.device,
            dtype=torch.long,
        )
    )
    delta = (
        subsampled_min(policy.critic(boot_obs, boot_actions, boot_exec_frac).float())
        - start
    )
    drift_rms = rms(delta)
    return (
        drift_rms / (rms(start) + 1e-8),
        drift_rms / (rms(batch["raw_returns"].to(device)) + 1e-8),
    )


def sample_random_baseline(
    evaluator: Evaluator, env: Any, max_episodes: int = 30, seed: int | None = None
) -> float:
    """Random-policy return, the divergence reference."""
    if seed is not None:
        env.action_space.seed(int(seed))
    baseline_evaluator = Evaluator(
        env,
        eval_seed=evaluator.eval_seed,
        max_episodes=max_episodes,
        max_steps=evaluator.max_steps,
    )
    results = baseline_evaluator.evaluate(
        lambda _obs: env.action_space.sample(), out_dir=None
    )
    if not results:
        return 0.0
    return float(np.mean([r["return"] for r in results]))

def make_vec_env(env_name: str, num_envs: int, record: dict | None = None):
    """Build a vectorised env with ``forkserver`` workers."""
    import gymnasium as gym

    def make_one():
        """Build one worker env, registering third-party namespaces locally."""
        if env_name.startswith("AdroitHand"):
            try:
                import gymnasium_robotics  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    f"{env_name} needs gymnasium-robotics: {exc}"
                ) from exc
        return gym.make(env_name)

    env_fns = [make_one for _ in range(num_envs)]
    try:
        env = gym.vector.AsyncVectorEnv(env_fns, context="forkserver")
        if record is not None:
            record["vector_env"] = "forkserver"
    except (ValueError, RuntimeError, ImportError) as exc:
        warnings.warn(
            f"AsyncVectorEnv(env_name={env_name!r}, num_envs={num_envs}, "
            f"context='forkserver') failed: {exc!r}. Falling back to "
            "SyncVectorEnv, which runs the envs serially in this process. "
            "Expect a large fps drop; the run is otherwise correct.",
            RuntimeWarning,
            stacklevel=2,
        )
        if record is not None:
            record["vector_env"] = "sync"
        return gym.vector.SyncVectorEnv(env_fns)
    else:
        return env


def collect_rollouts(policy, train_env, cfg, action_low, action_high, steps_to_go):
    collect_steps = min(cfg.chunks_per_update * cfg.action_horizon, steps_to_go)
    batch = policy.generate_rollout(
        train_env, collect_steps, action_low=action_low, action_high=action_high,
    )
    rollout = compute_rollout_diagnostics(batch)
    steps_this_rollout = int(rollout["executed_action_steps"])
    if steps_this_rollout <= 0:
        raise RuntimeError("rollout returned zero executed steps; aborting.")
    jerk_action = batch["action"].detach().cpu().numpy()
    jerk_pad = batch["action_is_pad"].detach().cpu().numpy()
    rollout_jerk = compute_mean_chunk_jerk(jerk_action, jerk_pad)
    rollout_seam_jerk = compute_seam_jerk(jerk_action, jerk_pad)
    return batch, rollout, steps_this_rollout, rollout_jerk, rollout_seam_jerk


def train_update(policy, optimizer, batch, tracker, cfg, grad_groups):
    policy.train()
    epoch_infos: list[dict[str, float]] = []
    grad_norms_pre: list[dict[str, torch.Tensor]] = []
    above_clamp_fracs: list[torch.Tensor] = []
    both_clamped_fracs: list[torch.Tensor] = []
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
    full, remainder = divmod(len(microbatches), accumulation_steps)
    group_sizes = [accumulation_steps] * full + ([remainder] if remainder else [])
    optimizer.zero_grad(set_to_none=True)
    minibatch: dict[str, torch.Tensor] = {}
    finite_error = torch.zeros(
        (), device=batch["raw_advantages"].device, dtype=torch.bool
    )
    for micro_step, mb_idx in enumerate(microbatches):
        group_size = group_sizes[micro_step // accumulation_steps]
        minibatch = sample_minibatch(batch, mb_idx, minibatch)
        loss, epoch_info = policy(minibatch, compute_info=tracker.enabled, check_finite=False)
        finite_error |= ~torch.isfinite(loss)
        if loss.dim() != 0:
            raise ValueError(f"Loss must be scalar before backward; got {tuple(loss.shape)}.")
        if not bool(torch.isfinite(loss).all()):
            raise MyfpoAborted(f"Non-finite loss at step {micro_step}: {loss}")
        (loss / group_size).backward()
        is_boundary = (
            (micro_step + 1) % accumulation_steps == 0 or micro_step + 1 == len(microbatches)
        )
        if is_boundary:
            norms = {
                role: torch.nn.utils.clip_grad_norm_(
                    params, cfg.max_grad_norm
                ).detach()
                for role, params in grad_groups.items()
            }
            # The groups partition the parameters, so this is the joint pre-clip
            # norm. It keeps train/grad_norm_before_clip comparable with history.
            norms["total"] = torch.linalg.vector_norm(
                torch.stack([n.reshape(()) for n in norms.values()])
            )
            grad_norms_pre.append(norms)
            above_clamp_fracs.append(policy.cfm_above_clamp_frac)
            both_clamped_fracs.append(policy.cfm_both_clamped_frac)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            policy.step_count += 1
            apply_lr_schedule(optimizer, policy.step_count)
        if epoch_info is not None:
            epoch_infos.append(epoch_info)
    if bool(finite_error):
        raise MyfpoAborted("Non-finite loss detected during gradient accumulation")
    return epoch_infos, grad_norms_pre, above_clamp_fracs, both_clamped_fracs


def evaluate_policy(policy, global_step, out_dir, evaluator, eval_policies, cfg, best_return=None):
    displaced = policy.swap_in_ema(global_step)
    try:
        results_by_sampling = {
            name: evaluator.evaluate(adapter, out_dir=out_dir if name == cfg.eval_sampling else None, append_csv=(name == cfg.eval_sampling))
            for name, adapter in eval_policies.items()
        }
    finally:
        policy.swap_out_ema(displaced)
    window_obj = summarise_window(
        results_by_sampling,
        reported_name=cfg.eval_sampling,
        stop_name=cfg.stop_eval_sampling,
        eval_seed=evaluator.eval_seed,
        summarise=evaluator.summarize,
        best_return=best_return,
    )
    summaries = window_obj.summaries
    eval_results = window_obj.reported_results
    mean_return = window_obj.mean_return
    stop_return = window_obj.stop_return
    result = {
        "mean_return": mean_return,
        "stop_return": stop_return,
        "improved": window_obj.improved,
        "displaced": bool(displaced),
        "eval_results": eval_results,
    }
    return result, summaries


def save_checkpoint(out_dir, policy, optimizer, global_step, name):
    ckpt_dir = out_dir / CHECKPOINTS_DIR / name
    save_policy_checkpoint(
        ckpt_dir,
        policy=policy,
        optimizer=optimizer,
        global_step=global_step,
        ema_state=policy.ema_state_dict(),
    )
    return str(ckpt_dir / "model.safetensors"), ckpt_dir


def run_training(
    env: Any,
    out_dir: str | Path,
    cfg: MyfpoConfig | None = None,
    *,
    seed: int = 3407,
    total_steps: int = 2000,
    eval_steps: int = 1000,
    save_steps: int = 1000,
    obs_dim: int | None = None,
    action_dim: int | None = None,
    resume_from_checkpoint: str | Path | None = None,
    runtime: dict | None = None,
) -> dict[str, Any]:
    """One myfpo training run. Env builds before any CUDA init, so forked
    vector-env workers inherit the right device state.
    """
    cfg = cfg or MyfpoConfig()
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
        train_env = make_vec_env(env_name, cfg.num_envs, record=runtime)
    else:
        if runtime is not None:
            runtime["vector_env"] = "single"
        if cfg.num_envs > 1:
            warnings.warn(
                f"num_envs={cfg.num_envs} requested but env_name={env_name!r} "
                "(no spec.id on the passed env), so vectorisation was skipped "
                "and the run uses a single env. Expect a large throughput drop; "
                "the run is otherwise correct.",
                RuntimeWarning,
                stacklevel=2,
            )
    if runtime is not None:
        runtime["eval_sampling"] = cfg.eval_sampling
        runtime["stop_eval_sampling"] = cfg.stop_eval_sampling
        (out_dir / "run_metadata.json").write_text(
            json.dumps(runtime, indent=2, sort_keys=True), encoding="utf-8"
        )

    action_low = np.atleast_1d(np.asarray(env.action_space.low, dtype=np.float32))
    action_high = np.atleast_1d(np.asarray(env.action_space.high, dtype=np.float32))

    global_step = 0
    if resume_from_checkpoint is not None:
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
        f"myfpo policy: actor_depth={cfg.dit_depth} hidden={cfg.hidden_dim} "
        f"params={sum(p.numel() for p in policy.parameters()):,}",
        file=sys.stderr,
    )
    eval_seed = 10000 + seed
    evaluator = Evaluator(
        env, eval_seed=eval_seed, max_episodes=cfg.eval_max_episodes
    )
    eval_policies = {
        "zero_start": _EvalPolicy(policy, start_from_zero=True),
        "noise_start": _EvalPolicy(policy, start_from_zero=False),
    }
    r_rand = sample_random_baseline(evaluator, env, seed=seed)
    tracker = build_tracker(
        cfg,
        out_dir=out_dir,
        method="myfpo",
        env_name=env_name,
        seed=seed,
        total_steps=total_steps,
    )

    next_eval = eval_steps
    next_save = save_steps

    above_clamp_updates = 0
    both_clamped_updates = 0
    best_return: float | None = None
    steps_to_best: int | None = None
    eval_count = 0
    recent_eval_returns: deque[float] = deque(maxlen=10)
    all_eval_returns: list[float] = []
    artifacts: list[str] = []
    best_artifact: str | None = None
    elapsed = 0.0
    fps = 0.0
    t_start = time.time()

    grad_groups = split_grad_params(policy)

    try:
        while global_step < total_steps:
            steps_to_go = total_steps - global_step
            batch, rollout, steps_this_rollout, rollout_jerk, rollout_seam_jerk = collect_rollouts(
                policy, train_env, cfg, action_low, action_high, steps_to_go
            )
            global_step += steps_this_rollout
            action_abs_max = rollout["action_abs_max"]
            if action_abs_max > 1.0 + 1e-3:
                raise ValueError(
                    f"Post-clip action magnitude exceeded 1: max_abs={action_abs_max:.3f}."
                )

            epoch_infos, grad_norms_pre, above_clamp_fracs, both_clamped_fracs = train_update(
                policy, optimizer, batch, tracker, cfg, grad_groups
            )

            drift = None
            if cfg.log_drift_diagnostics and tracker.enabled:
                drift = measure_value_drift(policy, batch, device)

            policy.update_ema(global_step)

            policy.update_observation_stats(batch["normalizer_observations"])

            if epoch_infos:
                info = {
                    k: sum(d[k] for d in epoch_infos) / len(epoch_infos)
                    for k in epoch_infos[0]
                }
            else:
                info = None
            scalars = [
                torch.stack(above_clamp_fracs).mean(),
                torch.stack(both_clamped_fracs).mean(),
            ]
            if drift is not None:
                scalars.extend(drift)
            (
                above_clamp_frac,
                both_clamped_frac,
                *drift_means,
            ) = torch.stack(scalars).tolist()
            grad_norms = {
                k: float(torch.stack([n[k] for n in grad_norms_pre]).mean())
                for k in ("total", "actor", "critic")
            }

            if tracker.enabled and info is not None:
                tracker.log(
                    build_train_metrics(
                        cfg=cfg,
                        info=info,
                        batch=batch,
                        rollout=rollout,
                        grad_norms=grad_norms,
                        above_clamp_frac=above_clamp_frac,
                        both_clamped_frac=both_clamped_frac,
                        drift_means=drift_means,
                        rollout_jerk=rollout_jerk,
                        rollout_seam_jerk=rollout_seam_jerk,
                    ),
                    step=global_step,
                )

            l_new_mean = info["l_new_mean"] if info is not None else float("nan")
            if both_clamped_frac > BOTH_CLAMPED_ABORT_FRAC:
                both_clamped_updates += 1
                if both_clamped_updates >= ABOVE_CLAMP_ABORT_UPDATES:
                    raise MyfpoAborted(
                        f"{ABOVE_CLAMP_ABORT_UPDATES} consecutive updates with >"
                        f"{BOTH_CLAMPED_ABORT_FRAC:.0%} of ratio elements whose two "
                        f"CFM losses both sit above cfm_clamp={cfg.cfm_clamp} "
                        f"(frac={both_clamped_frac:.2f}, "
                        f"l_new_mean={l_new_mean:.4f}, "
                        f"cfm_loss_reduction={cfg.cfm_loss_reduction!r}, "
                        f"action_horizon={cfg.action_horizon}). The ratio is pinned "
                        "at 1 and the policy gets no gradient. Raise cfm_clamp, "
                        "use reduction='mean', or lower action_horizon."
                    )
            elif above_clamp_frac > 0.5:
                above_clamp_updates += 1
                both_clamped_updates = 0
                if above_clamp_updates >= ABOVE_CLAMP_ABORT_UPDATES:
                    raise MyfpoAborted(
                        f"{ABOVE_CLAMP_ABORT_UPDATES} consecutive updates with >50% "
                        f"of CFM samples above cfm_clamp={cfg.cfm_clamp} "
                        f"(frac={above_clamp_frac:.2f}, "
                        f"cfm_loss_reduction={cfg.cfm_loss_reduction!r}). The policy "
                        "ratio is pinned at 1 and gets no gradient. Raise "
                        "cfm_clamp, or use a reduction whose scale fits under it."
                    )
            else:
                above_clamp_updates = 0
                both_clamped_updates = 0

            if global_step >= next_eval or global_step >= total_steps:
                result, summaries = evaluate_policy(
                    policy, global_step, out_dir, evaluator, eval_policies, cfg, best_return=best_return
                )
                mean_return = result.get("mean_return", 0.0)
                stop_return = result.get("stop_return", mean_return)
                all_eval_returns.append(mean_return)
                recent_eval_returns.append(mean_return)
                eval_count += 1
                if result.get("improved", False):
                    best_return, steps_to_best = stop_return, global_step
                    best_artifact_path, _ = save_checkpoint(out_dir, policy, optimizer, global_step, "best")
                    best_artifact = best_artifact_path
                if tracker.enabled:
                    eval_summary = summaries.get(cfg.eval_sampling, {})
                    tracker.log(
                        {
                            "eval/ema_active": float(bool(result.get("displaced", False))),
                            "eval/return_ci_low": eval_summary.get("return_ci_low", 0.0),
                            "eval/return_ci_high": eval_summary.get("return_ci_high", 0.0),
                            "eval/return_ci_width": (
                                eval_summary.get("return_ci_high", 0.0) - eval_summary.get("return_ci_low", 0.0)
                            ),
                            "eval/return_median": eval_summary.get("return_median", 0.0),
                            "eval/return_n": eval_summary.get("return_n", 0),
                            "train/policy_lr": optimizer.param_groups[0]["lr"],
                            "train/critic_lr": optimizer.param_groups[-1]["lr"],
                            **{
                                f"eval/{name}_return_mean": summaries.get(name, {}).get("return_mean", 0.0)
                                for name in eval_policies
                            },
                        },
                        step=global_step,
                    )
                    tracker.log(
                        compute_eval_summary_metrics(
                            mean_return,
                            result.get("eval_results", []),
                            all_eval_returns,
                            eval_count,
                            r_rand=r_rand,
                            best_return=best_return,
                            steps_to_best=steps_to_best,
                        ) if result.get("eval_results") else {},
                        step=global_step,
                    )
                next_eval = max(global_step + eval_steps, next_eval + eval_steps)

            if global_step >= next_save or global_step >= total_steps:
                artifact_path, _ = save_checkpoint(out_dir, policy, optimizer, global_step, f"step_{global_step}")
                artifacts.append(artifact_path)
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
        tracker.finish()

    final_path, _ = save_checkpoint(out_dir, policy, optimizer, global_step, "final")
    artifacts.append(final_path)
    if best_artifact is not None and best_artifact not in artifacts:
        artifacts.append(best_artifact)

    csv_path = out_dir / "eval_history.csv"
    if csv_path.exists():
        artifacts.append(str(csv_path))

    return {
        "method": "myfpo",
        "env_name": env_name,
        "seed": seed,
        "reward_mean_last10": (
            float(np.mean(recent_eval_returns)) if recent_eval_returns else 0.0
        ),
        "reward_median_last10": (
            float(np.median(recent_eval_returns)) if recent_eval_returns else 0.0
        ),
        "fps": fps,
        "artifacts": artifacts,
        "best_artifact": best_artifact,
        "best_return": best_return,
        "global_step": global_step,
    }
