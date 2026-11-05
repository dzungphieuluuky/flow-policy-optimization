"""Owns: online loop, checkpointing, metrics, log backends. NOT: policy
architecture. Key: run_training_loop, configure_run.
myfpo online training entry (PyTorch, MuJoCo-only).
"""

from __future__ import annotations
import math

import dataclasses
import json
import os
import sys
import time
import warnings
from collections import deque
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from myfpo.checkpointing import (
    CHECKPOINT_FILE,
    ResumeError,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from myfpo.rollout import resolve_action_bounds
from myfpo.configuration import MyfpoConfig
from myfpo.evaluator import (
    Evaluator,
    VectorEvaluator,
    compute_mean_chunk_jerk,
    compute_seam_jerk,
)
from myfpo.modeling import TrainingAborted, MyfpoPolicy, make_optimizer
from myfpo.fpo_core import compute_eval_summary_metrics, subsampled_min
from myfpo.processor import env_action_chunk
from myfpo.layers import (
    autocast_context,
    cudagraph_step_begin,
    resolve_device,
    rms,
    set_seed,
)
from tqdm.auto import tqdm

CHECKPOINTS_DIR = "checkpoints"


class EvalPolicy:
    def __init__(self, policy, start_from_zero=True):
        self.policy = policy
        self.start = start_from_zero

    @classmethod
    def for_sampling(cls, policy, sampling):
        """Build the wrapper that matches the ``eval_sampling`` mode."""
        return cls(policy, start_from_zero=(sampling == "zero_start"))

    def reset(self):
        """Clear the wrapped policy's frame and action buffers."""
        self.policy.reset()

    def __call__(self, obs):
        """Select one action. The evaluator pins eval mode for the whole run."""
        r = self.policy.select_action({"observation.state": obs}, start_from_zero=self.start)
        return np.atleast_1d(r.detach().cpu().numpy())

    def plan(self, window):
        """A batched raw chunk for the GPU evaluator: ``window`` is ``[E, T, obs]``.

        ``select_action`` is single-env by construction (one action buffer, one
        frame buffer), so the vectorized path plans a whole chunk per env
        instead of one action at a time. The evaluator executes it open-loop,
        which is what the collector does during training. The evaluator pins
        eval mode for the whole run.
        """
        return self.policy.predict_action_chunk(
            {"observation.state": window}, start_from_zero=self.start
        )

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





def updates_per_env_steps(cfg) -> int:
    """Env steps one scheduler update covers.

    One update collects ``chunks_per_update`` chunks of ``action_horizon``
    actions, so the loop steps the scheduler once per that many env steps.
    """
    return max(1, int(cfg.chunks_per_update) * int(cfg.action_horizon))


def make_scheduler(optimizer, cfg, total_steps: int):
    """Build the warmup and cosine ``LambdaLR``. Return None when the schedule is off.

    ``total_steps`` and ``lr_warmup_steps`` count env steps. The scheduler
    steps once per update, so both convert to update counts here.
    """
    if cfg.lr_schedule == "none":
        return None
    per_update = updates_per_env_steps(cfg)
    # The loop runs one update per collected batch, so a partial batch at the
    # end is still an update. A floor here would decay the LR to zero on it.
    total_updates = max(1, math.ceil(total_steps / per_update))
    # Round up to one update: an env-step warmup shorter than a single update
    # would otherwise disable the warmup completely.
    warmup_steps = int(cfg.lr_warmup_steps)
    warmup = max(1, math.ceil(warmup_steps / per_update)) if warmup_steps > 0 else 0
    if total_updates <= warmup:
        # The run ends inside its warmup. A constant LR is the honest reading,
        # and a smoke run must not train at a near-zero LR.
        return None

    def lr_lambda(step):
        """Scale the LR linearly during warmup, then apply the cosine decay."""
        if warmup and step < warmup:
            # ``LambdaLR`` steps once in its constructor, so update 0 must
            # already carry a nonzero LR.
            return float(step + 1) / float(warmup)
        progress = float(step - warmup) / float(max(1, total_updates - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def restore_scheduler(scheduler, state, global_step: int, cfg) -> None:
    """Put a resumed run back on its LR curve.

    ``state`` is the saved scheduler state. A checkpoint written before the
    scheduler was persisted has none, so the curve restarts from the counter
    instead. Either way the LR for the next update is applied now, because
    ``LambdaLR`` starts at epoch 0 and the loop applies the LR before it steps.
    """
    if scheduler is None:
        return
    if state is not None:
        # load_state_dict pops a key, so do not hand it the caller's dict.
        scheduler.load_state_dict(dict(state))
    else:
        scheduler.last_epoch = global_step // updates_per_env_steps(cfg)
    for i, group in enumerate(scheduler.optimizer.param_groups):
        group["lr"] = scheduler.base_lrs[i] * scheduler.lr_lambdas[i](scheduler.last_epoch)


def make_policy(
    obs_dim: int,
    action_dim: int,
    cfg: MyfpoConfig,
    device=None,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
) -> MyfpoPolicy:
    """Attach the runtime dims, then build the policy."""
    device = torch.device(device) if device is not None else resolve_device()
    policy_cfg = dataclasses.replace(
        cfg, obs_dim=obs_dim, action_dim=action_dim
    )
    policy = MyfpoPolicy(policy_cfg).to(device)
    if action_low is not None and action_high is not None:
        policy.set_action_bounds(action_low, action_high)
    # MyfpoPolicy compiles the whole loss path lazily (loss_and_info), so
    # there is nothing to wrap at build time.
    return policy


def sample_minibatch(
    batch: dict[str, torch.Tensor],
    indices: torch.Tensor,
    out: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Slice the leading dimension of every per-chunk batch value.

    ``step_rewards`` is a flat per-step array, not per chunk, so a minibatch
    indexes it with chunk ids. Nothing in the update reads the slice, and
    slicing it every minibatch is both waste and a trap for a future reader.
    """
    if out is None:
        out = {}
    passthrough = ("normalizer_observations", "gae_diagnostics", "step_rewards")
    for k, v in batch.items():
        if k in passthrough:
            out[k] = v
        else:
            out[k] = v[indices]
    return out


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


def executed_chunk(
    batch: dict[str, torch.Tensor], action_low, action_high
) -> torch.Tensor:
    """The env-facing action of a stored raw chunk.

    The batch holds the flow's raw output, which is what the policy was
    trained on. The env received the squash of it.
    """
    low, high = resolve_action_bounds(action_low, action_high)
    action = batch["action"]
    device = action.device
    return env_action_chunk(
        action.float(),
        low.to(device=device, dtype=torch.float32),
        high.to(device=device, dtype=torch.float32),
    )


@torch.no_grad()
def compute_rollout_diagnostics(
    batch: dict[str, torch.Tensor], *, action_low, action_high,
    env_action: torch.Tensor | None = None,
) -> dict[str, float]:
    """Read the post-rollout scalars with one device-to-host transfer.

    Two action spaces are reported: the raw chunk the flow produced, which
    is what the policy is trained on, and the chunk the env received.
    """
    action = batch["action"].float()
    is_pad = batch["action_is_pad"]
    executed = ~is_pad
    if env_action is None:
        env_action = executed_chunk(batch, action_low, action_high)
    exec_action = env_action[executed]
    scalars = [
        executed.sum(),
        action.abs().max(),
        action.abs().mean(),
        exec_action.abs().mean(),
        exec_action.abs().max(),
    ]
    keys = [
        "executed_action_steps",
        "raw_action_abs_max",
        "raw_action_abs_mean",
        "executed_action_abs_mean",
        "executed_action_abs_max",
    ]
    return dict(zip(keys, torch.stack(scalars).tolist()))




@torch.no_grad()
def measure_value_drift(policy, batch, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the RMS of the bootstrap delta relative to the start values and to the raw returns."""
    cfg = policy.config
    boot_obs = batch["boot_observations"].to(device=device, dtype=torch.float32)
    boot_actions_raw = batch["boot_actions"].to(device=device, dtype=torch.float32)
    start = batch["boot_values_start"].to(device).reshape(-1)
    boot_exec_frac = torch.full(
        (boot_obs.shape[0],),
        cfg.n_action_steps / float(cfg.action_horizon),
        device=boot_obs.device,
        dtype=torch.float32,
    )
    # Match the masking in bootstrap_values: squash, then zero out steps
    # beyond n_action_steps, so live and target critics read the same chunk.
    boot_mask = (
        torch.arange(cfg.action_horizon, device=boot_obs.device)
        .unsqueeze(0) < cfg.n_action_steps
    ).unsqueeze(-1).float()
    boot_actions = env_action_chunk(
        boot_actions_raw, policy.action_low, policy.action_high, boot_mask
    )
    critic_was_training = policy.critic.training
    policy.critic.eval()
    try:
        delta = (
            subsampled_min(policy.critic(boot_obs, boot_actions, boot_exec_frac).float())
            - start
        )
    finally:
        policy.critic.train(critic_was_training)
    drift_rms = rms(delta)
    return (
        drift_rms / (rms(start) + 1e-8),
        drift_rms / (rms(batch["raw_returns"].to(device)) + 1e-8),
    )


def sample_random_baseline(
    evaluator: Evaluator | VectorEvaluator, max_episodes: int = 30
) -> float:
    """Random-policy return, the divergence reference.

    The evaluator builds the baseline, so the same code covers the single-env
    numpy path and the batched GPU path. Each evaluator knows how to make a
    random policy for its own env: a per-step action for a numpy env, a raw
    chunk for a torch env.
    """
    baseline = evaluator.clone(max_episodes=max_episodes)
    results = baseline.evaluate(baseline.random_policy(baseline.env), out_dir=None)
    if not results:
        return 0.0
    return float(np.mean([r["return"] for r in results]))

def make_vec_env(env_name: str, num_envs: int, record: dict | None = None):
    """Build a vectorized env with ``forkserver`` workers."""
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


@dataclasses.dataclass
class RolloutStep:
    """One rollout, its diagnostics, and the trajectory metrics the loop logs."""

    batch: dict[str, torch.Tensor]
    diagnostics: dict[str, float]
    executed_steps: int
    jerk: float | None
    seam_jerk: float | None


def rollout_with_diagnostics(
    policy, train_env, cfg, action_low, action_high, steps_to_go
):
    """Run one rollout, run its diagnostics, and guard the env-facing action magnitude."""
    collect_steps = min(cfg.chunks_per_update * cfg.action_horizon, steps_to_go)
    batch = policy.generate_rollout(
        train_env, collect_steps, action_low=action_low, action_high=action_high,
    )
    env_action = executed_chunk(batch, action_low, action_high)
    rollout = compute_rollout_diagnostics(
        batch, action_low=action_low, action_high=action_high, env_action=env_action
    )
    steps_this_rollout = int(rollout["executed_action_steps"])
    if steps_this_rollout <= 0:
        raise RuntimeError("rollout returned zero executed steps; aborting.")
    # The guard is on the action the env receives, which the squash bounds.
    # The stored chunk is the raw one and is unbounded.
    env_abs_max = rollout["executed_action_abs_max"]
    env_bound = float(np.abs(np.concatenate([action_low, action_high])).max())
    if env_abs_max > env_bound + 1e-3:
        raise ValueError(
            f"Post-squash action magnitude exceeded the env bound: "
            f"max_abs={env_abs_max:.3f} > {env_bound:.3f}."
        )
    # The jerk is a property of the trajectory the env saw, so it is measured
    # on the env-facing action, not on the raw chunk.
    jerk_action = env_action.detach().cpu().numpy()
    jerk_pad = batch["action_is_pad"].detach().cpu().numpy()
    # Chunks are written in closing order, so adjacent rows cross envs once
    # more than one env runs. A seam only exists within one env.
    jerk_env = batch["chunk_env_id"].detach().cpu().numpy()
    rollout_jerk = compute_mean_chunk_jerk(jerk_action, jerk_pad)
    rollout_seam_jerk = compute_seam_jerk(jerk_action, jerk_pad, jerk_env)
    return RolloutStep(
        batch=batch,
        diagnostics=rollout,
        executed_steps=steps_this_rollout,
        jerk=rollout_jerk,
        seam_jerk=rollout_seam_jerk,
    )


def policy_update_step(policy, optimizer, batch, backends, cfg, grad_groups):
    """Run all minibatch epochs. Each accumulation boundary clips per role and applies one optimizer step."""
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
    # The policy carries no .device attribute; its parameters carry the device.
    # Passing a CPU device here would silently disable bf16 autocast on CUDA.
    with autocast_context(next(policy.actor.parameters()).device):
        for micro_step, mb_idx in enumerate(microbatches):
            group_size = group_sizes[micro_step // accumulation_steps]
            minibatch = sample_minibatch(batch, mb_idx, minibatch)
            # One minibatch is one cudagraph step. Its backward runs before the
            # next forward, so the manager may recycle the step's memory.
            cudagraph_step_begin()
            if backends:
                loss, staged = policy.compute_loss_and_probes(
                    minibatch, reduction="mean", check_finite=False
                )
            else:
                loss = policy(minibatch, check_finite=False)
                staged = None
            if loss.dim() != 0:
                raise ValueError(f"Loss must be scalar before backward; got {tuple(loss.shape)}.")
            # bool(isfinite(...).all()) syncs the device. One check per update,
            # or on every microbatch when MYFPO_CHECK_FINITE=1.
            if (
                micro_step == 0 or os.environ.get("MYFPO_CHECK_FINITE") == "1"
            ) and not bool(torch.isfinite(loss).all()):
                raise TrainingAborted(f"Non-finite loss at step {micro_step}: {loss}")
            (loss / group_size).backward()
            # The metrics' host transfer runs after backward, never between the
            # forward and the backward.
            if staged is not None:
                epoch_infos.append(policy.pack_metrics(staged))
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
    return epoch_infos, grad_norms_pre, above_clamp_fracs, both_clamped_fracs


def evaluate_policy(
    policy, global_step, out_dir, evaluator, eval_policies, cfg, best_return=None,
    video=None,
):
    """Swap in the EMA weights, evaluate every eval policy, and record the optional video."""
    displaced = policy.swap_in_ema(global_step)
    try:
        results_by_sampling = {
            name: evaluator.evaluate(adapter, out_dir=out_dir if name == cfg.eval_sampling else None, append_csv=(name == cfg.eval_sampling))
            for name, adapter in eval_policies.items()
        }
        if video is not None:
            # Inside the EMA swap, so the frames show the weights that were
            # scored. The recorder has its own env and seed, so this neither
            # perturbs the numbers above nor reuses their RNG stream.
            recorder, video_dir = video
            recorder.evaluate(
                eval_policies[cfg.eval_sampling],
                video_path=Path(video_dir) / f"step_{global_step}.mp4",
            )
    finally:
        policy.swap_out_ema(displaced)
    window_obj = summarize_window(
        results_by_sampling,
        reported_name=cfg.eval_sampling,
        stop_name=cfg.stop_eval_sampling,
        eval_seed=evaluator.eval_seed,
        summarize=evaluator.summarize,
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


def save_checkpoint(
    out_dir, policy, optimizer, global_step, name, use_ema=False, scheduler=None
):
    """Save one checkpoint, with the live model weights and the EMA shadow."""
    ckpt_dir = out_dir / CHECKPOINTS_DIR / name
    # The best and final scores are measured on the EMA weights. Saving the
    # live weights instead would pair the score with unmeasured weights.
    displaced = policy.swap_in_ema(global_step) if use_ema else {}
    try:
        save_policy_checkpoint(
            ckpt_dir,
            policy=policy,
            optimizer=optimizer,
            global_step=global_step,
            ema_state=policy.ema_state_dict(),
            scheduler_state=scheduler.state_dict() if scheduler is not None else None,
        )
    finally:
        if displaced:
            policy.swap_out_ema(displaced)
    return str(ckpt_dir / CHECKPOINT_FILE), ckpt_dir


@dataclasses.dataclass
class RunState:
    """The values the training loop carries from one update to the next."""

    global_step: int = 0
    next_eval: int = 0
    next_save: int = 0
    best_return: float | None = None
    steps_to_best: int | None = None
    above_clamp_updates: int = 0
    both_clamped_updates: int = 0
    best_artifact: str | None = None
    recent_eval_returns: deque[float] = dataclasses.field(
        default_factory=lambda: deque(maxlen=10)
    )
    all_eval_returns: list[float] = dataclasses.field(default_factory=list)
    artifacts: list[str] = dataclasses.field(default_factory=list)


def make_or_resume(
    cfg,
    device,
    obs_dim: int,
    action_dim: int,
    action_low,
    action_high,
    resume_from_checkpoint,
):
    """The policy, the optimizer, and any saved scheduler state. A resume takes the checkpoint's config."""
    if resume_from_checkpoint is not None:
        policy, optimizer, counters = load_policy_checkpoint(
            resume_from_checkpoint,
            device=device,
            action_low=action_low,
            action_high=action_high,
        )
        if optimizer is None:
            raise ResumeError(
                f"{resume_from_checkpoint} carries no optimizer state, so it "
                "cannot resume training. Load it for inference instead."
            )
        policy.train()
        return (
            policy,
            optimizer,
            int(counters["global_step"]),
            counters.get("scheduler"),
        )
    policy = make_policy(
        obs_dim,
        action_dim,
        cfg,
        device=device,
        action_low=action_low,
        action_high=action_high,
    )
    return policy, make_optimizer(policy), 0, None



@dataclasses.dataclass
class UpdateMetrics:
    """The mean metrics of one optimizer update."""

    info: dict[str, float] | None
    grad_norms: dict[str, float]
    above_clamp_frac: float
    both_clamped_frac: float
    drift_means: list[float]
    rollout_jerk: float | None
    rollout_seam_jerk: float | None

    @property
    def l_new_mean(self) -> float:
        """The mean of the new CFM loss. NaN when there is no info dict."""
        if self.info is None:
            return float("nan")
        return self.info["l_new_mean"]




def update_metrics(
    epoch_infos: list[dict[str, float]],
    above_clamp_fracs,
    both_clamped_fracs,
    drift,
    grad_norms_pre: list[dict[str, torch.Tensor]],
    rollout_jerk: float | None = None,
    rollout_seam_jerk: float | None = None,
) -> UpdateMetrics:
    """The mean of the per-minibatch metrics of one update.

    A key is averaged over the minibatches that carry it. The per-position
    probe runs on some minibatches only, so a key can be missing from the
    others. Reading it from every minibatch would raise.
    """
    if not epoch_infos:
        info = None
    else:
        keys = {k for entry in epoch_infos for k in entry}
        info = {}
        for key in sorted(keys):
            values = [entry[key] for entry in epoch_infos if key in entry]
            info[key] = sum(values) / len(values)
    scalars = [
        torch.stack(above_clamp_fracs).mean(),
        torch.stack(both_clamped_fracs).mean(),
    ]
    if drift is not None:
        scalars.extend(drift)
    above_clamp_frac, both_clamped_frac, *drift_means = torch.stack(scalars).tolist()
    grad_norms = {
        k: float(torch.stack([n[k] for n in grad_norms_pre]).mean())
        for k in ("total", "actor", "critic")
    }
    return UpdateMetrics(
        info=info,
        grad_norms=grad_norms,
        above_clamp_frac=above_clamp_frac,
        both_clamped_frac=both_clamped_frac,
        drift_means=drift_means,
        rollout_jerk=rollout_jerk,
        rollout_seam_jerk=rollout_seam_jerk,
    )


def log_update(setup, state, cfg, batch, rollout, update: UpdateMetrics) -> None:
    """Log the train rows of one update."""
    backends = setup.backends
    if not backends or update.info is None:
        return
    optimizer = setup.optimizer
    log_metrics(
        backends,
        make_train_metrics(
            cfg=cfg,
            info=update.info,
            batch=batch,
            rollout=rollout,
            grad_norms=update.grad_norms,
            above_clamp_frac=update.above_clamp_frac,
            both_clamped_frac=update.both_clamped_frac,
            drift_means=update.drift_means,
            rollout_jerk=update.rollout_jerk,
            rollout_seam_jerk=update.rollout_seam_jerk,
            policy_lr=optimizer.param_groups[0]["lr"],
            critic_lr=optimizer.param_groups[-1]["lr"],
        ),
        step=state.global_step,
    )


def check_clamp_guards(
    cfg,
    above_clamp_frac: float,
    both_clamped_frac: float,
    l_new_mean: float,
    state: RunState,
) -> None:
    """Abort when the ratio is pinned at 1 and the policy gets no gradient."""
    if both_clamped_frac > BOTH_CLAMPED_ABORT_FRAC:
        state.both_clamped_updates += 1
        if state.both_clamped_updates >= ABOVE_CLAMP_ABORT_UPDATES:
            raise TrainingAborted(
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
        state.above_clamp_updates += 1
        state.both_clamped_updates = 0
        if state.above_clamp_updates >= ABOVE_CLAMP_ABORT_UPDATES:
            raise TrainingAborted(
                f"{ABOVE_CLAMP_ABORT_UPDATES} consecutive updates with >50% "
                f"of CFM samples above cfm_clamp={cfg.cfm_clamp} "
                f"(frac={above_clamp_frac:.2f}, "
                f"cfm_loss_reduction={cfg.cfm_loss_reduction!r}). The policy "
                "ratio is pinned at 1 and gets no gradient. Raise "
                "cfm_clamp, or use a reduction whose scale fits under it."
            )
    else:
        state.above_clamp_updates = 0
        state.both_clamped_updates = 0


def log_eval_window(
    backends,
    cfg,
    eval_policies,
    result: dict,
    summaries: dict,
    state: RunState,
    r_rand: float,
) -> None:
    """Log the eval rows of one window."""
    eval_summary = summaries.get(cfg.eval_sampling, {})
    log_metrics(
        backends,
# SETUP
        {
            "eval/ema_active": float(bool(result.get("displaced", False))),
            "eval/return_ci_low": eval_summary.get("return_ci_low", 0.0),
            "eval/return_ci_high": eval_summary.get("return_ci_high", 0.0),
            "eval/return_ci_width": (
                eval_summary.get("return_ci_high", 0.0)
                - eval_summary.get("return_ci_low", 0.0)
            ),
            "eval/return_median": eval_summary.get("return_median", 0.0),
            "eval/return_n": eval_summary.get("return_n", 0),
            # The learning rate rides the train/* rows, which
            # land on every update, not only on the eval steps.
            **{
                f"eval/{name}_return_mean": summaries.get(name, {}).get(
                    "return_mean", 0.0
                )
                for name in eval_policies
            },
        },
        step=state.global_step,
    )
    log_metrics(
        backends,
        compute_eval_summary_metrics(
            result.get("mean_return", 0.0),
            result.get("eval_results", []),
            state.all_eval_returns,
            r_rand=r_rand,
            best_return=state.best_return,
            steps_to_best=state.steps_to_best,
        )
        if result.get("eval_results")
        else {},
        step=state.global_step,
    )


def eval_window(
    policy,
    optimizer,
    state: RunState,
    out_dir,
    evaluator,
    eval_policies,
    cfg,
    backends,
    eval_steps: int,
    r_rand: float,
    video=None,
    scheduler=None,
) -> None:
    """Evaluate both flow starts, log the window, and keep the best checkpoint."""
    result, summaries = evaluate_policy(
        policy,
        state.global_step,
        out_dir,
        evaluator,
        eval_policies,
        cfg,
        best_return=state.best_return,
        video=video,
    )
    mean_return = result.get("mean_return", 0.0)
    stop_return = result.get("stop_return", mean_return)
    state.all_eval_returns.append(mean_return)
    state.recent_eval_returns.append(mean_return)
    if result.get("improved", False):
        state.best_return, state.steps_to_best = stop_return, state.global_step
        best_artifact_path, _ = save_checkpoint(
            out_dir,
            policy,
            optimizer,
            state.global_step,
            "best",
            use_ema=True,
            scheduler=scheduler,
        )
        state.best_artifact = best_artifact_path
    if backends:
        log_eval_window(
            backends, cfg, eval_policies, result, summaries, state, r_rand
        )
    state.next_eval = max(
        state.global_step + eval_steps, state.next_eval + eval_steps
    )


def finalize_run(
    out_dir, policy, optimizer, state: RunState, env_name, seed, fps, scheduler=None
):
    """The final checkpoint, the artifact list, and the run summary."""
    final_path, _ = save_checkpoint(
        out_dir,
        policy,
        optimizer,
        state.global_step,
        "final",
        use_ema=True,
        scheduler=scheduler,
    )
    state.artifacts.append(final_path)
    if state.best_artifact is not None and state.best_artifact not in state.artifacts:
        state.artifacts.append(state.best_artifact)

    csv_path = out_dir / "eval_history.csv"
    if csv_path.exists():
        state.artifacts.append(str(csv_path))

    return {
        "method": "myfpo",
        "env_name": env_name,
        "seed": seed,
        "reward_mean_last10": (
            float(np.mean(state.recent_eval_returns))
            if state.recent_eval_returns
            else 0.0
        ),
        "reward_median_last10": (
            float(np.median(state.recent_eval_returns))
            if state.recent_eval_returns
            else 0.0
        ),
        "fps": fps,
        "artifacts": state.artifacts,
        "best_artifact": state.best_artifact,
        "best_return": state.best_return,
        "global_step": state.global_step,
    }


@dataclasses.dataclass
class RunSetup:
    """What one run needs before the training loop starts."""

    policy: Any
    optimizer: Any
    train_env: Any
    device: torch.device
    action_low: np.ndarray
    action_high: np.ndarray
    evaluator: Evaluator
    eval_policies: dict
    r_rand: float
    backends: list
    env_name: str
    global_step: int
    eval_env: Any = None
    video_env: Any = None
    video_evaluator: Any = None
    video_dir: Path | None = None
    scheduler: Any = None


# Rows in the recording env. ``render()`` renders every row, so this is kept
# small on purpose: the training width would be hundreds of MB per frame.
VIDEO_ENVS = 4

# Device memory a SAPIEN GPU sim context needs before it can start, measured
# (~1.3 GiB for a 4-row sim, almost independent of ``num_envs``). The recording
# env is a third context on top of training and evaluation. Below this the
# context creation faults and takes the whole run with it, so the budget is
# checked rather than discovered.
VIDEO_CONTEXT_BYTES = 1536 * 2**20


def make_train_env(cfg, env, env_name, runtime: dict | None):
    """The env the policy collects from. A vector env when the config asks."""
    if callable(getattr(env, "step_torch", None)):
        # A torch env already batches its own envs on the device. Forking
        # workers for it would start one SAPIEN context per worker.
        if runtime is not None:
            runtime["vector_env"] = "maniskill_gpu"
        return env
    if cfg.num_envs > 1 and env_name != "unknown":
        return make_vec_env(env_name, cfg.num_envs, record=runtime)
    if runtime is not None:
        runtime["vector_env"] = "single"
    if cfg.num_envs > 1:
        warnings.warn(
            f"num_envs={cfg.num_envs} requested but env_name={env_name!r} "
            "(no spec.id on the passed env), so vectorization was skipped "
            "and the run uses a single env. Expect a large throughput drop; "
            "the run is otherwise correct.",
            RuntimeWarning,
            stacklevel=3,
        )
    return env


def configure_run(
    env,
    out_dir: Path,
    cfg: MyfpoConfig,
    *,
    seed: int,
    total_steps: int,
    obs_dim: int,
    action_dim: int,
    resume_from_checkpoint,
    runtime: dict | None,
    video_dir: Path | None = None,
) -> RunSetup:
    """The env, the policy, the evaluator, and the log backends."""
    unwrapped = getattr(env, "unwrapped", env)
    spec = getattr(unwrapped, "spec", None)
    env_name = getattr(spec, "id", "unknown") if spec is not None else "unknown"

    set_seed(seed)
    device = resolve_device()

    train_env = make_train_env(cfg, env, env_name, runtime)
    if runtime is not None:
        runtime["eval_sampling"] = cfg.eval_sampling
        runtime["stop_eval_sampling"] = cfg.stop_eval_sampling
        (out_dir / "run_metadata.json").write_text(
            json.dumps(runtime, indent=2, sort_keys=True), encoding="utf-8"
        )

    action_low = np.atleast_1d(np.asarray(env.action_space.low, dtype=np.float32))
    action_high = np.atleast_1d(np.asarray(env.action_space.high, dtype=np.float32))

    policy, optimizer, global_step, scheduler_state = make_or_resume(
        cfg,
        device,
        obs_dim,
        action_dim,
        action_low,
        action_high,
        resume_from_checkpoint,
    )
    # A resume takes the checkpoint's config. Every later build here must read
    # that one, not the CLI's, or the scheduler and the evaluators disagree
    # with the policy they wrap. The train env is already built, so say so.
    if resume_from_checkpoint is not None:
        if policy.config.num_envs != cfg.num_envs:
            print(
                f"myfpo: resume: the checkpoint trained with num_envs="
                f"{policy.config.num_envs}, the CLI asked for {cfg.num_envs}; "
                f"the train env keeps {cfg.num_envs}.",
                file=sys.stderr,
            )
        cfg = policy.config
    scheduler = make_scheduler(optimizer, cfg, total_steps)
    if scheduler_state is not None or global_step > 0:
        restore_scheduler(scheduler, scheduler_state, global_step, cfg)

    print(
        f"myfpo policy: actor_depth={cfg.dit_depth} hidden={cfg.hidden_dim} "
        f"params={sum(p.numel() for p in policy.parameters() if p.requires_grad):,}",
        file=sys.stderr,
    )
    print(policy, file=sys.stderr)

    eval_seed = 10000 + seed
    eval_env = None
    video_env = None
    video_evaluator = None
    if callable(getattr(train_env, "make_eval_env", None)):
        eval_env = train_env.make_eval_env()
        evaluator = VectorEvaluator(
            eval_env,
            n_obs_steps=cfg.n_obs_steps,
            n_action_steps=cfg.n_action_steps,
            action_horizon=cfg.action_horizon,
            eval_seed=eval_seed,
            max_episodes=cfg.eval_max_episodes,
        )
        if video_dir is not None:
            # A separate, small render env: rendering costs GPU time per row and
            # returns every row, so the training width is the wrong thing to
            # record. Same task and seed, so the video is the same policy.
            free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
            if free < VIDEO_CONTEXT_BYTES:
                print(
                    f"myfpo: skipping video: {free / 2**20:.0f} MiB free, a render "
                    f"context needs ~{VIDEO_CONTEXT_BYTES / 2**20:.0f} MiB. "
                    "Free the GPU or lower --myfpo.num_envs.",
                    file=sys.stderr,
                )
            else:
                try:
                    video_env = train_env.make_eval_env(
                        num_envs=min(VIDEO_ENVS, cfg.num_envs),
                        render_mode="rgb_array",
                    )
                except Exception as exc:  # noqa: BLE001 - a video must not end the run
                    print(
                        "myfpo: skipping video, could not start a render env: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    video_env = None
            if video_env is not None:
                video_evaluator = VectorEvaluator(
                    video_env,
                    n_obs_steps=cfg.n_obs_steps,
                    n_action_steps=cfg.n_action_steps,
                    action_horizon=cfg.action_horizon,
                    eval_seed=eval_seed,
                    # One round of rows is one episode each: enough for a video.
                    max_episodes=min(VIDEO_ENVS, cfg.num_envs),
                )
    else:
        evaluator = Evaluator(
            env, eval_seed=eval_seed, max_episodes=cfg.eval_max_episodes
        )
    if video_dir is not None and video_evaluator is None:
        # Rather than fail the run over a diagnostic, say so and carry on: the
        # CPU envs have no batched renderer to record from.
        print(
            "myfpo: video requested but this env cannot render in batch; "
            "no video will be written (needs an MS-* env)",
            file=sys.stderr,
        )
    eval_policies = {
        "zero_start": EvalPolicy(policy, start_from_zero=True),
        "noise_start": EvalPolicy(policy, start_from_zero=False),
    }
    r_rand = sample_random_baseline(evaluator)
    backends = make_backends(
        cfg,
        out_dir=out_dir,
        method="myfpo",
        env_name=env_name,
        seed=seed,
        total_steps=total_steps,
    )
    return RunSetup(
        policy=policy,
        optimizer=optimizer,
        train_env=train_env,
        device=device,
        action_low=action_low,
        action_high=action_high,
        evaluator=evaluator,
        eval_policies=eval_policies,
        r_rand=r_rand,
        backends=backends,
        env_name=env_name,
        global_step=global_step,
        eval_env=eval_env,
        video_env=video_env,
        video_evaluator=video_evaluator,
        video_dir=Path(video_dir) if video_dir is not None else None,
        scheduler=scheduler,
    )




def invalidate_shared_env_carry(policy, evaluator, train_env) -> bool:
    """Drop the collector's carry when the evaluator used the training env.

    A single-env run evaluates on the same env the collector drives. The
    evaluator resets it and steps whole episodes, so the cached windows and
    autoreset flags describe a state that no longer exists. Returning True
    means the carry was dropped.

    The carry lives on the host rollout buffer, which is the object the
    collector reads it from. ``RolloutBuffers`` holds no copy of it.
    """
    if getattr(evaluator, "env", None) is not train_env:
        return False
    buffers = getattr(policy, "rollout_buffers", None)
    host = getattr(buffers, "host", None)
    if host is None:
        return False
    host.carry = None
    return True


def run_training_loop(
    setup: RunSetup, state: RunState, out_dir: Path,
    *, total_steps: int, eval_steps: int, save_steps: int,
) -> float:
    """The training loop. Return the frames per second of the run."""
    policy = setup.policy
    # On a resume the checkpoint config wins: make_or_resume already ignores
    # CLI scalar overrides, and the loop must not read two configs.
    cfg = policy.config
    optimizer = setup.optimizer
    train_env = setup.train_env
    action_low = setup.action_low
    action_high = setup.action_high
    evaluator = setup.evaluator
    eval_policies = setup.eval_policies
    r_rand = setup.r_rand
    backends = setup.backends
    device = setup.device
    grad_groups = split_grad_params(policy)
    t_start = time.time()

    scheduler = setup.scheduler
    with tqdm(total=total_steps, desc="train", unit="step") as pbar:
        while state.global_step < total_steps:
            steps_to_go = total_steps - state.global_step
            t0 = time.perf_counter()
            collected = rollout_with_diagnostics(
                policy, train_env, cfg, action_low, action_high, steps_to_go
            )
            batch, rollout = collected.batch, collected.diagnostics
            steps_this_rollout = collected.executed_steps
            rollout_jerk, rollout_seam_jerk = collected.jerk, collected.seam_jerk
            rollout["collect_s"] = time.perf_counter() - t0
            if cfg.normalize_rewards and hasattr(policy, "reward_normalizer"):
                rew = rollout.get("step_rewards") or batch.get("step_rewards")
                if rew is not None:
                    policy.reward_normalizer.update(rew.reshape(-1, 1))
            state.global_step += steps_this_rollout
            pbar.update(steps_this_rollout)


            epoch_infos, grad_norms_pre, above_clamp_fracs, both_clamped_fracs = policy_update_step(
                policy, optimizer, batch, backends, cfg, grad_groups
            )
            if scheduler is not None:
                scheduler.step()

            drift = None
            if cfg.log_drift_diagnostics and backends:
                drift = measure_value_drift(policy, batch, device)

            policy.update_ema(state.global_step)
            # Normaliser updated at end of cycle: rollout + update used frozen stats.
            policy.update_observation_stats(batch["normalizer_observations"])
            policy.update_action_stats(batch["action"], batch["action_is_pad"])
            policy.update_target_critic()

            update = update_metrics(
                epoch_infos,
                above_clamp_fracs,
                both_clamped_fracs,
                drift,
                grad_norms_pre,
                rollout_jerk,
                rollout_seam_jerk,
            )
            rollout["actor_step_s"] = policy.actor_step_s
            rollout["critic_step_s"] = policy.critic_step_s
            log_update(setup, state, cfg, batch, rollout, update)

            check_clamp_guards(
                cfg,
                update.above_clamp_frac,
                update.both_clamped_frac,
                update.l_new_mean,
                state,
            )




            if state.global_step >= state.next_eval or state.global_step >= total_steps:
                eval_window(
                policy,
                optimizer,
                state,
                out_dir,
                evaluator,
                eval_policies,
                cfg,
                backends,
                eval_steps,
                r_rand,
                video=(
                    (setup.video_evaluator, setup.video_dir)
                    if setup.video_evaluator is not None
                    else None
                ),
                scheduler=scheduler,
                )
                invalidate_shared_env_carry(policy, evaluator, train_env)

            if state.global_step >= state.next_save or state.global_step >= total_steps:
                artifact_path, _ = save_checkpoint(
                    out_dir,
                    policy,
                    optimizer,
                    state.global_step,
                    f"step_{state.global_step}",
                    scheduler=scheduler,
                )
                state.artifacts.append(artifact_path)
                state.next_save = max(
                    state.global_step + save_steps, state.next_save + save_steps
                )
            elapsed = time.time() - t_start
        fps = state.global_step / max(elapsed, 1e-6)
        if backends:
            log_metrics(
                backends,
                {"timing/fps": fps, "timing/elapsed_s": elapsed},
                step=state.global_step,
            )
    return fps


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
    video_dir: str | Path | None = None,
) -> dict[str, Any]:
    """One myfpo training run. Env builds before any CUDA init, so forked
    vector-env workers inherit the right device state.

    ``video_dir`` records the eval rollout of every eval window as an mp4
    (``step_<N>.mp4``). It needs a torch env that can render, and a first eval
    window: nothing is written if the run never evaluates.
    """
    cfg = cfg or MyfpoConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if obs_dim is None or action_dim is None:
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))

    setup = configure_run(
        env,
        out_dir,
        cfg,
        seed=seed,
        total_steps=total_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        resume_from_checkpoint=resume_from_checkpoint,
        runtime=runtime,
        video_dir=video_dir,
    )
    policy = setup.policy
    optimizer = setup.optimizer
    train_env = setup.train_env
    backends = setup.backends
    env_name = setup.env_name

    state = RunState(
        global_step=setup.global_step, next_eval=eval_steps, next_save=save_steps
    )
    try:
        fps = run_training_loop(
            setup,
            state,
            out_dir,
            total_steps=total_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
        )
    finally:
        if train_env is not env:
            train_env.close()
        if setup.eval_env is not None and setup.eval_env is not train_env:
            setup.eval_env.close()
        if setup.video_env is not None and setup.video_env is not train_env:
            setup.video_env.close()
        finish_backends(backends)

    return finalize_run(
        out_dir,
        policy,
        optimizer,
        state,
        env_name,
        seed,
        fps,
        scheduler=setup.scheduler,
    )

# TRACKING


class WandbBackend:
    """Writes scalars to a wandb run."""

    def __init__(
        self,
        *,
        name: str,
        method: str,
        env_name: str,
        seed: int,
        total_steps: int,
        cfg_dict: dict,
    ):
        import wandb

        self.wandb = wandb
        wandb.init(
            project=method,
            group=f"{method}_{env_name}",
            name=name,
            tags=["mujoco"],
            config={**cfg_dict, "method": method, "seed": seed, "env": env_name},
            job_type="smoke" if total_steps <= 10000 else "train",
            mode=os.environ.get("WANDB_MODE", "online"),
            reinit="finish_previous",
        )

    def log(self, data: dict[str, float], step: int) -> None:
        """Log the scalars at the given step."""
        self.wandb.log(data, step=step)

    def finish(self, exit_code: int) -> None:
        """Finish the wandb run with the given exit code."""
        self.wandb.finish(exit_code=exit_code)


class TensorboardBackend:
    """Writes scalars to a tensorboard event file."""

    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir, flush_secs=30)

    def log(self, data: dict[str, float], step: int) -> None:
        """Log each scalar at the given step. None values are skipped."""
        for key, value in data.items():
            if value is None:
                continue
            self.writer.add_scalar(key, float(value), step)

    def finish(self, _exit_code: int) -> None:
        """Close the event writer."""
        self.writer.close()


def make_run_name(method: str, env_name: str, seed: int, cfg_dict: dict) -> str:
    """The run name: method, env, seed, the knobs that differ, and a timestamp.

    The timestamp tells two runs of the same config apart in the wandb list.
    """
    tags: list[str] = []
    get = cfg_dict.get
    if (v := get("action_horizon")) is not None:
        tags.append(f"h{v}")
    if (v := get("hidden_dim")) is not None and (d := get("dit_depth")) is not None:
        tags.append(f"d{v}x{d}")
    if (v := get("num_learning_epochs")) is not None:
        tags.append(f"ep{v}")
    if (v := get("policy_lr")) is not None:
        tags.append(f"lr{v:g}" if isinstance(v, float) else f"lr{v}")
    if (v := get("clip_param")) is not None:
        tags.append(f"eps{v:g}" if isinstance(v, float) else f"eps{v}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return "_".join([f"{method}_{env_name}_s{seed}", *tags, stamp])


def make_backends(
    cfg,
    *,
    out_dir: str | Path,
    method: str,
    env_name: str,
    seed: int,
    total_steps: int,
) -> list:
    """The backends a run logs to. An empty list means no logging."""
    report_to = cfg.report_to
    if report_to == "none":
        return []

    cfg_dict = dataclasses.asdict(cfg)
    name = make_run_name(method, env_name, seed, cfg_dict)
    backends: list = []
    if report_to in ("wandb", "all"):
        backends.append(
            WandbBackend(
                name=name,
                method=method,
                env_name=env_name,
                seed=seed,
                total_steps=total_steps,
                cfg_dict=cfg_dict,
            )
        )
    if report_to in ("tensorboard", "all"):
        log_dir = Path(out_dir) / "tensorboard" / name
        backends.append(TensorboardBackend(str(log_dir)))
    return backends


def log_metrics(backends: list, data: dict[str, float], step: int) -> None:
    """Write one scalar dict to every backend."""
    for backend in backends:
        backend.log(data, step)


def finish_backends(backends: list, exit_code: int = 0) -> None:
    """Close every backend."""
    for backend in backends:
        backend.finish(exit_code)

# METRICS


@dataclass
class WindowSummary:
    """What one eval window found."""

    reported_results: list[dict]
    summaries: dict[str, dict]
    mean_return: float
    stop_return: float
    improved: bool


def summarize_window(
    results_by_sampling: dict[str, list[dict]],
    *,
    reported_name: str,
    stop_name: str,
    eval_seed: int,
    summarize,
    best_return: float | None,
) -> WindowSummary:
    """Summarise every start, then read the two means the runner needs."""
    summaries = {
        name: summarize(rows, seed=eval_seed)
        for name, rows in results_by_sampling.items()
    }
    return WindowSummary(
        reported_results=results_by_sampling[reported_name],
        summaries=summaries,
        mean_return=summaries[reported_name]["return_mean"],
        stop_return=summaries[stop_name]["return_mean"],
        improved=best_return is None or summaries[stop_name]["return_mean"] > best_return,
    )


GAE_DIAG_LOG_KEYS = {
    "linked_frac": "gae_linked_frac",
    "advantage_std": "gae_advantage_std",
    "return_rms": "returns_rms",
    "advantage_std_over_return_std": "advantage_std_over_return_std",
}

ROLLOUT_ABS_KEYS = (
    "raw_action_abs_mean",
    "raw_action_abs_max",
    "executed_action_abs_mean",
    "executed_action_abs_max",
)


def make_train_metrics(
    *,
    cfg: MyfpoConfig,
    info: dict,
    batch: dict,
    rollout: dict,
    grad_norms: dict[str, float],
    above_clamp_frac: float,
    both_clamped_frac: float,
    drift_means: list[float],
    rollout_jerk: float | None,
    rollout_seam_jerk: float | None,
    policy_lr: float,
    critic_lr: float,
) -> dict:
    """Return the ``train/*`` rows for one optimizer step."""
    data = {f"train/{k}": v for k, v in info.items()}
    data["train/grad_norm_before_clip"] = grad_norms["total"]
    data["train/grad_norm_actor_before_clip"] = grad_norms["actor"]
    data["train/grad_norm_critic_before_clip"] = grad_norms["critic"]
    data["train/cfm_above_clamp_frac"] = above_clamp_frac
    data["train/cfm_both_clamped_frac"] = both_clamped_frac
    # The learning rate of the step that is about to run. It is logged here,
    # on every update, so the curve has one point per update and not one per
    # evaluation.
    data["train/policy_lr"] = policy_lr
    data["train/critic_lr"] = critic_lr
    if cfg.log_drift_diagnostics and drift_means:
        data["train/boot_value_relative_drift"] = drift_means[0]
        data["train/value_target_drift"] = drift_means[1]
    if rollout_jerk is not None:
        data["train/rollout_jerk"] = rollout_jerk
    if rollout_seam_jerk is not None:
        data["train/rollout_seam_jerk"] = rollout_seam_jerk
    for key in ROLLOUT_ABS_KEYS:
        data[f"train/{key}"] = rollout[key]
    for key in ("collect_s", "actor_step_s", "critic_step_s"):
        data[f"train/{key}"] = rollout[key]
    gae_diag = batch.get("gae_diagnostics")
    if gae_diag:
        data.update(
            {
                f"train/{log}": gae_diag[src]
                for src, log in GAE_DIAG_LOG_KEYS.items()
            }
        )
    data["train/total_loss"] = (
        info["policy_loss"] + cfg.value_loss_coef * info["value_loss"]
    )
    return data



