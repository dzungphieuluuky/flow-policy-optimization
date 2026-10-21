"""myfpo online training entry (LeRobot-native, MuJoCo-only)."""

from __future__ import annotations

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
from myfpo.collector import resolve_action_bounds
from myfpo.configuration import MyfpoConfig
from myfpo.evaluator import (
    Evaluator,
    compute_mean_chunk_jerk,
    compute_seam_jerk,
)
from myfpo.modeling import MyfpoAborted, MyfpoPolicy, build_optimizer
from myfpo.metrics import compute_eval_summary_metrics, subsampled_min
from myfpo.processor import squash_to_env_bounds
from lerobot.utils.constants import CHECKPOINTS_DIR
from myfpo.utils import cudagraph_step_begin, resolve_device, rms, to_numpy_1d
from myfpo.tracking import build_backends, finish_backends, log_metrics
from myfpo.runner import build_train_metrics, summarise_window


class EvalPolicy:
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
    if cfg.compile:
        policy.actor = torch.compile(policy.actor, mode="reduce-overhead")
        policy.encoder = torch.compile(policy.encoder, mode="reduce-overhead")
        policy.critic = torch.compile(policy.critic, mode="reduce-overhead")
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


OBS_WARMUP_STEPS = 128


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
    for _ in range(OBS_WARMUP_STEPS):
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
    return squash_to_env_bounds(
        action.float(),
        low.to(device=device, dtype=torch.float32),
        high.to(device=device, dtype=torch.float32),
    )


@torch.no_grad()
def compute_rollout_diagnostics(
    batch: dict[str, torch.Tensor], *, action_low, action_high
) -> dict[str, float]:
    """Read the post-rollout scalars with one device-to-host transfer.

    Two action spaces are reported: the raw chunk the flow produced, which
    is what the policy is trained on, and the chunk the env received.
    """
    action = batch["action"].float()
    is_pad = batch["action_is_pad"]
    executed = ~is_pad
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
    rollout = compute_rollout_diagnostics(
        batch, action_low=action_low, action_high=action_high
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
    jerk_action = (
        executed_chunk(batch, action_low, action_high).detach().cpu().numpy()
    )
    jerk_pad = batch["action_is_pad"].detach().cpu().numpy()
    rollout_jerk = compute_mean_chunk_jerk(jerk_action, jerk_pad)
    rollout_seam_jerk = compute_seam_jerk(jerk_action, jerk_pad)
    return batch, rollout, steps_this_rollout, rollout_jerk, rollout_seam_jerk


def train_update(policy, optimizer, batch, backends, cfg, grad_groups):
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
        # One minibatch is one cudagraph step. Its backward runs before the
        # next forward, so the manager may recycle the step's memory.
        cudagraph_step_begin()
        if backends:
            loss, epoch_info = policy.loss_with_info(minibatch, check_finite=False)
        else:
            loss = policy(minibatch, check_finite=False)
            epoch_info = None
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


@dataclasses.dataclass
class RunState:
    """The values the training loop carries from one update to the next."""

    global_step: int = 0
    next_eval: int = 0
    next_save: int = 0
    best_return: float | None = None
    steps_to_best: int | None = None
    eval_count: int = 0
    above_clamp_updates: int = 0
    both_clamped_updates: int = 0
    best_artifact: str | None = None
    recent_eval_returns: deque[float] = dataclasses.field(
        default_factory=lambda: deque(maxlen=10)
    )
    all_eval_returns: list[float] = dataclasses.field(default_factory=list)
    artifacts: list[str] = dataclasses.field(default_factory=list)


def build_or_resume(
    cfg,
    device,
    obs_dim: int,
    action_dim: int,
    action_low,
    action_high,
    resume_from_checkpoint,
):
    """The policy and the optimizer. A resume takes the checkpoint's config."""
    if resume_from_checkpoint is not None:
        policy, optimizer, counters = load_policy_checkpoint(
            resume_from_checkpoint,
            device=device,
            action_low=action_low,
            action_high=action_high,
        )
        policy.train()
        return policy, optimizer, int(counters["global_step"])
    policy = build_policy(
        obs_dim,
        action_dim,
        cfg,
        device=device,
        action_low=action_low,
        action_high=action_high,
    )
    return policy, build_optimizer(policy), 0



@dataclasses.dataclass
class UpdateMetrics:
    """The mean metrics of one optimiser update."""

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
    """The mean of the per-minibatch metrics of one update."""
    info = (
        {
            k: sum(d[k] for d in epoch_infos) / len(epoch_infos)
            for k in epoch_infos[0]
        }
        if epoch_infos
        else None
    )
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
        build_train_metrics(
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
        state.above_clamp_updates += 1
        state.both_clamped_updates = 0
        if state.above_clamp_updates >= ABOVE_CLAMP_ABORT_UPDATES:
            raise MyfpoAborted(
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
            state.eval_count,
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
    )
    mean_return = result.get("mean_return", 0.0)
    stop_return = result.get("stop_return", mean_return)
    state.all_eval_returns.append(mean_return)
    state.recent_eval_returns.append(mean_return)
    state.eval_count += 1
    if result.get("improved", False):
        state.best_return, state.steps_to_best = stop_return, state.global_step
        best_artifact_path, _ = save_checkpoint(
            out_dir, policy, optimizer, state.global_step, "best"
        )
        state.best_artifact = best_artifact_path
    if backends:
        log_eval_window(
            backends, cfg, eval_policies, result, summaries, state, r_rand
        )
    state.next_eval = max(
        state.global_step + eval_steps, state.next_eval + eval_steps
    )


def finalize_run(out_dir, policy, optimizer, state: RunState, env_name, seed, fps):
    """The final checkpoint, the artifact list, and the run summary."""
    final_path, _ = save_checkpoint(
        out_dir, policy, optimizer, state.global_step, "final"
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


def make_train_env(cfg, env, env_name, runtime: dict | None):
    """The env the policy collects from. A vector env when the config asks."""
    if cfg.num_envs > 1 and env_name != "unknown":
        return make_vec_env(env_name, cfg.num_envs, record=runtime)
    if runtime is not None:
        runtime["vector_env"] = "single"
    if cfg.num_envs > 1:
        warnings.warn(
            f"num_envs={cfg.num_envs} requested but env_name={env_name!r} "
            "(no spec.id on the passed env), so vectorisation was skipped "
            "and the run uses a single env. Expect a large throughput drop; "
            "the run is otherwise correct.",
            RuntimeWarning,
            stacklevel=3,
        )
    return env


def setup_run(
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

    policy, optimizer, global_step = build_or_resume(
        cfg,
        device,
        obs_dim,
        action_dim,
        action_low,
        action_high,
        resume_from_checkpoint,
    )

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
    print(policy, file=sys.stderr)

    eval_seed = 10000 + seed
    evaluator = Evaluator(
        env, eval_seed=eval_seed, max_episodes=cfg.eval_max_episodes
    )
    eval_policies = {
        "zero_start": EvalPolicy(policy, start_from_zero=True),
        "noise_start": EvalPolicy(policy, start_from_zero=False),
    }
    r_rand = sample_random_baseline(evaluator, env, seed=seed)
    backends = build_backends(
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
    )


def train_loop(
    setup: RunSetup,
    state: RunState,
    cfg: MyfpoConfig,
    out_dir: Path,
    *,
    total_steps: int,
    eval_steps: int,
    save_steps: int,
) -> float:
    """The training loop. Return the frames per second of the run."""
    policy = setup.policy
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

    while state.global_step < total_steps:
        steps_to_go = total_steps - state.global_step
        batch, rollout, steps_this_rollout, rollout_jerk, rollout_seam_jerk = collect_rollouts(
            policy, train_env, cfg, action_low, action_high, steps_to_go
        )
        state.global_step += steps_this_rollout

        epoch_infos, grad_norms_pre, above_clamp_fracs, both_clamped_fracs = train_update(
            policy, optimizer, batch, backends, cfg, grad_groups
        )

        drift = None
        if cfg.log_drift_diagnostics and backends:
            drift = measure_value_drift(policy, batch, device)

        policy.update_ema(state.global_step)

        policy.update_observation_stats(batch["normalizer_observations"])

        update = update_metrics(
            epoch_infos,
            above_clamp_fracs,
            both_clamped_fracs,
            drift,
            grad_norms_pre,
            rollout_jerk,
            rollout_seam_jerk,
        )
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
            )

        if state.global_step >= state.next_save or state.global_step >= total_steps:
            artifact_path, _ = save_checkpoint(
                out_dir, policy, optimizer, state.global_step, f"step_{state.global_step}"
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
) -> dict[str, Any]:
    """One myfpo training run. Env builds before any CUDA init, so forked
    vector-env workers inherit the right device state.
    """
    cfg = cfg or MyfpoConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if obs_dim is None or action_dim is None:
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))

    setup = setup_run(
        env,
        out_dir,
        cfg,
        seed=seed,
        total_steps=total_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        resume_from_checkpoint=resume_from_checkpoint,
        runtime=runtime,
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
        fps = train_loop(
            setup,
            state,
            cfg,
            out_dir,
            total_steps=total_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
        )
    finally:
        if train_env is not env:
            train_env.close()
        finish_backends(backends)

    return finalize_run(out_dir, policy, optimizer, state, env_name, seed, fps)


