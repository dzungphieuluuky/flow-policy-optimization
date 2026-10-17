"""SJFPO online training entry (LeRobot-native, MuJoCo-only).

``generate_rollout`` gathers a batch, ``forward`` computes the loss. This loop
steps the optimizer, evaluates, and checkpoints. It also warms the obs stats,
sets the LR, updates the EMA, folds frames into the normalizer, and reads the
rollout diagnostics. Not loss math -- that lives in the policy.
"""

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
from myfpo.configuration_sjfpo import FLOW_STARTS, SJFPOConfig
from myfpo.evaluator import (
    Evaluator,
    compute_mean_chunk_jerk,
    compute_seam_jerk,
)
from myfpo.modeling_sjfpo import SJFPOAborted, SJFPOPolicy, build_optimizer
from lerobot.utils.constants import CHECKPOINTS_DIR
from myfpo.utils import resolve_device, rms, to_numpy_1d
from myfpo.tracking import build_tracker, compute_eval_summary_metrics
from myfpo.runner import build_train_metrics, summarise_window


# Consecutive above-clamp updates before aborting. clamp() has zero gradient
# above its max, so the policy stops learning while the critic keeps training.
ABOVE_CLAMP_ABORT_UPDATES = 5

# Share of ratio elements whose two losses both sit above cfm_clamp. At that
# share the ratio is exactly 1.0, so the policy gradient is zero. The share
# reads high when the loss level sits above the clamp, which the reduction and
# the action horizon both set. The value is half, so an ordinary run that keeps
# a gradient stays well under it.
BOTH_CLAMPED_ABORT_FRAC = 0.5

def generate_mini_batch_indices(
    n: int,
    num_mini_batches: int,
    device: torch.device | str,
    num_learning_epochs: int = 1,
):
    """Yield shuffled minibatch indices, one tensor per epoch.

    Every epoch reshuffles. The splits use the proportional form
    ``(i * n) // n_mb``. So a remainder spreads over the frequencies instead of
    being dropped.
    """
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

class _EvalPolicy:
    """A single-action eval adapter over :meth:`SJFPOPolicy.select_action`.

    The value ``start_from_zero`` picks the start point of the flow. The value
    ``True`` starts from ``eps=0``. The value ``False`` starts from a drawn
    ``eps``, and the training rollouts use it.

    The argument was named ``zero_sampling``, which read as a sampling method.
    It is a start point. The name also collided with ``noise_sampling_mode``,
    which is a real method.

    One vocabulary covers the config, the logs, and this adapter:

    ==========================  =========================================
    Config value                Meaning
    ==========================  =========================================
    ``zero_start``              ``start_from_zero=True``. Chain starts at
                                 ``eps=0``.
    ``noise_start``             ``start_from_zero=False``. Chain starts at
                                 a drawn ``eps``.
    ==========================  =========================================

    Running both adapters over the same seeds is the only way to see the two
    distributions diverge.
    """

    def __init__(self, policy: SJFPOPolicy, start_from_zero: bool = True):
        self.policy = policy
        self.start_from_zero = start_from_zero

    @classmethod
    def for_sampling(cls, policy: SJFPOPolicy, sampling: str) -> "_EvalPolicy":
        """Build the adapter for the config value ``sampling``."""
        if sampling not in FLOW_STARTS:
            raise ValueError(
                f"sampling={sampling!r} must be one of "
                f"{', '.join(repr(k) for k in FLOW_STARTS)}"
            )
        return cls(policy, start_from_zero=(sampling == "zero_start"))

    def reset(self) -> None:
        """Drop the policy's cached chunk state before a new episode."""
        self.policy.reset()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        was_training = self.policy.training
        self.policy.eval()
        try:
            action = self.policy.select_action(
                {"observation.state": obs},
                start_from_zero=self.start_from_zero,
            )
        finally:
            self.policy.train(was_training)
        return to_numpy_1d(action)


def build_policy(
    obs_dim: int,
    action_dim: int,
    cfg: SJFPOConfig,
    device=None,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
) -> SJFPOPolicy:
    """Attach runtime features, then build the policy."""
    device = torch.device(device) if device is not None else resolve_device()
    policy_cfg = dataclasses.replace(
        cfg,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (obs_dim,))
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (action_dim,))},
    )
    policy = SJFPOPolicy(policy_cfg).to(device)
    if action_low is not None and action_high is not None:
        policy.set_action_bounds(action_low, action_high)
    if getattr(cfg, "compile", False):
        # The mode comes from the capability, as the PyTorch docs state.
        # sm_89 and above give max-autotune. A lower capability gives
        # default or reduce-overhead. The value fullgraph=True needs static
        # graphs.
        mode = "default"
        try:
            if device.type == "cuda":
                idx = int(device.index or 0)
                props = torch.cuda.get_device_properties(idx)
                if (props.major, props.minor) >= (8, 9):
                    mode = "max-autotune"
                elif props.major >= 7:
                    mode = "reduce-overhead"
        except Exception:
            pass
        # fullgraph=False avoids recompilation on varying chunk/MC shapes.
        # Assignment breaks EMA/state_dict identity (gate 0b/0c).
        policy.actor = torch.compile(policy.actor, dynamic=True, mode=mode, fullgraph=False)
        policy.encoder = torch.compile(policy.encoder, dynamic=True, mode=mode, fullgraph=False)
        policy.critic = torch.compile(policy.critic, dynamic=True, mode=mode, fullgraph=False)
    return policy


def sample_minibatch(
    batch: dict[str, torch.Tensor],
    indices: torch.Tensor,
    out: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Slice the leading dimension of every per-chunk batch value.

    They all have the shape ``[B, ...]``. So one index tensor keeps them
    consistent.

    The values ``normalizer_observations`` and ``gae_diagnostics`` pass through.
    They have one row per executed env step, not one row per chunk.
    """
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
    """Apply the linear warmup, starting from the bases.

    The rule is ``lr = base_lr * min(1, step / warmup)``.

    The function recomputes the value instead of compounding it, so the value
    depends only on ``step_count`` and never on the order of the calls.

    The function used to take ``lr_decays`` and multiply by
    ``cfg.lr_decay**lr_decays``. A regression flag was the only caller that
    raised the count, so the multiplier was always one and the parameter was
    unreachable. Both the flag and the two config fields are gone.

    FPO++ reaches the warmup half through a scheduler
    (finetune_online_rl.py). This is the same ramp without the object.
    """
    for group in optimizer.param_groups:
        base = group["base_lr"]
        warmup = group.get("lr_warmup_steps", 0)
        ramp = 1.0 if warmup <= 0 else min(1.0, step_count / warmup)
        group["lr"] = base * ramp


def warm_observation_stats(policy, env, *, obs_dim: int, seed: int) -> int:
    """Fold random-policy frames into the running obs stats.

    The normalizer is the identity until its first update, and it updates only
    after a rollout. So without this function, the whole first rollout runs on
    raw obs. The env resets after.
    """
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
    """Read the post-rollout scalars with one device-to-host transfer.

    A separate ``float()`` call costs one CUDA sync each. The key
    ``action_abs_max`` blocks on one of those syncs.

    The collector stores the full scaled chunk. So the unexecuted rows still
    carry a real proposed action. The all-rows keys keep that meaning. The
    ``_executed`` keys apply the mask first.

    The key ``action_abs_mean`` holds mean|a|. It does not hold an L2 norm. The
    name pairs it with ``action_abs_max``, and that key is a max.

    The key ``executed_action_steps`` is a count, not a magnitude. It counts the
    executed action steps in this rollout, and it is what advances
    ``global_step``. The key was named ``step_delta``, which reads as the change
    in a step count. Nothing is subtracted here, and nothing was.
    """
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
    clip_counts = batch.get("chunk_clip_counts")
    if clip_counts is not None:
        scalars.append(clip_counts.float().mean())
        keys.append("chunk_clip_counts")
    return dict(zip(keys, torch.stack(scalars).tolist()))


@torch.no_grad()
def compute_action_clip_rate(batch: dict[str, torch.Tensor]) -> dict[str, float]:
    """Return how much of the proposal the action bounds had to truncate.

    This function replaces the ``action_abs_max > 1.05`` guard. That guard could
    never fire. ``scale_and_clip_chunk`` clamps into [-1, 1] before the value
    reaches the buffer. So the observable magnitude is capped at 1.0 whatever
    the policy proposed. The clip count is the only place saturation shows.

    The key ``action_clip_frac`` covers all proposed coordinates. So it compares
    across envs and across action dims.
    """
    clip_counts = batch.get("chunk_clip_counts")
    if clip_counts is None:
        return {}
    total = float(clip_counts.sum())
    rows = max(1, int(batch["action_is_pad"].numel()))
    dims = int(batch["action"].shape[-1])
    return {
        "action_clip_count": total,
        "action_clip_frac": total / (rows * dims),
    }


def measure_pre_clamp_projection(policy: SJFPOPolicy) -> dict[str, float]:
    """The share of coordinates the policy's own clamp had to move.

    The key ``action_clip_frac`` is structurally zero. ``scale_and_clip_chunk``
    runs in the collector, and the value it receives has already been clamped to
    [-1, 1] by ``predict_action_chunk`` and then mapped onto the env bounds by
    the postprocessor. So nothing can reach it out of range.

    This key reads the clamp that actually fires. It covers the last proposal of
    the rollout, because ``predict_action_chunk`` overwrites the value on each
    call. So it is a sample, and not a rollout average.
    """
    frac = getattr(policy, "action_pre_clamp_frac", None)
    if frac is None:
        return {}
    return {"action_pre_clamp_frac": float(frac.detach().cpu())}


@torch.no_grad()
def measure_value_drift(policy, batch, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Both critic-drift metrics off one critic forward.

    ``boot_value_relative_drift``: the live critic against the bootstrap value in
    use, as ``rms(move) / rms(bootstrap)``. It is a ratio. It was logged as
    ``boot_value_delta``, and "delta" reads as ``V_new - V_old``, which is a
    difference and not a ratio. Small means staleness is not what limits the
    critic.
    ``value_target_drift``: the same move against the return RMS, so it reads in
    loss units. Large against V can still be negligible against the return.
    """
    # The buffer stores boot_observations and boot_actions in the storage
    # dtype. The critic weights are fp32. This code runs outside autocast. So
    # the cast goes up to fp32.
    boot_obs = batch["boot_observations"].to(device=device, dtype=torch.float32)
    boot_actions = batch["boot_actions"].to(device=device, dtype=torch.float32)
    start = batch["boot_values_start"].to(device).reshape(-1)
    # Must be the same fraction generate_rollout fed the bootstrap, or this
    # compares Q(s',a',f1) against Q(s',a',f2) and reports the gap between two
    # different functions as staleness.
    boot_exec_frac = policy.exec_length_frac(
        torch.full(
            (boot_obs.shape[0],),
            policy.config.n_action_steps,
            device=boot_obs.device,
            dtype=torch.long,
        )
    )
    delta = (
        policy.critic(boot_obs, boot_actions, boot_exec_frac).float().amin(dim=0)
        - start
    )
    drift_rms = rms(delta)
    return (
        drift_rms / (rms(start) + 1e-8),
        drift_rms / (rms(batch["raw_returns"].to(device)) + 1e-8),
    )


def _accumulation_group_sizes(
    n_microbatches: int, accumulation_steps: int
) -> list[int]:
    """Return the microbatch count for each gradient-accumulation group.

    Every group but the last is ``accumulation_steps`` wide. The last group is
    short when the total does not divide. The loop flushes that group on its
    own. It does not drop it.

    Each microbatch is scaled by the size of its own group. So a short last
    group still contributes one full gradient instead of a fraction of one.
    """
    if n_microbatches < 1 or accumulation_steps < 1:
        raise ValueError(
            f"n_microbatches={n_microbatches} and accumulation_steps="
            f"{accumulation_steps} must both be >= 1"
        )
    full, remainder = divmod(n_microbatches, accumulation_steps)
    return [accumulation_steps] * full + ([remainder] if remainder else [])


def sample_random_baseline(
    evaluator: Evaluator, env: Any, max_episodes: int = 30, seed: int | None = None
) -> float:
    """Random-policy return, the divergence reference.

    ``0.0`` when no episode evaluated. Callers gate on ``baseline > 0``, so that
    disables divergence detection instead of falsely flagging every later run.

    The argument ``seed`` seeds the action space before the draw. Without it the
    value changes between calls in one process, so two runs of one config set
    different divergence floors. A floor that moves cannot support a comparison
    between runs.

    The random actions come from ``env.action_space``, while the episodes run on
    the evaluator's own copy of the environment. The two are different objects.
    Seeding ``env`` therefore covers the actions, and the episode seeds still
    come from the evaluator.
    """
    if seed is not None:
        env.action_space.seed(int(seed))
    baseline_evaluator = Evaluator(
        env,
        eval_seed=evaluator.eval_seed,
        max_episodes=max_episodes,
        max_steps=evaluator.max_steps,
    )
    results = baseline_evaluator.evaluate(
        lambda obs: env.action_space.sample(), out_dir=None
    )
    if not results:
        return 0.0
    return float(np.mean([r["return"] for r in results]))

def make_vec_env(env_name: str, num_envs: int, record: dict | None = None):
    """Build a vectorised env with ``forkserver`` workers.

    The value ``context="forkserver"`` is not the platform default. Linux
    defaults to ``fork``. By the time this function runs, ``resolve_device()``
    has already called ``cuda_available()``.

    Forking a process that holds a CUDA runtime is unsupported. It can deadlock
    or can corrupt the context in the child. ``forkserver`` forks from a clean
    single-threaded server instead. It costs less than ``spawn``, because the
    workers do not import the world again.

    The function falls back to ``SyncVectorEnv`` with a warning. It does not
    fall back silently. The sync path is one to two orders of magnitude
    slower. The ``fps`` field is the only other sign of it.
    """
    import gymnasium as gym

    def make_one():
        """Build one worker env, registering third-party namespaces locally.

        The import of ``gymnasium_robotics`` registers the ``AdroitHand*`` ids
        as a side effect. The script ``scripts/train.py`` runs that import in
        the *parent* process only.

        Under ``forkserver``, the workers fork from a clean interpreter that
        never ran the import. So the lambda failed at
        ``gym.make("AdroitHandDoor-v1")``. It raised
        ``NameNotFound: Environment 'AdroitHandDoor' doesn't exist``. The worker
        died. Then the parent surfaced a misleading
        ``ConnectionResetError: [Errno 104]``.

        The import inside the closure fixes the cause. Each worker performs the
        registration itself. So the parent and the children agree on the
        registry, whatever the start method is. The change is harmless for the
        MuJoCo ids, because gymnasium registers those at core import.
        """
        if env_name.startswith("AdroitHand"):
            try:
                import gymnasium_robotics  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    f"{env_name} needs gymnasium-robotics: {exc}"
                ) from exc
        return gym.make(env_name)

    env_fns = [make_one for _ in range(num_envs)]
    # The autoreset_mode must survive vectorisation in either branch. The
    # collector validates NEXT_STEP. See validate_autoreset_mode.
    try:
        env = gym.vector.AsyncVectorEnv(env_fns, context="forkserver")
        if record is not None:
            record["vector_env"] = "forkserver"
        return env
    except (ValueError, RuntimeError, ImportError) as exc:
        # Fallback only on env/vectorisation errors, not user interruption.
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
    runtime: dict | None = None,
) -> dict[str, Any]:
    """One SJFPO training run. Env builds before any CUDA init, so forked
    workers do not inherit the parent's CUDA context (cleanrl / SB3 ordering).

    Accumulation contract: backward every microbatch, step only at the group
    boundary, always flush a final partial group.

    One clip, both roles. ``clip_grad_norm_`` gets a flat list of every
    parameter -- actor and critic together -- so a big critic norm eats the
    actor's budget. Not per-optimizer, not per-group.
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

    train_env = env
    if cfg.num_envs > 1 and env_name != "unknown":
        train_env = make_vec_env(env_name, cfg.num_envs, record=runtime)
    else:
        # Recorded even when num_envs == 1, so the artifact says which path ran.
        if runtime is not None:
            runtime["vector_env"] = "single"
        if cfg.num_envs > 1:
            # num_envs is a throughput knob, not a correctness one, so this falls
            # back rather than failing. Silently, it used to be a run that claimed
            # 64 envs and used one.
            warnings.warn(
                f"num_envs={cfg.num_envs} requested but env_name={env_name!r} "
                "(no spec.id on the passed env), so vectorisation was skipped "
                "and the run uses a single env. Expect a large throughput drop; "
                "the run is otherwise correct.",
                RuntimeWarning,
                stacklevel=2,
            )
    if runtime is not None:
        # The tracker takes floats only, so a string constant cannot be a
        # per-step scalar. It goes here instead. A run states which flow start
        # produced each reported number.
        runtime["eval_sampling"] = cfg.eval_sampling
        runtime["stop_eval_sampling"] = cfg.stop_eval_sampling
        # Audits every auto-applied decision, so a run is self-describing after
        # the fact. Written before the first rollout, not at the end.
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

    def _save(dir_name: str) -> str:
        """Save one checkpoint under `out_dir/checkpoints/` -> artifact path.

        The three call sites (best / step_N / final) each need the same code.
        Only the directory name differs.
        """
        ckpt_dir = out_dir / CHECKPOINTS_DIR / dir_name
        save_policy_checkpoint(
            ckpt_dir,
            policy=policy,
            optimizer=optimizer,
            global_step=global_step,
            ema_state=policy.ema_state_dict(),
        )
        return str(ckpt_dir / "model.safetensors")

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
        f"SJFPO policy: actor_depth={cfg.dit_depth} hidden={cfg.hidden_dim} "
        f"params={sum(p.numel() for p in policy.parameters()):,}",
        file=sys.stderr,
    )
    eval_seed = 10000 + seed
    evaluator = Evaluator(
        env, eval_seed=eval_seed, max_episodes=cfg.eval_max_episodes
    )
    # One adapter per named flow start. Both run over the same eval seeds, so the
    # episode pairs line up and the two scores differ only by the start point.
    eval_policies = {
        "zero_start": _EvalPolicy(policy, start_from_zero=True),
        "noise_start": _EvalPolicy(policy, start_from_zero=False),
    }
    reported_name = cfg.eval_sampling
    stop_name = cfg.stop_eval_sampling
    r_rand = sample_random_baseline(evaluator, env, seed=seed)
    tracker = build_tracker(
        cfg,
        out_dir=out_dir,
        method="sjfpo",
        env_name=env_name,
        seed=seed,
        total_steps=total_steps,
    )

    next_eval = eval_steps
    next_save = save_steps

    eval_count = 0
    above_clamp_updates = 0
    both_clamped_updates = 0
    best_return: float | None = None
    steps_to_best: int | None = None
    recent_eval_returns: deque[float] = deque(maxlen=10)
    all_eval_returns: list[float] = []
    artifacts: list[str] = []
    best_artifact: str | None = None
    elapsed = 0.0
    fps = 0.0
    t_start = time.time()

    clip_params = [p for group in optimizer.param_groups for p in group["params"]]

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
            rollout = compute_rollout_diagnostics(batch)
            steps_this_rollout = int(rollout["executed_action_steps"])
            if steps_this_rollout <= 0:
                raise RuntimeError(
                    "rollout returned zero executed steps; aborting to avoid "
                    "an infinite training loop."
                )
            global_step += steps_this_rollout
            action_abs_max = rollout["action_abs_max"]

            # generate_rollout always emits both keys; the jerk reads them raw.
            jerk_action = batch["action"].detach().cpu().numpy()
            jerk_pad = batch["action_is_pad"].detach().cpu().numpy()
            rollout_jerk = compute_mean_chunk_jerk(jerk_action, jerk_pad)
            rollout_seam_jerk = compute_seam_jerk(jerk_action, jerk_pad)
            action_clip_rates = compute_action_clip_rate(batch)
            action_clip_rates.update(measure_pre_clamp_projection(policy))
            if action_abs_max > 1.0 + 1e-3:
                # Only reachable if the scaling affine itself is wrong.
                raise ValueError(
                    f"Post-clip action magnitude exceeded 1: max_abs="
                    f"{action_abs_max:.3f}. The stored chunk should already be "
                    "scaled to [-1, 1]; check scale_and_clip_chunk and the "
                    "action bounds."
                )

            policy.train()

            epoch_infos: list[dict[str, float]] = []
            grad_norms_pre: list[torch.Tensor] = []
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
            group_sizes = _accumulation_group_sizes(
                len(microbatches), accumulation_steps
            )
            optimizer.zero_grad(set_to_none=True)
            minibatch: dict[str, torch.Tensor] = {}
            finite_error = torch.zeros(
                (), device=batch["raw_advantages"].device, dtype=torch.bool
            )
            for micro_step, mb_idx in enumerate(microbatches):
                # Every group but the last is exactly accumulation_steps wide, so
                # the group index is the microbatch index divided down. The last
                # group may be short, and it still indexes within the list.
                group_size = group_sizes[micro_step // accumulation_steps]
                minibatch = sample_minibatch(batch, mb_idx, minibatch)
                loss, epoch_info = policy(
                    minibatch,
                    compute_info=tracker.enabled,
                    check_finite=False,
                )
                finite_error |= ~torch.isfinite(loss)
                if loss.dim() != 0:
                    raise ValueError(
                        f"Loss must be scalar before backward; got {tuple(loss.shape)}."
                    )
                (loss / group_size).backward()
                is_boundary = (
                    (micro_step + 1) % accumulation_steps == 0
                    or micro_step + 1 == len(microbatches)
                )
                if is_boundary:
                    # One clip over actor and critic together, as FPO++ does.
                    # A flat parameter list, not optimizer.param_groups:
                    # clip_grad_norm_ iterates its argument as tensors and
                    # raises AttributeError on a param-group dict.
                    pre_clip_total = torch.nn.utils.clip_grad_norm_(
                        clip_params, cfg.max_grad_norm
                    )
                    grad_norms_pre.append(pre_clip_total.detach())
                    above_clamp_fracs.append(policy.cfm_above_clamp_frac)
                    both_clamped_fracs.append(policy.cfm_both_clamped_frac)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    policy.step_count += 1
                    apply_lr_schedule(optimizer, policy.step_count)
                if epoch_info is not None:
                    epoch_infos.append(epoch_info)

            # The code checks once per update and not once per boundary. A call
            # to bool() on a device tensor causes a sync. NaN gradients make the
            # clip a no-op anyway. The run is already lost by the time this
            # raises.
            if bool(finite_error):
                raise SJFPOAborted(
                    "Non-finite loss detected during gradient accumulation"
                )

            # Once per rollout, not per boundary: a full critic pass over
            # 2 * n_chunks windows, and the number only grows with steps taken.
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
                torch.stack(grad_norms_pre).mean(),
                torch.stack(above_clamp_fracs).mean(),
                torch.stack(both_clamped_fracs).mean(),
            ]
            if drift is not None:
                scalars.extend(drift)
            (
                grad_norm_mean_pre,
                above_clamp_frac,
                both_clamped_frac,
                *drift_means,
            ) = torch.stack(scalars).tolist()

            if tracker.enabled and info is not None:
                tracker.log(
                    build_train_metrics(
                        cfg=cfg,
                        info=info,
                        batch=batch,
                        rollout=rollout,
                        action_clip_rates=action_clip_rates,
                        grad_norm_mean_pre=grad_norm_mean_pre,
                        above_clamp_frac=above_clamp_frac,
                        both_clamped_frac=both_clamped_frac,
                        drift_means=drift_means,
                        rollout_jerk=rollout_jerk,
                        rollout_seam_jerk=rollout_seam_jerk,
                    ),
                    step=global_step,
                )

            # Named here because the abort message needs it. The tracker can
            # be off, in which case info is None.
            l_new_mean = info["l_new_mean"] if info is not None else float("nan")
            if both_clamped_frac > BOTH_CLAMPED_ABORT_FRAC:
                # The stricter test. Both losses clamp to the same bound, so the
                # ratio is exactly 1.0. The policy receives no gradient at all,
                # while the critic keeps training.
                both_clamped_updates += 1
                if both_clamped_updates >= ABOVE_CLAMP_ABORT_UPDATES:
                    raise SJFPOAborted(
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
                    raise SJFPOAborted(
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
                # The EMA actor once it is warm: at decay 0.95 the shadow averages
                # the last ~20 optimizer steps, which smooths a curve this noisy.
                # The live weights go back right after.
                displaced = policy.swap_in_ema(global_step)
                try:
                    # Both starts run over the same seeds. The one the config
                    # names writes the history. The other is a diagnostic, so it
                    # writes nothing. Each start runs once.
                    results_by_sampling = {
                        name: evaluator.evaluate(
                            adapter,
                            out_dir=out_dir if name == reported_name else None,
                            append_csv=(name == reported_name),
                        )
                        for name, adapter in eval_policies.items()
                    }
                finally:
                    policy.swap_out_ema(displaced)
                window = summarise_window(
                    results_by_sampling,
                    reported_name=reported_name,
                    stop_name=stop_name,
                    eval_seed=eval_seed,
                    summarise=evaluator.summarize,
                    best_return=best_return,
                )
                summaries = window.summaries
                eval_summary = summaries[reported_name]
                eval_results = window.reported_results
                mean_return = window.mean_return
                stop_return = window.stop_return
                all_eval_returns.append(mean_return)
                recent_eval_returns.append(mean_return)
                eval_count += 1
                if window.improved:
                    best_return = stop_return
                    steps_to_best = global_step
                    # The peak has to reach the disk when it happens. With a
                    # save_steps cadence, the nearest periodic checkpoint can
                    # sit 100k steps away. The run always ends on the last
                    # checkpoint. So the peak needs its own file.
                    best_artifact = _save("best")
                if tracker.enabled:
                    tracker.log(
                        compute_eval_summary_metrics(
                            mean_return,
                            eval_results,
                            all_eval_returns,
                            eval_count,
                            r_rand=r_rand,
                            best_return=best_return,
                            steps_to_best=steps_to_best,
                        ),
                        step=global_step,
                    )
                    tracker.log(
                        {
                            "eval/ema_active": float(bool(displaced)),
                            # Logged every eval even though they almost always
                            # read 0. A flag that appears only when true leaves a
                            # gap, and a gap is indistinguishable from the
                            # tracker dropping the key.
                            # The interval around the score the stopper reads.
                            # A width near zero means the episodes agreed. A width
                            # that spans the threshold means the run continues.
                            "eval/return_ci_low": eval_summary["return_ci_low"],
                            "eval/return_ci_high": eval_summary["return_ci_high"],
                            "eval/return_ci_width": (
                                eval_summary["return_ci_high"]
                                - eval_summary["return_ci_low"]
                            ),
                            "eval/return_median": eval_summary["return_median"],
                            "eval/return_n": eval_summary["return_n"],
                            "train/policy_lr": optimizer.param_groups[0]["lr"],
                            "train/critic_lr": optimizer.param_groups[-1]["lr"],
                            # Both starts are logged under their own names, so a
                            # run states which distribution each number came from.
                            **{
                                f"eval/{name}_return_mean": summaries[name][
                                    "return_mean"
                                ]
                                for name in FLOW_STARTS
                            },
                        },
                        step=global_step,
                    )
                    for b in tracker.backends:
                        getattr(b, "log_eval_table", lambda *_: None)(eval_results, global_step)
                next_eval = max(global_step + eval_steps, next_eval + eval_steps)

            if global_step >= next_save or global_step >= total_steps:
                artifacts.append(_save(f"step_{global_step}"))
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

    artifacts.append(_save("final"))
    if best_artifact is not None and best_artifact not in artifacts:
        artifacts.append(best_artifact)

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
        # The median of the same window. It resists one collapsed episode, which
        # the mean does not. The field is additive. The mean stays for
        # continuity with the runs already recorded.
        "reward_median_last10": (
            float(np.median(recent_eval_returns)) if recent_eval_returns else 0.0
        ),
        "fps": fps,
        "artifacts": artifacts,
        "best_artifact": best_artifact,
        "best_return": best_return,
        "global_step": global_step,
    }
