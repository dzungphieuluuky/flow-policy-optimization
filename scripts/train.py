#!/usr/bin/env python3
"""Single-run training CLI for flow_policy_optimization (PyTorch).

One run per invocation. Draccus builds the run config. The config takes
singular values only.

    python scripts/train.py --env Hopper-v5 --seed 0 --total_steps 2000
    python scripts/train.py --method flow_policy_optimization --env Hopper-v5 --flow_policy_optimization.policy_lr 3e-4
    python scripts/train.py --method ppo --env Hopper-v5 --seed 0  # plain PPO baseline
    python scripts/train.py --method flow_policy_optimization --env Hopper-v5 --seed 0 \
        --resume outputs/flow_policy_optimization_Hopper-v5_seed0/checkpoints/checkpoint_1000 \
        --total_steps 4000

``--resume`` is flow_policy_optimization-only. It restores the model, the optimizer and the
global step from a ``checkpoints/<name>/checkpoint.pt`` directory; the config
is NOT restored (the lean pole has no per-run config drift to reconcile), so
CLI overrides apply to the fresh run as written.

--method is 'flow_policy_optimization' (lean reimplementation) or 'ppo' (plain baseline).
Exit codes: 0 success, 1 usage error, 2 runtime/divergence abort or draccus parse error, 4 missing artifact.
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import draccus

from flow_policy_optimization.flow_policy_optimization import (
    FlowPolicyOptimizationConfig,
    is_maniskill_id,
    make_maniskill_env,
    run_flow_policy_optimization_training,
)
from flow_policy_optimization.ppo import run_ppo_training


def configure_torch_cuda() -> bool:
    """Apply torch CUDA tuning when torch is importable and CUDA is present. All math is strict fp32."""
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "max_split_size_mb:128,expandable_segments:True",
    )
    return True


def apply_runtime_settings() -> dict:
    """Apply settings (no-op on unsupported hardware)."""
    try:
        import torch
    except ImportError:
        return {"tf32_enabled": False}

    torch.set_float32_matmul_precision("highest")
    return {
        "tf32_enabled": False,
        "tf32_matmul_precision": torch.get_float32_matmul_precision(),
    }


@dataclass
class TrainConfig:
    """Run one training job (single values only)."""

    method: Literal["flow_policy_optimization", "ppo"] = "flow_policy_optimization"
    env: str | None = None
    seed: int = 3407
    total_steps: int | None = None
    eval_steps: int | None = None
    save_steps: int | None = None
    obs_dim: int | None = None
    action_dim: int | None = None
    out_dir: str = "outputs/"

    # Resume a lean-flow_policy_optimization checkpoint dir (checkpoints/<name>/). Restores the
    # model, the optimizer and the global step into a FRESH run dir
    # (<base>_resume[/2...]); the config is NOT restored.
    resume: str | None = None
    video: bool = False
    flow_policy_optimization: FlowPolicyOptimizationConfig = field(default_factory=FlowPolicyOptimizationConfig)


def default_intervals(total: int, args_val: int | None, cap: int) -> int:
    """Resolve an interval: the CLI value, else a capped default."""
    if args_val:
        return args_val
    return min(cap, total)


def resolve_run_dir(
    out_base: str | Path, method: str, env: str, seed: int, resume: str | None
) -> Path:
    """Return the run directory for one invocation. A resumed run gets a fresh
    directory with a suffix, so a resume never overwrites the source run."""
    base = Path(out_base) / f"{method}_{env}_seed{seed}"
    if not resume:
        return base
    candidate = Path(str(base) + "_resume")
    i = 2
    while candidate.exists():
        candidate = Path(f"{base}_resume{i}")
        i += 1
    return candidate


def run_single(
    env_id: str,
    seed: int,
    out_dir: Path,
    cfg,
    total_steps: int,
    eval_steps: int,
    save_steps: int,
    obs_dim: int | None,
    action_dim: int | None,
    resume_from_checkpoint: str | None = None,
    runtime: dict | None = None,
    video_dir: Path | None = None,
    method: str = "flow_policy_optimization",
) -> dict:
    """Dispatch one run to the matching backend's training entry."""
    import gymnasium as gym

    if env_id.startswith("AdroitHand"):
        try:
            import gymnasium_robotics  # noqa: F401 - registers the Adroit env ids
        except ImportError as exc:
            raise RuntimeError(
                f"{env_id} needs gymnasium-robotics (pip install gymnasium-robotics)"
            ) from exc

    if is_maniskill_id(env_id):
        # ManiSkill's GPU env batches its own sims on one device, so it is built
        # here rather than through gym.make/make_vec_env. The spaces it exposes
        # are the single-env ones, which is what the checks below size from.
        env = make_maniskill_env(env_id, num_envs=cfg.num_envs)
    elif env_id.startswith("Meta-World/"):
        # Meta-World tasks are not individually registered; they are built via
        # the MT1 benchmark wrapper with the task name as env_name.
        import metaworld  # noqa: F401 - registers the Meta-World namespace

        task = env_id[len("Meta-World/"):]
        env = gym.make("Meta-World/MT1", env_name=task)
    else:
        env = gym.make(env_id)
    try:
        live_obs = int(math.prod(env.observation_space.shape))
        live_act = int(math.prod(env.action_space.shape))
        obs_dim = live_obs if obs_dim is None else obs_dim
        action_dim = live_act if action_dim is None else action_dim
        if (obs_dim, action_dim) != (live_obs, live_act):
            raise ValueError(
                f"requested dims {(obs_dim, action_dim)} != live {env_id} dims "
                f"{(live_obs, live_act)}"
            )

        if method == "ppo":
            return run_ppo_training(
                env=env,
                env_id=env_id,
                out_dir=out_dir,
                cfg=cfg,
                seed=seed,
                total_steps=total_steps,
                eval_steps=eval_steps,
                save_steps=save_steps,
                obs_dim=obs_dim,
                action_dim=action_dim,
                runtime=runtime,
                video_dir=video_dir,
            )

        if video_dir is not None:
            print("flow_policy_optimization: --video is not supported by the lean pole; skipping video", file=sys.stderr)

        return run_flow_policy_optimization_training(
            env=env,
            env_id=env_id,
            out_dir=out_dir,
            cfg=cfg,
            seed=seed,
            total_steps=total_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
            resume_from_checkpoint=resume_from_checkpoint,
            runtime=runtime,
        )
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    configure_torch_cuda()
    runtime = apply_runtime_settings()

    argv = list(sys.argv[1:] if argv is None else argv)
    args = draccus.parse(config_class=TrainConfig, args=argv)
    if not args.env:
        print("ERROR: --env is required", file=sys.stderr)
        return 1

    total = args.total_steps or 2000

    eval_s = default_intervals(total, args.eval_steps, 1000)
    ckpt_i = default_intervals(total, args.save_steps, 1000000)

    if args.resume and args.method != "flow_policy_optimization":
        print(
            f"ERROR: --resume is flow_policy_optimization-only, got --method {args.method!r}",
            file=sys.stderr,
        )
        return 1
    if args.resume and not Path(args.resume).is_dir():
        print(f"ERROR: --resume dir not found: {args.resume}", file=sys.stderr)
        return 1
    if args.resume:
        print(
            f"flow_policy_optimization: resume from {args.resume}; the checkpoint restores model, "
            "optimizer and step only — the config is not restored",
            file=sys.stderr,
        )

    run_dir = resolve_run_dir(
        args.out_dir, args.method, args.env, args.seed, args.resume
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = run_single(
            args.env,
            args.seed,
            run_dir,
            args.flow_policy_optimization,
            total,
            eval_s,
            ckpt_i,
            args.obs_dim,
            args.action_dim,
            args.resume,
            runtime,
            video_dir=run_dir / "videos" if args.video else None,
            method=args.method,
        )
    except Exception as exc:  # noqa: BLE001 - a run can fail any way; the traceback is printed
        traceback.print_exc(file=sys.stderr)
        print(
            json.dumps(
                {
                    "method": args.method,
                    "env": args.env,
                    "seed": args.seed,
                    "error": str(exc),
                }
            )
        )
        return 2

    for artifact in result.get("artifacts", []):
        if not Path(artifact).exists():
            print(f"MISSING ARTIFACT: {artifact}", file=sys.stderr)
            return 4
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
