#!/usr/bin/env python3
"""Single-run training CLI for myfpo (PyTorch).

One run per invocation. Draccus builds the run config. The config takes
singular values only.

One call runs one run. So call the script once per env and seed combination that
you want.

    python scripts/train.py --env Hopper-v5 --seed 0 --total_steps 2000
    python scripts/train.py --method myfpo --env Hopper-v5 --myfpo.policy_lr 3e-4
    python scripts/train.py --method myfpo --env Hopper-v5 --seed 0 \
        --resume outputs/myfpo_Hopper-v5_seed0/checkpoint_1000 \
        --total_steps 4000

Flags use underscores (draccus convention, same as lerobot-train).

The config is the ``MyfpoConfig`` dataclass. Every field is a CLI flag under
``--myfpo.``, so the command line is the only override.


The exit codes are these. 0 means success. 1 means a usage error. 2 means a
runtime or divergence abort. 4 means a required artifact is missing. A parse
error from draccus exits with 2.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import draccus

from myfpo.configuration import (
    KNOWN_ENVS,
    get_observation_action_dim,
)
from myfpo.configuration import MyfpoConfig


def configure_torch_cuda() -> bool:
    """Apply torch CUDA tuning when torch is importable and CUDA is present.

    Returns True when torch is usable. Never raises, so a torch-less
    environment still reaches the import error further down with context.
    """
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.backends.cudnn.rnn.fp32_precision = "tf32"
    torch.backends.cudnn.benchmark = True
    # Memory allocation hint. It is harmless if the user environment already set
    # it.
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "max_split_size_mb:128,expandable_segments:True",
    )
    return True


def apply_runtime_settings() -> dict:
    """Apply settings (no-op on unsupported hardware).

    Returns applied settings for metadata.
    """
    import torch

    from myfpo.layers import (
        detect_gpu,
        resolve_compute_dtype,
        resolve_storage_dtype,
    )

    gpu = detect_gpu()
    tf32 = bool(gpu and gpu["supports"]["tf32"])
    # TF32 is an instruction set of sm_80 and above. A request for TF32 on
    # Turing is accepted and ignored. So the "highest" branch exists to make that
    # no-op visible in the metadata. It does not change the behaviour.
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    # After 8 shape variants, dynamo silently falls back to eager. Then you keep
    # paying the compile cost and get none of the benefit. The chunk count of the
    # rollout varies every run, so the default of 8 is reached quickly.

    return {
        "gpu": gpu,
        "tf32_enabled": tf32,
        "tf32_matmul_precision": torch.get_float32_matmul_precision(),
        "dynamo_cache_size_limit": torch._dynamo.config.cache_size_limit,
        "storage_dtype": str(resolve_storage_dtype()),
        "compute_dtype": str(resolve_compute_dtype()),
    }


@dataclass
class TrainConfig:
    """Run one training job (single values only)."""

    method: Literal["myfpo"] = "myfpo"
    env: str | None = None
    seed: int = 3407
    total_steps: int | None = None
    eval_steps: int | None = None
    save_steps: int | None = None
    obs_dim: int | None = None
    action_dim: int | None = None
    out_dir: str = "outputs/"

    # Resume a myfpo checkpoint dir (checkpoint_<step>/ or checkpoint_final/).
    # The run continues the policy, the optimizer and the counters into a FRESH
    # run dir (<base>_resume[/2...]). total_steps stays an absolute env-step
    # budget.
    resume: str | None = None
    myfpo: MyfpoConfig = field(default_factory=MyfpoConfig)


def default_intervals(total: int, args_val: int | None, cap: int) -> int:
    """Resolve an interval: the CLI value, else a capped default.

    The cap keeps smoke runs sane.
    """
    if args_val:
        return args_val
    return min(cap, total)



def resumed_cfg_overrides(cfg) -> list[str]:
    """Scalar fields differing from dataclass defaults.

    Used to warn with --resume: the checkpoint config wins there, so CLI
    overrides of these fields are silently ignored. Only plain scalars are
    compared (dict/tensor features have no meaningful CLI equality).
    """
    try:
        fresh = type(cfg)()
    except TypeError:
        return []
    touched = []
    for f in dataclasses.fields(cfg):
        try:
            cur, default = getattr(cfg, f.name), getattr(fresh, f.name)
        except AttributeError:
            continue
        if (
            (isinstance(cur, (bool, int, float, str)) or cur is None)
            and (isinstance(default, (bool, int, float, str)) or default is None)
            and cur != default
        ):
            touched.append(f.name)
    return sorted(touched)


def resolve_run_dir(
    out_base: str | Path, method: str, env: str, seed: int, resume: str | None
) -> Path:
    """Return the run directory for one invocation. A resumed run gets a fresh
    directory with a suffix.

    ``<method>_<env>_seed<seed>`` normally, else the first free
    ``<base>_resume``, ``<base>_resume2``, ... so a resume never overwrites
    the original run (or an earlier resume).
    """
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

    env = gym.make(env_id)
    try:
        # The registry fills the dims that the config leaves unset. The live
        # space is the ground truth on a mismatch.
        reg_obs, reg_act = get_observation_action_dim(env_id)
        live_obs = int(math.prod(env.observation_space.shape))
        live_act = int(math.prod(env.action_space.shape))
        if (reg_obs, reg_act) != (live_obs, live_act):
            raise ValueError(
                f"registry dims {(reg_obs, reg_act)} != live {env_id} dims "
                f"{(live_obs, live_act)}; update configs/envs.yaml"
            )
        obs_dim = reg_obs if obs_dim is None else obs_dim
        action_dim = reg_act if action_dim is None else action_dim
        if (obs_dim, action_dim) != (live_obs, live_act):
            raise ValueError(
                f"requested dims {(obs_dim, action_dim)} != live {env_id} dims "
                f"{(live_obs, live_act)}"
            )

        from myfpo.training import run_training

        return run_training(
            env=env,
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
    if args.env not in KNOWN_ENVS:
        print(
            f"ERROR: unknown env {args.env!r}; choices: {', '.join(KNOWN_ENVS)}",
            file=sys.stderr,
        )
        return 1

    total = args.total_steps or 2000

    eval_s = default_intervals(total, args.eval_steps, 1000)
    ckpt_i = default_intervals(total, args.save_steps, 1000)
    cfg = args.myfpo

    if args.resume and args.method != "myfpo":
        print(
            f"ERROR: --resume is myfpo-only, got --method {args.method!r}",
            file=sys.stderr,
        )
        return 1
    if args.resume and not Path(args.resume).is_dir():
        print(f"ERROR: --resume dir not found: {args.resume}", file=sys.stderr)
        return 1
    if args.resume:
        overridden = resumed_cfg_overrides(cfg)
        if overridden:
            print(
                "WARNING: --resume restores the checkpoint config; "
                f"CLI overrides ignored for: {', '.join(overridden)}",
                file=sys.stderr,
            )

    # The run dir gets a suffix when resuming, so the source run is never
    # overwritten.
    run_dir = resolve_run_dir(
        args.out_dir, args.method, args.env, args.seed, args.resume
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = run_single(
            args.env,
            args.seed,
            run_dir,
            cfg,
            total,
            eval_s,
            ckpt_i,
            args.obs_dim,
            args.action_dim,
            args.resume,
            runtime,
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