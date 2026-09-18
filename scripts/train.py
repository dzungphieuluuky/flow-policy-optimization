#!/usr/bin/env python3
"""Single-run training CLI for SJFPO/PPO/SAC (MuJoCo-only, CUDA-default, fp32).

One run per invocation; draccus builds the run config (singular values only).
Sweeps live in scripts/sweep.sh: one call here per method x env x seed.

    python scripts/train.py --method sjfpo --env Hopper-v5 --seed 0 --total_steps 2000
    python scripts/train.py --method sjfpo --env Hopper-v5 --sjfpo.optimizer_lr 3e-4
    python scripts/train.py --method sjfpo --env Hopper-v5 --seed 0 \
        --resume .omo/evidence/train/sjfpo_Hopper-v5_seed0/checkpoint_1000 \
        --total_steps 4000
    METHODS="sjfpo ppo" ENVS=Hopper-v5 SEEDS="0 1" scripts/sweep.sh

Flags use underscores (draccus convention, same as lerobot-train).

Precedence: CLI flags > --config YAML (flat, legacy) > dataclass defaults.
The flat YAML only fills method fields the CLI left at defaults. Tuple/dict
fields (hidden, ...) are YAML-only (draccus takes one token each).

Exit codes: 0 success, 1 usage error, 2 runtime/divergence abort,
4 required artifact missing (draccus parse errors exit 2).
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import draccus
import torch  # noqa: E402
torch.backends.fp32_precision = "ieee"
torch.backends.cuda.matmul.fp32_precision = "ieee"
torch.backends.cudnn.fp32_precision = "ieee"
torch.backends.cudnn.conv.fp32_precision = "tf32"
torch.backends.cudnn.rnn.fp32_precision = "tf32"
# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.config import (  # noqa: E402
    KNOWN_ENVS,
    PPOConfig,
    SACConfig,
    lookup_env_dims,
)
from sjfpo.configuration_sjfpo import SJFPOConfig  # noqa: E402

VALID_ENVS = KNOWN_ENVS


@dataclass
class TrainConfig:
    """One training run (singular values only; sweep axes live in sweep.sh)."""

    method: Literal["sjfpo", "ppo", "sac"] = "sjfpo"
    env: str | None = None
    seed: int = 3407
    total_steps: int | None = None
    eval_steps: int | None = None
    save_steps: int | None = None
    obs_dim: int | None = None
    action_dim: int | None = None
    out_dir: str = ".omo/evidence/train"
    config: str | None = None  # legacy flat YAML overlay for the method config
    # Resume an sjfpo checkpoint dir (checkpoint_<step>/ or checkpoint_final/):
    # continues policy + optimizer + counters into a FRESH run dir
    # (<base>_resume[/2...]). total_steps stays an absolute env-step budget.
    resume: str | None = None
    sjfpo: SJFPOConfig = field(default_factory=SJFPOConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sac: SACConfig = field(default_factory=SACConfig)


def load_yaml_dict(path: str | Path) -> dict:
    """Load YAML via safe_load only; friendly error when pyyaml is missing."""
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "ERROR: pyyaml is required for --config (pip install pyyaml)"
        ) from exc
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"ERROR: config {path} must be a YAML mapping")
    return data


def coerce_value(cfg_cls: type, raw: dict) -> dict:
    fields = {f.name for f in dataclasses.fields(cfg_cls)}
    out = {k: v for k, v in raw.items() if k in fields}
    for key in ("hidden",):
        if isinstance(out.get(key), list):
            out[key] = tuple(out[key])
    return out


def apply_flat_yaml(method_cfg, yaml_dict: dict):
    """Legacy flat --config YAML fills method fields the CLI left at defaults."""
    raw = coerce_value(type(method_cfg), yaml_dict)
    if not raw:
        return method_cfg
    fresh = type(method_cfg)()
    cli_touched = {
        f.name
        for f in dataclasses.fields(method_cfg)
        if getattr(method_cfg, f.name) != getattr(fresh, f.name)
    }
    overrides = {k: v for k, v in raw.items() if k not in cli_touched}
    return dataclasses.replace(method_cfg, **overrides) if overrides else method_cfg


def select_method_cfg(args: TrainConfig, yaml_dict: dict):
    """Nested draccus sub-config for the method, overlaid with flat YAML."""
    method_cfg = {"sjfpo": args.sjfpo, "ppo": args.ppo, "sac": args.sac}[args.method]
    return apply_flat_yaml(method_cfg, yaml_dict)


def resolve_run_dir(out_base: str | Path, method: str, env: str, seed: int, resume: str | None) -> Path:
    """Run dir for one invocation; resumed runs get a fresh suffixed dir.

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
    method: str,
    env_id: str,
    seed: int,
    out_dir: Path,
    cfg,
    total_steps: int,
    eval_steps: int,
    save_steps: int,
    obs_dim: int | None,
    action_dim: int | None,
    n_obs_steps: int = 16,
    resume_from_checkpoint: str | None = None,
) -> dict:
    """Dispatch one run to train_sjfpo_run/ppo_train_run/sac_train_run."""
    import gymnasium as gym

    if env_id.startswith("AdroitHand"):
        try:
            import gymnasium_robotics  # noqa: F401 (registers Adroit env ids)
        except ImportError:
            raise RuntimeError(
                f"{env_id} needs gymnasium-robotics (pip install gymnasium-robotics)"
            )
    env = gym.make(env_id)
    try:
        # Registry fills unset dims; the live space is ground truth on mismatch.
        reg_obs, reg_act = lookup_env_dims(env_id)
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
        from sjfpo.baselines import ppo_train_run, sac_train_run
        from sjfpo.train_sjfpo import train_sjfpo_run

        runners = {
            "sjfpo": (train_sjfpo_run, {"obs_dim": obs_dim, "action_dim": action_dim, "resume_from_checkpoint": resume_from_checkpoint}),
            "ppo": (ppo_train_run, {"n_obs_steps": n_obs_steps}),
            "sac": (sac_train_run, {"n_obs_steps": n_obs_steps}),
        }
        if resume_from_checkpoint is not None and method != "sjfpo":
            raise ValueError(f"--resume is sjfpo-only, got method={method!r}")
        run_fn, extra = runners[method]
        return run_fn(
            env=env,
            out_dir=out_dir,
            cfg=cfg,
            seed=seed,
            total_steps=total_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            **extra,
        )
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    # Tensor-core fp32 matmuls (TF32) for leftover fp32 GEMMs; autocast still
    # handles bf16. Precision impact is far below the numeric gates.
    torch.set_float32_matmul_precision("high")
    args = draccus.parse(config_class=TrainConfig, args=argv)
    if not args.env:
        print("ERROR: --env is required", file=sys.stderr)
        return 1
    if args.env not in VALID_ENVS:
        print(
            f"ERROR: unknown env {args.env!r}; choices: {', '.join(VALID_ENVS)}",
            file=sys.stderr,
        )
        return 1
    yaml_dict: dict = load_yaml_dict(args.config) if args.config else {}
    total = args.total_steps or int(yaml_dict.get("total_steps", 2000))
    eval_s = args.eval_steps or int(yaml_dict.get("eval_steps", min(1000, total)))
    ckpt_i = args.save_steps or int(yaml_dict.get("save_steps", min(1000, total)))
    n_obs_steps = int(yaml_dict.get("n_obs_steps", 16))
    cfg = select_method_cfg(args, yaml_dict)
    if args.resume and args.method != "sjfpo":
        print(f"ERROR: --resume is sjfpo-only, got --method {args.method!r}", file=sys.stderr)
        return 1
    if args.resume and not Path(args.resume).is_dir():
        print(f"ERROR: --resume dir not found: {args.resume}", file=sys.stderr)
        return 1
    # Same layout the old matrix used (sweep.format_run_directory_name),
    # suffixed when resuming so the source run is never overwritten.
    run_dir = resolve_run_dir(args.out_dir, args.method, args.env, args.seed, args.resume)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = run_single(
            args.method,
            args.env,
            args.seed,
            run_dir,
            cfg,
            total,
            eval_s,
            ckpt_i,
            args.obs_dim,
            args.action_dim,
            n_obs_steps,
            args.resume,
        )
    except Exception as exc:
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
