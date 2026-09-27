#!/usr/bin/env python3
"""Single-run training CLI for SJFPO (PyTorch).

One run per invocation; draccus builds the run config (singular values only).
One call per run; run each env/seed combination you want.

    python scripts/train.py --env Hopper-v5 --seed 0 --total_steps 2000
    python scripts/train.py --method sjfpo --env Hopper-v5 --sjfpo.policy_lr 3e-4
    python scripts/train.py --method sjfpo --env Hopper-v5 --seed 0 \
        --resume outputs/sjfpo_Hopper-v5_seed0/checkpoint_1000 \
        --total_steps 4000

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
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import draccus

# Make src/ importable before any sjfpo.* import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.config import (  # noqa: E402
    KNOWN_ENVS,
    get_observation_action_dim,
)
from sjfpo.configuration_sjfpo import SJFPOConfig  # noqa: E402


VALID_ENVS = KNOWN_ENVS


def configure_torch_cuda() -> bool:
    """Apply torch CUDA tuning when torch is importable and CUDA is present.

    Returns True when torch is usable. Never raises, so a torch-less
    environment still reaches the import error further down with context.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.backends.cudnn.rnn.fp32_precision = "tf32"
    torch.backends.cudnn.benchmark = True
    # TF32 for leftover fp32 GEMMs; autocast still handles bf16 matmuls.
    torch.set_float32_matmul_precision("high")
    # Memory allocation hint (harmless if already set by user env).
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "max_split_size_mb:128,expandable_segments:True",
    )
    return True


@dataclass
class TrainConfig:
    """Run one training job (single values only)."""

    method: Literal["sjfpo"] = "sjfpo"
    env: str | None = None
    seed: int = 3407
    total_steps: int | None = None
    eval_steps: int | None = None
    save_steps: int | None = None
    obs_dim: int | None = None
    action_dim: int | None = None
    out_dir: str = "outputs/"
    wandb_project: str | None = None
    wandb_enabled: bool = True
    wandb_run_name: str | None = None
    config: str | None = None  # legacy flat YAML overlay for the method config
    # Resume an sjfpo checkpoint dir (checkpoint_<step>/ or
    # checkpoint_final/): continues policy + optimizer + counters into a FRESH
    # run dir (<base>_resume[/2...]). total_steps stays an absolute env-step
    # budget.
    resume: str | None = None
    sjfpo: SJFPOConfig = field(default_factory=SJFPOConfig)


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
    """Keep only fields the target dataclass knows; coerce list->tuple."""
    fields = {f.name for f in dataclasses.fields(cfg_cls)}
    out = {k: v for k, v in raw.items() if k in fields}
    for key in ("hidden",):
        if isinstance(out.get(key), list):
            out[key] = tuple(out[key])
    return out


def apply_flat_yaml(
    method_cfg, yaml_dict: dict, cli_fields: set[str] | None = None
):
    """Apply flat YAML after the dataclass defaults and before CLI overrides."""
    raw = coerce_value(type(method_cfg), yaml_dict)
    if not raw:
        return method_cfg
    fresh = type(method_cfg)()
    cli_touched = {
        f.name
        for f in dataclasses.fields(method_cfg)
        if getattr(method_cfg, f.name) != getattr(fresh, f.name)
    }
    cli_touched.update(cli_fields or ())
    overrides = {k: v for k, v in raw.items() if k not in cli_touched}
    return dataclasses.replace(method_cfg, **overrides) if overrides else method_cfg


def _sjfpo_cli_fields(argv: list[str]) -> set[str]:
    """Return SJFPO config fields explicitly present on the command line."""
    fields: set[str] = set()
    prefix = "--sjfpo."
    for index, token in enumerate(argv):
        if not token.startswith(prefix):
            continue
        name = token[len(prefix) :].split("=", 1)[0]
        if name:
            fields.add(name)
    return fields


def select_method_cfg(
    args: TrainConfig, yaml_dict: dict, cli_fields: set[str] | None = None
):
    """Nested draccus sub-config for the method, overlaid with flat YAML.

    """
    return apply_flat_yaml(args.sjfpo, yaml_dict, cli_fields=cli_fields)


def default_intervals(
    total: int, args_val: int | None, yaml_dict: dict, key: str, cap: int
) -> int:
    """CLI > YAML > capped default (the cap keeps smoke runs sane)."""
    if args_val:
        return args_val
    return int(yaml_dict.get(key, min(cap, total)))


def resumed_cfg_overrides(cfg) -> list[str]:
    """Scalar fields differing from dataclass defaults.

    Used to warn with --resume: the checkpoint config wins there, so CLI
    overrides of these fields are silently ignored. Only plain scalars are
    compared (dict/tensor features have no meaningful CLI equality).
    """
    try:
        fresh = type(cfg)()
    except Exception:
        return []
    touched = []
    for f in dataclasses.fields(cfg):
        try:
            cur, default = getattr(cfg, f.name), getattr(fresh, f.name)
        except Exception:
            continue
        if isinstance(cur, (bool, int, float, str)) or cur is None:
            if isinstance(default, (bool, int, float, str)) or default is None:
                if cur != default:
                    touched.append(f.name)
    return sorted(touched)


def resolve_run_dir(
    out_base: str | Path, method: str, env: str, seed: int, resume: str | None
) -> Path:
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
    resume_from_checkpoint: str | None = None,
    wandb_enabled: bool = True,
) -> dict:
    """Dispatch one run to the matching backend's training entry."""
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

        from sjfpo.train_sjfpo import train_sjfpo_run

        if resume_from_checkpoint is not None and method != "sjfpo":
            raise ValueError(f"--resume is sjfpo-only, got method={method!r}")

        return train_sjfpo_run(
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
            wandb_enabled=wandb_enabled,
        )
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    configure_torch_cuda()

    argv = list(sys.argv[1:] if argv is None else argv)
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

    eval_s = default_intervals(total, args.eval_steps, yaml_dict, "eval_steps", 1000)
    ckpt_i = default_intervals(total, args.save_steps, yaml_dict, "save_steps", 1000)
    cfg = select_method_cfg(args, yaml_dict, _sjfpo_cli_fields(argv))

    if args.resume and args.method != "sjfpo":
        print(
            f"ERROR: --resume is sjfpo-only, got --method {args.method!r}",
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

    # Suffixed when resuming so the source run is never overwritten.
    run_dir = resolve_run_dir(
        args.out_dir, args.method, args.env, args.seed, args.resume
    )
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
            args.resume,
            args.wandb_enabled,
        )
    except Exception as exc:
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