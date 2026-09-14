#!/usr/bin/env python3
"""Unified training CLI for SJFPO/PPO/SAC (MuJoCo-only, CUDA-default, fp32).

SJFPO trains through SJFPOPolicy (src/sjfpo/train_sjfpo.py).

Single-run (default):
    python scripts/train.py --method sjfpo --env Hopper-v5 --seed 0 --total-steps 2000
Matrix sweep (RUN_MATRIX order, scan_runs/run_pending):
    python scripts/train.py --matrix --methods sjfpo ppo --envs Hopper-v5 --seeds 0 1
    python scripts/train.py --matrix --tiny
Precedence (single): CLI flags > --config YAML > dataclass defaults.
Precedence (matrix): CLI scalars > --config YAML > shared budgets > dataclass defaults.

Exit codes: 0 success, 1 usage/argparse error, 2 runtime/divergence abort,
4 required artifact missing.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # graceful degrade when tqdm is not installed
    tqdm = None  # type: ignore[assignment]

import torch  # noqa: E402

    # Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.config import (  # noqa: E402
    CKPT_INTERVAL,
    ENVS,
    EVAL_INTERVAL,
    METHODS,
    SEEDS,
    PPOConfig,
    RunSpec,
    SACConfig,
    TOTAL_STEPS,
    build_filtered_matrix,
)
from sjfpo.configuration_sjfpo import SJFPOConfig  # noqa: E402

VALID_ENVS = ENVS
VALID_METHODS = METHODS
TINY_STEPS = 20_000
TINY_INTERVAL = 10_000


def present_value(argv: list[str] | None, *flags: str) -> bool:
    tokens = argv if argv is not None else sys.argv[1:]
    return any(t in flags for t in tokens)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args (single-run singular vs matrix plural enforced in main)."""
    parser = argparse.ArgumentParser(
        description="Train SJFPO/PPO/SAC on MuJoCo (unified CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        choices=VALID_METHODS,
        default="sjfpo",
        help="Algorithm for single-run mode",
    )
    parser.add_argument(
        "--env",
        choices=VALID_ENVS,
        default=None,
        help="MuJoCo env for single-run mode (required without --matrix)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (single-run)")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=VALID_METHODS,
        default=None,
        help="Methods for --matrix mode (default: all)",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        choices=VALID_ENVS,
        default=None,
        help="Envs for --matrix mode (default: all)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seeds for --matrix mode (default: all)",
    )
    parser.add_argument(
        "--matrix", action="store_true", help="Run sweep via scan_runs/run_pending"
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Tiny 3-run override (only with --matrix, warns over filters)",
    )
    parser.add_argument("--total-steps", type=int, default=None, help="Total env steps")
    parser.add_argument("--eval-interval", type=int, default=None, help="Eval interval")
    parser.add_argument(
        "--ckpt-interval", type=int, default=None, help="Checkpoint interval"
    )
    parser.add_argument("--obs-dim", type=int, default=None, help="Override obs dim")
    parser.add_argument("--act-dim", type=int, default=None, help="Override act dim")
    parser.add_argument(
        "--out-dir", type=str, default=".omo/evidence/train", help="Output directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config override (CLI > YAML > dataclass defaults)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar (enabled by default)",
    )
    return parser.parse_args(argv)


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
    if cfg_cls is SJFPOConfig and isinstance(out.get("tau_range"), list):
        out["tau_range"] = tuple(out["tau_range"])
    for key in ("hidden",):
        if isinstance(out.get(key), list):
            out[key] = tuple(out[key])
    return out


def build_cfg(method: str, yaml_dict: dict):
    """Build SJFPO/PPO/SAC config: YAML(filtered to fields) over dataclass defaults."""
    if method == "sjfpo":
        return SJFPOConfig(**coerce_value(SJFPOConfig, yaml_dict))
    if method == "ppo":
        return PPOConfig(**coerce_value(PPOConfig, yaml_dict))
    return SACConfig(**coerce_value(SACConfig, yaml_dict))


def run_single(
    method: str,
    env_id: str,
    seed: int,
    out_dir: Path,
    cfg,
    total_steps: int,
    eval_interval: int,
    ckpt_interval: int,
    obs_dim: int | None,
    act_dim: int | None,
    n_obs_steps: int = 16,
) -> dict:
    """Dispatch one run to train_sjfpo_run/ppo_train_run/sac_train_run."""
    import gymnasium as gym

    env = gym.make(env_id)
    try:
        if method == "sjfpo":
            from sjfpo.train_sjfpo import train_sjfpo_run

            return train_sjfpo_run(
                env=env,
                out_dir=out_dir,
                cfg=cfg,
                seed=seed,
                total_steps=total_steps,
                eval_interval=eval_interval,
                ckpt_interval=ckpt_interval,
                obs_dim=obs_dim,
                act_dim=act_dim,
            )
        if method == "ppo":
            from sjfpo.baselines import ppo_train_run

            return ppo_train_run(
                env=env,
                out_dir=out_dir,
                cfg=cfg,
                seed=seed,
                total_steps=total_steps,
                eval_interval=eval_interval,
                ckpt_interval=ckpt_interval,
                n_obs_steps=n_obs_steps,
            )
        from sjfpo.baselines import sac_train_run

        return sac_train_run(
            env=env,
            out_dir=out_dir,
            cfg=cfg,
            seed=seed,
            total_steps=total_steps,
            eval_interval=eval_interval,
            ckpt_interval=ckpt_interval,
            n_obs_steps=n_obs_steps,
        )
    finally:
        env.close()


def validate_arg_combinations(
    args: argparse.Namespace, argv: list[str] | None
) -> int | None:
    """Reject bad singular/plural flag mixes; return exit code or None."""
    tiny_no_matrix = args.tiny and not args.matrix
    if tiny_no_matrix:
        print("ERROR: --tiny requires --matrix", file=sys.stderr)
        return 1
    if args.matrix:
        if present_value(argv, "--method", "--env", "--seed"):
            print(
                "ERROR: --matrix takes plural --methods/--envs/--seeds, "
                "not singular --method/--env/--seed",
                file=sys.stderr,
            )
            return 1
    elif present_value(argv, "--methods", "--envs", "--seeds"):
        print(
            "ERROR: singular mode takes --method/--env/--seed; "
            "use --matrix for --methods/--envs/--seeds",
            file=sys.stderr,
        )
        return 1
    if not args.matrix and not args.env:
        print("ERROR: --env is required in single-run mode", file=sys.stderr)
        return 1
    return None


def resolve_matrix_settings(
    args: argparse.Namespace, argv: list[str] | None, yaml_dict: dict
):
    """Resolve matrix axes + budgets (tiny override or CLI/YAML/defaults)."""
    if args.tiny:
        if present_value(argv, "--methods", "--envs", "--seeds"):
            print(
                "WARNING: --tiny overrides --methods/--envs/--seeds filters",
                file=sys.stderr,
            )
        os.environ["SJFPO_TINY_MATRIX"] = "1"
        methods, envs, seeds = list(METHODS), ["Hopper-v5"], [0]
        total, eval_i, ckpt_i = TINY_STEPS, TINY_INTERVAL, TINY_INTERVAL
    else:
        methods = args.methods or list(METHODS)
        envs = args.envs or list(ENVS)
        seeds = args.seeds if args.seeds is not None else list(SEEDS)
        total = args.total_steps or int(yaml_dict.get("total_steps", TOTAL_STEPS))
        eval_i = args.eval_interval or int(
            yaml_dict.get("eval_interval", EVAL_INTERVAL)
        )
        ckpt_i = args.ckpt_interval or int(
            yaml_dict.get("ckpt_interval", CKPT_INTERVAL)
        )
    return methods, envs, seeds, total, eval_i, ckpt_i


def start_matrix_progress(matrix, out_base: Path, no_progress: bool):
    """Start the tqdm matrix bar; degrade to None when unavailable."""
    pbar = None
    if tqdm is not None and not no_progress:
        try:
            from sjfpo.sweep import scan_runs as scan_runs_for_pbar

            scan = scan_runs_for_pbar(matrix, out_base)
            total = len(scan.pending)
            if total > 0:
                pbar = tqdm(total=total, desc="matrix", unit="run", dynamic_ncols=True)
        except Exception:
            pbar = None
    return pbar


def run_matrix_mode(
    args: argparse.Namespace,
    argv: list[str] | None,
    yaml_dict: dict,
    n_obs_steps: int,
    out_base: Path,
) -> int:
    """Execute the --matrix sweep; return the process exit code."""
    methods, envs, seeds, total, eval_i, ckpt_i = resolve_matrix_settings(
        args, argv, yaml_dict
    )
    matrix = build_filtered_matrix(methods, envs, seeds, total, eval_i, ckpt_i)
    pbar = start_matrix_progress(matrix, out_base, args.no_progress)
    from sjfpo.sweep import format_run_directory_name, run_pending

    def run_fn(spec: RunSpec, base: Path) -> dict:
        run_dir = base / format_run_directory_name(spec)
        cfg = build_cfg(spec.method, yaml_dict)
        try:
            result = run_single(
                spec.method,
                spec.env,
                spec.seed,
                run_dir,
                cfg,
                spec.total_steps,
                spec.eval_interval,
                spec.ckpt_interval,
                args.obs_dim,
                args.act_dim,
                n_obs_steps,
            )
        finally:
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(f"{spec.method}/{spec.env} s{spec.seed}")
        return result

    try:
        summary = run_pending(matrix, out_base, run_fn)
    finally:
        if pbar is not None:
            pbar.close()
    out = {
        k: (v if not isinstance(v, list) else len(v))
        for k, v in summary.items()
        if k != "results"
    }
    out["failed"] = summary.get("failed", [])
    print(json.dumps(out))
    if summary.get("failed"):
        return 2  # runtime abort: one or more matrix runs failed
    return 0 if summary.get("completed", 0) else 1


def run_single_mode(
    args: argparse.Namespace, yaml_dict: dict, n_obs_steps: int, out_base: Path
) -> int:
    """Execute one single-run training; return the process exit code."""
    total = args.total_steps or int(yaml_dict.get("total_steps", 2000))
    eval_i = args.eval_interval or int(yaml_dict.get("eval_interval", min(1000, total)))
    ckpt_i = args.ckpt_interval or int(yaml_dict.get("ckpt_interval", min(1000, total)))
    cfg = build_cfg(args.method, yaml_dict)
    try:
        result = run_single(
            args.method,
            args.env,
            args.seed,
            out_base / args.method,
            cfg,
            total,
            eval_i,
            ckpt_i,
            args.obs_dim,
            args.act_dim,
            n_obs_steps,
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


def main(argv: list[str] | None = None) -> int:
    # Tensor-core fp32 matmuls (TF32) for leftover fp32 GEMMs; autocast still
    # handles bf16. Precision impact is far below the numeric gates.
    torch.set_float32_matmul_precision("high")
    args = parse_args(argv)
    rejection = validate_arg_combinations(args, argv)
    if rejection is not None:
        return rejection
    yaml_dict: dict = load_yaml_dict(args.config) if args.config else {}
    n_obs_steps = int(yaml_dict.get("n_obs_steps", 16))
    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    if args.matrix:
        return run_matrix_mode(args, argv, yaml_dict, n_obs_steps, out_base)
    return run_single_mode(args, yaml_dict, n_obs_steps, out_base)


if __name__ == "__main__":
    sys.exit(main())
