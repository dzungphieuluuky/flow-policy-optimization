#!/usr/bin/env python3
"""Multi-seed driver and comparison for SJFPO (PyTorch).

``train.py`` runs one seed per invocation. One seed cannot separate a method
from the noise. This script runs a list of seeds, then compares the arms with
a bootstrap interval over the seeds.

Two modes. ``run`` trains the seeds and writes one summary CSV. ``compare``
reads several summary CSVs and writes the comparison table.

    python scripts/multiseed.py run --env Hopper-v5 --seeds 3407,1,2,3,4 \
        --total_steps 20000 --out outputs/sweep_mean
    python scripts/multiseed.py run --env Hopper-v5 --seeds 3407,1,2,3,4 \
        --sjfpo.cfm_loss_reduction mean --out outputs/sweep_mean
    python scripts/multiseed.py compare \
        outputs/sweep_mean/sweep_summary.csv outputs/sweep_sum/sweep_summary.csv

The interval in ``compare`` covers the spread across training seeds. It is the
interval that decides whether two arms differ. A difference whose interval
straddles zero is not evidence of a difference.

Exit codes match ``train.py``: 0 ok, 1 usage, 2 runtime abort.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import draccus  # noqa: E402

from myfpo.config import KNOWN_ENVS  # noqa: E402
from myfpo.configuration_sjfpo import SJFPOConfig  # noqa: E402
from myfpo.evaluator import bootstrap_interval, median_of  # noqa: E402

from train import (  # noqa: E402
    _sjfpo_cli_fields,
    apply_flat_yaml,
    apply_runtime_settings,
    configure_torch_cuda,
    default_intervals,
    load_yaml_dict,
    resolve_run_dir,
    run_single,
)

SUMMARY_FIELDS = [
    "label",
    "env",
    "seed",
    "reward_mean_last10",
    "reward_median_last10",
    "best_return",
    "fps",
    "global_step",
    "diverged",
    "regressed",
    "lr_decays",
    "run_dir",
]

# The interval resamples over training seeds. The seed count is small, so the
# percentile method is kept. It makes no assumption about the shape.
BOOTSTRAP_RESAMPLES = 10_000


@dataclass
class RunArgs:
    """One multi-seed sweep request. The config takes singular values only."""

    env: str | None = None
    seeds: str | None = None
    seed_count: int = 5
    seed_start: int = 1
    label: str | None = None
    total_steps: int | None = None
    eval_steps: int | None = None
    save_steps: int | None = None
    out: str = "outputs/multiseed"
    sjfpo: SJFPOConfig = dataclasses.field(default_factory=SJFPOConfig)


def parse_seeds(raw: str | None, count: int, start: int) -> list[int]:
    """Read a seed list, or build ``count`` seeds starting at ``start``."""
    if raw:
        out: list[int] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                out.append(int(token))
            except ValueError:
                raise SystemExit(f"ERROR: --seeds holds a non-integer: {token!r}")
        if not out:
            raise SystemExit("ERROR: --seeds holds no integer")
        return out
    if count < 1:
        raise SystemExit("ERROR: --seed-count must be >= 1")
    return [start + i for i in range(count)]


def write_summary(path: Path, rows: list[dict]) -> Path:
    """Write one summary CSV. A fresh file replaces the old one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})
    return path


def read_summary(path: str | Path) -> list[dict]:
    """Read a summary CSV written by :func:`write_summary`."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"ERROR: summary CSV not found: {p}")
    with open(p, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"ERROR: summary CSV holds no row: {p}")
    return rows


def _finite(values: list[str]) -> list[float]:
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            out.append(f)
    return out


def aggregate(rows: list[dict], field: str) -> dict[str, float]:
    """The across-seed aggregate for one score field."""
    values = _finite([r.get(field, "") for r in rows])
    if not values:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "median": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    mean, low, high = bootstrap_interval(values, resamples=BOOTSTRAP_RESAMPLES)
    return {
        "n": float(len(values)),
        "mean": mean,
        "median": median_of(values),
        "ci_low": low,
        "ci_high": high,
    }


def bootstrap_difference(
    a: list[float], b: list[float], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0
) -> tuple[float, float, float]:
    """A percentile interval for ``mean(a) - mean(b)``, drawn independently.

    It returns ``(difference, low, high)``. The value ``nan`` stands for a
    part it cannot compute.

    The function resamples each list separately. The two lists come from
    different seed sets, so a paired draw would need the shared seed per pair.
    The independent draw is the weaker and correct default. It widens the
    interval when the seed sets differ, which is the honest result.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    xi = x[rng.integers(0, x.size, size=(resamples, x.size))].mean(axis=1)
    yi = y[rng.integers(0, y.size, size=(resamples, y.size))].mean(axis=1)
    diff = xi - yi
    return (
        float(x.mean() - y.mean()),
        float(np.percentile(diff, 2.5)),
        float(np.percentile(diff, 97.5)),
    )


def compare(paths: list[str], field: str, out: Path) -> int:
    """Write the comparison table for one score field."""
    rows_out: list[dict] = []
    per_arm: dict[str, list[float]] = {}
    for p in paths:
        rows = read_summary(p)
        label = rows[0].get("label") or Path(p).parent.name
        stats = aggregate(rows, field)
        per_arm.setdefault(label, []).extend(_finite([r.get(field, "") for r in rows]))
        rows_out.append(
            {
                "label": label,
                "field": field,
                "n_seeds": stats["n"],
                "mean": stats["mean"],
                "median": stats["median"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "n_diverged": sum(1 for r in rows if str(r.get("diverged")).lower() in ("true", "1")),
                "n_regressed": sum(1 for r in rows if str(r.get("regressed")).lower() in ("true", "1")),
            }
        )

    # Every arm against the first, so a run with three arms reports two pairs.
    labels = list(per_arm)
    for label in labels[1:]:
        diff, low, high = bootstrap_difference(per_arm[labels[0]], per_arm[label])
        separated = bool(np.isfinite(low) and np.isfinite(high) and (low > 0 or high < 0))
        rows_out.append(
            {
                "label": f"{labels[0]} minus {label}",
                "field": field,
                "n_seeds": float(len(per_arm[label])),
                "mean": diff,
                "median": float("nan"),
                "ci_low": low,
                "ci_high": high,
                "n_diverged": "",
                "n_regressed": "",
                "separated": separated,
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "field",
        "n_seeds",
        "mean",
        "median",
        "ci_low",
        "ci_high",
        "n_diverged",
        "n_regressed",
        "separated",
    ]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows_out:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(json.dumps({"comparison": str(out), "rows": len(rows_out)}, indent=2))
    for row in rows_out:
        print(
            f"{row['label']:38s} n={row['n_seeds']:.0f} "
            f"median={row['median']} "
            f"ci=[{row['ci_low']}, {row['ci_high']}]"
            + (
                f"  SEPARATED={row['separated']}"
                if "separated" in row and row["separated"] != ""
                else ""
            )
        )
    return 0


def run_mode(args) -> int:
    """Train every seed, then write the summary CSV."""
    total = args.total_steps or 2000
    eval_s = default_intervals(total, args.eval_steps, {}, "eval_steps", 1000)
    ckpt_i = default_intervals(total, args.save_steps, {}, "save_steps", 1000)
    argv = list(sys.argv[1:])
    cfg = apply_flat_yaml(args.sjfpo, {}, cli_fields=_sjfpo_cli_fields(argv))
    seeds = parse_seeds(args.seeds, args.seed_count, args.seed_start)
    label = args.label or f"sjfpo_{args.env}"

    out_root = Path(args.out)
    rows: list[dict] = []
    failures: list[dict] = []
    for seed in seeds:
        run_dir = resolve_run_dir(out_root, "sjfpo", args.env, seed, None)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== seed {seed} -> {run_dir} ===", flush=True)
        try:
            result = run_single(
                args.env, seed, run_dir, cfg, total, eval_s, ckpt_i,
                None, None, None, None,
            )
        except Exception as exc:  # noqa: BLE001 - one bad seed must not stop the rest
            traceback.print_exc(file=sys.stderr)
            failures.append({"seed": seed, "error": str(exc)})
            continue
        rows.append(
            {
                "label": label,
                "env": args.env,
                "seed": seed,
                "reward_mean_last10": result.get("reward_mean_last10"),
                "reward_median_last10": result.get("reward_median_last10"),
                "best_return": result.get("best_return"),
                "fps": result.get("fps"),
                "global_step": result.get("global_step"),
                "diverged": result.get("diverged"),
                "regressed": result.get("regressed"),
                "lr_decays": result.get("lr_decays"),
                "run_dir": str(run_dir),
            }
        )
        write_summary(out_root / "sweep_summary.csv", rows)

    if not rows:
        print(json.dumps({"error": "no seed completed", "failures": failures}, indent=2))
        return 2
    stats = aggregate(rows, "reward_median_last10")
    print(
        json.dumps(
            {
                "summary": str(out_root / "sweep_summary.csv"),
                "seeds_ok": len(rows),
                "seeds_failed": len(failures),
                "median_of_seed_medians": stats["median"],
                "ci": [stats["ci_low"], stats["ci_high"]],
            },
            indent=2,
        )
    )
    return 2 if failures else 0


def _split_compare_args(rest: list[str]) -> tuple[list[str], str, Path]:
    """Pull the flags out of ``rest`` and leave only the positional paths.

    A naive "not a flag" filter would read the value after ``--field`` or
    ``--out`` as a path, because that value does not start with a dash.
    """
    paths: list[str] = []
    field = "reward_median_last10"
    out = Path("outputs/comparison.csv")
    index = 0
    while index < len(rest):
        token = rest[index]
        if token == "--field":
            index += 1
            if index >= len(rest):
                raise SystemExit("ERROR: --field needs a value")
            field = rest[index]
        elif token.startswith("--field="):
            field = token.split("=", 1)[1]
        elif token == "--out":
            index += 1
            if index >= len(rest):
                raise SystemExit("ERROR: --out needs a value")
            out = Path(rest[index])
        elif token.startswith("--out="):
            out = Path(token.split("=", 1)[1])
        elif token.startswith("-"):
            raise SystemExit(f"ERROR: unknown flag for compare: {token!r}")
        else:
            paths.append(token)
        index += 1
    return paths, field, out


def main(argv: list[str] | None = None) -> int:
    configure_torch_cuda()
    runtime = apply_runtime_settings()
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv[0] if argv and not argv[0].startswith("-") else "run"
    if mode == "compare":
        paths, field, out = _split_compare_args(argv[1:])
        if not paths:
            print("ERROR: compare needs at least one summary CSV", file=sys.stderr)
            return 1
        return compare(paths, field, out)

    args = draccus.parse(config_class=RunArgs, args=argv[1:] if argv and not argv[0].startswith("-") else argv)
    if not args.env:
        print("ERROR: --env is required", file=sys.stderr)
        return 1
    if args.env not in KNOWN_ENVS:
        print(f"ERROR: unknown env {args.env!r}; choices: {', '.join(KNOWN_ENVS)}", file=sys.stderr)
        return 1
    out = run_mode(args)
    del runtime
    return out


if __name__ == "__main__":
    sys.exit(main())