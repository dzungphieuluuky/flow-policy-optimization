"""Benchmark analysis: bootstrap CIs, IQM, superiority, normalization, plots, CSV."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for tests
import matplotlib.pyplot as plt
import numpy as np

from sjfpo.config import RunSpec


class AnalysisError(Exception):
    """Analysis cannot proceed (missing/invalid data)."""

    pass


# Bootstrap confidence intervals


def bootstrap_ci(
    data: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI: (mean, ci_low, ci_high)."""
    if len(data) == 0:
        raise ValueError("data must not be empty")

    rng = np.random.default_rng(seed)
    n = len(data)
    boot_means = np.empty(n_boot)

    for i in range(n_boot):
        indices = rng.integers(0, n, size=n)
        boot_sample = data[indices]
        boot_means[i] = np.mean(boot_sample)

    alpha = (1 - ci) / 2
    ci_low = np.percentile(boot_means, 100 * alpha)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha))
    mean = np.mean(data)

    return float(mean), float(ci_low), float(ci_high)


# Interquartile Mean (IQM)


def compute_iqm(returns_per_seed: list[float]) -> float:
    """Mean after trimming top/bottom 25%; plain mean when n < 4."""
    if not returns_per_seed:
        raise ValueError("returns_per_seed must not be empty")

    arr = np.array(returns_per_seed, dtype=float)
    n = len(arr)

    if n < 4:
        return float(np.mean(arr))

    sorted_arr = np.sort(arr)
    trim_count = n // 4

    if trim_count == 0:
        return float(np.mean(arr))

    trimmed = sorted_arr[trim_count : n - trim_count]
    return float(np.mean(trimmed))


# Superiority testing


def is_superior(ci_low: float, ci_high: float) -> bool:
    """True iff the CI of the difference excludes 0 on the positive side."""
    return ci_low > 0


# Cross-environment normalization


def normalize_cross_env(
    results_by_env: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, list[float]]]:
    """Divide each env's returns by its max final return across methods."""
    normalized = {}

    for env_name, methods in results_by_env.items():
        # Find max final return across all methods for this env
        all_final_returns = []
        for method_returns in methods.values():
            if method_returns:
                all_final_returns.append(method_returns[-1])

        if not all_final_returns:
            normalized[env_name] = {m: [] for m in methods}
            continue

        max_return = max(all_final_returns)
        if max_return == 0:
            max_return = 1.0

        normalized[env_name] = {}
        for method, returns in methods.items():
            normalized[env_name][method] = [r / max_return for r in returns]

    return normalized


# Load evaluation histories


def load_eval_histories(run_dir: Path, *, eval_interval: int) -> list[dict[str, Any]]:
    """Scan run dirs for eval_history.csv files.

    Returns [{method, env, seed, global_step, return}]; global_step is
    episode_index * eval_interval (the CSV has no step column).
    Malformed rows/headers raise loudly, never default to zero.
    """
    histories = []

    for subdir in run_dir.iterdir():
        if not subdir.is_dir():
            continue

        parts = subdir.name.split("_")
        if len(parts) < 3:
            continue

        method = parts[0]
        env = "_".join(parts[1:-1])
        seed_str = parts[-1].removeprefix("seed")
        try:
            seed = int(seed_str)
        except ValueError:
            continue

        csv_path = subdir / "eval_history.csv"
        if not csv_path.exists():
            continue

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            missing = {"episode", "return"} - set(reader.fieldnames or [])
            if missing:
                raise AnalysisError(f"{csv_path} missing columns: {sorted(missing)}")
            for row in reader:
                histories.append(
                    {
                        "method": method,
                        "env": env,
                        "seed": seed,
                        "global_step": int(row["episode"]) * eval_interval,
                        "return": float(row["return"]),
                    }
                )

    return histories


@dataclass(frozen=True)
class AnalysisReport:
    """MuJoCo analysis results."""

    per_env_curves: dict[str, Path]
    aggregate_png: Path
    summary_csv: Path
    superiority: dict[str, bool]
    excluded_diverged: list[str]
    summary_stats: dict[tuple[str, str], tuple[float, float, float]] = field(
        default_factory=dict  # (method, env) -> (iqm, ci_low, ci_high)
    )


# Plotting functions


def plot_per_env_curves(
    env_name: str,
    curves_by_method: dict[str, dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Learning curves with bootstrap-CI shading for one env."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"sjfpo": "#1f77b4", "ppo": "#ff7f0e", "sac": "#2ca02c"}

    for method, data in curves_by_method.items():
        eval_points = data["eval_points"]
        returns_per_seed = data["returns_per_seed"]

        if not returns_per_seed:
            continue

        returns_array = np.array(returns_per_seed)
        mean_returns = np.mean(returns_array, axis=0)

        ci_low = np.empty(len(eval_points))
        ci_high = np.empty(len(eval_points))
        for i in range(len(eval_points)):
            point_returns = returns_array[:, i]
            if len(point_returns) >= 2:
                _, low, high = bootstrap_ci(point_returns, n_boot=500, ci=0.95, seed=42)
                ci_low[i] = low
                ci_high[i] = high
            else:
                ci_low[i] = mean_returns[i]
                ci_high[i] = mean_returns[i]

        color = colors.get(method, "#000000")
        ax.plot(
            eval_points, mean_returns, label=method.upper(), color=color, linewidth=2
        )
        ax.fill_between(eval_points, ci_low, ci_high, color=color, alpha=0.2)

    ax.set_xlabel("Global Step")
    ax.set_ylabel("Return")
    ax.set_title(f"Learning Curves: {env_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    plt.tight_layout()
    png_path = output_dir / f"{env_name}_curves.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return png_path


def plot_aggregate(
    aggregate_data: dict[str, dict[str, list[float]]],
    output_dir: Path,
) -> Path:
    """Normalized cross-env bar chart with bootstrap-CI error bars."""
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = set()
    for env_data in aggregate_data.values():
        methods.update(env_data.keys())
    methods = sorted(methods)

    method_means = {}
    method_cis = {}

    for method in methods:
        all_returns = []
        for env_data in aggregate_data.values():
            if method in env_data:
                all_returns.extend(env_data[method])

        if all_returns:
            mean, ci_low, ci_high = bootstrap_ci(
                np.array(all_returns), n_boot=2000, ci=0.95, seed=42
            )
            method_means[method] = mean
            method_cis[method] = (ci_low, ci_high)
        else:
            method_means[method] = 0.0
            method_cis[method] = (0.0, 0.0)

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"sjfpo": "#1f77b4", "ppo": "#ff7f0e", "sac": "#2ca02c"}
    x_pos = np.arange(len(methods))
    means = [method_means[m] for m in methods]
    ci_lows = [method_cis[m][0] for m in methods]
    ci_highs = [method_cis[m][1] for m in methods]

    bars = ax.bar(
        x_pos,
        means,
        yerr=[
            np.array(means) - np.array(ci_lows),
            np.array(ci_highs) - np.array(means),
        ],
        capsize=5,
        color=[colors.get(m, "#000000") for m in methods],
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_ylabel("Normalized Return")
    ax.set_title("Aggregate Cross-Environment Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    png_path = output_dir / "aggregate_comparison.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return png_path


# CSV export


def write_summary_csv(report: AnalysisReport, output_dir: Path) -> None:
    """Summary CSV (method,env,iqm,ci_low,ci_high,status) from report cells."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results_summary.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "env", "iqm", "ci_low", "ci_high", "status"])
        for (method, env), (iqm, ci_low, ci_high) in sorted(
            report.summary_stats.items()
        ):
            status = "complete"
            if any(
                d.startswith(f"{method}_{env}_seed") for d in report.excluded_diverged
            ):
                status = "diverged"
            writer.writerow([method, env, iqm, ci_low, ci_high, status])


# Main analysis entry point


def analyze_results(
    matrix: tuple[RunSpec, ...],
    base_dir: Path,
    output_dir: Path,
    *,
    eval_interval: int,
) -> AnalysisReport:
    """Run the full analysis: histories → curves → aggregate → plots → CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    histories = load_eval_histories(base_dir, eval_interval=eval_interval)

    if not histories:
        raise AnalysisError("no eval histories")

    from collections import defaultdict

    grouped = defaultdict(lambda: defaultdict(list))

    for h in histories:
        key = (h["method"], h["env"], h["seed"])
        grouped[key]["returns"].append(h["return"])
        grouped[key]["eval_points"].append(h["global_step"])

    # Diverged runs (diverged.json) are excluded from curves.
    excluded_diverged = []
    for subdir in base_dir.iterdir():
        if not subdir.is_dir():
            continue
        manifest_path = subdir / "diverged.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            if manifest.get("diverged", False):
                excluded_diverged.append(subdir.name)

    # Compute per-env curves
    per_env_curves = {}
    curves_by_env_method = defaultdict(lambda: defaultdict(list))

    for (method, env, seed), data in grouped.items():
        if f"{method}_{env}_{seed}" in excluded_diverged:
            continue
        curves_by_env_method[env][method].append(
            {
                "seed": seed,
                "eval_points": data["eval_points"],
                "returns": data["returns"],
            }
        )

    for env, methods in curves_by_env_method.items():
        curves_by_method = {}
        for method, seeds_data in methods.items():
            if seeds_data:
                curves_by_method[method] = {
                    "eval_points": seeds_data[0]["eval_points"],
                    "returns_per_seed": [d["returns"] for d in seeds_data],
                }
        if curves_by_method:
            png_path = plot_per_env_curves(env, curves_by_method, output_dir)
            per_env_curves[env] = png_path

    # Compute aggregate comparison
    # Collect final returns per method per env
    final_returns_by_env = defaultdict(lambda: defaultdict(list))
    for (method, env, seed), data in grouped.items():
        if f"{method}_{env}_{seed}" in excluded_diverged:
            continue
        if data["returns"]:
            final_returns_by_env[env][method].append(data["returns"][-1])

    # Per-cell summary stats for the CSV (real IQM + bootstrap CI).
    summary_stats: dict[tuple[str, str], tuple[float, float, float]] = {}
    for env, methods in final_returns_by_env.items():
        for method, returns in methods.items():
            _, ci_low, ci_high = bootstrap_ci(
                np.array(returns, dtype=float), n_boot=2000, ci=0.95, seed=42
            )
            summary_stats[(method, env)] = (compute_iqm(returns), ci_low, ci_high)

    # Normalize cross-env
    normalized = normalize_cross_env(final_returns_by_env)

    # Plot aggregate
    aggregate_png = plot_aggregate(normalized, output_dir)

    # Compute superiority
    superiority = {}
    for method in ["sjfpo", "ppo", "sac"]:
        baseline_methods = [m for m in ["sjfpo", "ppo", "sac"] if m != method]
        if not baseline_methods:
            superiority[method] = False
            continue

        method_returns = []
        for env_data in final_returns_by_env.values():
            if method in env_data:
                method_returns.extend(env_data[method])

        if not method_returns:
            superiority[method] = False
            continue

        best_baseline_returns = []
        for baseline in baseline_methods:
            baseline_returns = []
            for env_data in final_returns_by_env.values():
                if baseline in env_data:
                    baseline_returns.extend(env_data[baseline])
            if baseline_returns:
                best_baseline_returns.append((np.mean(baseline_returns), baseline))

        if not best_baseline_returns:
            superiority[method] = False
            continue

        best_baseline_returns.sort(reverse=True)
        best_baseline_method = best_baseline_returns[0][1]

        baseline_returns = []
        for env_data in final_returns_by_env.values():
            if best_baseline_method in env_data:
                baseline_returns.extend(env_data[best_baseline_method])

        if not baseline_returns:
            superiority[method] = False
            continue

        min_len = min(len(method_returns), len(baseline_returns))
        if min_len < 2:
            superiority[method] = False
            continue

        diff = np.array(method_returns[:min_len]) - np.array(baseline_returns[:min_len])
        _, ci_low, ci_high = bootstrap_ci(diff, n_boot=2000, ci=0.95, seed=42)
        superiority[method] = is_superior(ci_low, ci_high)

    summary_csv = output_dir / "results_summary.csv"
    write_summary_csv(
        AnalysisReport(
            per_env_curves=per_env_curves,
            aggregate_png=aggregate_png,
            summary_csv=summary_csv,
            superiority=superiority,
            excluded_diverged=excluded_diverged,
        ),
        output_dir,
    )

    return AnalysisReport(
        per_env_curves=per_env_curves,
        aggregate_png=aggregate_png,
        summary_csv=summary_csv,
        superiority=superiority,
        excluded_diverged=excluded_diverged,
        summary_stats=summary_stats,
    )
