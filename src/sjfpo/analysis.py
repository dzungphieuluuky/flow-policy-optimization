"""Analysis module for SJFPO benchmark results.

Provides bootstrap confidence intervals, IQM computation, superiority testing,
cross-environment normalization, plotting, and CSV export.
"""

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
    """Raised when analysis cannot proceed due to missing or invalid data."""

    pass


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    data: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval using percentile method.

    Parameters
    ----------
    data : np.ndarray
        1D array of values (e.g., per-seed IQMs).
    n_boot : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (e.g., 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float, float]
        (mean, ci_low, ci_high) of the bootstrap distribution.
    """
    if len(data) == 0:
        raise ValueError("data must not be empty")

    rng = np.random.default_rng(seed)
    n = len(data)
    boot_means = np.empty(n_boot)

    for i in range(n_boot):
        # Stratified bootstrap over seeds: resample with replacement
        indices = rng.integers(0, n, size=n)
        boot_sample = data[indices]
        boot_means[i] = np.mean(boot_sample)

    alpha = (1 - ci) / 2
    ci_low = np.percentile(boot_means, 100 * alpha)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha))
    mean = np.mean(data)

    return float(mean), float(ci_low), float(ci_high)


# ---------------------------------------------------------------------------
# Interquartile Mean (IQM)
# ---------------------------------------------------------------------------


def compute_iqm(returns_per_seed: list[float]) -> float:
    """Compute Interquartile Mean (IQM) of final returns.

    IQM = mean after trimming bottom 25% and top 25% of values.
    If fewer than 4 values, return the mean (can't trim meaningfully).

    Parameters
    ----------
    returns_per_seed : list[float]
        List of final-episode returns, one per seed.

    Returns
    -------
    float
        Interquartile mean.
    """
    if not returns_per_seed:
        raise ValueError("returns_per_seed must not be empty")

    arr = np.array(returns_per_seed, dtype=float)
    n = len(arr)

    if n < 4:
        return float(np.mean(arr))

    # Sort and trim bottom 25% and top 25% by count
    sorted_arr = np.sort(arr)
    trim_count = n // 4  # floor division for 25%

    if trim_count == 0:
        return float(np.mean(arr))

    trimmed = sorted_arr[trim_count : n - trim_count]
    return float(np.mean(trimmed))


# ---------------------------------------------------------------------------
# Superiority testing
# ---------------------------------------------------------------------------


def is_superior(ci_low: float, ci_high: float) -> bool:
    """Check if 95% CI of difference excludes 0 on the positive side.

    Returns True if ci_low > 0 (significantly better).
    Returns False if CI crosses 0 or is entirely negative.

    Parameters
    ----------
    ci_low : float
        Lower bound of confidence interval.
    ci_high : float
        Upper bound of confidence interval.

    Returns
    -------
    bool
        True if significantly better (CI entirely above 0), False otherwise.
    """
    return ci_low > 0


# ---------------------------------------------------------------------------
# Cross-environment normalization
# ---------------------------------------------------------------------------


def normalize_cross_env(
    results_by_env: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, list[float]]]:
    """Normalize returns across environments by max running return baseline.

    Per env, divide all method returns by max running return baseline
    (max of final returns across methods). Pool normalized values across envs.

    Parameters
    ----------
    results_by_env : dict[str, dict[str, list[float]]]
        Nested dict: env_name -> method -> list of returns per seed.

    Returns
    -------
    dict[str, dict[str, list[float]]]
        Normalized results with same structure.
    """
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
            max_return = 1.0  # avoid division by zero

        normalized[env_name] = {}
        for method, returns in methods.items():
            normalized[env_name][method] = [r / max_return for r in returns]

    return normalized


# ---------------------------------------------------------------------------
# Load evaluation histories
# ---------------------------------------------------------------------------


def load_eval_histories(run_dir: Path, *, eval_interval: int) -> list[dict[str, Any]]:
    """Scan a run directory for eval_history.csv files.

    Returns list of dicts with keys: method, env, seed, global_step, return.
    ``global_step`` is ``episode_index * eval_interval`` — the CSV schema
    ([episode, return, length]) carries no step column, so the interval must
    be stated explicitly by the caller (no magic constants here).

    Malformed rows/headers raise loudly (AnalysisError/ValueError), never
    default to zero.

    Parameters
    ----------
    run_dir : Path
        Directory containing per-run subdirectories.
    eval_interval : int
        Eval cadence in env steps (required keyword-only).

    Returns
    -------
    list[dict]
        List of evaluation history records.
    """
    histories = []

    for subdir in run_dir.iterdir():
        if not subdir.is_dir():
            continue

        # Parse method, env, seed from directory name
        # Expected format: {method}_{env}_{seed}
        parts = subdir.name.split("_")
        if len(parts) < 3:
            continue

        method = parts[0]
        env = "_".join(parts[1:-1])  # env may contain underscores
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


# ---------------------------------------------------------------------------
# Analysis report dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisReport:
    """Container for MuJoCo analysis results."""

    per_env_curves: dict[str, Path]
    aggregate_png: Path
    summary_csv: Path
    superiority: dict[str, bool]
    excluded_diverged: list[str]
    # (method, env) -> (iqm, ci_low, ci_high) of final returns across seeds.
    summary_stats: dict[tuple[str, str], tuple[float, float, float]] = field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def plot_per_env_curves(
    env_name: str,
    curves_by_method: dict[str, dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Plot learning curves for a single environment.

    Parameters
    ----------
    env_name : str
        Environment name.
    curves_by_method : dict
        Dict mapping method name to dict with keys:
        - 'eval_points': list of x values
        - 'returns_per_seed': list of lists (one per seed)
    output_dir : Path
        Directory to save the plot.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
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

        # Bootstrap CI for shading
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

    # Format x-axis in scientific notation
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
    """Plot normalized cross-environment comparison bar chart.

    Parameters
    ----------
    aggregate_data : dict
        Normalized data: env_name -> method -> list of normalized returns.
    output_dir : Path
        Directory to save the plot.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all methods
    methods = set()
    for env_data in aggregate_data.values():
        methods.update(env_data.keys())
    methods = sorted(methods)

    # Compute mean and CI per method across all envs and seeds
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


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def write_summary_csv(report: AnalysisReport, output_dir: Path) -> None:
    """Write results summary CSV from the report's per-cell statistics.

    Columns: method,env,iqm,ci_low,ci_high,status. One row per (method, env)
    present in ``report.summary_stats`` (no hardcoded grid, no placeholders).

    Parameters
    ----------
    report : AnalysisReport
        Analysis report containing results.
    output_dir : Path
        Directory to write the CSV file.
    """
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


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------


def analyze_results(
    matrix: tuple[RunSpec, ...],
    base_dir: Path,
    output_dir: Path,
    *,
    eval_interval: int,
) -> AnalysisReport:
    """Main analysis entry point.

    Parameters
    ----------
    matrix : tuple[RunSpec]
        Run specifications.
    base_dir : Path
        Directory containing per-run subdirectories.
    output_dir : Path
        Directory to write outputs.
    eval_interval : int
        Eval cadence in env steps (required keyword-only; forwarded to
        load_eval_histories for the episode → global_step mapping).

    Returns
    -------
    AnalysisReport
        Analysis results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all evaluation histories
    histories = load_eval_histories(base_dir, eval_interval=eval_interval)

    if not histories:
        raise AnalysisError("no eval histories")

    # Group by method, env, seed
    from collections import defaultdict

    grouped = defaultdict(lambda: defaultdict(list))

    for h in histories:
        key = (h["method"], h["env"], h["seed"])
        grouped[key]["returns"].append(h["return"])
        grouped[key]["eval_points"].append(h["global_step"])

    # Check for diverged runs
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
        # Compare against best baseline (excluding self)
        baseline_methods = [m for m in ["sjfpo", "ppo", "sac"] if m != method]
        if not baseline_methods:
            superiority[method] = False
            continue

        # Pool all seeds across envs for this method
        method_returns = []
        for env_data in final_returns_by_env.values():
            if method in env_data:
                method_returns.extend(env_data[method])

        if not method_returns:
            superiority[method] = False
            continue

        # Find best baseline
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

        # Get baseline returns
        baseline_returns = []
        for env_data in final_returns_by_env.values():
            if best_baseline_method in env_data:
                baseline_returns.extend(env_data[best_baseline_method])

        if not baseline_returns:
            superiority[method] = False
            continue

        # Compute difference CI
        min_len = min(len(method_returns), len(baseline_returns))
        if min_len < 2:
            superiority[method] = False
            continue

        diff = np.array(method_returns[:min_len]) - np.array(baseline_returns[:min_len])
        _, ci_low, ci_high = bootstrap_ci(diff, n_boot=2000, ci=0.95, seed=42)
        superiority[method] = is_superior(ci_low, ci_high)

    # Write summary CSV
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
