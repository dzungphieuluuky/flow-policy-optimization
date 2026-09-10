"""Tests for src/mfpo/analysis.py (plan task 15).

Written FIRST (TDD) — all tests must pass after implementation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from mfpo.analysis import (
    bootstrap_ci,
    compute_iqm,
    is_superior,
    normalize_cross_env,
    load_eval_histories,
    analyze_results,
    AnalysisReport,
    plot_per_env_curves,
    plot_aggregate,
    write_summary_csv,
    AnalysisError,
)


# ---------------------------------------------------------------------------
# 1. test_bootstrap_ci_contains_median
# ---------------------------------------------------------------------------


def test_bootstrap_ci_contains_median():
    """Bootstrap CI should contain the sample median."""
    rng = np.random.default_rng(42)
    data = rng.normal(100, 15, size=50)
    mean, ci_low, ci_high = bootstrap_ci(data, n_boot=2000, ci=0.95, seed=42)
    sample_median = np.median(data)
    assert ci_low <= sample_median <= ci_high, (
        f"CI [{ci_low:.2f}, {ci_high:.2f}] does not contain median {sample_median:.2f}"
    )


# ---------------------------------------------------------------------------
# 2. test_bootstrap_ci_width_decreases_with_n
# ---------------------------------------------------------------------------


def test_bootstrap_ci_width_decreases_with_n():
    """CI width for n=100 should be smaller than n=10."""
    rng = np.random.default_rng(123)
    data_10 = rng.normal(50, 10, size=10)
    data_100 = rng.normal(50, 10, size=100)

    _, ci_low_10, ci_high_10 = bootstrap_ci(data_10, n_boot=2000, ci=0.95, seed=42)
    _, ci_low_100, ci_high_100 = bootstrap_ci(data_100, n_boot=2000, ci=0.95, seed=42)

    width_10 = ci_high_10 - ci_low_10
    width_100 = ci_high_100 - ci_low_100

    assert width_100 < width_10, (
        f"CI width with n=100 ({width_100:.2f}) should be smaller than n=10 ({width_10:.2f})"
    )


# ---------------------------------------------------------------------------
# 3. test_iqm_computation
# ---------------------------------------------------------------------------


def test_iqm_computation():
    """Manual test: values = [1,2,3,4,5,6,7,8] → IQM = mean([3,4,5,6]) = 4.5."""
    values = [1, 2, 3, 4, 5, 6, 7, 8]
    iqm = compute_iqm(values)
    assert iqm == pytest.approx(4.5), f"Expected 4.5, got {iqm}"


# ---------------------------------------------------------------------------
# 4. test_iqm_few_values
# ---------------------------------------------------------------------------


def test_iqm_few_values():
    """values = [1,2,3] → IQM = mean([1,2,3]) = 2.0 (can't trim < 4 values)."""
    values = [1, 2, 3]
    iqm = compute_iqm(values)
    assert iqm == pytest.approx(2.0), f"Expected 2.0, got {iqm}"


# ---------------------------------------------------------------------------
# 5. test_superiority_true
# ---------------------------------------------------------------------------


def test_superiority_true():
    """ci_low=0.5, ci_high=2.0 → is_superior=True (CI entirely above 0)."""
    assert is_superior(0.5, 2.0) is True
    assert is_superior(0.1, 1.0) is True  # small positive CI
    assert is_superior(-2.0, -0.5) is False  # negative CI = not superior


# ---------------------------------------------------------------------------
# 6. test_superiority_false_overlapping
# ---------------------------------------------------------------------------


def test_superiority_false_overlapping():
    """ci_low=-0.5, ci_high=2.0 → is_superior=False (CI crosses 0)."""
    assert is_superior(-0.5, 2.0) is False
    assert is_superior(-2.0, 0.5) is False
    assert is_superior(0.0, 2.0) is False  # boundary case


# ---------------------------------------------------------------------------
# 7. test_planted_effect_detection
# ---------------------------------------------------------------------------


def test_planted_effect_detection():
    """Use synthetic_curves generator with rng seed 7.
    Assert the superior-set == {'strong'} EXACTLY with tie excluded.
    """
    from tests.fixtures.synthetic_curves import generate_curves

    curves = generate_curves()

    # Compute final returns per seed for each method
    final_returns = {}
    for method, seeds_data in curves.items():
        final_returns[method] = [d["returns"][-1] for d in seeds_data]

    # Compute IQM per method
    iqms = {method: compute_iqm(returns) for method, returns in final_returns.items()}

    # Bootstrap CI for each method
    cis = {}
    for method, returns in final_returns.items():
        mean, ci_low, ci_high = bootstrap_ci(
            np.array(returns), n_boot=2000, ci=0.95, seed=42
        )
        cis[method] = (ci_low, ci_high)

    # Find best baseline (highest IQM among weak/tie)
    baseline_methods = ["weak", "tie"]
    best_baseline = max(baseline_methods, key=lambda m: iqms[m])

    # Check superiority of each method vs best baseline
    superior = {}
    for method in ["strong", "weak", "tie"]:
        if method == best_baseline:
            superior[method] = False
            continue
        # Compute difference CI
        diff_returns = np.array(final_returns[method]) - np.array(
            final_returns[best_baseline]
        )
        _, ci_low, ci_high = bootstrap_ci(diff_returns, n_boot=2000, ci=0.95, seed=42)
        superior[method] = is_superior(ci_low, ci_high)

    # Assert: only 'strong' is superior
    assert superior == {"strong": True, "weak": False, "tie": False}, (
        f"Expected {{'strong': True, 'weak': False, 'tie': False}}, got {superior}"
    )


# ---------------------------------------------------------------------------
# 8. test_per_env_png_created
# ---------------------------------------------------------------------------


def test_per_env_png_created(tmp_path):
    """Generate curves, plot them, assert PNG exists with size > 10KB."""
    from tests.fixtures.synthetic_curves import generate_curves

    curves = generate_curves()

    # Convert to format expected by plot_per_env_curves
    curves_by_method = {}
    for method, seeds_data in curves.items():
        curves_by_method[method] = {
            "eval_points": seeds_data[0]["eval_points"],
            "returns_per_seed": [d["returns"] for d in seeds_data],
        }

    png_path = plot_per_env_curves("TestEnv", curves_by_method, tmp_path)

    assert png_path.exists(), f"PNG not created at {png_path}"
    assert png_path.stat().st_size > 10000, (
        f"PNG too small: {png_path.stat().st_size} bytes"
    )


# ---------------------------------------------------------------------------
# 9. test_summary_csv_schema
# ---------------------------------------------------------------------------


def test_summary_csv_schema(tmp_path):
    """Create a minimal analysis, verify CSV has exactly the columns."""
    # Create a minimal report
    report = AnalysisReport(
        per_env_curves={"TestEnv": tmp_path / "test_curves.png"},
        aggregate_png=tmp_path / "aggregate.png",
        summary_csv=tmp_path / "results_summary.csv",
        superiority={"mfpo": True, "ppo": False, "sac": False},
        excluded_diverged=[],
    )

    # Create dummy PNG files
    (tmp_path / "test_curves.png").write_bytes(b"x" * 11000)
    (tmp_path / "aggregate.png").write_bytes(b"x" * 11000)

    write_summary_csv(report, tmp_path)

    csv_path = tmp_path / "results_summary.csv"
    assert csv_path.exists()

    with open(csv_path) as f:
        header = f.readline().strip()

    expected_columns = "method,env,iqm,ci_low,ci_high,wallclock_h,status"
    assert header == expected_columns, f"Expected '{expected_columns}', got '{header}'"


# ---------------------------------------------------------------------------
# 10. test_empty_history_raises
# ---------------------------------------------------------------------------


def test_empty_history_raises(tmp_path):
    """Call analyze_results on empty dir → AnalysisError('no eval histories') raised."""
    from mfpo.config import RunSpec

    matrix = (
        RunSpec(
            method="mfpo",
            env="TestEnv",
            seed=0,
            total_steps=1000,
            eval_interval=500,
            ckpt_interval=500,
        ),
    )

    with pytest.raises(AnalysisError, match="no eval histories"):
        analyze_results(matrix, tmp_path, tmp_path / "output")


# ---------------------------------------------------------------------------
# 11. test_diverged_excluded_and_reported
# ---------------------------------------------------------------------------


def test_diverged_excluded_and_reported(tmp_path):
    """Create mock run with diverged manifest → excluded from analysis, listed in report."""
    from mfpo.config import RunSpec

    # Create a run directory with a diverged manifest
    run_dir = tmp_path / "mfpo_TestEnv_0"
    run_dir.mkdir(parents=True)

    # Write eval_history.csv
    import csv

    csv_path = run_dir / "eval_history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "return", "length"])
        writer.writeheader()
        writer.writerow({"episode": 0, "return": 10.0, "length": 100})

    # Write diverged manifest
    manifest_path = run_dir / "diverged.json"
    import json

    with open(manifest_path, "w") as f:
        json.dump({"diverged": True, "reason": "below 10% random baseline"}, f)

    matrix = (
        RunSpec(
            method="mfpo",
            env="TestEnv",
            seed=0,
            total_steps=1000,
            eval_interval=500,
            ckpt_interval=500,
        ),
    )

    report = analyze_results(matrix, tmp_path, tmp_path / "output")

    assert "mfpo_TestEnv_0" in report.excluded_diverged
    assert report.superiority.get("mfpo") is False  # excluded methods not superior


# ---------------------------------------------------------------------------
# Additional helper tests
# ---------------------------------------------------------------------------


def test_normalize_cross_env():
    """Test cross-environment normalization."""
    results_by_env = {
        "Env1": {"mfpo": [100, 110, 120], "ppo": [80, 90, 100]},
        "Env2": {"mfpo": [200, 220, 240], "ppo": [150, 160, 170]},
    }

    normalized = normalize_cross_env(results_by_env)

    # Check that values are normalized by max per env
    # Env1 max = 120, Env2 max = 240
    assert normalized["Env1"]["mfpo"] == pytest.approx(
        [100 / 120, 110 / 120, 120 / 120]
    )
    assert normalized["Env1"]["ppo"] == pytest.approx([80 / 120, 90 / 120, 100 / 120])
    assert normalized["Env2"]["mfpo"] == pytest.approx(
        [200 / 240, 220 / 240, 240 / 240]
    )
    assert normalized["Env2"]["ppo"] == pytest.approx([150 / 240, 160 / 240, 170 / 240])


def test_load_eval_histories(tmp_path):
    """Test loading eval histories from run directories."""
    # Create run directories with eval_history.csv
    run1 = tmp_path / "mfpo_HalfCheetah-v5_0"
    run1.mkdir(parents=True)
    import csv

    with open(run1 / "eval_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "return", "length"])
        writer.writeheader()
        writer.writerow({"episode": 0, "return": 100.0, "length": 100})
        writer.writerow({"episode": 1, "return": 150.0, "length": 100})

    run2 = tmp_path / "ppo_HalfCheetah-v5_0"
    run2.mkdir(parents=True)
    with open(run2 / "eval_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "return", "length"])
        writer.writeheader()
        writer.writerow({"episode": 0, "return": 80.0, "length": 100})

    histories = load_eval_histories(tmp_path)

    assert len(histories) == 3
    methods = {h["method"] for h in histories}
    assert methods == {"mfpo", "ppo"}
    envs = {h["env"] for h in histories}
    assert envs == {"HalfCheetah-v5"}
