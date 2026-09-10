"""Unit tests for src/mfpo/calibration.py (plan task 13).

Written BEFORE the calibration implementation (TDD): the first pytest run
must fail RED on the missing module, then go GREEN once calibration.py is
implemented.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from mfpo import config  # noqa: E402
from mfpo.config import build_legacy_run_matrix


def _make_fps(mfpo: float, ppo: float, sac: float) -> dict[str, float]:
    return {"mfpo": mfpo, "ppo": ppo, "sac": sac}


class TestGateReport:
    """Tests for the GateReport dataclass and its markdown rendering."""

    def test_gate_passes_within_budget(self):
        from mfpo.calibration import GateReport, calibrate_and_gate

        # High FPS so total < 120h and per-run < 10h
        fps = _make_fps(mfpo=50000.0, ppo=80000.0, sac=60000.0)
        report = calibrate_and_gate(build_legacy_run_matrix(), fps)

        assert report.gate_pass is True
        assert report.fallback_ladder == []
        assert report.total_hours < config.BUDGET_TOTAL_HOURS
        for hours in report.per_run_hours.values():
            assert hours <= config.BUDGET_PER_RUN_HOURS

    def test_gate_fails_total_over_budget(self):
        from mfpo.calibration import calibrate_and_gate

        # Low FPS so total > 120h (45 runs, need ~2.67h/run average)
        # 1M steps / fps / 3600 * 1.15 * 1.4 > 2.67
        # fps < 1M / 3600 / 2.67 / 1.15 / 1.4 ≈ 86
        fps = _make_fps(mfpo=80.0, ppo=100.0, sac=90.0)
        report = calibrate_and_gate(build_legacy_run_matrix(), fps)

        assert report.gate_pass is False
        assert len(report.fallback_ladder) >= 3
        assert report.total_hours > config.BUDGET_TOTAL_HOURS

    def test_gate_fails_per_run_over_budget(self):
        from mfpo.calibration import calibrate_and_gate

        # One method extremely slow -> single run > 10h
        # 1M / fps / 3600 * 1.15 * 1.4 > 10
        # fps < 1M / 3600 / 10 / 1.15 / 1.4 ≈ 17
        fps = _make_fps(mfpo=15.0, ppo=80000.0, sac=60000.0)
        report = calibrate_and_gate(build_legacy_run_matrix(), fps)

        assert report.gate_pass is False
        # Check that the message about USER SIGN-OFF REQUIRED appears
        markdown = report.markdown
        assert "USER SIGN-OFF REQUIRED" in markdown

    def test_markdown_contains_projected(self):
        from mfpo.calibration import calibrate_and_gate

        fps = _make_fps(mfpo=50000.0, ppo=80000.0, sac=60000.0)
        report = calibrate_and_gate(build_legacy_run_matrix(), fps)

        markdown = report.markdown
        assert "Projected" in markdown
        assert "mfpo" in markdown
        assert "ppo" in markdown
        assert "sac" in markdown

    def test_fallback_ladder_order(self):
        from mfpo.calibration import calibrate_and_gate

        fps = _make_fps(mfpo=80.0, ppo=100.0, sac=90.0)
        report = calibrate_and_gate(build_legacy_run_matrix(), fps)

        ladder = report.fallback_ladder
        assert len(ladder) >= 3
        # Order: seeds -> envs -> steps
        assert "Reduce seeds" in ladder[0]
        assert "Reduce envs" in ladder[1]
        assert "Reduce steps" in ladder[2]

    def test_missing_fps_key_raises(self):
        from mfpo.calibration import calibrate_and_gate

        fps = {"mfpo": 50000.0, "ppo": 80000.0}  # missing "sac"
        with pytest.raises(KeyError) as exc_info:
            calibrate_and_gate(build_legacy_run_matrix(), fps)
        assert "sac" in str(exc_info.value)

    def test_empty_matrix(self):
        from mfpo.calibration import calibrate_and_gate

        fps = _make_fps(mfpo=50000.0, ppo=80000.0, sac=60000.0)
        report = calibrate_and_gate((), fps)

        assert report.gate_pass is True
        assert report.total_hours == 0.0
        assert report.per_run_hours == {}
        assert report.fallback_ladder == []

    def test_safety_margin_applied(self):
        from mfpo.calibration import calibrate_and_gate

        # Create a minimal matrix with known values
        spec = config.RunSpec(
            method="mfpo",
            env="HalfCheetah-v5",
            seed=0,
            total_steps=1_000_000,
            eval_interval=50_000,
            ckpt_interval=25_000,
        )
        # 10000 fps -> 1_000_000 / 10000 / 3600 = 0.02777... hours base
        # With 15% margin: * 1.15 = 0.03194...
        # Eval overhead: 1_000_000 / 50_000 = 20 evals * 0.02 = 0.4 -> factor 1.4
        # Total: 0.02777 * 1.15 * 1.4 = 0.0446...
        fps = {"mfpo": 10000.0}
        report = calibrate_and_gate((spec,), fps)

        expected_base = 1_000_000 / 10000 / 3600
        expected_with_margin = expected_base * 1.15
        eval_overhead = 1.0 + (1_000_000 / 50_000) * 0.02
        expected_total = expected_with_margin * eval_overhead

        actual = report.per_run_hours["mfpo_HalfCheetah-v5_0"]
        assert abs(actual - expected_total) < 1e-6, f"Expected {expected_total}, got {actual}"


class TestRenderMarkdown:
    """Tests for the standalone render_markdown function."""

    def test_render_markdown_standalone(self):
        from mfpo.calibration import GateReport, render_markdown

        report = GateReport(
            per_run_hours={"mfpo_HalfCheetah-v5_0": 1.5, "ppo_HalfCheetah-v5_0": 2.0},
            total_hours=3.5,
            gate_pass=True,
            fallback_ladder=[],
        )
        markdown = render_markdown(report)
        assert "Projected" in markdown
        assert "mfpo" in markdown
        assert "ppo" in markdown
        assert "1.5" in markdown
        assert "2.0" in markdown
        assert "3.5" in markdown

    def test_render_markdown_with_fallback(self):
        from mfpo.calibration import GateReport, render_markdown

        report = GateReport(
            per_run_hours={"mfpo_HalfCheetah-v5_0": 12.0},
            total_hours=12.0,
            gate_pass=False,
            fallback_ladder=["Reduce seeds from 5 to 3"],
        )
        markdown = render_markdown(report)
        assert "USER SIGN-OFF REQUIRED" in markdown
        assert "Reduce seeds from 5 to 3" in markdown
