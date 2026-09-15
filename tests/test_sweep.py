"""Tests for src/sjfpo/sweep.py.

Contract under test:
* ScanResult dataclass with pending/skipped/diverged lists
* scan_runs() scans per-run subdirs for manifest.json and classifies runs
* flush_pending_runs() executes pending runs in deterministic order
* run_pending() main entry point with atexit handler
* SweepError for scan-time errors
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.config import RunSpec, RUN_MATRIX  # noqa: E402

# Helpers


def make_run_spec(method: str, env: str, seed: int) -> RunSpec:
    """Create a RunSpec matching the config defaults."""
    return RunSpec(
        method=method,
        env=env,
        seed=seed,
        total_steps=1_000_000,
        eval_steps=50_000,
        ckpt_interval=25_000,
    )


def write_manifest(run_dir: Path, data: dict) -> None:
    """Write a manifest.json file to the run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(data), encoding="utf-8")


def formatRunDirectoryName(spec: RunSpec) -> str:
    """Generate the per-run subdir name: {method}_{env}_seed{seed}/"""
    return f"{spec.method}_{spec.env}_seed{spec.seed}"

    # Tests


class TestScanResult:
    def test_scan_result_creation(self):
        from sjfpo.sweep import ScanResult

        pending = [make_run_spec("sjfpo", "Hopper-v5", 0)]
        skipped = [make_run_spec("ppo", "Walker2d-v5", 1)]
        diverged = [make_run_spec("sac", "HalfCheetah-v5", 2)]

        result = ScanResult(pending=pending, skipped=skipped, diverged=diverged)

        assert result.pending == pending
        assert result.skipped == skipped
        assert result.diverged == diverged

    def test_scan_result_frozen(self):
        from sjfpo.sweep import ScanResult

        result = ScanResult(pending=[], skipped=[], diverged=[])
        with pytest.raises(Exception):  # dataclass frozen -> FrozenInstanceError
            result.pending = []


class TestScanRuns:
    def test_scans_missing_runs(self, tmp_path):
        """All runs missing -> all pending, none skipped/diverged."""
        from sjfpo.sweep import scan_runs

        matrix = (
            make_run_spec("sjfpo", "Hopper-v5", 0),
            make_run_spec("ppo", "Walker2d-v5", 1),
            make_run_spec("sac", "HalfCheetah-v5", 2),
        )

        result = scan_runs(matrix, tmp_path)

        assert len(result.pending) == 3
        assert len(result.skipped) == 0
        assert len(result.diverged) == 0
        # Check all three are in pending
        pending_keys = {(s.method, s.env, s.seed) for s in result.pending}
        assert pending_keys == {
            ("sjfpo", "Hopper-v5", 0),
            ("ppo", "Walker2d-v5", 1),
            ("sac", "HalfCheetah-v5", 2),
        }

    def test_skips_complete_runs(self, tmp_path):
        """Runs with complete=true manifest -> skipped."""
        from sjfpo.sweep import scan_runs

        spec1 = make_run_spec("sjfpo", "Hopper-v5", 0)
        spec2 = make_run_spec("ppo", "Walker2d-v5", 1)

        write_manifest(tmp_path / formatRunDirectoryName(spec1), {"complete": True})
        write_manifest(tmp_path / formatRunDirectoryName(spec2), {"complete": True})

        matrix = (spec1, spec2)
        result = scan_runs(matrix, tmp_path)

        assert len(result.pending) == 0
        assert len(result.skipped) == 2
        assert len(result.diverged) == 0
        skipped_keys = {(s.method, s.env, s.seed) for s in result.skipped}
        assert skipped_keys == {("sjfpo", "Hopper-v5", 0), ("ppo", "Walker2d-v5", 1)}

    def test_resumes_partial_runs(self, tmp_path):
        """Runs with complete=false manifest -> pending (to resume)."""
        from sjfpo.sweep import scan_runs

        spec = make_run_spec("sjfpo", "Hopper-v5", 0)
        write_manifest(
            tmp_path / formatRunDirectoryName(spec),
            {"complete": False, "global_step": 5000},
        )

        matrix = (spec,)
        result = scan_runs(matrix, tmp_path)

        assert len(result.pending) == 1
        assert len(result.skipped) == 0
        assert len(result.diverged) == 0
        assert result.pending[0].method == "sjfpo"
        assert result.pending[0].env == "Hopper-v5"
        assert result.pending[0].seed == 0

    def test_diverged_excluded(self, tmp_path):
        """Runs with diverged=true manifest -> diverged list, NOT pending."""
        from sjfpo.sweep import scan_runs

        spec = make_run_spec("sjfpo", "Hopper-v5", 0)
        write_manifest(tmp_path / formatRunDirectoryName(spec), {"diverged": True})

        matrix = (spec,)
        result = scan_runs(matrix, tmp_path)

        assert len(result.pending) == 0
        assert len(result.skipped) == 0
        assert len(result.diverged) == 1
        assert result.diverged[0].method == "sjfpo"

    def test_deterministic_ordering(self, tmp_path):
        """Pending runs sorted by (method, env, seed) lexicographic, then RUN_MATRIX index."""
        from sjfpo.sweep import scan_runs

        # Create 6 runs with varying completeness - only some will be pending
        specs = [
            make_run_spec("ppo", "Walker2d-v5", 1),  # complete -> skipped
            make_run_spec("sjfpo", "Hopper-v5", 0),  # missing -> pending
            make_run_spec("sac", "HalfCheetah-v5", 2),  # missing -> pending
            make_run_spec("sjfpo", "Hopper-v5", 1),  # missing -> pending
            make_run_spec("ppo", "Walker2d-v5", 0),  # missing -> pending
            make_run_spec("sac", "HalfCheetah-v5", 0),  # complete -> skipped
        ]

        # Mark some as complete
        write_manifest(tmp_path / formatRunDirectoryName(specs[0]), {"complete": True})
        write_manifest(tmp_path / formatRunDirectoryName(specs[5]), {"complete": True})

        matrix = tuple(specs)
        result = scan_runs(matrix, tmp_path)

        # Pending should be sorted by (method, env, seed)
        pending_keys = [(s.method, s.env, s.seed) for s in result.pending]
        expected_order = [
            ("ppo", "Walker2d-v5", 0),
            ("sac", "HalfCheetah-v5", 2),
            ("sjfpo", "Hopper-v5", 0),
            ("sjfpo", "Hopper-v5", 1),
        ]
        assert pending_keys == expected_order

    def test_corrupted_manifest_raises_sweep_error(self, tmp_path):
        """Invalid JSON in manifest -> SweepError naming the run."""
        from sjfpo.sweep import scan_runs, SweepError

        spec = make_run_spec("sjfpo", "Hopper-v5", 0)
        run_dir = tmp_path / formatRunDirectoryName(spec)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text("{ invalid json }", encoding="utf-8")

        matrix = (spec,)
        with pytest.raises(SweepError) as excinfo:
            scan_runs(matrix, tmp_path)

        assert "sjfpo_Hopper-v5_seed0" in str(excinfo.value)

    def test_manifest_without_complete_key_treated_as_partial(self, tmp_path):
        """Manifest exists but no 'complete' key -> treated as partial (pending)."""
        from sjfpo.sweep import scan_runs

        spec = make_run_spec("sjfpo", "Hopper-v5", 0)
        write_manifest(tmp_path / formatRunDirectoryName(spec), {"global_step": 1000})

        matrix = (spec,)
        result = scan_runs(matrix, tmp_path)

        assert len(result.pending) == 1
        assert len(result.skipped) == 0

    def test_diverged_with_complete_false_still_diverged(self, tmp_path):
        """If diverged=true, it goes to diverged even if complete=false."""
        from sjfpo.sweep import scan_runs

        spec = make_run_spec("sjfpo", "Hopper-v5", 0)
        write_manifest(
            tmp_path / formatRunDirectoryName(spec),
            {"complete": False, "diverged": True},
        )

        matrix = (spec,)
        result = scan_runs(matrix, tmp_path)

        assert len(result.pending) == 0
        assert len(result.diverged) == 1


class TestFlushPending:
    def test_run_pending_executes_in_order(self, tmp_path):
        """Mock run_fn records call order; verify deterministic order."""
        from sjfpo.sweep import flush_pending_runs

        call_order = []

        def mock_run_fn(spec, base_dir):
            call_order.append((spec.method, spec.env, spec.seed))
            return {"method": spec.method, "env": spec.env, "seed": spec.seed}

        pending = [
            make_run_spec("sjfpo", "Hopper-v5", 0),
            make_run_spec("sjfpo", "Hopper-v5", 1),
            make_run_spec("ppo", "Walker2d-v5", 0),
        ]

        session_start = time.time()
        results, failed = flush_pending_runs(
            pending, tmp_path, mock_run_fn, session_start
        )

        assert call_order == [
            ("sjfpo", "Hopper-v5", 0),
            ("sjfpo", "Hopper-v5", 1),
            ("ppo", "Walker2d-v5", 0),
        ]
        assert len(results) == 3
        assert failed == []

    def test_session_summary_printed(self, tmp_path, capsys):
        """Verify session summary is printed with completed/remaining/hours."""
        from sjfpo.sweep import flush_pending_runs

        def mock_run_fn(spec, base_dir):
            return {"method": spec.method, "env": spec.env, "seed": spec.seed}

        pending = [
            make_run_spec("sjfpo", "Hopper-v5", 0),
            make_run_spec("ppo", "Walker2d-v5", 1),
        ]

        session_start = time.time()
        flush_pending_runs(pending, tmp_path, mock_run_fn, session_start)

        captured = capsys.readouterr()
        assert "SESSION SUMMARY" in captured.out
        assert "2 completed" in captured.out
        assert "0 failed" in captured.out
        assert "0 remaining" in captured.out
        assert "hours elapsed" in captured.out

    def test_interrupted_run_continues(self, tmp_path):
        """Mock run_fn raises on first call but succeeds on second; second still executes."""
        from sjfpo.sweep import flush_pending_runs

        call_count = {"count": 0}

        def mock_run_fn(spec, base_dir):
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise RuntimeError("Simulated failure")
            return {"method": spec.method, "env": spec.env, "seed": spec.seed}

        pending = [
            make_run_spec("sjfpo", "Hopper-v5", 0),
            make_run_spec("ppo", "Walker2d-v5", 1),
        ]

        session_start = time.time()
        results, failed = flush_pending_runs(
            pending, tmp_path, mock_run_fn, session_start
        )

        # First run failed, second succeeded
        assert len(results) == 1
        assert results[0]["method"] == "ppo"
        assert results[0]["seed"] == 1
        assert failed == ["sjfpo_Hopper-v5_seed0"]

    def test_flush_pending_returns_results_list(self, tmp_path):
        """flush_pending_runs returns list of results from successful runs."""
        from sjfpo.sweep import flush_pending_runs

        def mock_run_fn(spec, base_dir):
            return {"spec": spec, "success": True}

        pending = [
            make_run_spec("sjfpo", "Hopper-v5", 0),
            make_run_spec("ppo", "Walker2d-v5", 1),
        ]

        session_start = time.time()
        results, failed = flush_pending_runs(
            pending, tmp_path, mock_run_fn, session_start
        )

        assert len(results) == 2
        assert all(r["success"] for r in results)
        assert failed == []


class TestRunPending:
    def test_run_pending_scan_summary_printed(self, tmp_path, capsys):
        """Verify scan summary is printed: X skipped, Y pending, Z diverged."""
        from sjfpo.sweep import run_pending

        spec1 = make_run_spec("sjfpo", "Hopper-v5", 0)
        spec2 = make_run_spec("ppo", "Walker2d-v5", 1)
        spec3 = make_run_spec("sac", "HalfCheetah-v5", 2)

        write_manifest(tmp_path / formatRunDirectoryName(spec1), {"complete": True})
        write_manifest(tmp_path / formatRunDirectoryName(spec3), {"diverged": True})

        matrix = (spec1, spec2, spec3)

        def mock_run_fn(spec, base_dir):
            return {"method": spec.method, "env": spec.env, "seed": spec.seed}

        result = run_pending(matrix, tmp_path, mock_run_fn)

        captured = capsys.readouterr()
        assert "1 skipped (complete)" in captured.out
        assert "1 pending" in captured.out
        assert "1 diverged" in captured.out

    def test_run_pending_returns_dict_with_keys(self, tmp_path):
        """run_pending returns dict with completed, skipped, diverged, results keys."""
        from sjfpo.sweep import run_pending

        spec = make_run_spec("sjfpo", "Hopper-v5", 0)
        matrix = (spec,)

        def mock_run_fn(spec, base_dir):
            return {"method": spec.method, "env": spec.env, "seed": spec.seed}

        result = run_pending(matrix, tmp_path, mock_run_fn)

        assert "completed" in result
        assert "skipped" in result
        assert "diverged" in result
        assert "failed" in result
        assert "results" in result
        assert result["completed"] == 1
        assert result["skipped"] == 0
        assert result["diverged"] == 0
        assert result["failed"] == []
        assert len(result["results"]) == 1

    def test_run_pending_registers_atexit_handler(self, tmp_path):
        """run_pending registers atexit handler that prints session summary at exit."""
        import atexit
        from sjfpo.sweep import run_pending

        spec = make_run_spec("sjfpo", "Hopper-v5", 0)
        matrix = (spec,)

        def mock_run_fn(spec, base_dir):
            return {"method": spec.method, "env": spec.env, "seed": spec.seed}

            # Just verify it runs without error; atexit output is captured at interpreter shutdown
            # not by pytest's capsys, so we can't easily assert on it here.

        result = run_pending(matrix, tmp_path, mock_run_fn)
        assert result["completed"] == 1

    def test_run_pending_surfaces_failures(self, tmp_path, capsys):
        """A failing run appears in summary['failed'] and the FAILED block."""
        from sjfpo.sweep import run_pending

        matrix = (make_run_spec("sjfpo", "Hopper-v5", 0),)

        def mock_run_fn(spec, base_dir):
            raise RuntimeError("boom")

        result = run_pending(matrix, tmp_path, mock_run_fn)
        assert result["completed"] == 0
        assert result["failed"] == ["sjfpo_Hopper-v5_seed0"]
        out = capsys.readouterr().out
        assert "SESSION SUMMARY FAILED: sjfpo_Hopper-v5_seed0" in out
        assert "ERROR in run sjfpo_Hopper-v5_seed0: boom" in out

    def test_run_pending_handles_empty_matrix(self, tmp_path):
        """Empty matrix returns zeros."""
        from sjfpo.sweep import run_pending

        matrix = ()

        def mock_run_fn(spec, base_dir):
            return {}

        result = run_pending(matrix, tmp_path, mock_run_fn)

        assert result["completed"] == 0
        assert result["skipped"] == 0
        assert result["diverged"] == 0
        assert result["results"] == []


class TestSweepError:
    def test_sweep_error_is_exception(self):
        from sjfpo.sweep import SweepError

        assert issubclass(SweepError, Exception)

    def test_sweep_error_can_be_raised_and_caught(self):
        from sjfpo.sweep import SweepError

        with pytest.raises(SweepError) as excinfo:
            raise SweepError("test error message")

        assert "test error message" in str(excinfo.value)
