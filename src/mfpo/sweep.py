"""Reentrant sweep executor state machine (plan task 14).

Scans per-run directories for checkpoint manifests, classifies runs as
pending/skipped/diverged, and executes pending runs sequentially with
deterministic ordering. Provides atexit safety net for SIGTERM'd sessions.
"""

from __future__ import annotations

import atexit
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mfpo.config import RunSpec


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SweepError(RuntimeError):
    """Raised for scan-time errors (corrupted manifest, missing files, etc.)."""

    pass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanResult:
    """Result of scanning the run matrix against the filesystem.

    Attributes:
        pending: Runs to execute (MISSING + PARTIAL).
        skipped: Already COMPLETE runs.
        diverged: Runs flagged as diverged (excluded from pending).
    """

    pending: list[RunSpec]
    skipped: list[RunSpec]
    diverged: list[RunSpec]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def format_run_directory_name(spec: RunSpec) -> str:
    """Generate the per-run subdir name: {method}_{env}_seed{seed}/"""
    return f"{spec.method}_{spec.env}_seed{spec.seed}"
def read_run_manifest(run_dir: Path) -> dict[str, Any]:
    """Read and parse manifest.json, raising SweepError on failure."""
    manifest_path = run_dir / "manifest.json"
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SweepError(f"corrupt manifest in {run_dir}: {e}") from e
    except OSError as e:
        raise SweepError(f"failed to read manifest in {run_dir}: {e}") from e


def classify_run_status(
    run_dir: Path,
    dir_name: str,
    manifest_reader: Callable[[Path], dict[str, Any]],
) -> str:
    """Classify one run dir: missing | diverged | complete | partial."""
    if not run_dir.is_dir():
        return "missing"  # subdir doesn't exist
    if not (run_dir / "manifest.json").is_file():
        return "missing"  # directory exists but no manifest
    try:
        manifest = manifest_reader(run_dir)
    except SweepError:
        raise SweepError(f"failed to scan run {dir_name}") from None
    if manifest.get("diverged", False) is True:
        return "diverged"  # takes precedence over complete
    if manifest.get("complete", False) is True:
        return "complete"
    return "partial"  # manifest exists but not complete


def print_session_summary(
    completed_count: int, total_count: int, session_start_time: float
) -> None:
    """Print completed/remaining/hours session summary."""
    elapsed_hours = (time.time() - session_start_time) / 3600.0
    remaining = total_count - completed_count
    print(
        f"SESSION SUMMARY: {completed_count} completed, "
        f"{remaining} remaining, {elapsed_hours:.2f} hours elapsed"
    )


def scan_runs(
    matrix: tuple[RunSpec, ...],
    base_dir: str | Path,
    dir_name_fn: Callable[[RunSpec], str] = format_run_directory_name,
    manifest_reader: Callable[[Path], dict[str, Any]] = read_run_manifest,
    sort_key: Callable[[RunSpec], tuple[Any, ...]] = lambda s: (s.method, s.env, s.seed),
) -> ScanResult:
    """Scan the run matrix, classifying each run as pending/skipped/diverged."""
    base = Path(base_dir)
    pending: list[RunSpec] = []
    skipped: list[RunSpec] = []
    diverged: list[RunSpec] = []
    index_map = {spec: i for i, spec in enumerate(matrix)}  # tiebreaker
    for spec in matrix:
        run_dir = base / dir_name_fn(spec)
        status = classify_run_status(run_dir, dir_name_fn(spec), manifest_reader)
        if status == "missing" or status == "partial":
            pending.append(spec)
        elif status == "complete":
            skipped.append(spec)
        else:
            diverged.append(spec)
    # Deterministic ordering: sort_key(spec) then matrix index tiebreaker
    pending.sort(key=lambda s: sort_key(s) + (index_map.get(s, 0),))
    return ScanResult(pending=pending, skipped=skipped, diverged=diverged)


# ---------------------------------------------------------------------------
# Internal: flush_pending_runs
# ---------------------------------------------------------------------------


def flush_pending_runs(
    pending_runs: list[RunSpec],
    base_dir: str | Path,
    run_fn: Callable[[RunSpec, Path], dict[str, Any]],
    session_start_time: float,
    dir_name_fn: Callable[[RunSpec], str] = format_run_directory_name,
) -> list[dict[str, Any]]:
    """Execute pending runs sequentially; skips failures, always summarizes."""
    base = Path(base_dir)
    results: list[dict[str, Any]] = []
    completed_count = 0
    try:
        for spec in pending_runs:
            try:
                result = run_fn(spec, base)
                results.append(result)
                completed_count += 1
            except Exception as e:
                # Log error and continue with next run
                print(f"ERROR in run {dir_name_fn(spec)}: {e}")
                continue
    finally:
        print_session_summary(completed_count, len(pending_runs), session_start_time)
    return results


def announce_scan_results(scan_result: ScanResult) -> None:
    """Print the skipped/pending/diverged scan summary."""
    print(
        f"{len(scan_result.skipped)} skipped (complete), "
        f"{len(scan_result.pending)} pending, "
        f"{len(scan_result.diverged)} diverged"
    )


def register_atexit_summary(pending_count: int, session_start: float) -> None:
    """Register belt-and-braces atexit summary for SIGTERM'd sessions."""
    def print_atexit_summary() -> None:
        elapsed_hours = (time.time() - session_start) / 3600.0
        print(
            f"SESSION SUMMARY (atexit): {pending_count} pending at start, "
            f"{elapsed_hours:.2f} hours elapsed"
        )

    atexit.register(print_atexit_summary)


# ---------------------------------------------------------------------------
# Public API: run_pending (main entry point)
# ---------------------------------------------------------------------------


def run_pending(
    matrix: tuple[RunSpec, ...],
    base_dir: str | Path,
    run_fn: Callable[[RunSpec, Path], dict[str, Any]],
    scan_fn: Callable[[tuple[RunSpec, ...], str | Path], ScanResult] = scan_runs,
    dir_name_fn: Callable[[RunSpec], str] = format_run_directory_name,
) -> dict[str, Any]:
    """Main entry point: scan, report, execute, and register atexit handler.

    Args:
        matrix: Tuple of RunSpec from config.RUN_MATRIX.
        base_dir: Parent directory containing per-run subdirs.
        run_fn: Function to execute each run: run_fn(spec, base_dir) -> result dict.
        scan_fn: Scan function used to classify runs (defaults to scan_runs).
        dir_name_fn: Maps a RunSpec to its per-run subdir name (for error logs).

    Returns:
        Dict with keys: completed, skipped, diverged, results.
    """
    base = Path(base_dir)
    session_start = time.time()

    # Scan
    scan_result = scan_fn(matrix, base)

    announce_scan_results(scan_result)

    register_atexit_summary(len(scan_result.pending), session_start)

    # Execute pending runs
    results = flush_pending_runs(
        scan_result.pending, base, run_fn, session_start, dir_name_fn
    )

    return {
        "completed": len(results),
        "skipped": len(scan_result.skipped),
        "diverged": len(scan_result.diverged),
        "results": results,
    }
