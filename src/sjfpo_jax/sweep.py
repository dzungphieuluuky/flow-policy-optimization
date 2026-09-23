"""Reentrant sweep executor: scan run dirs, execute pending runs in order."""

from __future__ import annotations

import atexit
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sjfpo.config import RunSpec


class SweepError(RuntimeError):
    """Scan-time errors (corrupt manifest, missing files)."""

    pass


@dataclass(frozen=True)
class ScanResult:
    """Scan outcome: pending (missing+partial), skipped (complete), diverged."""

    pending: list[RunSpec]
    skipped: list[RunSpec]
    diverged: list[RunSpec]


def format_run_directory_name(spec: RunSpec) -> str:
    """Per-run subdir: {method}_{env}_seed{seed}/."""
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
        return "missing"
    if not (run_dir / "manifest.json").is_file():
        return "missing"
    try:
        manifest = manifest_reader(run_dir)
    except SweepError:
        raise SweepError(f"failed to scan run {dir_name}") from None
    if manifest.get("diverged", False) is True:
        return "diverged"
    if manifest.get("complete", False) is True:
        return "complete"
    return "partial"


def print_session_summary(
    completed_count: int,
    total_count: int,
    session_start_time: float,
    failed: list[str] | None = None,
) -> None:
    """Print completed/failed/remaining/hours session summary."""
    elapsed_hours = (time.time() - session_start_time) / 3600.0
    failed = list(failed or [])
    remaining = total_count - completed_count - len(failed)
    print(
        f"SESSION SUMMARY: {completed_count} completed, "
        f"{len(failed)} failed, {remaining} remaining, "
        f"{elapsed_hours:.2f} hours elapsed"
    )
    for name in failed:
        print(f"SESSION SUMMARY FAILED: {name}")


def scan_runs(
    matrix: tuple[RunSpec, ...],
    base_dir: str | Path,
    dir_name_fn: Callable[[RunSpec], str] = format_run_directory_name,
    manifest_reader: Callable[[Path], dict[str, Any]] = read_run_manifest,
    sort_key: Callable[[RunSpec], tuple[Any, ...]] = lambda s: (
        s.method,
        s.env,
        s.seed,
    ),
) -> ScanResult:
    """Scan the run matrix, classifying each run as pending/skipped/diverged."""
    base = Path(base_dir)
    pending: list[RunSpec] = []
    skipped: list[RunSpec] = []
    diverged: list[RunSpec] = []
    index_map = {spec: i for i, spec in enumerate(matrix)}
    for spec in matrix:
        run_dir = base / dir_name_fn(spec)
        status = classify_run_status(run_dir, dir_name_fn(spec), manifest_reader)
        if status == "missing" or status == "partial":
            pending.append(spec)
        elif status == "complete":
            skipped.append(spec)
        else:
            diverged.append(spec)
    pending.sort(key=lambda s: sort_key(s) + (index_map.get(s, 0),))
    return ScanResult(pending=pending, skipped=skipped, diverged=diverged)


def flush_pending_runs(
    pending_runs: list[RunSpec],
    base_dir: str | Path,
    run_fn: Callable[[RunSpec, Path], dict[str, Any]],
    session_start_time: float,
    dir_name_fn: Callable[[RunSpec], str] = format_run_directory_name,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Execute pending runs sequentially; failures are recorded, never silent."""
    base = Path(base_dir)
    results: list[dict[str, Any]] = []
    failed: list[str] = []
    completed_count = 0
    try:
        for spec in pending_runs:
            name = dir_name_fn(spec)
            try:
                result = run_fn(spec, base)
                results.append(result)
                completed_count += 1
            except Exception as e:
                failed.append(name)
                print(f"ERROR in run {name}: {e}")
                continue
    finally:
        print_session_summary(
            completed_count, len(pending_runs), session_start_time, failed
        )
    return results, failed


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


def run_pending(
    matrix: tuple[RunSpec, ...],
    base_dir: str | Path,
    run_fn: Callable[[RunSpec, Path], dict[str, Any]],
    scan_fn: Callable[[tuple[RunSpec, ...], str | Path], ScanResult] = scan_runs,
    dir_name_fn: Callable[[RunSpec], str] = format_run_directory_name,
) -> dict[str, Any]:
    """Scan, report, execute pending runs, register atexit handler."""
    base = Path(base_dir)
    session_start = time.time()

    scan_result = scan_fn(matrix, base)

    announce_scan_results(scan_result)

    register_atexit_summary(len(scan_result.pending), session_start)

    results, failed = flush_pending_runs(
        scan_result.pending, base, run_fn, session_start, dir_name_fn
    )

    return {
        "completed": len(results),
        "skipped": len(scan_result.skipped),
        "diverged": len(scan_result.diverged),
        "failed": failed,
        "results": results,
    }
