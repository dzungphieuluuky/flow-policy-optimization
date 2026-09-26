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
    """Build the conventional subdirectory name for one run.

    Args:
        spec: Run spec to name.

    Returns:
        Name of the form ``{method}_{env}_seed{seed}``.
    """
    return f"{spec.method}_{spec.env}_seed{spec.seed}"


def read_run_manifest(run_dir: Path) -> dict[str, Any]:
    """Read and parse a run's ``manifest.json``.

    Args:
        run_dir: Directory of the run to read.

    Returns:
        The parsed manifest dict.

    Raises:
        SweepError: If the manifest is corrupt or unreadable.
    """
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
    """Classify a run directory from its manifest.

    Args:
        run_dir: Directory of the run to classify.
        dir_name: Run directory name, used in the error message.
        manifest_reader: Callable that reads and parses the manifest.

    Returns:
        One of ``missing``, ``diverged``, ``complete``, or ``partial``.

    Raises:
        SweepError: If the manifest exists but cannot be read.
    """
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
    """Print a completed/failed/remaining/hours summary for the session.

    Args:
        completed_count: Runs that finished successfully.
        total_count: Runs attempted this session.
        session_start_time: ``time.time()`` value from session start.
        failed: Names of the runs that failed.
    """
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
    """Scan the run matrix and bucket each run by its on-disk status.

    Runs are re-sorted by ``sort_key`` with their original matrix index as the
    tiebreaker, so a resumed sweep keeps the canonical ordering.

    Args:
        matrix: Run specs making up the sweep.
        base_dir: Parent directory holding the per-run subdirectories.
        dir_name_fn: Callable mapping a spec to its subdirectory name.
        manifest_reader: Callable that reads and parses a manifest.
        sort_key: Callable producing the sort key for a spec.

    Returns:
        The pending, skipped, and diverged runs.

    Raises:
        SweepError: If any run's manifest exists but cannot be read.
    """
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
    """Execute pending runs sequentially, recording failures as they happen.

    A failing run is logged and the sweep continues, so one broken run cannot
    abort the rest. The session summary is printed even if the sweep is
    interrupted.

    Args:
        pending_runs: Run specs to execute, in order.
        base_dir: Parent directory the runs write into.
        run_fn: Callable executing one spec and returning its result dict.
        session_start_time: ``time.time()`` value from session start.
        dir_name_fn: Callable mapping a spec to its subdirectory name.

    Returns:
        Tuple of the successful result dicts and the names of failed runs.
    """
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
    """Print the skipped/pending/diverged scan summary.

    Args:
        scan_result: Scan outcome to summarise.
    """
    print(
        f"{len(scan_result.skipped)} skipped (complete), "
        f"{len(scan_result.pending)} pending, "
        f"{len(scan_result.diverged)} diverged"
    )


def register_atexit_summary(pending_count: int, session_start: float) -> None:
    """Register a fallback session summary for interrupted sessions.

    The normal summary is skipped when the process is killed by SIGTERM, so an
    ``atexit`` handler reports what was still pending and how long the session
    ran.

    Args:
        pending_count: Runs pending when the handler was registered.
        session_start: ``time.time()`` value from session start.
    """

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
    """Scan the matrix, report the outcome, and execute the pending runs.

    The sweep is reentrant: scanning first means an interrupted session
    resumes by skipping whatever already completed.

    Args:
        matrix: Run specs making up the sweep.
        base_dir: Parent directory holding the per-run subdirectories.
        run_fn: Callable executing one spec and returning its result dict.
        scan_fn: Callable that buckets the matrix into a :class:`ScanResult`.
        dir_name_fn: Callable mapping a spec to its subdirectory name.

    Returns:
        Session result dict with the completed, failed, and remaining runs.
    """
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
