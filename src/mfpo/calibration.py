"""Calibration and budget-gate logic for MFPO benchmark (plan task 13).

Provides:
- GateReport: frozen dataclass with per-run projections, totals, pass/fail,
  fallback ladder, and markdown rendering.
- calibrate_and_gate(): computes projected hours from measured FPS and
  evaluates against budget constants.
- render_markdown(): standalone table renderer for GateReport.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mfpo.config import (
    BUDGET_PER_RUN_HOURS,
    BUDGET_TOTAL_HOURS,
    RunSpec,
)


class CalibrationError(Exception):
    """Raised when calibration cannot proceed due to missing or invalid data."""

    def __init__(self, message: str, method: str | None = None) -> None:
        super().__init__(message)
        self.method = method

    def __str__(self) -> str:
        if self.method:
            return f"{super().__str__()} (method: {self.method})"
        return super().__str__()


@dataclass(frozen=True)
class GateReport:
    """Budget gate report with per-run projections and fallback recommendations."""

    per_run_hours: dict[str, float]
    total_hours: float
    gate_pass: bool
    fallback_ladder: list[str] = field(default_factory=list)
    markdown: str = field(default="", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "markdown", render_markdown(self))


def project_run_hours(
    matrix: tuple[RunSpec, ...], measured_fps: dict[str, float]
) -> tuple[dict[str, float], float]:
    """Project per-run hours from measured FPS (+15% margin, +2%/eval)."""
    per_run_hours: dict[str, float] = {}
    for spec in matrix:
        if spec.method not in measured_fps:
            raise KeyError(f"Missing FPS for method: {spec.method}")
        fps = measured_fps[spec.method]
        projected = spec.total_steps / fps / 3600.0 * 1.15
        num_evals = spec.total_steps / spec.eval_interval
        projected *= 1.0 + num_evals * 0.02
        key = f"{spec.method}_{spec.env}_{spec.seed}"
        per_run_hours[key] = projected
    return per_run_hours, sum(per_run_hours.values())


def build_fallback_ladder(matrix: tuple[RunSpec, ...]) -> list[str]:
    """Recommend seed → env → step reductions when the budget gate fails."""
    fallback_ladder: list[str] = []
    methods = {s.method for s in matrix}
    envs = {s.env for s in matrix}
    seeds = {s.seed for s in matrix}
    if len(seeds) > 1:
        new_seeds = max(1, len(seeds) - 2)
        fallback_ladder.append(
            f"Reduce seeds from {len(seeds)} to {new_seeds} "
            f"({len(methods)} method × {len(envs)} env × {new_seeds} seeds = "
            f"{len(methods) * len(envs) * new_seeds} runs)"
        )
    if len(envs) > 1:
        new_envs = max(1, len(envs) - 1)
        fallback_ladder.append(
            f"Reduce envs from {len(envs)} to {new_envs} "
            f"({len(methods)} method × {new_envs} env × {len(seeds)} seeds = "
            f"{len(methods) * new_envs * len(seeds)} runs)"
        )
    if matrix:
        ref_steps = matrix[0].total_steps
        new_steps = max(100_000, int(ref_steps * 0.6))
        fallback_ladder.append(
            f"Reduce steps from {ref_steps:,} to {new_steps:,} "
            f"(all methods × envs × seeds)"
        )
    return fallback_ladder


def calibrate_and_gate(
    matrix: tuple[RunSpec, ...], measured_fps: dict[str, float]
) -> GateReport:
    """Compute projected hours and evaluate budget gate (MuJoCo path)."""
    per_run_hours, total_hours = project_run_hours(matrix, measured_fps)
    per_run_ok = all(h <= BUDGET_PER_RUN_HOURS for h in per_run_hours.values())
    total_ok = total_hours <= BUDGET_TOTAL_HOURS
    gate_pass = per_run_ok and total_ok
    fallback_ladder = build_fallback_ladder(matrix) if not gate_pass else []
    return GateReport(
        per_run_hours=per_run_hours,
        total_hours=total_hours,
        gate_pass=gate_pass,
        fallback_ladder=fallback_ladder,
    )


def render_markdown(report: GateReport) -> str:
    """Render GateReport as a markdown table string.

    Includes 'Projected' in header and per-env rows.
    """
    lines: list[str] = []

    # Header with 'Projected' keyword
    lines.append("| Method | Suite | Seed | Projected Hours |")
    lines.append("|--------|-------|------|-----------------|")

    # Sort keys for consistent output
    for key in sorted(report.per_run_hours.keys()):
        hours = report.per_run_hours[key]
        # Parse key: method_env_seed
        parts = key.split("_")
        if len(parts) >= 3:
            method = parts[0]
            # Env token is the middle segment (method first, seed last)
            # Reconstruct: method is first, seed is last, env is middle
            seed = parts[-1]
            suite = "_".join(parts[1:-1])
        else:
            method, suite, seed = key, "", ""
        lines.append(f"| {method} | {suite} | {seed} | {hours:.3f} |")

    # Total row
    lines.append(f"| **Total** |  |  | **{report.total_hours:.3f}** |")

    # Gate status
    if report.gate_pass:
        lines.append("")
        lines.append(
            "✅ **GATE PASSED** — Total within budget, all runs within per-run limit."
        )
    else:
        lines.append("")
        lines.append("❌ **GATE FAILED** — **USER SIGN-OFF REQUIRED**")
        lines.append("")
        lines.append("**Fallback Ladder (apply in order):**")
        for i, step in enumerate(report.fallback_ladder, 1):
            lines.append(f"{i}. {step}")

    return "\n".join(lines)
