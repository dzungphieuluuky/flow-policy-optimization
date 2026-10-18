"""One eval window, as a function."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["WindowSummary", "summarise_window"]


@dataclass
class WindowSummary:
    """What one eval window found."""

    reported_results: list[dict]
    summaries: dict[str, dict]
    mean_return: float
    stop_return: float
    improved: bool


def summarise_window(
    results_by_sampling: dict[str, list[dict]],
    *,
    reported_name: str,
    stop_name: str,
    eval_seed: int,
    summarise,
    best_return: float | None,
) -> WindowSummary:
    """Summarise every start, then read the two means the runner needs."""
    summaries = {
        name: summarise(rows, seed=eval_seed)
        for name, rows in results_by_sampling.items()
    }
    return WindowSummary(
        reported_results=results_by_sampling[reported_name],
        summaries=summaries,
        mean_return=summaries[reported_name]["return_mean"],
        stop_return=summaries[stop_name]["return_mean"],
        improved=best_return is None or summaries[stop_name]["return_mean"] > best_return,
    )