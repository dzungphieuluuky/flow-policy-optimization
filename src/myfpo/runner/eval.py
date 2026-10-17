"""One eval window, as a function.

``train_sjfpo_run`` used to hold the eval window inline: about ninety lines
between the last abort check and the save block. The window swapped the policy
to its EMA weights, ran both flow starts, summarised each, decided whether the
run had set a new peak, wrote the peak to disk, and logged two metric dicts.

The arithmetic in the middle of that block has no side effects and no policy
state. It is :func:`summarise_window`. The rest is orchestration, and it stayed
in the runner.

:func:`summarise_window` is pure, so a test can drive it with two dicts and no
policy. The peak decision belongs here rather than in the runner because it is
the one rule the best checkpoint depends on: the peak reads
``stop_eval_sampling``, not the reported distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["WindowSummary", "summarise_window"]


@dataclass
class WindowSummary:
    """What one eval window found.

    The attribute ``results`` is the reported start's episode rows. The runner
    needs them after the block, for the tracker's eval table.
    """

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
    """Summarise every start, then read the two means the runner needs.

    The argument ``summarise`` is ``evaluator.summarize``. It is passed in so
    this module does not import the evaluator, and so a test can pass a stub.

    Two means come out, and they are not the same mean. ``mean_return`` is the
    reported score. ``stop_return`` is the peak candidate. They differ when
    ``stop_eval_sampling`` differs from ``eval_sampling``, which is the point of
    having both fields.

    The interval in the summary covers the spread across episodes. It does not
    cover the spread across training seeds. So it bounds the noise from one
    window, and not the noise from the whole run.
    """
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