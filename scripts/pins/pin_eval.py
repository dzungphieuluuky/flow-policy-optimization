"""Pin the eval statistics and the jerk measures, at ``atol=0``.

What this covers
    ``compute_eval_summary_metrics`` over a fixed set of windows, the action
    jerk, the bootstrap interval and ``summarize_results``. Two groupings
    the legacy pin carried are gone with the functions they measured
    (``compute_seam_jerk`` and ``compute_mean_chunk_jerk`` left the lean pole).

What this does not cover
    Running an environment. Every input here is a number this file supplies.

Usage
    PYTHONPATH=src .venv/bin/python scripts/pins/pin_eval.py --write
    PYTHONPATH=src .venv/bin/python scripts/pins/pin_eval.py

See ``README.md`` for which baseline is tracked and how to regenerate it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shared import compare, flatten

BASELINE = Path(__file__).with_name("eval_old.pt")

# Fixed inputs. The values are uneven on purpose: a constant window makes a
# standard deviation of zero, and a pin that only ever sees zero cannot tell a
# broken reduction from a real one.
RETURNS = [
    -18.4, 12.1, 47.9, 3.2, 88.6, -2.7, 61.3, 5.0, 30.8, -11.9,
    74.2, 19.6, 2.4, 55.1, -7.3, 41.8, 0.9, 63.5, 26.7, -4.2,
]


def _episodes(returns):
    """One episode dict per return, with a varying length and jerk."""
    out = []
    for i, value in enumerate(returns):
        out.append({
            "return": value,
            "length": 100 + 7 * i,
            "success": None if i % 5 else float(i % 2),
            "jerk": None if i % 3 else 0.5 * i - 2.0,
        })
    return out


def _windows():
    """Several windows, so a peak or a streak has something to find."""
    windows = []
    for cut in (0, 5, 10, 15, 20):
        block = RETURNS[:cut] if cut else RETURNS
        windows.append((
            sum(block) / len(block),
            _episodes(block),
            list(block),
        ))
    return windows


def collect() -> dict:
    from myfpo.myfpo import (
        bootstrap_interval,
        compute_action_jerk,
        compute_eval_summary_metrics,
        summarize_results,
    )

    out: dict = {}
    for index, (mean, episodes, all_returns) in enumerate(_windows()):
        for r_rand in (None, 0.0, 25.0):
            tag = f"summary.w{index}.rand{r_rand}"
            out.update(flatten(compute_eval_summary_metrics(
                mean, episodes, all_returns,
                r_rand=r_rand,
                best_return=max(all_returns),
                steps_to_best=all_returns.index(max(all_returns)),
            ), tag + "."))

    for seed in (0, 1, 2):
        for lo_q, hi_q in ((0.025, 0.975), (0.05, 0.95), (0.005, 0.995)):
            tag = f"boot.seed{seed}.q{lo_q}_{hi_q}"
            out.update(flatten(bootstrap_interval(
                RETURNS, resamples=2000, low_q=lo_q, high_q=hi_q, seed=seed,
            ), tag + "."))

    for tag_seed in (0,):
        out.update(flatten(summarize_results(_episodes(RETURNS), seed=tag_seed), "summary_results."))

    # Jerk. One action array [H, D] from one chunk window.
    n, horizon, action_dim = 4, 8, 3
    gen = torch.Generator().manual_seed(11)
    chunk = torch.randn(n, horizon, action_dim, generator=gen)
    out.update(flatten(compute_action_jerk(chunk[0]), "jerk.action."))
    out.update(flatten(compute_action_jerk(chunk[0, :2]), "jerk.short_action."))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true",
        help="Rewrite the baseline from the current code. Use only on purpose.",
    )
    args = parser.parse_args()

    new = collect()
    if args.write:
        torch.save(new, BASELINE)
        print(f"wrote {BASELINE} ({len(new)} keys)")
        return 0

    old = torch.load(BASELINE, weights_only=False) if BASELINE.exists() else None
    return 1 if compare("pin_eval", new, old) else 0


if __name__ == "__main__":
    raise SystemExit(main())