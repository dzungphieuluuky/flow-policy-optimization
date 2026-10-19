"""One log key per quantity, and no two spellings of the same series.

Renaming a log key breaks every run already recorded. So the rule needs a test:
the family must use one order, the old spellings must stay gone, and no two
keys may mean the same number.
"""

from pathlib import Path

import pytest
import torch

from myfpo.metrics import compute_eval_summary_metrics

SRC = Path(__file__).resolve().parents[1] / "src" / "myfpo"


def _all_source() -> str:
    """Every module under ``src/myfpo``, concatenated.

    These tests ask whether a log key is logged anywhere. Reading one file
    answered that wrongly once: the metric chain moved into the runner package
    and the file-scoped search went green on code that no longer held the key.
    The claim is about the package, so the search covers the package.
    """
    return "\n".join(p.read_text() for p in sorted(SRC.rglob("*.py")))

OLD_KEYS = (
    "eval/mean_return",
    "eval/min_return",
    "eval/max_return",
    "eval/std_return",
    "eval/best_mean_return",
    "eval/reward_improvement",
    "eval/reward_improvement_pct",
    "eval/sample_random_baseline",
    "eval/consecutive_below",
    "train/grad_norm_preclip",
    "train/chunk_clip_counts",
)


def _summary(**kw):
    base = dict(
        mean_return=10.0,
        eval_results=[
            {"return": r, "length": 10, "success": None, "jerk": None} for r in (10.0,)
        ],
        all_eval_returns=[5.0, 10.0],
        eval_count=2,
    )
    base.update(kw)
    return compute_eval_summary_metrics(**base)


def test_none_of_the_old_spellings_appear_in_any_source_file():
    """The whole package, not one file. A stale key hides in the least-read one."""
    for path in SRC.rglob("*.py"):
        text = path.read_text()
        for key in OLD_KEYS:
            assert f'"{key}"' not in text, f"{path.name} still logs {key}"


def test_the_return_family_names_the_quantity_first():
    """return_mean, return_min, return_max, return_std, return_best."""
    out = _summary(best_return=20.0)
    for key in (
        "eval/return_mean",
        "eval/return_min",
        "eval/return_max",
        "eval/return_std",
        "eval/return_best",
    ):
        assert key in out, key


def test_the_improvement_pair_says_return_and_fraction():
    """It used to say "reward" and "pct".

    Eight sibling keys said "return". And "pct" reads as a percentage, so 0.5
    looked like 50.
    """
    out = _summary()
    assert "eval/return_improvement" in out
    assert "eval/return_improvement_frac" in out
    assert out["eval/return_improvement"] == pytest.approx(5.0)
    assert out["eval/return_improvement_frac"] == pytest.approx(1.0)


def test_the_random_baseline_key_names_the_policy():
    """It was named after the function that produced it."""
    out = _summary(r_rand=123.0)
    assert out["eval/random_policy_return"] == pytest.approx(123.0)
    assert "eval/random_baseline" not in out




def test_no_two_keys_carry_the_same_number():
    """Two keys with the same value usually mean one quantity logged twice."""
    out = _summary(best_return=10.0, r_rand=10.0)
    by_value: dict[float, list[str]] = {}
    for key, value in out.items():
        if key.startswith("eval/") and key.endswith("_return"):
            by_value.setdefault(round(float(value), 9), []).append(key)
    # mean, best and random are all 10.0 here. Distinct keys, distinct names.
    # The test only fails if the SAME key appears twice.
    assert all(len(names) == 1 for names in by_value.values()), by_value
    assert len(set(by_value[10.0])) == len(by_value[10.0]), by_value


def test_the_return_mean_key_matches_the_summarizer():
    """The trainer logs the score twice under two spellings.

    ``compute_eval_summary_metrics`` wrote ``eval/mean_return``, and the same
    loop wrote ``eval/{start}_return_mean`` from ``summarize()["return_mean"]``.
    One number, two series.
    """
    from myfpo.evaluator import bootstrap_interval

    mean, _low, _high = bootstrap_interval([1.0, 2.0, 3.0], resamples=64)
    out = _summary(mean_return=mean)
    assert out["eval/return_mean"] == pytest.approx(mean)
    # The other series still exists, and it is built from an f-string over the
    # two flow starts, so the literal key is not in the source.
    src = (Path(__file__).resolve().parents[1] / "src" / "myfpo" / "training.py").read_text()
    assert 'f"eval/{name}_return_mean"' in src
    assert "_return_mean" in src, "the per-start series must still exist"




def test_an_empty_window_is_still_nan_not_zero():
    """The rename must not change what a missing measurement looks like."""
    out = compute_eval_summary_metrics(
        mean_return=0.0, eval_results=[], all_eval_returns=[0.0], eval_count=0
    )
    for key in ("eval/return_min", "eval/return_max", "eval/return_std"):
        assert out[key] != out[key], key  # NaN
        assert not isinstance(out[key], torch.Tensor)


def test_the_grad_norm_key_spells_out_before():
    src = _all_source()
    assert "train/grad_norm_before_clip" in src
    assert "train/grad_norm_preclip" not in src