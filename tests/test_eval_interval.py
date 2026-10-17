"""The evaluation interval, and the stopping rules that read it.

The stopping rules used to act on one mean from one seed. The repo records a
same-seed drift, so the rules could fire inside the noise. They now read a
bootstrap interval, and they refuse to fire when the interval straddles the
threshold.

This file pins the interval, the straddle test, the rule behaviour with and
without an interval, and the determinism of the random baseline.
"""

import dataclasses

import numpy as np
import pytest

from myfpo.configuration_sjfpo import SJFPOConfig
from myfpo.evaluator import (
    BOOTSTRAP_HIGH_Q,
    BOOTSTRAP_LOW_Q,
    BOOTSTRAP_RESAMPLES,
    Evaluator,
    bootstrap_interval,
    check_divergence,
    check_regression,
    interval_straddles,
    median_of,
)

NAN = float("nan")


# --- the interval ---------------------------------------------------------


def test_the_interval_brackets_the_mean():
    """A percentile interval must contain the point estimate."""
    rng = np.random.default_rng(0)
    values = rng.normal(loc=100.0, scale=5.0, size=40).tolist()
    mean, low, high = bootstrap_interval(values)
    assert low < mean < high, f"{low} !< {mean} !< {high}"


def test_the_interval_narrows_as_the_episode_count_grows():
    """More episodes must give a tighter bound. Otherwise the interval is noise."""
    rng = np.random.default_rng(1)
    widths = []
    for n in (10, 100, 2000):
        values = rng.normal(loc=50.0, scale=10.0, size=n).tolist()
        _, low, high = bootstrap_interval(values)
        widths.append(high - low)
    assert widths[0] > widths[1] > widths[2], widths


def test_a_single_episode_gives_no_bounds():
    """One episode cannot support an interval. NaN marks it, not zero."""
    mean, low, high = bootstrap_interval([7.0])
    assert mean == 7.0
    assert np.isnan(low) and np.isnan(high), f"{low} {high}"


def test_an_empty_list_gives_nan_everywhere():
    mean, low, high = bootstrap_interval([])
    assert all(np.isnan(v) for v in (mean, low, high))


def test_non_finite_returns_are_dropped_before_resampling():
    """A NaN episode must not poison the interval."""
    values = [10.0, 11.0, 12.0, NAN, 13.0]
    mean, low, high = bootstrap_interval(values)
    clean, clow, chigh = bootstrap_interval([10.0, 11.0, 12.0, 13.0])
    assert (mean, low, high) == (clean, clow, chigh)


def test_the_same_input_gives_the_same_interval():
    """Without a fixed seed two calls would disagree for no reason."""
    values = [3.0, 9.0, 1.0, 7.0, 4.0]
    assert bootstrap_interval(values, seed=3) == bootstrap_interval(values, seed=3)


def test_a_different_seed_gives_a_different_interval():
    """The resampling must actually depend on the seed."""
    values = list(np.random.default_rng(2).normal(size=60))
    a = bootstrap_interval(values, seed=0)
    b = bootstrap_interval(values, seed=1)
    assert a[1] != b[1] or a[2] != b[2], "the seed did not reach the resampler"


def test_the_interval_is_two_sided_at_the_stated_quantiles():
    assert BOOTSTRAP_LOW_Q == 0.025
    assert BOOTSTRAP_HIGH_Q == 0.975
    assert BOOTSTRAP_RESAMPLES == 10_000


def test_the_quantiles_must_be_ordered():
    with pytest.raises(ValueError, match="low_q"):
        bootstrap_interval([1.0, 2.0, 3.0], low_q=0.9, high_q=0.1)


def test_the_resample_count_must_be_positive():
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_interval([1.0, 2.0, 3.0], resamples=0)


# --- the median -----------------------------------------------------------


def test_the_median_resists_one_collapsed_episode():
    """The reason the median is reported beside the mean."""
    values = [100.0, 101.0, 102.0, 103.0, -1000.0]
    assert float(np.mean(values)) < 50.0
    # Sorted: -1000, 100, 101, 102, 103. The middle value is 101.
    assert median_of(values) == 101.0


def test_the_median_of_an_empty_list_is_nan():
    assert np.isnan(median_of([]))


# --- the straddle test ----------------------------------------------------


def test_a_straddling_interval_is_reported_as_one():
    assert interval_straddles(50.0, 60.0, 40.0, 80.0)
    assert interval_straddles(50.0, 60.0, 60.0, 80.0), "the bound counts"
    assert interval_straddles(50.0, 60.0, 40.0, 60.0), "the bound counts"


def test_a_clear_interval_does_not_straddle():
    assert not interval_straddles(50.0, 60.0, 40.0, 59.0)
    assert not interval_straddles(50.0, 60.0, 61.0, 80.0)


def test_an_unknown_interval_does_not_straddle():
    """An absent interval must not silence a real signal."""
    assert not interval_straddles(10.0, 60.0, NAN, NAN)
    assert not interval_straddles(10.0, 60.0, NAN, 80.0)


# --- divergence -----------------------------------------------------------


def test_divergence_still_fires_without_an_interval():
    """The old behaviour must survive when no interval is supplied."""
    streak = 0
    for _ in range(3):
        streak, flag = check_divergence(1.0, 100.0, streak)
    assert streak == 3 and flag


def test_divergence_is_suppressed_while_the_interval_straddles():
    """The new behaviour: noise alone must not end a run."""
    streak = 0
    for _ in range(6):
        streak, flag = check_divergence(
            1.0, 100.0, streak, ci_low=0.5, ci_high=50.0
        )
        assert not flag, "the run stopped on an interval that straddles the floor"
    assert streak == 0, "a straddling eval must reset the streak"


def test_divergence_fires_once_the_interval_clears():
    """It must not become permanently suppressed."""
    streak, flag = check_divergence(1.0, 100.0, 0, ci_low=0.5, ci_high=2.0)
    assert streak == 1 and not flag
    streak, flag = check_divergence(1.0, 100.0, 2, ci_low=0.5, ci_high=2.0)
    assert streak == 3 and flag


def test_divergence_stays_off_without_a_positive_baseline():
    assert check_divergence(1.0, 0.0, 4) == (0, False)


# --- regression -----------------------------------------------------------


def test_regression_still_fires_without_an_interval():
    streak = 0
    for _ in range(5):
        streak, flag = check_regression(
            50.0, 100.0, streak, return_frac=0.7, patience=5
        )
    assert flag


def test_regression_is_suppressed_while_the_interval_straddles():
    streak = 0
    for _ in range(8):
        streak, flag = check_regression(
            50.0, 100.0, streak, return_frac=0.7, patience=5,
            ci_low=40.0, ci_high=80.0,
        )
        assert not flag, "a drop inside the noise must not stop the run"
    assert streak == 0


def test_regression_fires_once_the_interval_clears():
    """It must not become permanently suppressed."""
    streak, flags = 0, []
    for _ in range(4):
        streak, flag = check_regression(
            50.0, 100.0, streak, return_frac=0.7, patience=2,
            ci_low=40.0, ci_high=55.0,
        )
        flags.append(flag)
    assert flags[-1] is True, flags


def test_regression_stays_off_without_a_peak():
    assert check_regression(1.0, None, 3, return_frac=0.7, patience=1) == (0, False)
    assert check_regression(1.0, 0.0, 3, return_frac=0.7, patience=1) == (0, False)


# --- the seeded random baseline ------------------------------------------


def test_the_random_baseline_is_reproducible_once_seeded():
    """The divergence floor must not move between runs."""
    import gymnasium as gym

    from myfpo.train_sjfpo import sample_random_baseline

    def draw() -> float:
        env = gym.make("Hopper-v5")
        try:
            ev = Evaluator(env, eval_seed=5, max_episodes=2)
            return sample_random_baseline(ev, env, max_episodes=3, seed=11)
        finally:
            env.close()

    assert draw() == draw(), "the same seed gave two different floors"


def test_different_seeds_give_different_baselines():
    import gymnasium as gym

    from myfpo.train_sjfpo import sample_random_baseline

    def draw(seed: int) -> float:
        env = gym.make("Hopper-v5")
        try:
            ev = Evaluator(env, eval_seed=5, max_episodes=2)
            return sample_random_baseline(ev, env, max_episodes=3, seed=seed)
        finally:
            env.close()

    assert draw(1) != draw(2), "the seed did not reach the action space"


def test_the_trainer_seeds_the_floor_with_the_run_seed():
    """The call site must pass the seed, or the fix does nothing."""
    import inspect

    from myfpo.train_sjfpo import train_sjfpo_run

    src = inspect.getsource(train_sjfpo_run)
    assert "sample_random_baseline(evaluator, env, seed=seed)" in src, (
        "the trainer must seed the random baseline with the run seed"
    )


# --- the summary ----------------------------------------------------------


def test_summarize_reports_every_key_on_a_real_eval():
    import gymnasium as gym

    ev = Evaluator(gym.make("Hopper-v5"), eval_seed=3, max_episodes=4)
    results = ev.evaluate(lambda obs: np.zeros(3, dtype=np.float32))
    s = ev.summarize(results, seed=1)
    for key in (
        "return_mean",
        "return_median",
        "return_ci_low",
        "return_ci_high",
        "return_n",
    ):
        assert key in s, key
    assert s["return_n"] == 4.0
    assert s["return_ci_low"] < s["return_mean"] < s["return_ci_high"]


def test_summarize_gives_nan_bounds_for_one_episode():
    ev = Evaluator.__new__(Evaluator)
    results = [{"return": 12.0}]
    s = ev.summarize(results, seed=0)
    assert s["return_mean"] == 12.0
    assert np.isnan(s["return_ci_low"])


def test_summarize_on_no_episodes():
    ev = Evaluator.__new__(Evaluator)
    s = ev.summarize([], seed=0)
    assert all(np.isnan(s[k]) for k in ("return_mean", "return_median"))
    assert s["return_n"] == 0.0


def test_the_summary_does_not_touch_the_evaluator_seeds():
    """Summarizing must not shift the seed sequence of the next eval."""
    import gymnasium as gym

    env = gym.make("Hopper-v5")
    try:
        ev = Evaluator(env, eval_seed=17, max_episodes=3)
        first = ev.evaluate(lambda o: np.zeros(3, dtype=np.float32))
        ev.summarize(first, seed=0)
        ev.summarize(first, seed=99)
        second = ev.evaluate(lambda o: np.zeros(3, dtype=np.float32))
        assert [r["return"] for r in first] == [r["return"] for r in second]
    finally:
        env.close()


def test_the_trainer_passes_the_interval_to_both_rules():
    import inspect

    from myfpo.train_sjfpo import train_sjfpo_run

    src = inspect.getsource(train_sjfpo_run)
    assert src.count("ci_low=stop_summary") == 2, (
        "both check_divergence and check_regression must receive an interval"
    )
    assert src.count("ci_high=stop_summary") == 2


def test_the_trainer_logs_the_interval():
    import inspect

    from myfpo.train_sjfpo import train_sjfpo_run

    src = inspect.getsource(train_sjfpo_run)
    for key in ("return_ci_low", "return_ci_high", "return_ci_width", "return_median"):
        assert f'eval/{key}' in src, f"the tracker does not log eval/{key}"


def test_the_return_dict_carries_the_median():
    """The mean stays for continuity. The median is additive."""
    import inspect

    from myfpo.train_sjfpo import train_sjfpo_run

    src = inspect.getsource(train_sjfpo_run)
    assert "reward_median_last10" in src
    assert "reward_mean_last10" in src, "the existing field must not be removed"


def test_the_config_default_does_not_change():
    """Item 5 adds no config field, so the released defaults still hold."""
    cfg = SJFPOConfig()
    assert dataclasses.replace(cfg).eval_max_episodes == 50
    assert cfg.stop_return_frac == 0.7
    assert cfg.stop_patience == 5