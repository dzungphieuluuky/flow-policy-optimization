"""SJFPO evaluator. The module runs deterministic rollouts and writes
``eval_history.csv``.

A rollout calls ``env.reset(seed=...)`` and steps to termination, to truncation,
or to the step budget. The code stores the actions per step in fp32.
``compute_action_jerk`` is the mean squared second-difference. It is a scale-free
smoothness probe.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


CSV_FIELDS = ["episode", "return", "length", "success", "jerk"]

# Bootstrap resamples for the percentile interval. The interval resamples the
# episode returns that the evaluator already holds. It does not resample the
# rollout seeds. So it covers the spread across episodes, and not the spread
# across environments. A run with one episode gets a zero width interval.
BOOTSTRAP_RESAMPLES = 10_000
# The percentile interval reads two tails. The value is 0.975 for a 95 percent
# two-sided interval.
BOOTSTRAP_LOW_Q = 0.025
BOOTSTRAP_HIGH_Q = 0.975


def bootstrap_interval(
    returns: list[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    low_q: float = BOOTSTRAP_LOW_Q,
    high_q: float = BOOTSTRAP_HIGH_Q,
    seed: int = 0,
) -> tuple[float, float, float]:
    """A percentile interval for the mean of ``returns``.

    It returns ``(mean, low, high)``. The value ``nan`` stands for any part it
    cannot compute.

    The function draws ``resamples`` indices with replacement, takes the mean
    of each draw, and reports the two percentiles of those means. It makes no
    assumption about the shape of the return distribution.

    The value ``returns`` empty, or a single value, gives ``nan`` bounds. One
    episode cannot support an interval. So ``nan`` marks the case. The caller
    must treat an unknown bound as unknown, not as a wide one.

    The argument ``seed`` fixes the resampling. The same input gives the same
    interval. Without it, the interval moves between calls and two runs of one
    policy disagree for no reason.
    """
    values = np.asarray([float(r) for r in returns], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if values.size < 2:
        return mean, float("nan"), float("nan")
    if not 0.0 <= low_q < high_q <= 1.0:
        raise ValueError(f"need 0 <= low_q < high_q <= 1, got ({low_q}, {high_q})")
    if resamples < 1:
        raise ValueError(f"resamples={resamples} must be >= 1")
    rng = np.random.default_rng(seed)
    # One [resamples, n] index matrix, then one mean over axis 1. The loop would
    # cost resamples Python iterations.
    idx = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[idx].mean(axis=1)
    low = float(np.percentile(means, 100.0 * low_q))
    high = float(np.percentile(means, 100.0 * high_q))
    return mean, low, high


def median_of(returns: list[float]) -> float:
    """The median of ``returns``. It returns ``nan`` for an empty input.

    The median resists one bad episode. The mean does not. A run whose eval
    window holds one collapsed episode reads a lower mean, and the stopper acts
    on that lower mean.
    """
    values = np.asarray([float(r) for r in returns], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def write_eval_csv(
    results: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    append: bool = False,
) -> Path:
    """Write ``eval_history.csv``, columns ``[episode, return, length, success, jerk]``.

    With ``append=True``, old rows stay and the header is written only if the
    file is new.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "eval_history.csv"
    file_exists = csv_path.is_file()
    mode = "a" if append and file_exists else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not (append and file_exists):
            writer.writeheader()
        writer.writerows(results)
    return csv_path


def compute_action_jerk(actions: np.ndarray) -> float | None:
    """Mean squared second-difference of actions ``[T, D]``. None if T < 3.

    Scales with amplitude squared, so compare within a run, never across action
    scales. Exactly ``0.0`` for constant or constant-velocity motion.
    """
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] < 3:
        return None
    second_diff = a[2:] - 2.0 * a[1:-1] + a[:-2]
    return float(np.square(second_diff).mean())


def compute_seam_jerk(actions: np.ndarray, action_is_pad: np.ndarray) -> float | None:
    """Mean squared second-difference across consecutive chunk boundaries.

    ``actions`` is ``[N, H, D]`` and ``action_is_pad`` is ``[N, H]``. A seam
    needs two executed steps before it and one after. The function returns None
    if no seam qualifies.

    :func:`compute_mean_chunk_jerk` reduces inside each chunk. It therefore
    scores 0 for a policy that steps at every seam. The buffer appends in
    wavefront order, so consecutive rows are often different environments. Thus
    this function over-counts. The result is a conservative upper bound, never a
    false negative.
    """
    a, pad = _jerk_inputs(actions, action_is_pad)
    n = a.shape[0]
    seams: list[float] = []
    for i in range(n - 1):
        tail = np.nonzero(~pad[i])[0]
        head = np.nonzero(~pad[i + 1])[0]
        if tail.size < 2 or head.size < 1:
            continue
        window = np.stack([a[i, tail[-2]], a[i, tail[-1]], a[i + 1, head[0]]])
        seams.append(float(np.square(window[2:] - 2.0 * window[1:-1] + window[:-2]).mean()))
    return float(np.mean(seams)) if seams else None


def _jerk_inputs(actions, action_is_pad) -> tuple[np.ndarray, np.ndarray]:
    """Shared validation for the two jerk functions."""
    a = np.asarray(actions)
    pad = np.asarray(action_is_pad, dtype=bool)
    if a.ndim != 3 or pad.shape != a.shape[:2]:
        raise ValueError(
            f"expected actions [N, H, D] and action_is_pad [N, H]; "
            f"got {a.shape} and {pad.shape}"
        )
    return a, pad


def extract_success(info: dict) -> float | None:
    """Task success flag out of an info dict.

    1.0 or 0.0 if reported, else None when the env reports nothing.
    """
    if not isinstance(info, dict) or info.get("success") is None:
        return None
    return float(bool(info["success"]))


def compute_mean_chunk_jerk(actions: np.ndarray, action_is_pad: np.ndarray) -> float | None:
    """Average :func:`compute_action_jerk` over the executed prefixes of the
    chunks.

    The argument ``actions`` has the shape ``[N, H, D]``. The argument
    ``action_is_pad`` has the shape ``[N, H]``. The function skips every chunk
    with fewer than 3 executed steps. It returns None if no chunk qualifies.

    The function covers the inside of a chunk only. Use
    :func:`compute_seam_jerk` for the boundaries between chunks.
    """
    a, pad = _jerk_inputs(actions, action_is_pad)
    jerks: list[float] = []
    for chunk, mask in zip(a, pad):
        n_exec = int((~mask).sum())
        if n_exec >= 3:
            j = compute_action_jerk(chunk[:n_exec])
            if j is not None:
                jerks.append(j)
    return float(np.mean(jerks)) if jerks else None


def interval_straddles(threshold: float, low: float, high: float) -> bool:
    """Does the interval ``[low, high]`` cross ``threshold``?

    It returns True when ``low`` is at or below ``threshold`` and ``high`` is at
    or above ``threshold``. So the data cannot tell which side of the threshold
    the policy sits on.

    The bounds ``low`` and ``high`` may both be ``nan``. Then the function
    returns False. A caller with no interval keeps the old behaviour, which
    acts on the point estimate alone. An unknown bound must not silence a real
    signal, so it must not invent one either.

    The function took a first argument ``value``, the point estimate the
    interval was built around. The body never read it. A caller could pass an
    interval built around a different value, and the mismatch was invisible. So
    the parameter is gone. A caller must now pass only the bounds it means.
    """
    if not (np.isfinite(low) and np.isfinite(high)):
        return False
    return bool(low <= threshold <= high)


def check_divergence(
    mean_return: float,
    baseline: float,
    consecutive: int,
    *,
    threshold_frac: float = 0.10,
    patience: int = 3,
    ci_low: float = float("nan"),
    ci_high: float = float("nan"),
) -> tuple[int, bool]:
    """Increase a sub-threshold streak. Flag divergence after ``patience`` rows
    in a row.

    The baseline must be strictly positive. If it is not, the check is off and
    the counter resets. The function returns the new streak and whether it
    flagged divergence.

    The values ``ci_low`` and ``ci_high`` are the bounds of the interval around
    ``mean_return``. When they straddle the threshold, the streak resets. The
    evidence does not separate collapse from noise, so the run continues. The
    function needs ``ci_low <= mean_return <= ci_high``. A caller that passes an
    interval from a different value gets a reset it cannot interpret.
    """
    if baseline <= 0:
        return 0, False
    threshold = threshold_frac * baseline
    if mean_return < threshold and not interval_straddles(threshold, ci_low, ci_high):
        consecutive += 1
    else:
        consecutive = 0
    return consecutive, consecutive >= patience


def check_regression(
    mean_return: float,
    best_return: float | None,
    consecutive: int,
    *,
    return_frac: float,
    patience: int,
    ci_low: float = float("nan"),
    ci_high: float = float("nan"),
) -> tuple[int, bool]:
    """Increase a below-peak streak. Flag it once one ``patience`` window
    completes.

    Any eval that matches the peak resets the streak.

    This function catches forgetting. It does not catch collapse.
    The function :func:`check_divergence` catches collapse. A run that peaked
    and then drifted to a third of its peak never reaches a floor. That floor is
    set at a fraction of the random return.

    The values ``ci_low`` and ``ci_high`` behave as in :func:`check_divergence`.
    An interval that straddles the threshold resets the streak. So a run whose
    drop sits inside the noise keeps its best checkpoint and keeps training.
    """
    if best_return is None or best_return <= 0:
        return 0, False
    threshold = return_frac * best_return
    if mean_return >= threshold or interval_straddles(threshold, ci_low, ci_high):
        return 0, False
    consecutive += 1
    if consecutive < patience:
        return consecutive, False
    return 0, True


def render_episode_frames(
    env_id: str,
    policy: Callable[[np.ndarray], np.ndarray],
    seed: int = 0,
    max_steps: int = 1000,
) -> tuple[list[np.ndarray], float]:
    """Roll out one episode, collect rendered frames.

    ``policy`` gets ``reset()`` first if it has one. Returns the uint8 frames
    and the episode return.
    """
    import gymnasium as gym

    env = gym.make(env_id, render_mode="rgb_array")
    try:
        obs, _ = env.reset(seed=seed)
        if hasattr(policy, "reset"):
            policy.reset()
        frames: list[np.ndarray] = []
        ret = 0.0
        for _ in range(max_steps):
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ret += float(reward)
            frames.append(np.asarray(env.render(), dtype=np.uint8))
            if terminated or truncated:
                break
        return frames, ret
    finally:
        env.close()


class Evaluator:
    """Deterministic evaluator for flat-obs (MuJoCo) envs."""

    def __init__(
        self,
        env: Any,
        eval_seed: int = 0,
        max_episodes: int = 12,
        max_steps: int | None = None,

    ):
        if env is None:
            raise ValueError("Evaluator requires a non-None env")
        if hasattr(env, "observation_space"):
            try:
                import gymnasium as gym
            except ImportError as e:
                raise ImportError(
                    "gymnasium is required for Evaluator obs-contract validation"
                ) from e
            if isinstance(env.observation_space, gym.spaces.Dict):
                raise AssertionError("Evaluator expects flat Box obs")
        if max_episodes < 0:
            raise ValueError(f"max_episodes={max_episodes} must be >= 0")
        if max_steps is not None and max_steps < 0:
            raise ValueError(f"max_steps={max_steps} must be >= 0 or None")
        self.env = env
        self.eval_seed = eval_seed
        self.max_episodes = max_episodes
        self.max_steps = max_steps

    def rollout_single_episode(
        self,
        policy: Callable[[np.ndarray], np.ndarray],
        seed: int,
        episode: int,
        step_budget: int | None,
    ) -> tuple[dict[str, Any], int]:
        """One deterministic episode. Returns ``(result, steps_taken)``.

        ``step_budget``, when set, caps the episode and the returned length
        reflects the truncated count. ``policy`` gets ``reset()`` first if it
        has one.
        """
        obs, _ = self.env.reset(seed=seed)
        if hasattr(policy, "reset"):
            policy.reset()
        ep_return = 0.0
        ep_length = 0
        ep_actions: list[np.ndarray] = []
        ep_success: float | None = None
        done = False
        while not done:
            if step_budget is not None and ep_length >= step_budget:
                break
            action = policy(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            ep_return += float(reward)
            ep_length += 1
            done = bool(terminated) or bool(truncated)
            ep_actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
            success = extract_success(info)
            if success is not None:
                ep_success = success
        jerk = compute_action_jerk(np.stack(ep_actions)) if ep_actions else None
        return (
            {
                "episode": episode,
                "return": ep_return,
                "length": ep_length,
                "success": ep_success,
                "jerk": jerk,
            },
            ep_length,
        )

    def evaluate(
        self,
        policy: Callable[[np.ndarray], np.ndarray],
        *,
        out_dir: str | Path | None = None,
        append_csv: bool = False,
    ) -> list[dict[str, Any]]:
        """Deterministic rollouts, optionally writing ``eval_history.csv``.

        Each episode draws a seed from a generator seeded by ``self.eval_seed``,
        so repeat evals of one policy match while successive episodes differ.
        With ``self.max_steps`` set, the loop stops early once that budget is
        spent. One dict per episode, keys in :data:`CSV_FIELDS`.

        The caller reads the interval through :meth:`summarize`. The episodes
        themselves carry no interval, so a stored row means one episode.
        """
        rng = np.random.default_rng(self.eval_seed)
        results: list[dict[str, Any]] = []
        total_steps = 0
        for ep in range(self.max_episodes):
            if self.max_steps is not None and total_steps >= self.max_steps:
                break
            seed = int(rng.integers(0, 2**31))
            budget = None if self.max_steps is None else self.max_steps - total_steps
            result, taken = self.rollout_single_episode(policy, seed, ep, budget)
            results.append(result)
            total_steps += taken
        if out_dir is not None:
            write_eval_csv(results, out_dir, append=append_csv)
        return results

    def summarize(
        self, results: list[dict[str, Any]], seed: int = 0
    ) -> dict[str, float]:
        """The mean, the median, the interval, and the count for one eval.

        It returns the keys ``return_mean``, ``return_median``,
        ``return_ci_low``, ``return_ci_high``, and ``return_n``.

        The bounds come from :func:`bootstrap_interval`. They are ``nan`` when
        fewer than two episodes carry a finite return. So the caller can tell an
        unmeasured bound from a wide one.

        The argument ``seed`` reaches the resampler. A caller that compares two
        policies must pass the same value to both, so the comparison does not
        turn on the resampling.
        """
        returns = [
            float(r["return"]) for r in results if r.get("return") is not None
        ]
        mean, low, high = bootstrap_interval(returns, seed=seed)
        finite = sum(1 for r in returns if np.isfinite(r))
        return {
            "return_mean": mean,
            "return_median": median_of(returns),
            "return_ci_low": low,
            "return_ci_high": high,
            "return_n": float(finite),
        }
