"""myfpo evaluator. The module runs deterministic rollouts and writes a per-eval
CSV, and computes the jerk diagnostics that expose chunk-boundary artifacts.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


CSV_FIELDS = ["episode", "return", "length", "success", "jerk"]

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_LOW_Q = 0.025
BOOTSTRAP_HIGH_Q = 0.975


def bootstrap_interval(
    returns: list[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    low_q: float = BOOTSTRAP_LOW_Q,
    high_q: float = BOOTSTRAP_HIGH_Q,
    seed: int = 0,
) -> tuple[float, float, float]:
    """A percentile interval for the mean of ``returns``."""
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
    idx = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[idx].mean(axis=1)
    low = float(np.percentile(means, 100.0 * low_q))
    high = float(np.percentile(means, 100.0 * high_q))
    return mean, low, high


def median_of(returns: list[float]) -> float:
    """The median of ``returns``. It returns ``nan`` for an empty input."""
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
    """Write ``eval_history.csv``, columns ``[episode, return, length, success, jerk]``."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "eval_history.csv"
    file_exists = csv_path.is_file()
    mode = "a" if append and file_exists else "w"
    with csv_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not (append and file_exists):
            writer.writeheader()
        writer.writerows(results)
    return csv_path


def compute_action_jerk(actions: np.ndarray) -> float | None:
    """Mean squared second-difference of actions ``[T, D]``. None if T < 3."""
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] < 3:
        return None
    second_diff = a[2:] - 2.0 * a[1:-1] + a[:-2]
    return float(np.square(second_diff).mean())


def compute_seam_jerk(actions: np.ndarray, action_is_pad: np.ndarray) -> float | None:
    """Mean squared second-difference across consecutive chunk boundaries."""
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
    """Task success flag out of an info dict."""
    if not isinstance(info, dict) or info.get("success") is None:
        return None
    return float(bool(info["success"]))


def compute_mean_chunk_jerk(actions: np.ndarray, action_is_pad: np.ndarray) -> float | None:
    """Average :func:`compute_action_jerk` over the executed prefixes of the chunks.
    Returns ``None`` when no chunk executed at least three steps.
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
        """One deterministic episode. Returns ``(result, steps_taken)``."""
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
        """Deterministic rollouts, optionally writing ``eval_history.csv``."""
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
        """The mean, the median, the interval, and the count for one eval."""
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
