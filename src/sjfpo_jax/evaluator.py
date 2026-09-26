"""SJFPO evaluator — deterministic rollouts, ``eval_history.csv``.

Rollouts call ``env.reset(seed=...)`` and step until termination/truncation
or the step budget is exhausted. Actions are stored per step in fp32;
``action_jerk`` reports mean squared second-difference as a scale-free
smoothness probe.

"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


CSV_FIELDS = ["episode", "return", "length", "success", "jerk"]


def write_eval_csv(
    results: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    append: bool = False,
) -> Path:
    """Write ``eval_history.csv`` with columns ``[episode, return, length, success, jerk]``.

    When ``append=True``, existing rows are preserved and the header is
    only written if the file doesn't already exist.
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


def action_jerk(actions: np.ndarray) -> float | None:
    """Mean squared second-difference of actions ``[T, D]``; ``None`` when T < 3.

    Scale-free smoothness probe: constant or constant-velocity motion yields
    exactly ``0.0``; larger values indicate jitter between consecutive steps.
    """
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] < 3:
        return None
    second_diff = a[2:] - 2.0 * a[1:-1] + a[:-2]
    return float(np.square(second_diff).mean())


def extract_success(info: dict) -> float | None:
    """Env-reported task success (``info["success"]``); ``None`` when absent."""
    if not isinstance(info, dict) or info.get("success") is None:
        return None
    return float(bool(info["success"]))


def mean_chunk_jerk(actions: np.ndarray, is_pad: np.ndarray) -> float | None:
    """Mean ``action_jerk`` over chunks' executed prefixes.

    ``actions`` is ``[N, H, D]`` real-space chunks and ``is_pad`` is ``[N, H]``
    with True marking padding past the executed prefix. Chunks with fewer
    than 3 executed steps are skipped; returns ``None`` if no chunk
    qualifies.
    """
    a = np.asarray(actions)
    pad = np.asarray(is_pad, dtype=bool)
    if a.ndim != 3 or pad.shape != a.shape[:2]:
        raise ValueError(
            f"expected actions [N, H, D] and is_pad [N, H]; "
            f"got {a.shape} and {pad.shape}"
        )
    jerks: list[float] = []
    for chunk, mask in zip(a, pad):
        n_exec = int((~mask).sum())
        if n_exec >= 3:
            j = action_jerk(chunk[:n_exec])
            if j is not None:
                jerks.append(j)
    return float(np.mean(jerks)) if jerks else None


def check_divergence(
    mean_return: float,
    baseline: float,
    consecutive: int,
    *,
    threshold_frac: float = 0.10,
    patience: int = 3,
) -> tuple[int, bool]:
    """Increment a sub-threshold counter; flag divergence after ``patience`` in a row.

    The baseline must be strictly positive for the check to apply — a
    non-positive baseline disables divergence detection and resets the
    counter.
    """
    if baseline > 0 and mean_return < threshold_frac * baseline:
        consecutive += 1
    else:
        consecutive = 0
    return consecutive, consecutive >= patience




class Evaluator:
    """Deterministic evaluator for flat-observation (MuJoCo) environments."""

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
        """Roll one deterministic episode; return ``(result, steps_taken)``.

        ``step_budget`` (when set) caps the episode at that many steps;
        the returned length reflects the truncated count.
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
                ep_success = success  # last report wins (terminal info)
        jerk = action_jerk(np.stack(ep_actions)) if ep_actions else None
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
        """Run deterministic rollouts, optionally writing ``eval_history.csv``."""
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
