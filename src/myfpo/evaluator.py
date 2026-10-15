"""SJFPO evaluator: deterministic rollouts, ``eval_history.csv``.

Rollouts call ``env.reset(seed=...)`` and step to termination, truncation, or
the step budget. Actions store per step in fp32. ``compute_action_jerk`` is the mean
squared second-difference, a scale-free smoothness probe.
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
    """Mean squared second-difference *across* consecutive chunk boundaries.

    ``actions`` ``[N, H, D]``, ``action_is_pad`` ``[N, H]``. A seam needs two
    executed steps before it and one after. None if none qualify.

    :func:`compute_mean_chunk_jerk` reduces inside each chunk and so scores 0
    for a policy that steps at every seam. The buffer appends in wavefront
    order, so consecutive rows are often different envs and this over-counts:
    a conservative upper bound, never a false negative.
    """
    a = np.asarray(actions)
    pad = np.asarray(action_is_pad, dtype=bool)
    if a.ndim != 3 or pad.shape != a.shape[:2]:
        raise ValueError(
            f"expected actions [N, H, D] and action_is_pad [N, H]; "
            f"got {a.shape} and {pad.shape}"
        )
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
    a = np.asarray(actions)
    pad = np.asarray(action_is_pad, dtype=bool)
    if a.ndim != 3 or pad.shape != a.shape[:2]:
        raise ValueError(
            f"expected actions [N, H, D] and action_is_pad [N, H]; "
            f"got {a.shape} and {pad.shape}"
        )
    jerks: list[float] = []
    for chunk, mask in zip(a, pad):
        n_exec = int((~mask).sum())
        if n_exec >= 3:
            j = compute_action_jerk(chunk[:n_exec])
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
    """Increase a sub-threshold streak. Flag divergence after ``patience`` rows
    in a row.

    The baseline must be strictly positive. If it is not, the check is off and
    the counter resets. The function returns the new streak and whether it
    flagged divergence.
    """
    if baseline > 0 and mean_return < threshold_frac * baseline:
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
) -> tuple[int, bool]:
    """Increase a below-peak streak. Flag it once one ``patience`` window
    completes.

    Any eval that matches the peak resets the streak.

    This function catches forgetting. It does not catch collapse.
    The function :func:`check_divergence` catches collapse. A run that peaked
    and then drifted to a third of its peak never reaches a floor. That floor is
    set at a fraction of the random return.
    """
    if best_return is None or best_return <= 0:
        return 0, False
    if mean_return >= return_frac * best_return:
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
