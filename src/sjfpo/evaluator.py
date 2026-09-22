"""SJFPO evaluator — deterministic rollouts, ``eval_history.csv``."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def write_eval_csv(results: list[dict[str, Any]], out_dir: str | Path) -> None:
    """Write ``eval_history.csv`` with columns ``[episode, return, length, success, jerk]``."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "eval_history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["episode", "return", "length", "success", "jerk"]
        )
        writer.writeheader()
        writer.writerows(results)


def action_jerk(actions: np.ndarray) -> float | None:
    """Mean squared second-difference of actions [T, D] (None when T < 3).      Scale-free smoothness probe: constant or consta"""
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] < 3:
        return None
    return float(np.square(a[2:] - 2.0 * a[1:-1] + a[:-2]).mean())


def extract_success(info: dict) -> float | None:
    """Env-reported task success (``info["success"]``), None when absent."""
    if not isinstance(info, dict) or info.get("success") is None:
        return None
    return float(bool(info["success"]))


def mean_chunk_jerk(actions: np.ndarray, is_pad: np.ndarray) -> float | None:
    """Mean ``action_jerk`` over chunks' executed prefixes.      ``actions`` [N, H, D] real-space chunks, ``is_pad`` [N, H] tai"""
    a = np.asarray(actions)
    pad = np.asarray(is_pad, dtype=bool)
    jerks = []
    for chunk, mask in zip(a, pad):
        n_exec = int((~mask).sum())
        if n_exec >= 3:
            j = action_jerk(chunk[:n_exec])
            if j is not None:
                jerks.append(j)
    return float(np.mean(jerks)) if jerks else None


def render_episode_frames(
    env_id: str,
    policy: Callable[[np.ndarray], np.ndarray],
    seed: int = 0,
    max_steps: int = 1000,
) -> tuple[list[np.ndarray], float]:
    """One deterministic episode frames [T,H,W,C] uint8 via rgb_array + return."""
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


def check_divergence(
    mean_return: float, baseline: float, consecutive: int
) -> tuple[int, bool]:
    """Count sub-10% evals; diverged after 3 straight (never on baseline <= 0)."""
    if baseline > 0 and mean_return < 0.10 * baseline:
        consecutive += 1
    else:
        consecutive = 0
    return consecutive, consecutive >= 3


def compute_random_baseline(env: Any, seed: int, n_obs_steps: int) -> float:
    """Random-policy baseline return for divergence detection (1.0 if empty/zero)."""
    import gymnasium as gym

    from sjfpo.baselines import FrameStackWrapper

    fresh_env = (
        gym.make(env.unwrapped.spec.id)
        if hasattr(env, "unwrapped")
        and hasattr(env.unwrapped, "spec")
        and env.unwrapped.spec is not None
        else type(env.unwrapped)()
    )
    random_env = FrameStackWrapper(fresh_env, n_obs_steps=n_obs_steps)
    results = Evaluator(random_env, eval_seed=10000 + seed, max_episodes=12).evaluate(
        lambda obs: env.action_space.sample(), out_dir=None
    )
    r_rand = np.mean([r["return"] for r in results]) if results else 1.0
    random_env.close()
    return r_rand or 1.0


class Evaluator:
    """Deterministic evaluator for flat-observation (MuJoCo) envs."""

    def __init__(
        self,
        env: Any,
        eval_seed: int = 0,
        max_episodes: int = 12,
        max_steps: int | None = None,
    ):
        if env is not None and hasattr(env, "observation_space"):
            try:
                import gymnasium as gym
            except ImportError as e:
                raise ImportError(
                    "gymnasium is required for Evaluator obs-contract validation"
                ) from e
            if isinstance(getattr(env, "observation_space"), gym.spaces.Dict):
                raise AssertionError("Evaluator expects flat Box obs")
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
        """Roll one deterministic episode; return (result, steps taken)."""
        obs, _ = self.env.reset(seed=seed)
        if hasattr(policy, "reset"):
            policy.reset()
        ep_return = 0.0
        ep_length = 0
        steps_taken = 0
        done = False
        ep_actions: list[np.ndarray] = []
        ep_success: float | None = None
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            ep_return += float(reward)
            ep_length += 1
            steps_taken += 1
            done = terminated or truncated
            ep_actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
            success = extract_success(info)
            if success is not None:
                ep_success = success  # last report wins (terminal info)
            if step_budget is not None and steps_taken >= step_budget:
                break
        return {
            "episode": episode,
            "return": ep_return,
            "length": ep_length,
            "success": ep_success,
            "jerk": action_jerk(np.stack(ep_actions)) if ep_actions else None,
        }, steps_taken

    def evaluate(
        self,
        policy: Callable[[np.ndarray], np.ndarray],
        *,
        out_dir: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Run deterministic rollouts, optionally writing ``eval_history.csv``."""
        rng = np.random.RandomState(self.eval_seed)
        results: list[dict[str, Any]] = []
        total_steps = 0
        for ep in range(self.max_episodes):
            if self.max_steps is not None and total_steps >= self.max_steps:
                break
            seed = int(rng.randint(0, 2**31))
            budget = None if self.max_steps is None else self.max_steps - total_steps
            result, taken = self.rollout_single_episode(policy, seed, ep, budget)
            results.append(result)
            total_steps += taken
        if out_dir is not None:
            write_eval_csv(results, out_dir)
        return results
