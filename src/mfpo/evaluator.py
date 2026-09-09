"""MFPO evaluator — deterministic rollouts with temporal metrics.

Plan task 10. Runs deterministic (zero-noise) rollouts and writes
``eval_history.csv`` per run directory. Adapters convert method-specific
policy interfaces to a uniform ``callable(obs) -> action`` signature.

Tensor / array conventions:
    H: horizon / chunk length (env-dependent for MuJoCo)
    Da: action dimension (env-dependent)
    B: batch (unused here; rollouts are single-env sequential)
    CSV schema: [episode, return, length]
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mfpo.dct_utils import dct, idct


def write_eval_csv(results: list[dict[str, Any]], out_dir: str | Path) -> None:
    """Write ``eval_history.csv`` with columns ``[episode, return, length]``."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "eval_history.csv"  # [episode, return, length]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "return", "length"])
        writer.writeheader()  # header
        writer.writerows(results)  # rows


def check_divergence(mean_return: float, baseline: float, consecutive: int) -> tuple[int, bool]:
    """Single divergence guard: count sub-10% evals; diverged after 3 straight.

    Non-positive baselines never diverge (baseline > 0 guard).
    Shared by the MFPO loop (runner) and the skrl loops (baselines); callers
    own state plumbing and logging, this owns only the threshold + count.
    """
    if baseline > 0 and mean_return < 0.10 * baseline:
        consecutive += 1
    else:
        consecutive = 0
    return consecutive, consecutive >= 3


def compute_random_baseline(env: Any, seed: int, context_len: int) -> float:
    """Random-policy baseline return for divergence detection (1.0 if empty/zero)."""
    import gymnasium as gym

    from mfpo.baselines import FrameStackWrapper

    fresh_env = (
        gym.make(env.unwrapped.spec.id)
        if hasattr(env, "unwrapped")
        and hasattr(env.unwrapped, "spec")
        and env.unwrapped.spec is not None
        else type(env.unwrapped)()
    )
    random_env = FrameStackWrapper(fresh_env, context_len=context_len)
    results = Evaluator(random_env, eval_seed=10000 + seed, max_episodes=12).evaluate(
        lambda obs: env.action_space.sample(), out_dir=None
    )
    r_rand = np.mean([r["return"] for r in results]) if results else 1.0
    random_env.close()
    return r_rand or 1.0


class Evaluator:
    """Unified deterministic evaluator for flat-observation (MuJoCo) envs.

    Enforces flat ``Box`` observation contract.

    Attributes:
        env: Gymnasium env with flat ``observation_space``.
        eval_seed: RNG seed for deterministic rollout seeding.
        max_episodes: Episode cap per ``evaluate`` call.
        max_steps: Optional total-step cap across episodes.

    Args:
        env: Gymnasium environment (or ``None`` for unit tests).
        eval_seed: Seed for ``RandomState`` used to seed ``env.reset(seed=)``.
        max_episodes: Number of deterministic rollouts.
        max_steps: Optional total step budget.

    Raises:
        AssertionError: If ``env.observation_space`` is a ``gym.spaces.Dict``.
    """

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

                if isinstance(getattr(env, "observation_space"), gym.spaces.Dict):
                    raise AssertionError("Evaluator expects flat Box obs")
            except Exception as e:
                if "Evaluator expects flat" in str(e):
                    raise
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
        obs, _ = self.env.reset(seed=seed)  # [obs_dim] or dict
        if hasattr(policy, "reset"):
            policy.reset()
        ep_return = 0.0  # scalar
        ep_length = 0  # scalar
        steps_taken = 0  # scalar
        done = False
        while not done:
            action = policy(obs)  # [act_dim]
            obs, reward, terminated, truncated, _info = self.env.step(action)  # step
            ep_return += float(reward)  # scalar
            ep_length += 1
            steps_taken += 1
            done = terminated or truncated
            if step_budget is not None and steps_taken >= step_budget:
                break
        return {"episode": episode, "return": ep_return, "length": ep_length}, steps_taken

    def evaluate(
        self,
        policy: Callable[[np.ndarray], np.ndarray],
        *,
        out_dir: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Run deterministic rollouts, optionally writing ``eval_history.csv``."""
        rng = np.random.RandomState(self.eval_seed)  # deterministic
        results: list[dict[str, Any]] = []
        total_steps = 0  # scalar
        for ep in range(self.max_episodes):  # [0, max_episodes)
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
