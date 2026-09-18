"""Chunk-level semi-MDP GAE + returns (pure numpy fp64).

Each executed chunk is one decision epoch. For chunk k with length L_k:
    R_k     = Σᵢ γⁱ·r_{k,i}            (executed steps only)
    δ_k     = R_k + γ^{L_k}·V_{k+1}·(1−terminated_k) − V_k
    A_k     = δ_k + (γλ)^{L_k}·(1−terminated_k)·A_{k+1}
    G_k     = A_k + V_k
terminated cuts bootstrap and recursion; truncated (TimeLimit) bootstraps δ
but cuts the recursion into the post-reset episode. Advantages are centered
afterwards; returns are not.
"""

from __future__ import annotations

import numpy as np

__all__ = ["compute_advantages", "compute_returns", "normalize_advantages"]


def as_chunk_arrays(rewards, values, next_values, executed_n_action_steps, terminated):
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    executed_n_action_steps = np.asarray(executed_n_action_steps, dtype=np.int64)
    terminated = np.asarray(terminated, dtype=bool)
    nv = np.asarray(next_values, dtype=np.float64)
    n = values.shape[0]
    if nv.ndim == 0:
        if n > 1:
            nv = np.concatenate([values[1:], nv.reshape(1)])
        else:
            nv = nv.reshape(1)
    return rewards, values, nv, executed_n_action_steps, terminated, n


def validate_chunk_stream(
    rewards, values, next_values, executed_n_action_steps, terminated, truncated, n
):
    if (
        next_values.shape != (n,)
        or executed_n_action_steps.shape != (n,)
        or terminated.shape != (n,)
        or truncated.shape != (n,)
    ):
        raise ValueError(
            f"shape mismatch: values {values.shape}, next_values {next_values.shape}, "
            f"executed_n_action_steps {executed_n_action_steps.shape}, terminated {terminated.shape}, "
            f"truncated {truncated.shape}"
        )
    if executed_n_action_steps.ndim != 1 or n == 0:
        raise ValueError("need a non-empty 1-D stream of chunks")
    if (executed_n_action_steps <= 0).any():
        raise ValueError("every chunk must have >= 1 executed step")
    if int(executed_n_action_steps.sum()) != rewards.shape[0]:
        raise ValueError(
            f"rewards has {rewards.shape[0]} steps but sum(executed_n_action_steps)={int(executed_n_action_steps.sum())}"
        )


def compute_returns(advantages, values):
    return np.asarray(advantages, dtype=np.float64) + np.asarray(
        values, dtype=np.float64
    )


def normalize_advantages(advantages, clamp: float | None = 100.0):
    advantages = np.asarray(advantages, dtype=np.float64)
    normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    if clamp is not None:
        # OFF advantage_clamp (100, 100): insurance against pathological skew.
        normalized = np.clip(normalized, -clamp, clamp)
    return normalized


def compute_advantages(
    rewards,
    values,
    next_values,
    executed_n_action_steps,
    terminated,
    gamma: float = 0.99,
    lam: float = 0.95,
    normalize: bool = True,
    truncated=None,
):
    """Semi-MDP chunk-level GAE: (advantages [N], returns [N]) float64.

    next_values [N] is the bootstrap candidate (0 for terminated chunks).
    executed_n_action_steps [N] holds EXECUTED steps per chunk. truncated [N] (or
    None) marks TimeLimit hits. Returns = raw advantages + V.
    """
    rewards, values, next_values, executed_n_action_steps, terminated, n = as_chunk_arrays(
        rewards, values, next_values, executed_n_action_steps, terminated
    )
    if truncated is None:
        truncated = np.zeros(n, dtype=bool)
    else:
        truncated = np.asarray(truncated, dtype=bool)
    validate_chunk_stream(
        rewards, values, next_values, executed_n_action_steps, terminated, truncated, n
    )

    offsets = np.concatenate(([0], np.cumsum(executed_n_action_steps)))
    # ------- intra-chunk discounted rewards -------
    r_chunk = np.empty(n, dtype=np.float64)
    for k in range(n):
        steps = rewards[offsets[k] : offsets[k + 1]]
        r_chunk[k] = float(
            np.dot(gamma ** np.arange(steps.shape[0], dtype=np.float64), steps)
        )

    # ------- backward GAE recursion over chunks -------
    advantages = np.empty(n, dtype=np.float64)
    next_advantage = 0.0
    for k in range(n - 1, -1, -1):
        length = int(executed_n_action_steps[k])
        if terminated[k]:
            delta, continuation = r_chunk[k] - values[k], 0.0
        else:
            delta = r_chunk[k] + (gamma**length) * next_values[k] - values[k]
            continuation = 0.0 if truncated[k] else (gamma * lam) ** length
        advantages[k] = delta + continuation * next_advantage
        next_advantage = advantages[k]

    returns = compute_returns(advantages, values)
    if normalize:
        advantages = normalize_advantages(advantages)
    return advantages, returns
