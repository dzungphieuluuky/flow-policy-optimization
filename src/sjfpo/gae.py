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

__all__ = ["compute_advantages"]


def as_chunk_arrays(rewards, values, next_values, n_action_steps, terminated):
    """Coerce inputs to arrays; expand a scalar stream-end bootstrap."""
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n_action_steps = np.asarray(n_action_steps, dtype=np.int64)
    terminated = np.asarray(terminated, dtype=bool)
    nv = np.asarray(next_values, dtype=np.float64)
    n = values.shape[0]
    if nv.ndim == 0:
        if n > 1:
            nv = np.concatenate([values[1:], nv.reshape(1)])
        else:
            nv = nv.reshape(1)
    return rewards, values, nv, n_action_steps, terminated, n


def validate_chunk_stream(rewards, values, next_values, n_action_steps, terminated, truncated, n):
    """Raise ValueError on shape/count mismatches in the chunk stream."""
    if (
        next_values.shape != (n,)
        or n_action_steps.shape != (n,)
        or terminated.shape != (n,)
        or truncated.shape != (n,)
    ):
        raise ValueError(
            f"shape mismatch: values {values.shape}, next_values {next_values.shape}, "
            f"n_action_steps {n_action_steps.shape}, terminated {terminated.shape}, "
            f"truncated {truncated.shape}"
        )
    if n_action_steps.ndim != 1 or n == 0:
        raise ValueError("need a non-empty 1-D stream of chunks")
    if (n_action_steps <= 0).any():
        raise ValueError("every chunk must have >= 1 executed step")
    if int(n_action_steps.sum()) != rewards.shape[0]:
        raise ValueError(
            f"rewards has {rewards.shape[0]} steps but sum(n_action_steps)={int(n_action_steps.sum())}"
        )


def discount_chunk_rewards(rewards, n_action_steps, gamma, n):
    """R_k = Σᵢ γⁱ·r_{k,i} over executed steps only."""
    offsets = np.concatenate(([0], np.cumsum(n_action_steps)))
    r_chunk = np.empty(n, dtype=np.float64)
    for k in range(n):
        steps = rewards[offsets[k] : offsets[k + 1]]
        discounts = gamma ** np.arange(steps.shape[0], dtype=np.float64)
        r_chunk[k] = float(np.dot(discounts, steps))
    return r_chunk


def backward_lambda_pass(
    r_chunk, values, next_values, n_action_steps, terminated, truncated, gamma, lam, n
):
    """Backward GAE recursion; terminated cuts all, truncated cuts recursion only."""
    advantages = np.empty(n, dtype=np.float64)
    next_advantage = 0.0
    for k in range(n - 1, -1, -1):
        length = int(n_action_steps[k])
        if terminated[k]:
            delta = r_chunk[k] - values[k]
            continuation = 0.0
        elif truncated[k]:
            # Timeout: bootstrap pre-reset V, but the post-reset A_{k+1}
            # belongs to a fresh episode — cut it from the recursion.
            delta = r_chunk[k] + (gamma**length) * next_values[k] - values[k]
            continuation = 0.0
        else:
            delta = r_chunk[k] + (gamma**length) * next_values[k] - values[k]
            continuation = (gamma * lam) ** length
        advantages[k] = delta + continuation * next_advantage
        next_advantage = advantages[k]
    return advantages


def finalize_gae_outputs(advantages, values, normalize):
    """Form returns from raw advantages; optionally standardize the advantages."""
    returns = advantages + values
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


def compute_advantages(
    rewards,
    values,
    next_values,
    n_action_steps,
    terminated,
    gamma: float = 0.99,
    lam: float = 0.95,
    normalize: bool = True,
    truncated=None,
):
    """Semi-MDP chunk-level GAE: (advantages [N], returns [N]) float64.

    next_values [N] is the bootstrap candidate (0 for terminated chunks).
    n_action_steps [N] holds EXECUTED steps per chunk. truncated [N] (or
    None) marks TimeLimit hits. Returns = raw advantages + V.
    """
    rewards, values, next_values, n_action_steps, terminated, n = as_chunk_arrays(
        rewards, values, next_values, n_action_steps, terminated
    )
    if truncated is None:
        truncated = np.zeros(n, dtype=bool)
    else:
        truncated = np.asarray(truncated, dtype=bool)
    validate_chunk_stream(rewards, values, next_values, n_action_steps, terminated, truncated, n)

    r_chunk = discount_chunk_rewards(rewards, n_action_steps, gamma, n)
    advantages = backward_lambda_pass(
        r_chunk, values, next_values, n_action_steps, terminated, truncated, gamma, lam, n
    )
    return finalize_gae_outputs(advantages, values, normalize)
