"""Chunk-level semi-MDP GAE + returns (plan task 7).

BINDING semantics (plan Scope §Chunk-level semi-MDP GAE, Metis majors #8/#11):
each executed chunk is ONE semi-MDP decision epoch (Schulman GAE adapted to
semi-MDP decision epochs). For chunk k with executed length L_k:

    R_k     = sum_{i=0}^{L_k-1} gamma^i * r_{k,i}     (ONLY executed steps;
                                                       internally discounted gamma^i)
    delta_k = R_k + gamma^{L_k} * V_{k+1} * (1 - terminated_k) - V_k
    A_k     = delta_k + (gamma*lam)^{L_k} * (1 - terminated_k) * A_{k+1}
    G_k     = A_k + V_k

- gamma compounds ONCE per chunk transition: the next-chunk bootstrap is
  multiplied by gamma^{L_k} (gamma^8 for full chunks), and the lambda
  recursion carries (gamma*lam)^{L_k}. Variable executed lengths (mid-chunk
  death / mid-chunk TimeLimit) are handled by exponentiating with THIS
  chunk's length.
- terminated chunks NEVER bootstrap from V(next obs) and cut the recursion.
- truncated (TimeLimit) chunks DO bootstrap from V(next obs); the stream
  continues, so only their own flag masks nothing (they are not terminated).
- advantages are centered afterwards (mean 0); returns are NOT recentered.

Per-chunk V values are consumed as black-box scalars: any history/frame-stack
conditioning inside the critic is legitimate information-state dependence and
irrelevant to the exactness of this computation.

Pure numpy fp64; no torch dependency (the trainer casts as needed).
"""

from __future__ import annotations

import numpy as np

__all__ = ["chunk_gae"]


def as_chunk_arrays(rewards, values, next_values, n_action_steps, terminated):
    """Coerce stream inputs to numpy arrays; expand scalar stream-end bootstrap."""
    rewards = np.asarray(rewards, dtype=np.float64)  # [S] flat stream order
    values = np.asarray(values, dtype=np.float64)  # [N] V(s_k)
    n_action_steps = np.asarray(n_action_steps, dtype=np.int64)  # [N] executed steps
    terminated = np.asarray(terminated, dtype=bool)  # [N] terminal flags
    nv = np.asarray(next_values, dtype=np.float64)  # [N] or scalar bootstrap
    n = values.shape[0]  # scalar stream length
    if nv.ndim == 0:  # scalar: values[1:] + final_value for the last chunk
        if n > 1:  # multi-chunk stream
            nv = np.concatenate([values[1:], nv.reshape(1)])  # [N]
        else:  # single-chunk stream
            nv = nv.reshape(1)  # [1]
    return rewards, values, nv, n_action_steps, terminated, n


def validate_chunk_stream(rewards, values, next_values, n_action_steps, terminated, n):
    """Raise ValueError on shape/count mismatches in the chunk stream."""
    if (
        next_values.shape != (n,)
        or n_action_steps.shape != (n,)
        or terminated.shape != (n,)
    ):
        raise ValueError(
            f"shape mismatch: values {values.shape}, next_values {next_values.shape}, "
            f"n_action_steps {n_action_steps.shape}, terminated {terminated.shape}"
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
    """Compute R_k = sum_i gamma^i * r_{k,i} over executed steps only."""
    offsets = np.concatenate(([0], np.cumsum(n_action_steps)))  # [N+1] stream offsets
    r_chunk = np.empty(n, dtype=np.float64)  # [N] intradiscounted returns
    for k in range(n):  # decision epochs in stream order
        steps = rewards[offsets[k] : offsets[k + 1]]  # [L_k] executed rewards
        discounts = gamma ** np.arange(steps.shape[0], dtype=np.float64)  # [L_k]
        r_chunk[k] = float(np.dot(discounts, steps))  # scalar R_k
    return r_chunk  # [N]


def backward_lambda_pass(
    r_chunk, values, next_values, n_action_steps, terminated, gamma, lam, n
):
    """Backward GAE recursion over decision epochs; terminated cuts bootstrap."""
    advantages = np.empty(n, dtype=np.float64)  # [N] chunk advantages
    next_advantage = 0.0  # A_{k+1} scalar; zero past the end of the stream
    for k in range(n - 1, -1, -1):  # reversed decision epochs
        length = int(n_action_steps[k])  # scalar executed steps this chunk
        if terminated[k]:  # terminal epoch: no bootstrap, recursion cut
            delta = r_chunk[k] - values[k]  # scalar TD residual
            continuation = 0.0  # scalar recursion weight
        else:  # continuing epoch: bootstrap with gamma^L_k
            delta = r_chunk[k] + (gamma**length) * next_values[k] - values[k]  # scalar
            continuation = (gamma * lam) ** length  # scalar recursion weight
        advantages[k] = delta + continuation * next_advantage  # scalar A_k
        next_advantage = advantages[k]  # scalar carry to previous epoch
    return advantages  # [N]


def finalize_gae_outputs(advantages, values, center):
    """Form returns from raw advantages; optionally center the advantages."""
    returns = advantages + values
    if center:
        advantages = advantages - advantages.mean()
    return advantages, returns


def chunk_gae(
    rewards,
    values,
    next_values,
    n_action_steps,
    terminated,
    gamma: float = 0.99,
    lam: float = 0.95,
    center: bool = True,
):
    """Semi-MDP chunk-level GAE over a stream of executed chunks.

    Args:
        rewards: flat [sum(n_action_steps)] per-step rewards in stream order.
        values: [N] V(s_k) at each chunk start.
        next_values: [N] V(obs after chunk k ends) — bootstrap candidate;
            multiplied by 0 for terminated chunks.
        n_action_steps: [N] EXECUTED steps per chunk (variable allowed; a chunk
            that dies mid-way stores only its executed length).
        terminated: [N] bool — True when chunk k ended the episode by
            termination (no bootstrap, recursion cut). Truncated/ongoing
            chunks are False and bootstrap normally.
        gamma: discount (config: 0.99).
        lam: GAE lambda (config: 0.95).
        center: subtract the mean of the advantages afterwards.

    Returns:
        (advantages [N], returns [N]) float64; returns = raw advantages + V
        (computed before centering).
    """
    rewards, values, next_values, n_action_steps, terminated, n = as_chunk_arrays(
        rewards, values, next_values, n_action_steps, terminated
    )
    validate_chunk_stream(rewards, values, next_values, n_action_steps, terminated, n)
    r_chunk = discount_chunk_rewards(rewards, n_action_steps, gamma, n)
    advantages = backward_lambda_pass(
        r_chunk, values, next_values, n_action_steps, terminated, gamma, lam, n
    )
    return finalize_gae_outputs(advantages, values, center)
