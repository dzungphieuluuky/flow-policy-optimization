"""Chunked open-loop rollout collector (plan task 6 + v4 transition dict).

Collects environment interaction in H-step predicted action chunks: the policy is
queried ONCE per chunk (conditioned on the stacked state window) and the first
``n_action_steps`` proposed actions are executed OPEN-LOOP — replanning happens
exactly every n env steps unless the episode ends first (1 <= n <= H enforced).

Transition storage is a plain tensor dict with LeRobot-style keys
(``observation.state`` / ``action`` / ``next.*``). The ``action`` anchor is the
orthonormal DCT-II of the FULL proposed scaled chunk (shape [H, act] always),
so every transition stacks into fixed-shape batches regardless of how many
steps were actually executed. All chunks train (policy and value); the trainer
zero-pads unexecuted tail rows so only executed steps are fitted.

Boundary semantics (plan Scope §Chunk-level semi-MDP GAE, BINDING):
    terminated  -> stop executing immediately, DISCARD all remaining
                   proposed actions of the chunk, next.done=True,
                   V-bootstrap FORBIDDEN (next.truncated=False)
    truncated   -> gymnasium TimeLimit semantics; stop executing,
                   next.done=False, next.truncated=True (bootstrap-eligible)
    neither     -> chunk ran to completion (or the step budget ran out
                   mid-chunk), next.done=False, next.truncated=False

Next-action semantics (chunk-level SARSA bootstrapping): every transition
carries ``next.action`` — the DCT of the chunk proposed at the NEXT replan
([H, act], same pipeline as ``action``). For ``continue`` chunks this is the
next record's ``action`` (filled once it is proposed); for ``truncated``
chunks it is an extra behavior-policy proposal on the pre-reset next-state
window; for ``terminated`` chunks (no bootstrap) and stream-end tails it is
zeros of the same shape.

Action-space handling (plan Scope §Flow space): the policy proposes raw
actions; elements outside ``[action_low, action_high]`` are clipped before
execution and EVERY clipped element is counted per chunk (never silent).
The stored action representation is the orthonormal DCT-II of the SCALED
full proposed chunk, where scaling maps bounds to [-1, 1]:
``scaled = 2·(a − low)/(high − low) − 1``.

Error path: if the env raises at any point, the optional
``on_error_flush`` checkpoint hook is attempted FIRST (its own failure is
swallowed) and the ORIGINAL exception is then re-raised untouched.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import TypedDict

import numpy as np
import torch

from sjfpo.config import SJFPO_CFG
from sjfpo.dct_utils import dct

__all__ = ["Transition", "collect_chunks"]


Transition = TypedDict(
    "Transition",
    {
        "observation.state": torch.Tensor,  # [T, obs] state window at replan time
        "action": torch.Tensor,  # [H, act] DCT of FULL SCALED proposed chunk
        "next.action": torch.Tensor,  # [H, act] DCT of NEXT proposed chunk
        "next.observation.state": torch.Tensor,  # [T, obs] window after execution
        "next.reward": torch.Tensor,  # [n_executed] per executed step
        "next.done": torch.Tensor,  # bool scalar, terminated only
        "next.truncated": torch.Tensor,  # bool scalar, bootstrap-eligible
        "n_action_steps": torch.Tensor,  # int64 scalar, actually executed steps
        "clip_count": torch.Tensor,  # int64 scalar, clipped proposal elements
    },
)


def as_float_tensor(x: object) -> torch.Tensor:
    """Coerce numpy arrays / torch tensors to a detached fp32 tensor.

    Args:
        x: Array-like with shape ``[... ]`` (obs ``[obs_dim]``, action ``[chunk_size, act_dim]``).

    Returns:
        Detached fp32 tensor ``[..., ]`` preserving shape.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32)  # [...] -> [...] fp32 detached
    return torch.as_tensor(np.asarray(x, dtype=np.float32))  # [...] -> [...] fp32


def scale_and_clip_chunk(
    proposed: torch.Tensor, low: torch.Tensor, high: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Clip a proposed chunk to bounds; return (executed, scaled, clip_count)."""
    clip_count = int(((proposed < low) | (proposed > high)).sum())
    executed = torch.clamp(proposed, min=low, max=high)
    scaled = 2.0 * (executed - low) / (high - low) - 1.0
    return executed, scaled, clip_count


def build_warmup_window(
    first_obs: torch.Tensor, n_obs_steps: int
) -> deque[torch.Tensor]:
    """Replicate the first obs EXACTLY n_obs_steps times."""
    return deque((first_obs.clone() for _ in range(n_obs_steps)), maxlen=n_obs_steps)


def reset_chunk_window(env, n_obs_steps: int) -> deque[torch.Tensor]:
    """Reset the env and rebuild the warmup window from the fresh obs."""
    obs, _ = env.reset()
    new_obs = as_float_tensor(obs)
    return deque((new_obs.clone() for _ in range(n_obs_steps)), maxlen=n_obs_steps)


def execute_open_loop_steps(
    env,
    executed: torch.Tensor,
    window: deque[torch.Tensor],
    executed_total: int,
    total_steps: int,
    n_action_steps: int,
) -> tuple[list[float], str]:
    """Execute the first n proposed actions open-loop; return (rewards, done_kind)."""
    rewards: list[float] = []
    done_kind = "continue"
    for i in range(n_action_steps):
        if executed_total >= total_steps:
            break  # step budget exhausted mid-chunk
        obs, reward, terminated, truncated, info = env.step(executed[i].numpy())
        executed_total += 1
        rewards.append(float(reward))
        window.append(as_float_tensor(obs))
        if terminated:  # checked first: termination wins, never bootstrap
            done_kind = "terminated"
            break
        if truncated:
            done_kind = "truncated"
            break
    return rewards, done_kind


def propose_action_coeffs(
    policy: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Propose a chunk on a state window; return DCT of scaled chunk [H, act].

        Same propose → clip → scale → DCT pipeline as the rollout path, without
    executing anything. Used for bootstrap-eligible boundary next-actions."""
    proposed = as_float_tensor(policy(state))
    executed, scaled, clip_count = scale_and_clip_chunk(proposed, low, high)
    if scaled.shape[0] != chunk_size:
        raise ValueError(
            f"proposed chunk has {scaled.shape[0]} rows, "
            f"expected chunk_size={chunk_size}"
        )
    return dct(scaled, dim=0)  # [H, act]


def finalize_transition(
    state: torch.Tensor,
    next_state: torch.Tensor,
    scaled: torch.Tensor,
    rewards: list[float],
    done_kind: str,
    clip_count: int,
    chunk_size: int,
    n_action_steps: int,
    next_action: torch.Tensor | None = None,
) -> Transition:
    """Pack executed-chunk fields into a Transition dict (LeRobot keys)."""
    if scaled.shape[0] != chunk_size:
        raise ValueError(
            f"proposed chunk has {scaled.shape[0]} rows, "
            f"expected chunk_size={chunk_size}"
        )
    if not 1 <= n_action_steps <= chunk_size:
        raise ValueError(
            f"n_action_steps={n_action_steps} must satisfy "
            f"1 <= n <= chunk_size={chunk_size}"
        )
    n_executed = len(rewards)
    coeff = dct(scaled, dim=0)  # DCT of FULL SCALED proposed chunk [H, act]
    if next_action is None:
        next_action = torch.zeros_like(coeff)  # placeholder; filled by chaining
    return {
        "observation.state": state,
        "action": coeff,
        "next.action": next_action,
        "next.observation.state": next_state,
        "next.reward": torch.tensor(rewards, dtype=torch.float32),
        "next.done": torch.tensor(done_kind == "terminated"),
        "next.truncated": torch.tensor(done_kind == "truncated"),
        "n_action_steps": torch.tensor(n_executed, dtype=torch.int64),
        "clip_count": torch.tensor(clip_count, dtype=torch.int64),
    }


def propose_and_execute_chunk(
    policy: Callable[[torch.Tensor], torch.Tensor],
    window: deque[torch.Tensor],
    low: torch.Tensor,
    high: torch.Tensor,
    env,
    executed_total: int,
    total_steps: int,
    chunk_size: int,
    n_action_steps: int,
) -> tuple[Transition, str]:
    """Propose one chunk and execute its first n actions; return (transition, done)."""
    state = torch.stack(tuple(window), dim=0)
    proposed = as_float_tensor(policy(state))
    executed, scaled, clip_count = scale_and_clip_chunk(proposed, low, high)
    rewards, done_kind = execute_open_loop_steps(
        env, executed, window, executed_total, total_steps, n_action_steps
    )
    next_state = torch.stack(tuple(window), dim=0)  # post-execution window, pre-reset
    if done_kind == "truncated":
        # Bootstrap-eligible boundary: behavior next-action on the pre-reset window.
        next_action = propose_action_coeffs(policy, next_state, low, high, chunk_size)
    else:
        next_action = None  # terminated: masked anyway; continue: chained later
    record = finalize_transition(
        state,
        next_state,
        scaled,
        rewards,
        done_kind,
        clip_count,
        chunk_size,
        n_action_steps,
        next_action,
    )
    return record, done_kind


def flush_on_collection_error(
    on_error_flush: Callable[[], None] | None,
) -> None:
    """Attempt the error-flush hook once; hook failures never mask the error."""
    if on_error_flush is not None:
        try:
            on_error_flush()
        except Exception:
            pass  # hook failure must never mask the original error


def collect_chunks(
    env,
    policy: Callable[[torch.Tensor], torch.Tensor],
    total_steps: int,
    action_low: float | np.ndarray,
    action_high: float | np.ndarray,
    *,
    n_obs_steps: int = SJFPO_CFG.n_obs_steps,
    n_action_steps: int,  # required: no config default; training passes chunk_size
    on_error_flush: Callable[[], None] | None = None,
) -> list[Transition]:
    """Roll out ``env`` with open-loop chunk execution for ``total_steps``."""
    low = as_float_tensor(action_low)
    high = as_float_tensor(action_high)
    chunk_size = SJFPO_CFG.chunk_size
    if not 1 <= n_action_steps <= chunk_size:
        raise ValueError(
            f"n_action_steps={n_action_steps} must satisfy "
            f"1 <= n <= chunk_size={chunk_size}"
        )
    try:
        obs, _ = env.reset()
        first_obs = as_float_tensor(obs)
        window = build_warmup_window(first_obs, n_obs_steps)  # warmup replication
        records: list[Transition] = []
        executed_total = 0
        prev: Transition | None = None
        prev_done = "continue"
        while executed_total < total_steps:
            record, done_kind = propose_and_execute_chunk(
                policy,
                window,
                low,
                high,
                env,
                executed_total,
                total_steps,
                chunk_size,
                n_action_steps,
            )
            if prev is not None and prev_done == "continue":
                # The next proposal IS the bootstrap action of the previous chunk.
                prev["next.action"] = record["action"].clone()
            executed_total += int(record["n_action_steps"])
            records.append(record)
            prev, prev_done = record, done_kind
            if done_kind != "continue":
                window = reset_chunk_window(env, n_obs_steps)
        if records and prev_done == "continue":
            # Budget exhausted mid-episode: behavior proposal on the live window.
            records[-1]["next.action"] = propose_action_coeffs(
                policy, torch.stack(tuple(window), dim=0), low, high, chunk_size
            )
        return records
    except Exception:
        flush_on_collection_error(on_error_flush)
        raise
