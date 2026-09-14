"""Chunked open-loop rollout collector.

The policy is queried ONCE per chunk; the first ``n_action_steps`` proposed
actions execute OPEN-LOOP (1 <= n <= H). Transitions are plain tensor dicts
with LeRobot-style keys; ``action``/``next.action`` are the FULL proposed
chunks in real scaled space ([H, action_dim], one row per predicted step;
DCT happens downstream in ``forward``).

Boundary semantics: terminated stops at once (no bootstrap, next.action
zeros); truncated stops with bootstrap-eligible next.action (fresh behavior
proposal on the pre-reset window); continue chains next.action from the next
proposal (budget-exhausted tails get a live-window proposal). Actions are
clipped to bounds before execution and every clipped element is counted.
On env error the ``on_error_flush`` hook runs first, then the original
exception re-raises.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import TypedDict

import numpy as np
import torch

from sjfpo.configuration_sjfpo import SJFPOConfig

_DEFAULTS = SJFPOConfig()  # bare policy config: single source for defaults

__all__ = ["Transition", "collect_chunks", "bootstrap_action_lengths"]


def bootstrap_action_lengths(chunks, chunk_size: int) -> list[int]:
    """Executed-step mask lengths for bootstrap next-actions.

    Interior-continue chunks use the NEXT chunk's executed length; truncated
    and tail proposals stay full; terminated entries are unused (full).
    """
    n = len(chunks)
    out: list[int] = []
    for i, c in enumerate(chunks):
        if not bool(c["next.done"]) and not bool(c["next.truncated"]) and i + 1 < n:
            out.append(int(chunks[i + 1]["n_action_steps"]))
        else:
            out.append(chunk_size)
    return out


Transition = TypedDict(
    "Transition",
    {
        "observation.state": torch.Tensor,  # [T, obs] state window at replan time
        "action": torch.Tensor,  # [H, action_dim] real SCALED FULL proposed chunk
        "next.action": torch.Tensor,  # [H, action_dim] real SCALED NEXT proposed chunk
        "next.observation.state": torch.Tensor,  # [T, obs] window after execution
        "next.reward": torch.Tensor,  # [n_executed] per executed step
        "next.done": torch.Tensor,  # bool scalar, terminated only
        "next.truncated": torch.Tensor,  # bool scalar, bootstrap-eligible
        "n_action_steps": torch.Tensor,  # int64 scalar, actually executed steps
        "clip_count": torch.Tensor,  # int64 scalar, clipped proposal elements
    },
)


def as_float_tensor(x: object) -> torch.Tensor:
    """Numpy/torch input -> detached fp32 tensor, shape preserved."""
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32)  # [...] -> [...] fp32 detached
    return torch.as_tensor(np.asarray(x, dtype=np.float32))  # [...] -> [...] fp32


def scale_and_clip_chunk(
    proposed: torch.Tensor, low: torch.Tensor, high: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Clip to bounds; return (executed, scaled-to-[-1,1], clip_count)."""
    clip_count = int(((proposed < low) | (proposed > high)).sum())
    executed = torch.clamp(proposed, min=low, max=high)
    scaled = 2.0 * (executed - low) / (high - low) - 1.0
    return executed, scaled, clip_count


def build_warmup_window(
    first_obs: torch.Tensor, n_obs_steps: int
) -> deque[torch.Tensor]:
    """First obs replicated exactly n_obs_steps times."""
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


def propose_action_chunk(
    policy: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Propose without executing; return the real scaled chunk (boundary next-actions)."""
    proposed = as_float_tensor(policy(state))
    executed, scaled, clip_count = scale_and_clip_chunk(proposed, low, high)
    if scaled.shape[0] != chunk_size:
        raise ValueError(
            f"proposed chunk has {scaled.shape[0]} rows, "
            f"expected chunk_size={chunk_size}"
        )
    return scaled


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
    if next_action is None:
        next_action = torch.zeros_like(scaled)  # placeholder; filled by chaining
    return {
        "observation.state": state,
        "action": scaled,
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
    next_state = torch.stack(tuple(window), dim=0)
    if done_kind == "truncated":
        next_action = propose_action_chunk(policy, next_state, low, high, chunk_size)
    else:
        next_action = None
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
            pass


def collect_chunks(
    env,
    policy: Callable[[torch.Tensor], torch.Tensor],
    total_steps: int,
    action_low: float | np.ndarray,
    action_high: float | np.ndarray,
    *,
    n_obs_steps: int = _DEFAULTS.n_obs_steps,
    n_action_steps: int,  # required: no config default; training passes chunk_size
    on_error_flush: Callable[[], None] | None = None,
) -> list[Transition]:
    """Roll out ``env`` with open-loop chunk execution for ``total_steps``."""
    low = as_float_tensor(action_low)
    high = as_float_tensor(action_high)
    chunk_size = _DEFAULTS.horizon
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
            records[-1]["next.action"] = propose_action_chunk(
                policy, torch.stack(tuple(window), dim=0), low, high, chunk_size
            )
        return records
    except Exception:
        flush_on_collection_error(on_error_flush)
        raise
