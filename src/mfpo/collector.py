"""Chunked open-loop rollout collector (plan task 6).

Collects environment interaction in H-step action chunks: the policy is
queried ONCE per chunk (conditioned on the stacked context window) and its
proposed chunk is executed OPEN-LOOP for up to ``chunk_len`` env steps —
replanning happens exactly every 8 env steps unless the episode ends first.

Boundary semantics (plan Scope §Chunk-level semi-MDP GAE, BINDING):
    terminated  -> stop executing immediately, DISCARD all remaining
                   proposed actions of the chunk, done='terminated',
                   V-bootstrap FORBIDDEN (bootstrap=False)
    truncated   -> gymnasium TimeLimit semantics; stop executing,
                   done='truncated', bootstrap-ELIGIBLE (bootstrap=True)
    neither     -> chunk ran to completion (or the step budget ran out
                   mid-chunk), done='continue', bootstrap=False

Action-space handling (plan Scope §Flow space): the policy proposes raw
actions; elements outside ``[action_low, action_high]`` are clipped before
execution and EVERY clipped element is counted per chunk (never silent).
The stored action representation is the orthonormal DCT-II of the SCALED
executed prefix, where scaling maps bounds to [-1, 1]:
``scaled = 2·(a − low)/(high − low) − 1``.

Error path: if the env raises at any point, the optional
``on_error_flush`` checkpoint hook is attempted FIRST (its own failure is
swallowed) and the ORIGINAL exception is then re-raised untouched.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import einx
import torch

from mfpo.config import MFPO_CFG
from mfpo.dct_utils import dct

__all__ = ["ChunkRecord", "collect_chunks"]


@dataclass(frozen=True)
class ChunkRecord:
    """One collected decision-time chunk (semi-MDP step)."""

    context: torch.Tensor  # [context_len, obs_dim] window at replan time
    action_dct: torch.Tensor  # [exec_len, act_dim] DCT of SCALED executed prefix
    rewards: list[float]  # reward per EXECUTED env step
    done: str  # 'continue' | 'terminated' | 'truncated'
    bootstrap: bool  # True iff done == 'truncated' (V(s_last) eligible)
    clip_count: int  # proposed elements outside action bounds this chunk
    exec_len: int  # number of executed env steps in this chunk


def as_float_tensor(x: object) -> torch.Tensor:
    """Coerce numpy arrays / torch tensors to a detached fp32 tensor.

    Args:
        x: Array-like with shape ``[... ]`` (obs ``[obs_dim]``, action ``[chunk_len, act_dim]``).

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
    first_obs: torch.Tensor, context_len: int
) -> deque[torch.Tensor]:
    """Replicate the first obs EXACTLY context_len times."""
    return deque((first_obs.clone() for _ in range(context_len)), maxlen=context_len)


def reset_chunk_window(env, context_len: int) -> deque[torch.Tensor]:
    """Reset the env and rebuild the warmup window from the fresh obs."""
    obs, _ = env.reset()
    new_obs = as_float_tensor(obs)
    return deque((new_obs.clone() for _ in range(context_len)), maxlen=context_len)


def execute_open_loop_steps(
    env,
    executed: torch.Tensor,
    window: deque[torch.Tensor],
    executed_total: int,
    total_steps: int,
) -> tuple[list[float], str]:
    """Execute one chunk open-loop; return (rewards, done_kind)."""
    rewards: list[float] = []
    done_kind = "continue"
    for i in range(MFPO_CFG.chunk_len):
        if executed_total >= total_steps:
            break  # step budget exhausted mid-chunk
        obs, reward, terminated, truncated, _info = env.step(executed[i].numpy())
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


def finalize_chunk_record(
    context: torch.Tensor,
    scaled: torch.Tensor,
    rewards: list[float],
    done_kind: str,
    clip_count: int,
) -> ChunkRecord:
    """Pack executed-chunk fields into a ChunkRecord."""
    exec_len = len(rewards)
    coeff = dct(scaled[:exec_len], dim=0)  # DCT of SCALED executed prefix
    return ChunkRecord(
        context=context,
        action_dct=coeff,
        rewards=rewards,
        done=done_kind,
        bootstrap=done_kind == "truncated",
        clip_count=clip_count,
        exec_len=exec_len,
    )


def propose_and_execute_chunk(
    policy: Callable[[torch.Tensor], torch.Tensor],
    window: deque[torch.Tensor],
    low: torch.Tensor,
    high: torch.Tensor,
    env,
    executed_total: int,
    total_steps: int,
) -> tuple[ChunkRecord, str]:
    """Propose one chunk and execute it open-loop; return (record, done_kind)."""
    context = torch.stack(tuple(window), dim=0)
    proposed = as_float_tensor(policy(context))
    executed, scaled, clip_count = scale_and_clip_chunk(proposed, low, high)
    rewards, done_kind = execute_open_loop_steps(
        env, executed, window, executed_total, total_steps
    )
    record = finalize_chunk_record(context, scaled, rewards, done_kind, clip_count)
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
    context_len: int = MFPO_CFG.context_len,
    on_error_flush: Callable[[], None] | None = None,
) -> list[ChunkRecord]:
    """Roll out ``env`` with open-loop chunk execution for ``total_steps``."""
    low = as_float_tensor(action_low)
    high = as_float_tensor(action_high)
    try:
        obs, _ = env.reset()
        first_obs = as_float_tensor(obs)
        window = build_warmup_window(first_obs, context_len)  # warmup replication
        records: list[ChunkRecord] = []
        executed_total = 0
        while executed_total < total_steps:
            record, done_kind = propose_and_execute_chunk(
                policy, window, low, high, env, executed_total, total_steps
            )
            executed_total += record.exec_len
            records.append(record)
            if done_kind != "continue":
                window = reset_chunk_window(env, context_len)
        return records
    except Exception:
        flush_on_collection_error(on_error_flush)
        raise
