"""Chunked open-loop rollout collector.

Performance notes
-----------------
* Actions are transferred host-side **once per step** (batched over envs) rather
  than once per env per step. The per-step `.numpy()` sync is the single
  biggest cost in the vec collector.
* Observation windows and rewards stay in **fp32** end-to-end; only parameters
  and stored policy outputs use ``STORAGE_DTYPE``. bf16 rewards corrupt the
  GAE target.
* Bounds are validated once per collector call and reduced to
  ``(low, range)``; ``scale_and_clip_chunk`` no longer re-validates.
* Warmup windows share one clone (deque holds N references to the same
  tensor) instead of cloning ``n_obs_steps`` copies; safe because the window
  is only read, never mutated in place.
* Tail truncation writes into the existing reward/done tensors rather than
  constructing a fresh scalar per tail.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import TypedDict

import numpy as np
import torch

from lerobot.processor.converters import to_tensor
from lerobot.utils.constants import DONE, OBS_STATE, REWARD, TRUNCATED

from sjfpo.configuration_sjfpo import SJFPOConfig

__all__ = ["Transition", "collect_chunks", "collect_chunks_vec"]


# Lazily-constructed config (import cost only on first use).
_DEFAULT_CFG: SJFPOConfig | None = None


def _defaults() -> SJFPOConfig:
    global _DEFAULT_CFG
    if _DEFAULT_CFG is None:
        _DEFAULT_CFG = SJFPOConfig()
    return _DEFAULT_CFG


# Dtypes: observations and rewards are fp32 end-to-end. Only policy outputs
# and stored params use STORAGE_DTYPE (bf16 on Blackwell).
_OBS_DTYPE = torch.float32
_REWARD_DTYPE = torch.float32
_INDEX_DTYPE = torch.int64


Transition = TypedDict(
    "Transition",
    {
        "observation.state": torch.Tensor,  # [T, obs] fp32
        "action": torch.Tensor,  # [H, action_dim] scaled
        "next.observation.state": torch.Tensor,  # [T, obs] fp32
        "next.reward": torch.Tensor,  # [n_exec] fp32
        "next.done": torch.Tensor,  # bool scalar
        "next.truncated": torch.Tensor,  # bool scalar
        "executed_n_action_steps": torch.Tensor,  # int64 scalar
        "clip_count": torch.Tensor,  # int64 scalar
    },
)


class _StepTransition(dict):
    """Transition dict with post-step windows stored outside public keys."""

    def __init__(self, record: Transition, step_next_states: torch.Tensor):
        super().__init__(record)
        # Keep the public eight-key Transition contract while retaining the
        # state window after every executed action for step-level GAE.
        self.step_next_states = step_next_states


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def as_float_tensor(x: object) -> torch.Tensor:
    """Coerce to an fp32 CPU/GPU tensor without forcing a device hop.

    A ``torch.Tensor`` input keeps its device (no implicit ``.cpu()``);
    numpy / Python inputs are lifted to a fresh CPU fp32 tensor.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().to(_OBS_DTYPE)
    return torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=_OBS_DTYPE)


def _bounds_to_tensor(
    low: float | np.ndarray | torch.Tensor,
    high: float | np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coerce, validate, and return ``(low, range = high - low)`` on fp32."""
    lo = torch.as_tensor(
        np.asarray(low, dtype=np.float32), dtype=torch.float32
    ).reshape(-1)
    hi = torch.as_tensor(
        np.asarray(high, dtype=np.float32), dtype=torch.float32
    ).reshape(-1)
    if lo.shape != hi.shape:
        raise ValueError(
            f"action_low shape {tuple(lo.shape)} != action_high shape {tuple(hi.shape)}"
        )
    rng = hi - lo
    if not bool((rng > 0).all()):
        raise ValueError("action bounds must satisfy high > low in every dim")
    return lo, rng


def scale_and_clip_chunk(
    proposed: torch.Tensor, low: torch.Tensor, range_: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Clamp a proposal to bounds and rescale to [-1, 1]; count clipped elements.

    ``range_`` is the precomputed ``high - low`` (see :func:`_bounds_to_tensor`).
    """
    clip_mask = (proposed < low) | (proposed > low + range_)
    clip_count = int(clip_mask.sum())
    executed = torch.clamp(proposed, min=low, max=low + range_)
    scaled = 2.0 * (executed - low) / range_ - 1.0
    return executed, scaled, clip_count


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------


def build_warmup_window(
    first_obs: torch.Tensor, n_obs_steps: int
) -> deque[torch.Tensor]:
    """Deque of ``n_obs_steps`` identical frames.

    Uses a single clone referenced N times — safe because the window is only
    read, never mutated in place. Appends past the initial fill replace the
    oldest reference, not the shared tensor.
    """
    shared = first_obs.detach().clone()
    return deque((shared for _ in range(n_obs_steps)), maxlen=n_obs_steps)


def reset_chunk_window(env, n_obs_steps: int) -> deque[torch.Tensor]:
    obs, _ = env.reset()
    return build_warmup_window(as_float_tensor(obs), n_obs_steps)


# ---------------------------------------------------------------------------
# Transition builder
# ---------------------------------------------------------------------------


def _bool_tensor(v: bool) -> torch.Tensor:
    return torch.tensor(v, dtype=torch.bool)


def _int_tensor(v: int) -> torch.Tensor:
    return torch.tensor(v, dtype=_INDEX_DTYPE)


def finalize_transition(
    state: torch.Tensor,
    next_state: torch.Tensor,
    scaled: torch.Tensor,
    rewards: list[float] | np.ndarray,
    done_kind: str,
    clip_count: int,
    chunk_size: int,
    n_action_steps: int,
    step_next_states: torch.Tensor,
) -> Transition:
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
    # Rewards land in fp32 — GAE targets must not be bf16-truncated.
    reward_t = torch.as_tensor(
        np.asarray(rewards, dtype=np.float32), dtype=_REWARD_DTYPE
    )
    n_executed = int(reward_t.shape[0])
    if step_next_states.shape != (n_executed, *next_state.shape):
        raise ValueError(
            f"step_next_states shape {tuple(step_next_states.shape)} != "
            f"{(n_executed, *next_state.shape)}"
        )
    record: Transition = {
        OBS_STATE: state,
        "action": scaled,
        "next.observation.state": next_state,
        REWARD: reward_t,
        DONE: _bool_tensor(done_kind == "terminated"),
        TRUNCATED: _bool_tensor(done_kind == "truncated"),
        "executed_n_action_steps": _int_tensor(n_executed),
        "clip_count": _int_tensor(clip_count),
    }
    return _StepTransition(record, step_next_states)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Single-env collector
# ---------------------------------------------------------------------------


def execute_open_loop_steps(
    env,
    executed_np: np.ndarray,  # [H, act_dim] on host — no per-step sync
    window: deque[torch.Tensor],
    executed_total: int,
    total_steps: int,
    n_action_steps: int,
) -> tuple[list[float], str, torch.Tensor]:
    rewards: list[float] = []
    step_next_states: list[torch.Tensor] = []
    done_kind = "continue"
    for i in range(n_action_steps):
        if executed_total >= total_steps:
            break
        obs, reward, terminated, truncated, _info = env.step(executed_np[i])
        executed_total += 1
        rewards.append(float(reward))
        window.append(as_float_tensor(obs))
        step_next_states.append(torch.stack(tuple(window), dim=0))
        if terminated:
            done_kind = "terminated"
            break
        if truncated:
            done_kind = "truncated"
            break
    return rewards, done_kind, torch.stack(step_next_states, dim=0)


def propose_and_execute_chunk(
    policy: Callable[[torch.Tensor], torch.Tensor],
    window: deque[torch.Tensor],
    low: torch.Tensor,
    range_: torch.Tensor,
    env,
    executed_total: int,
    total_steps: int,
    chunk_size: int,
    n_action_steps: int,
) -> tuple[Transition, str]:
    state = torch.stack(tuple(window), dim=0)
    proposed = as_float_tensor(policy(state))
    executed, scaled, clip_count = scale_and_clip_chunk(proposed, low, range_)
    # Single host transfer for the whole chunk, then index on CPU.
    executed_np = executed.detach().cpu().numpy()
    rewards, done_kind, step_next_states = execute_open_loop_steps(
        env, executed_np, window, executed_total, total_steps, n_action_steps
    )
    next_state = torch.stack(tuple(window), dim=0)
    record = finalize_transition(
        state,
        next_state,
        scaled,
        rewards,
        done_kind,
        clip_count,
        chunk_size,
        n_action_steps,
        step_next_states,
    )
    return record, done_kind


def flush_on_collection_error(
    on_error_flush: Callable[[], None] | None,
) -> None:
    if on_error_flush is not None:
        try:
            on_error_flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Vector collector
# ---------------------------------------------------------------------------


def collect_chunks_vec(
    envs,
    policy: Callable[[torch.Tensor], torch.Tensor],
    total_steps: int,
    action_low: float | np.ndarray,
    action_high: float | np.ndarray,
    *,
    n_obs_steps: int | None = None,
    n_action_steps: int,
    seed: int | None = None,
    on_error_flush: Callable[[], None] | None = None,
) -> list[Transition]:
    """Vector twin of :func:`collect_chunks` — same records, same chaining."""
    cfg = _defaults()
    n_obs_steps = cfg.n_obs_steps if n_obs_steps is None else n_obs_steps
    chunk_size = cfg.action_horizon

    n = int(envs.num_envs)
    target = -(-total_steps // n)  # ceil per-env budget

    low, range_ = _bounds_to_tensor(action_low, action_high)

    if not 1 <= n_action_steps <= chunk_size:
        raise ValueError(
            f"n_action_steps={n_action_steps} must satisfy "
            f"1 <= n <= chunk_size={chunk_size}"
        )
    if seed is None:
        seed = int(torch.randint(0, 2**31, (1,)).item())
    act_dim = int(np.asarray(envs.single_action_space.shape).prod())

    try:
        obs, _ = envs.reset(seed=[seed + i for i in range(n)])
        windows = [build_warmup_window(as_float_tensor(o), n_obs_steps) for o in obs]
        cur: list = [None] * n  # per-env in-flight chunk state
        pos = [0] * n
        rewards: list[list[float]] = [[] for _ in range(n)]
        step_states: list[list[torch.Tensor]] = [[] for _ in range(n)]
        prev: list[Transition | None] = [None] * n
        prev_done = ["continue"] * n
        executed_total = [0] * n
        awaiting = [False] * n
        records: list[list[Transition]] = [[] for _ in range(n)]

        while any(e < target for e in executed_total):
            need = [
                i
                for i in range(n)
                if cur[i] is None and not awaiting[i] and executed_total[i] < target
            ]
            if need:
                states = torch.stack(
                    [torch.stack(tuple(windows[i]), dim=0) for i in need]
                )
                proposed = as_float_tensor(policy(states))
                for k, i in enumerate(need):
                    executed, scaled, clip_count = scale_and_clip_chunk(
                        proposed[k], low, range_
                    )
                    cur[i] = (executed, scaled, clip_count, states[k])
                    pos[i] = 0
                    rewards[i] = []
                    step_states[i] = []

            # Gather all active actions into a tensor, then ONE host transfer.
            actions_t = torch.zeros((n, act_dim), dtype=torch.float32)
            any_active = False
            for i in range(n):
                if cur[i] is not None:
                    actions_t[i] = cur[i][0][pos[i]]
                    any_active = True
            if not any_active:
                # All envs over budget or awaiting: nothing left to step.
                break
            actions_np = actions_t.cpu().numpy()

            obs, rew, term, trunc, _ = envs.step(actions_np)
            term_np = np.asarray(term, dtype=bool)
            trunc_np = np.asarray(trunc, dtype=bool)
            done_np = term_np | trunc_np
            rew_np = np.asarray(rew, dtype=np.float32)

            for i in range(n):
                if awaiting[i]:
                    windows[i] = build_warmup_window(
                        as_float_tensor(obs[i]), n_obs_steps
                    )
                    awaiting[i] = False
                    continue
                if cur[i] is None:
                    continue
                rewards[i].append(float(rew_np[i]))
                windows[i].append(as_float_tensor(obs[i]))
                step_states[i].append(torch.stack(tuple(windows[i]), dim=0))
                pos[i] += 1
                if done_np[i] or pos[i] >= n_action_steps:
                    _executed, scaled, clip_count, replan = cur[i]
                    n_exec = pos[i]
                    if term_np[i]:
                        done_kind = "terminated"
                    elif trunc_np[i]:
                        done_kind = "truncated"
                    else:
                        done_kind = "continue"
                    next_state = torch.stack(tuple(windows[i]), dim=0)
                    record = finalize_transition(
                        replan,
                        next_state,
                        scaled,
                        rewards[i],
                        done_kind,
                        clip_count,
                        chunk_size,
                        n_action_steps,
                        torch.stack(step_states[i], dim=0),
                    )
                    executed_total[i] += n_exec
                    records[i].append(record)
                    prev[i], prev_done[i] = record, done_kind
                    cur[i] = None
                    if done_np[i]:
                        awaiting[i] = True

        # Budget-exhausted tails: mark as truncated so GAE bootstraps the cut.
        for i in range(n):
            if prev[i] is not None and prev_done[i] == "continue":
                prev[i][TRUNCATED].fill_(True)

        return [r for traj in records for r in traj]

    except Exception:
        flush_on_collection_error(on_error_flush)
        raise


# ---------------------------------------------------------------------------
# Single-env collector
# ---------------------------------------------------------------------------


def collect_chunks(
    env,
    policy: Callable[[torch.Tensor], torch.Tensor],
    total_steps: int,
    action_low: float | np.ndarray,
    action_high: float | np.ndarray,
    *,
    n_obs_steps: int | None = None,
    n_action_steps: int,
    on_error_flush: Callable[[], None] | None = None,
) -> list[Transition]:
    cfg = _defaults()
    n_obs_steps = cfg.n_obs_steps if n_obs_steps is None else n_obs_steps
    chunk_size = cfg.action_horizon

    low, range_ = _bounds_to_tensor(action_low, action_high)

    if not 1 <= n_action_steps <= chunk_size:
        raise ValueError(
            f"n_action_steps={n_action_steps} must satisfy "
            f"1 <= n <= chunk_size={chunk_size}"
        )
    try:
        obs, _ = env.reset()
        window = build_warmup_window(as_float_tensor(obs), n_obs_steps)
        records: list[Transition] = []
        executed_total = 0
        while executed_total < total_steps:
            record, done_kind = propose_and_execute_chunk(
                policy,
                window,
                low,
                range_,
                env,
                executed_total,
                total_steps,
                chunk_size,
                n_action_steps,
            )
            executed_total += int(record["executed_n_action_steps"])
            records.append(record)
            if done_kind != "continue":
                window = reset_chunk_window(env, n_obs_steps)
        return records
    except Exception:
        flush_on_collection_error(on_error_flush)
        raise
