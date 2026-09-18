"""Chunked open-loop rollout collector.

The policy is queried ONCE per chunk; the first ``n_action_steps`` proposed
actions execute OPEN-LOOP (1 <= n <= H). Transitions are plain tensor dicts
with LeRobot-style keys; ``action`` is the FULL proposed chunk in real scaled
space ([H, action_dim], one row per predicted step; DCT happens downstream
in ``forward``). V(s) bootstraps from the stored next window, so no
next-action proposal is collected.

Boundary semantics: terminated stops at once (no bootstrap); truncated stops
bootstrap-eligible; budget-exhausted tails are marked truncated so GAE never
recurses past the collected stream. Actions are clipped to bounds before
execution and every clipped element is counted. On env error the
``on_error_flush`` hook runs first, then the original exception re-raises.
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

_DEFAULTS = SJFPOConfig()  # bare policy config: single source for defaults

__all__ = ["Transition", "collect_chunks", "collect_chunks_vec"]


Transition = TypedDict(
    "Transition",
    {
        "observation.state": torch.Tensor,  # [T, obs] state window at replan time
        "action": torch.Tensor,  # [H, action_dim] real SCALED FULL proposed chunk
        "next.observation.state": torch.Tensor,  # [T, obs] window after execution
        "next.reward": torch.Tensor,  # [n_executed] per executed step
        "next.done": torch.Tensor,  # bool scalar, terminated only
        "next.truncated": torch.Tensor,  # bool scalar, bootstrap-eligible
        "executed_n_action_steps": torch.Tensor,  # int64 scalar, actually executed steps
        "clip_count": torch.Tensor,  # int64 scalar, clipped proposal elements
    },
)


def as_float_tensor(x: object) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32)
    return to_tensor(np.asarray(x, dtype=np.float32))


def scale_and_clip_chunk(
    proposed: torch.Tensor, low: torch.Tensor, high: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if bool(((high - low) <= 0).any()):
        raise ValueError("action bounds must satisfy high > low in every dim")
    clip_count = int(((proposed < low) | (proposed > high)).sum())
    executed = torch.clamp(proposed, min=low, max=high)
    scaled = 2.0 * (executed - low) / (high - low) - 1.0
    return executed, scaled, clip_count


def build_warmup_window(
    first_obs: torch.Tensor, n_obs_steps: int
) -> deque[torch.Tensor]:
    return deque((first_obs.clone() for _ in range(n_obs_steps)), maxlen=n_obs_steps)


def reset_chunk_window(env, n_obs_steps: int) -> deque[torch.Tensor]:
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
    rewards: list[float] = []
    done_kind = "continue"
    for i in range(n_action_steps):
        if executed_total >= total_steps:
            break
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


def finalize_transition(
    state: torch.Tensor,
    next_state: torch.Tensor,
    scaled: torch.Tensor,
    rewards: list[float],
    done_kind: str,
    clip_count: int,
    chunk_size: int,
    n_action_steps: int,
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
    n_executed = len(rewards)
    return {
        OBS_STATE: state,
        "action": scaled,
        "next.observation.state": next_state,
        REWARD: to_tensor(rewards, dtype=torch.float32),
        DONE: to_tensor(done_kind == "terminated", dtype=torch.bool),
        TRUNCATED: to_tensor(done_kind == "truncated", dtype=torch.bool),
        "executed_n_action_steps": to_tensor(n_executed, dtype=torch.int64),
        "clip_count": to_tensor(clip_count, dtype=torch.int64),
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
    state = torch.stack(tuple(window), dim=0)
    proposed = as_float_tensor(policy(state))
    executed, scaled, clip_count = scale_and_clip_chunk(proposed, low, high)
    rewards, done_kind = execute_open_loop_steps(
        env, executed, window, executed_total, total_steps, n_action_steps
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


def collect_chunks_vec(
    envs,
    policy: Callable[[torch.Tensor], torch.Tensor],
    total_steps: int,
    action_low: float | np.ndarray,
    action_high: float | np.ndarray,
    *,
    n_obs_steps: int = _DEFAULTS.n_obs_steps,
    n_action_steps: int,  # required: no config default; training passes chunk_size
    seed: int | None = None,
    on_error_flush: Callable[[], None] | None = None,
) -> list[Transition]:
    """Vector twin of :func:`collect_chunks` (same records, same chaining).

    ``envs`` is a ``gym.vector.VectorEnv`` (NextStep autoreset): each env runs
    its own chunk loop (own window, prev-chain, budget), but proposals are one
    batched policy call and stepping is one vector call. Per-env budget is
    ``ceil(total_steps / num_envs)``, so the merged batch matches the
    single-env size while collection scales ~linearly with env count.
    ``policy`` takes ``[M, T, obs]`` and returns ``[M, H, action_dim]``.
    """
    n = int(envs.num_envs)
    target = -(-total_steps // n)  # ceil: per-env executed-step budget
    low = as_float_tensor(action_low)
    high = as_float_tensor(action_high)
    chunk_size = _DEFAULTS.action_horizon
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
        windows = [
            build_warmup_window(as_float_tensor(o), n_obs_steps) for o in obs
        ]
        cur: list = [None] * n  # (executed, scaled, clip_count, replan_state)
        pos = [0] * n
        rewards: list[list[float]] = [[] for _ in range(n)]
        prev: list = [None] * n
        prev_done = ["continue"] * n
        executed_total = [0] * n
        awaiting = [False] * n  # done last step: rebuild window from reset obs
        records: list[list[Transition]] = [[] for _ in range(n)]
        while any(e < target for e in executed_total):
            need = [
                i
                for i in range(n)
                if cur[i] is None
                and not awaiting[i]
                and executed_total[i] < target
            ]
            if need:
                states = torch.stack(
                    [torch.stack(tuple(windows[i]), dim=0) for i in need]
                )
                proposed = as_float_tensor(policy(states))
                for k, i in enumerate(need):
                    executed, scaled, clip_count = scale_and_clip_chunk(
                        proposed[k], low, high
                    )
                    cur[i] = (executed, scaled, clip_count, states[k])
                    pos[i] = 0
                    rewards[i] = []
            actions = np.zeros((n, act_dim), dtype=np.float32)
            for i in range(n):
                if cur[i] is not None:
                    actions[i] = cur[i][0][pos[i]].numpy()
            obs, rew, term, trunc, _ = envs.step(actions)
            done = np.asarray(term, dtype=bool) | np.asarray(trunc, dtype=bool)
            for i in range(n):
                if awaiting[i]:
                    # Gap step after done: this obs is the fresh reset obs.
                    windows[i] = build_warmup_window(
                        as_float_tensor(obs[i]), n_obs_steps
                    )
                    awaiting[i] = False
                    continue
                if cur[i] is None:
                    continue  # over-budget env idling on dummy actions
                rewards[i].append(float(rew[i]))
                windows[i].append(as_float_tensor(obs[i]))
                pos[i] += 1
                if done[i] or pos[i] >= n_action_steps:
                    executed, scaled, clip_count, replan = cur[i]
                    n_exec = pos[i]
                    if bool(term[i]):
                        done_kind = "terminated"
                    elif bool(trunc[i]):
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
                    )
                    executed_total[i] += n_exec
                    records[i].append(record)
                    prev[i], prev_done[i] = record, done_kind
                    cur[i] = None
                    if done[i]:
                        awaiting[i] = True
        tails = [i for i in range(n) if prev[i] is not None and prev_done[i] == "continue"]
        for i in tails:
            # Budget exhausted mid-episode: bootstrap the cut, but never
            # recurse into another env's GAE.
            prev[i][TRUNCATED] = torch.tensor(True)
        return [record for trajectory in records for record in trajectory]
    except Exception:
        flush_on_collection_error(on_error_flush)
        raise


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
    low = as_float_tensor(action_low)
    high = as_float_tensor(action_high)
    chunk_size = _DEFAULTS.action_horizon
    if not 1 <= n_action_steps <= chunk_size:
        raise ValueError(
            f"n_action_steps={n_action_steps} must satisfy "
            f"1 <= n <= chunk_size={chunk_size}"
        )
    try:
        obs, _ = env.reset()
        first_obs = as_float_tensor(obs)
        window = build_warmup_window(first_obs, n_obs_steps)
        records: list[Transition] = []
        executed_total = 0
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
            executed_total += int(record["executed_n_action_steps"])
            records.append(record)
            if done_kind != "continue":
                window = reset_chunk_window(env, n_obs_steps)
        return records
    except Exception:
        flush_on_collection_error(on_error_flush)
        raise
