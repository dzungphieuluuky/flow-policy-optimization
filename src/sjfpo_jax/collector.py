"""Chunked open-loop rollout collector (JAX).

Two collectors mirror the torch API:
- ``collect_chunks`` for a single env.
- ``collect_chunks_vec`` for a vectorized env.

Both call a Python ``policy`` callable (a ``predict_action_chunk`` wrapper) and
write per-chunk ``Transition`` dicts whose values are ``jnp.ndarray``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from lerobot.utils.constants import DONE, OBS_STATE, REWARD, TRUNCATED

from sjfpo_jax.configuration_sjfpo import SJFPOConfig

__all__ = ["Transition", "collect_chunks", "collect_chunks_vec"]

_DEFAULT_CFG: SJFPOConfig | None = None


def _defaults() -> SJFPOConfig:
    global _DEFAULT_CFG
    if _DEFAULT_CFG is None:
        _DEFAULT_CFG = SJFPOConfig()
    return _DEFAULT_CFG


_OBS_DTYPE = jnp.float32
_REWARD_DTYPE = jnp.float32
_INDEX_DTYPE = jnp.int32


Transition = TypedDict(
    "Transition",
    {
        "observation.state": jnp.ndarray,
        "action": jnp.ndarray,
        "next.observation.state": jnp.ndarray,
        "next.reward": jnp.ndarray,
        "next.done": jnp.ndarray,
        "next.truncated": jnp.ndarray,
        "executed_n_action_steps": jnp.ndarray,
        "clip_count": jnp.ndarray,
    },
)


# ---------------------------------------------------------------------------
# Conversions / bounds
# ---------------------------------------------------------------------------


def as_float_array(x) -> jnp.ndarray:
    if isinstance(x, jnp.ndarray):
        return x.astype(_OBS_DTYPE)
    return jnp.asarray(np.asarray(x, dtype=np.float32), dtype=_OBS_DTYPE)


def _bounds_to_array(low, high):
    lo = jnp.asarray(np.asarray(low, dtype=np.float32)).reshape(-1)
    hi = jnp.asarray(np.asarray(high, dtype=np.float32)).reshape(-1)
    if lo.shape != hi.shape:
        raise ValueError(
            f"action_low shape {lo.shape} != action_high shape {hi.shape}"
        )
    rng = hi - lo
    if not bool(jnp.all(rng > 0)):
        raise ValueError("action bounds must satisfy high > low in every dim")
    return lo, rng


def scale_and_clip_chunk(proposed, low, range_):
    """Clamp proposal to bounds and rescale to [-1, 1]; count clipped elements."""
    clip_mask = (proposed < low) | (proposed > low + range_)
    clip_count = int(clip_mask.sum())
    executed = jnp.clip(proposed, low, low + range_)
    scaled = 2.0 * (executed - low) / range_ - 1.0
    return executed, scaled, clip_count


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def build_warmup_window(first_obs, n_obs_steps):
    shared = jnp.asarray(first_obs).astype(_OBS_DTYPE)
    return deque((shared for _ in range(n_obs_steps)), maxlen=n_obs_steps)


def reset_chunk_window(env, n_obs_steps):
    obs, _ = env.reset()
    return build_warmup_window(as_float_array(obs), n_obs_steps)


# ---------------------------------------------------------------------------
# Transition builder
# ---------------------------------------------------------------------------


def finalize_transition(
    state, next_state, scaled, rewards, done_kind,
    clip_count, chunk_size, n_action_steps,
):
    if scaled.shape[0] != chunk_size:
        raise ValueError(
            f"proposed chunk has {scaled.shape[0]} rows, expected {chunk_size}"
        )
    if not 1 <= n_action_steps <= chunk_size:
        raise ValueError(f"n_action_steps={n_action_steps} out of range")
    reward_t = jnp.asarray(np.asarray(rewards, dtype=np.float32))
    n_executed = int(reward_t.shape[0])
    return {
        OBS_STATE: state,
        "action": scaled,
        "next.observation.state": next_state,
        REWARD: reward_t,
        DONE: jnp.asarray(done_kind == "terminated"),
        TRUNCATED: jnp.asarray(done_kind == "truncated"),
        "executed_n_action_steps": jnp.asarray(n_executed, dtype=_INDEX_DTYPE),
        "clip_count": jnp.asarray(clip_count, dtype=_INDEX_DTYPE),
    }


# ---------------------------------------------------------------------------
# Single-env collector
# ---------------------------------------------------------------------------


def execute_open_loop_steps(
    env, executed_np, window, executed_total, total_steps, n_action_steps,
):
    rewards: list[float] = []
    done_kind = "continue"
    for i in range(n_action_steps):
        if executed_total >= total_steps:
            break
        obs, reward, terminated, truncated, _ = env.step(executed_np[i])
        executed_total += 1
        rewards.append(float(reward))
        window.append(as_float_array(obs))
        if terminated:
            done_kind = "terminated"
            break
        if truncated:
            done_kind = "truncated"
            break
    return rewards, done_kind


def propose_and_execute_chunk(
    policy, window, low, range_, env, executed_total, total_steps,
    chunk_size, n_action_steps,
):
    state = jnp.stack(tuple(window), axis=0)
    proposed = as_float_array(policy(state))
    executed, scaled, clip_count = scale_and_clip_chunk(proposed, low, range_)
    executed_np = np.asarray(executed)
    rewards, done_kind = execute_open_loop_steps(
        env, executed_np, window, executed_total, total_steps, n_action_steps
    )
    next_state = jnp.stack(tuple(window), axis=0)
    record = finalize_transition(
        state, next_state, scaled, rewards, done_kind, clip_count,
        chunk_size, n_action_steps,
    )
    return record, done_kind


# ---------------------------------------------------------------------------
# Vector collector
# ---------------------------------------------------------------------------


def collect_chunks_vec(
    envs, policy, total_steps, action_low, action_high,
    *, n_obs_steps=None, n_action_steps, key=None,
):
    cfg = _defaults()
    n_obs_steps = cfg.n_obs_steps if n_obs_steps is None else n_obs_steps
    chunk_size = cfg.action_horizon

    n = int(envs.num_envs)
    target = -(-total_steps // n)
    low, range_ = _bounds_to_array(action_low, action_high)

    if not 1 <= n_action_steps <= chunk_size:
        raise ValueError(f"n_action_steps={n_action_steps} out of range")

    if key is None:
        key = jax.random.PRNGKey(0)
    seed_max = max(1, np.iinfo(np.int32).max - n)
    seed = int(jax.random.randint(key, (), 0, seed_max, dtype=jnp.int32))
    act_dim = int(np.prod(envs.single_action_space.shape))

    obs, _ = envs.reset(seed=[seed + i for i in range(n)])
    windows = [build_warmup_window(as_float_array(o), n_obs_steps) for o in obs]
    cur: list = [None] * n
    pos = [0] * n
    rewards: list[list[float]] = [[] for _ in range(n)]
    prev: list = [None] * n
    prev_done = ["continue"] * n
    executed_total = [0] * n
    awaiting = [False] * n
    records: list[list] = [[] for _ in range(n)]

    while any(e < target for e in executed_total):
        need = [
            i for i in range(n)
            if cur[i] is None and not awaiting[i] and executed_total[i] < target
        ]
        if need:
            states = jnp.stack([jnp.stack(tuple(windows[i]), 0) for i in need])
            proposed = as_float_array(policy(states))
            for k, i in enumerate(need):
                executed, scaled, clip_count = scale_and_clip_chunk(
                    proposed[k], low, range_
                )
                cur[i] = (executed, scaled, clip_count, states[k])
                pos[i] = 0
                rewards[i] = []

        actions_t = jnp.zeros((n, act_dim), dtype=jnp.float32)
        any_active = False
        for i in range(n):
            if cur[i] is not None:
                actions_t = actions_t.at[i].set(cur[i][0][pos[i]])
                any_active = True
        if not any_active:
            break
        actions_np = np.asarray(actions_t)

        obs, rew, term, trunc, _ = envs.step(actions_np)
        term_np = np.asarray(term, dtype=bool)
        trunc_np = np.asarray(trunc, dtype=bool)
        done_np = term_np | trunc_np
        rew_np = np.asarray(rew, dtype=np.float32)

        for i in range(n):
            if awaiting[i]:
                windows[i] = build_warmup_window(as_float_array(obs[i]), n_obs_steps)
                awaiting[i] = False
                continue
            if cur[i] is None:
                continue
            rewards[i].append(float(rew_np[i]))
            windows[i].append(as_float_array(obs[i]))
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
                next_state = jnp.stack(tuple(windows[i]), axis=0)
                record = finalize_transition(
                    replan, next_state, scaled, rewards[i], done_kind,
                    clip_count, chunk_size, n_action_steps,
                )
                executed_total[i] += n_exec
                records[i].append(record)
                prev[i], prev_done[i] = record, done_kind
                cur[i] = None
                if done_np[i]:
                    awaiting[i] = True

    for i in range(n):
        if prev[i] is not None and prev_done[i] == "continue":
            prev[i][TRUNCATED] = jnp.asarray(True)

    return [r for traj in records for r in traj]


# ---------------------------------------------------------------------------
# Single-env collector
# ---------------------------------------------------------------------------


def collect_chunks(
    env, policy, total_steps, action_low, action_high,
    *, n_obs_steps=None, n_action_steps, key=None,
):
    cfg = _defaults()
    n_obs_steps = cfg.n_obs_steps if n_obs_steps is None else n_obs_steps
    chunk_size = cfg.action_horizon
    low, range_ = _bounds_to_array(action_low, action_high)

    if not 1 <= n_action_steps <= chunk_size:
        raise ValueError(f"n_action_steps={n_action_steps} out of range")

    obs, _ = env.reset()
    window = build_warmup_window(as_float_array(obs), n_obs_steps)
    records: list = []
    executed_total = 0
    while executed_total < total_steps:
        record, done_kind = propose_and_execute_chunk(
            policy, window, low, range_, env, executed_total,
            total_steps, chunk_size, n_action_steps,
        )
        executed_total += int(record["executed_n_action_steps"])
        records.append(record)
        if done_kind != "continue":
            window = reset_chunk_window(env, n_obs_steps)
    return records