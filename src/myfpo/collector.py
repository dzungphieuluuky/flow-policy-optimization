"""Chunked open-loop rollout collector."""

from __future__ import annotations

import contextlib
import dataclasses
from typing import NamedTuple

from collections.abc import Callable

import numpy as np
import torch

from myfpo.processor import squash_to_env_bounds

__all__ = ["RolloutBuffer", "collect_chunks_vec"]


OBS_DTYPE = torch.float32
REWARD_DTYPE = torch.float32
INDEX_DTYPE = torch.int64


class InFlightChunk(NamedTuple):
    """What one env is executing right now, while it holds a chunk.

    ``actions_raw`` is the chunk the flow produced, in the policy's own action
    space. The env received its squash.
    """

    actions_raw: torch.Tensor
    start_state: torch.Tensor
    step_limit: int


class RolloutBuffer:
    """Fixed-capacity CPU storage for chunk and executed-step rollout data."""

    def __init__(
        self,
        *,
        max_chunks: int,
        max_steps: int,
        obs_steps: int,
        obs_dim: int,
        action_horizon: int,
        action_dim: int,
        chunk_capacity: int | None = None,
        device: torch.device | str = "cpu",
        pin_memory: bool = False,
    ):
        if max_chunks < 1 or max_steps < 1:
            raise ValueError("max_chunks and max_steps must be positive")
        if obs_steps < 1 or obs_dim < 1 or action_horizon < 1 or action_dim < 1:
            raise ValueError("observation and action dimensions must be positive")
        self.device = torch.device(device)
        self.max_chunks = max_chunks
        self.max_steps = max_steps
        self.obs_steps = obs_steps
        self.obs_dim = obs_dim
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.pin_memory = pin_memory
        if chunk_capacity is None:
            chunk_capacity = max_chunks
        if chunk_capacity < 1:
            raise ValueError("chunk_capacity must be positive")
        self.chunk_capacity = chunk_capacity

        self.step_frames = self.zeros_for((max_steps, obs_dim), OBS_DTYPE)
        self.step_rewards = self.zeros_for((max_steps,), REWARD_DTYPE)

        self.chunk_state = self.zeros_for(
            (chunk_capacity, obs_steps, obs_dim), OBS_DTYPE
        )
        self.chunk_next_state = self.zeros_for(
            (chunk_capacity, obs_steps, obs_dim), OBS_DTYPE
        )
        self.chunk_actions = self.zeros_for(
            (chunk_capacity, action_horizon, action_dim), OBS_DTYPE
        )
        self.chunk_starts = self.zeros_for((chunk_capacity,), INDEX_DTYPE)
        self.chunk_lengths = self.zeros_for((chunk_capacity,), INDEX_DTYPE)
        self.chunk_terminated = self.zeros_for((chunk_capacity,), torch.bool)
        self.chunk_truncated = self.zeros_for((chunk_capacity,), torch.bool)
        self.chunk_env_id = self.zeros_for((chunk_capacity,), torch.int16)
        self.n_chunks = 0
        self.n_steps = 0

    PIN_MIN_BYTES = 1 << 20



    def zeros_for(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """One zero tensor in this buffer's layout, dtype and device."""
        nbytes = dtype.itemsize
        for dim in shape:
            nbytes *= dim
        return torch.zeros(
            *shape,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory and nbytes >= self.PIN_MIN_BYTES,
        )

    def shape_key(self) -> tuple[int, ...]:
        """The buffer layout signature: the six shape fields, and nothing else."""
        return (
            self.max_chunks,
            self.max_steps,
            self.obs_steps,
            self.obs_dim,
            self.action_horizon,
            self.action_dim,
        )

    def reset(self) -> None:
        """Reset cursors, keep pre-allocated storage."""
        self.n_chunks = 0
        self.n_steps = 0

    def add_chunk(
        self,
        *,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        rewards: list[float] | np.ndarray | torch.Tensor,
        step_frames: torch.Tensor,
        terminated: bool,
        truncated: bool,
        env_id: int = 0,
    ) -> int:
        """Append one executed chunk."""
        if self.n_chunks >= self.max_chunks:
            raise RuntimeError("RolloutBuffer chunk capacity exceeded")
        reward_t = torch.as_tensor(rewards, dtype=REWARD_DTYPE, device=self.device).reshape(-1)
        n_exec = int(reward_t.numel())
        if n_exec < 1 or self.n_steps + n_exec > self.max_steps:
            raise RuntimeError("RolloutBuffer step capacity exceeded")
        if state.shape != (self.obs_steps, self.obs_dim):
            raise ValueError(f"state shape {tuple(state.shape)} does not fit buffer")
        if next_state.shape != (self.obs_steps, self.obs_dim):
            raise ValueError(
                f"next_state shape {tuple(next_state.shape)} does not fit buffer"
            )
        if action.shape != (self.action_horizon, self.action_dim):
            raise ValueError(f"action shape {tuple(action.shape)} does not fit buffer")
        if tuple(step_frames.shape) != (n_exec, self.obs_dim):
            raise ValueError(
                f"step_frames shape {tuple(step_frames.shape)} does not fit buffer"
            )

        chunk_index = self.n_chunks
        offset = self.n_steps
        self.chunk_state[chunk_index].copy_(state)
        self.chunk_next_state[chunk_index].copy_(next_state)
        self.step_frames[offset : offset + n_exec].copy_(step_frames)
        self.step_rewards[offset : offset + n_exec].copy_(reward_t)
        self.chunk_actions[chunk_index].copy_(action)
        self.chunk_starts[chunk_index] = offset
        self.chunk_lengths[chunk_index] = n_exec
        self.chunk_terminated[chunk_index] = terminated
        self.chunk_truncated[chunk_index] = truncated
        self.chunk_env_id[chunk_index] = int(env_id)
        self.n_chunks += 1
        self.n_steps += n_exec
        return chunk_index

    def copy_to(self, other: "RolloutBuffer", *, non_blocking: bool = False) -> None:
        """Copy the active prefix into a compatible buffer, e.g. onto a GPU."""
        other.n_chunks = self.n_chunks
        other.n_steps = self.n_steps
        names = (
            "step_frames",
            "step_rewards",
            "chunk_state",
            "chunk_next_state",
            "chunk_actions",
            "chunk_starts",
            "chunk_lengths",
            "chunk_terminated",
            "chunk_truncated",
            "chunk_env_id",
        )
        def prefix(buffer, name):
            n = self.n_steps if name.startswith("step_") else self.n_chunks
            return getattr(buffer, name)[:n]

        torch._foreach_copy_(
            [prefix(other, name) for name in names],
            [prefix(self, name) for name in names],
            non_blocking=non_blocking,
        )

    def active(self) -> dict[str, torch.Tensor]:
        """Contiguous views of the populated prefix."""
        n_chunks, n_steps = self.n_chunks, self.n_steps
        return {
            "step_frames": self.step_frames[:n_steps],
            "step_rewards": self.step_rewards[:n_steps],
            "chunk_state": self.chunk_state[:n_chunks],
            "chunk_next_state": self.chunk_next_state[:n_chunks],
            "chunk_actions": self.chunk_actions[:n_chunks],
            "chunk_starts": self.chunk_starts[:n_chunks],
            "chunk_lengths": self.chunk_lengths[:n_chunks],
            "chunk_terminated": self.chunk_terminated[:n_chunks],
            "chunk_truncated": self.chunk_truncated[:n_chunks],
            "chunk_env_id": self.chunk_env_id[:n_chunks],
        }


def as_float_tensor(x: object) -> torch.Tensor:
    """fp32 CPU/GPU tensor, without forcing a device hop."""
    if isinstance(x, torch.Tensor):
        return x.detach().to(OBS_DTYPE)
    return torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=OBS_DTYPE)


def resolve_action_bounds(
    low: float | np.ndarray | torch.Tensor,
    high: float | np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coerce env action bounds into fp32 tensors, strict ``high > low``."""
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
    if not bool(((hi - lo) > 0).all()):
        raise ValueError("action bounds must satisfy high > low in every dim")
    return lo, hi


def resolve_chunk_args(
    n_obs_steps: int,
    n_action_steps: int,
    action_horizon: int,
) -> tuple[int, int]:
    """Check the window width and the execution span."""
    if not 1 <= n_action_steps <= action_horizon:
        raise ValueError(
            f"n_action_steps={n_action_steps} must satisfy "
            f"1 <= n <= action_horizon={action_horizon}"
        )
    return n_obs_steps, action_horizon


def flush_on_collection_error(
    on_error_flush: Callable[[], None] | None,
) -> None:
    """Call the error flush hook, swallowing any raise."""
    if on_error_flush is not None:
        with contextlib.suppress(ValueError, RuntimeError):
            on_error_flush()


def validate_autoreset_mode(envs) -> None:
    """Reject a vec env whose autoreset mode breaks the awaiting machine."""
    mode = getattr(envs, "autoreset_mode", None)
    if mode is None:
        return
    name = getattr(mode, "name", None) or str(mode)
    if name != "NEXT_STEP":
        raise ValueError(
            f"collect_chunks_vec requires AutoresetMode.NEXT_STEP (the "
            f"gymnasium default), got {mode!r}. The awaiting/reset state "
            "machine assumes the reset observation arrives on the step() "
            "call after done=True."
        )


@dataclasses.dataclass
class CollectState:
    """The per-env state of the collection loop."""

    windows: torch.Tensor
    reward_windows: torch.Tensor
    step_frames: torch.Tensor
    in_flight: list[InFlightChunk | None]
    pos_t: torch.Tensor
    plan: torch.Tensor
    actions_t: torch.Tensor
    executed_total: list[int]
    awaiting: list[bool]
    remaining: int
    env_rows: torch.Tensor
    idle_row: torch.Tensor
    roll_table: torch.Tensor


def make_state(
    n: int,
    n_obs_steps: int,
    obs_dim: int,
    action_horizon: int,
    act_dim: int,
    obs0: torch.Tensor,
    remaining: int,
) -> CollectState:
    """The state the collection loop starts with."""
    windows = torch.zeros(n, n_obs_steps, obs_dim, dtype=OBS_DTYPE)
    windows.copy_(obs0[:, None, :].expand_as(windows))
    env_rows = torch.arange(n)
    return CollectState(
        windows=windows,
        reward_windows=torch.zeros(n, action_horizon, dtype=REWARD_DTYPE),
        step_frames=torch.zeros(n, action_horizon, obs_dim, dtype=OBS_DTYPE),
        in_flight=[None] * n,
        pos_t=torch.zeros(n, dtype=torch.long),
        plan=torch.zeros(n + 1, action_horizon, act_dim, dtype=torch.float32),
        actions_t=torch.zeros((n, act_dim), dtype=torch.float32),
        executed_total=[0] * n,
        awaiting=[False] * n,
        remaining=remaining,
        env_rows=env_rows,
        idle_row=torch.full_like(env_rows, n),
        roll_table=(
            torch.arange(n_obs_steps)[:, None]
            + torch.arange(n_obs_steps)[None, :]
        )
        % n_obs_steps,
    )


def plan_chunks(
    state: CollectState,
    envs_needing_a_chunk: list[int],
    policy: Callable[[torch.Tensor], torch.Tensor],
    low: torch.Tensor,
    high: torch.Tensor,
    targets: list[int],
    n_action_steps: int,
) -> None:
    """Give every idle env a new chunk, and hold it until it runs out."""
    states = state.windows[envs_needing_a_chunk]
    # The policy returns the raw chunk. The env receives the squash
    # of it, and the buffer keeps the raw chunk: the flow is
    # trained on what it produced, not on the squash of it.
    raw_all = as_float_tensor(policy(states))
    executed_all = squash_to_env_bounds(raw_all, low, high)
    wavefront = torch.as_tensor(envs_needing_a_chunk)
    state.plan[wavefront] = executed_all
    state.pos_t[wavefront] = 0
    for k, i in enumerate(envs_needing_a_chunk):
        state.in_flight[i] = InFlightChunk(
            actions_raw=raw_all[k],
            start_state=states[k],
            step_limit=min(n_action_steps, targets[i] - state.executed_total[i]),
        )


def step_envs(state: CollectState, envs, action_horizon: int):
    """Send the next action of every env, and record what came back."""
    stepping = torch.tensor(
        [held is not None for held in state.in_flight], dtype=torch.bool
    )
    live_idx = state.env_rows[stepping]
    state.actions_t.copy_(
        state.plan[
            torch.where(stepping, state.env_rows, state.idle_row),
            state.pos_t.clamp(max=action_horizon - 1),
        ]
    )
    obs, rew, term, trunc, _ = envs.step(state.actions_t.cpu().numpy())
    term_np = np.asarray(term, dtype=bool)
    trunc_np = np.asarray(trunc, dtype=bool)
    done_np = term_np | trunc_np
    rew_np = np.asarray(rew, dtype=np.float32)
    obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32))

    if live_idx.numel():
        p = state.pos_t[live_idx]
        state.reward_windows[live_idx, p] = torch.from_numpy(rew_np)[live_idx]
        state.step_frames[live_idx, p] = obs_t[live_idx]
        state.windows[live_idx, p] = obs_t[live_idx]
        state.pos_t[live_idx] = p + 1
    return obs_t, term_np, trunc_np, done_np


def close_chunks(
    state: CollectState,
    obs_t: torch.Tensor,
    term_np: np.ndarray,
    trunc_np: np.ndarray,
    done_np: np.ndarray,
    buffer: RolloutBuffer,
    n_obs_steps: int,
) -> None:
    """Close every chunk that ran to its limit or hit a terminal."""
    for i in range(len(state.in_flight)):
        if state.awaiting[i]:
            state.windows[i].copy_(obs_t[i].expand_as(state.windows[i]))
            state.awaiting[i] = False
            continue
        if state.in_flight[i] is None:
            continue
        n_exec = int(state.pos_t[i])
        held = state.in_flight[i]
        if done_np[i] or n_exec >= held.step_limit:
            raw = held.actions_raw
            start_state = held.start_state
            state.windows[i].copy_(
                state.windows[i][state.roll_table[int(state.pos_t[i]) % n_obs_steps]]
            )
            state.executed_total[i] += n_exec
            state.remaining -= n_exec
            buffer.add_chunk(
                state=start_state,
                next_state=state.windows[i],
                action=raw,
                rewards=state.reward_windows[i, :n_exec],
                step_frames=state.step_frames[i, :n_exec],
                terminated=bool(term_np[i]),
                truncated=bool(trunc_np[i]),
                env_id=i,
            )
            state.in_flight[i] = None
            if done_np[i]:
                state.awaiting[i] = True


def collect_chunks_vec(
    envs,
    policy: Callable[[torch.Tensor], torch.Tensor],
    total_steps: int,
    action_low: float | np.ndarray,
    action_high: float | np.ndarray,
    *,
    n_obs_steps: int,
    n_action_steps: int,
    seed: int | None = None,
    on_error_flush: Callable[[], None] | None = None,
    buffer: RolloutBuffer,
) -> RolloutBuffer:
    """Collect executed action chunks from a vectorised env into ``buffer``."""
    n_obs_steps, action_horizon = resolve_chunk_args(
        n_obs_steps, n_action_steps, buffer.action_horizon
    )

    validate_autoreset_mode(envs)

    n = int(envs.num_envs)
    base_target, remainder = divmod(total_steps, n)
    targets = [base_target + int(i < remainder) for i in range(n)]

    low, high = resolve_action_bounds(action_low, action_high)

    if n_action_steps > n_obs_steps:
        raise ValueError(
            f"n_action_steps={n_action_steps} exceeds n_obs_steps={n_obs_steps}: "
            "the vec obs window is a ring of n_obs_steps rows and cannot hold a "
            "chunk longer than itself"
        )
    if seed is None:
        seed = int(torch.randint(0, 2**31, (1,)).item())
    act_dim = int(np.asarray(envs.single_action_space.shape).prod())

    try:
        obs, _ = envs.reset(seed=[seed + i for i in range(n)])
        obs_dim = int(obs.shape[-1])
        obs0 = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        state = make_state(
            n,
            n_obs_steps,
            obs_dim,
            action_horizon,
            act_dim,
            obs0,
            remaining=sum(targets),
        )
        while state.remaining:
            envs_needing_a_chunk = [
                i
                for i in range(n)
                if (
                    state.in_flight[i] is None
                    and not state.awaiting[i]
                    and state.executed_total[i] < targets[i]
                )
            ]
            if envs_needing_a_chunk:
                plan_chunks(
                    state,
                    envs_needing_a_chunk,
                    policy,
                    low,
                    high,
                    targets,
                    n_action_steps,
                )

            obs_t, term_np, trunc_np, done_np = step_envs(
                state, envs, action_horizon
            )

            close_chunks(
                state,
                obs_t,
                term_np,
                trunc_np,
                done_np,
                buffer,
                n_obs_steps,
            )

    except (ValueError, RuntimeError, KeyError):
        flush_on_collection_error(on_error_flush)
        raise
    else:
        return buffer
