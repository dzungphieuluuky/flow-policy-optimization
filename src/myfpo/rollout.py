"""Owns: chunk collection, advantage computation, bootstrap. Key: compute_chunk_advantages, run_rollout.
The rollout procedure, as a module function.
"""

from __future__ import annotations
import torch
import contextlib
import dataclasses
from typing import NamedTuple
from collections.abc import Callable
import numpy as np
from myfpo.processor import squash_to_env_bounds
from dataclasses import dataclass
from lerobot.utils.constants import ACTION, OBS_STATE
from myfpo.fpo_core import (
    compute_gae_diagnostics,
    sample_eps,
    sample_mc_timesteps,
    subsampled_min,
)
from myfpo.layers import (
    autocast_context,
    cudagraph_step_begin,
    resolve_storage_dtype,
    to_device_async,
)

__all__ = [
    "EnvCarry",
    "RolloutBuffer",
    "RolloutBuffers",
    "collect_chunks_vec",
    "run_rollout",
]

# ADVANTAGES
# ---- from gae.py ----
@torch.no_grad()
def compute_chunk_advantages(
    step_rewards: torch.Tensor,
    q_current_chunk: torch.Tensor,
    q_next_state: torch.Tensor,
    chunk_starts: torch.Tensor,
    chunk_lengths: torch.Tensor,
    *,
    chunk_terminated: torch.Tensor,
    chunk_size: int,
    gamma: float = 0.99,
    chunk_truncated: torch.Tensor | None = None,
    chunk_env_id: torch.Tensor | None = None,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``[n]`` advantages and the ``[n]`` regression target.

    The target is ``advantage + q_current_chunk``: a TD(lambda) backup over the
    macro-transitions defined by ``chunk_starts`` / ``chunk_lengths``. Chunks
    from different envs interleave in the buffer, so adjacency is resolved
    through ``chunk_env_id`` rather than by buffer position.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")

    device = q_current_chunk.device
    dtype = torch.float32

    rewards = step_rewards.to(device=device, dtype=dtype).reshape(-1)
    q_current_chunk = q_current_chunk.to(device=device, dtype=dtype).reshape(-1)
    q_next_state = q_next_state.to(device=device, dtype=dtype).reshape(-1)
    starts = chunk_starts.to(device=device, dtype=torch.long).reshape(-1)
    lengths = chunk_lengths.to(device=device, dtype=torch.long).reshape(-1)
    terminated = chunk_terminated.to(device=device, dtype=torch.bool).reshape(-1)

    n_chunks = starts.numel()
    if lengths.numel() != n_chunks or terminated.numel() != n_chunks or q_current_chunk.numel() != n_chunks or q_next_state.numel() != n_chunks:
        raise ValueError(f"chunk tensors must have {n_chunks} entries")

    if chunk_truncated is None:
        truncated = torch.zeros(n_chunks, device=device, dtype=torch.bool)
    else:
        truncated = chunk_truncated.to(device=device, dtype=torch.bool).reshape(-1)
        if truncated.numel() != n_chunks:
            raise ValueError(
                f"chunk_truncated must have {n_chunks} entries, got {truncated.numel()}"
            )

    if gae_lambda > 0.0:
        if chunk_truncated is None:
            raise ValueError(
                "chunk_truncated is required when gae_lambda > 0: without it the "
                "advantage trace crosses an episode reset"
            )
        if chunk_env_id is None:
            raise ValueError(
                "chunk_env_id is required when gae_lambda > 0: chunks from different "
                "envs interleave in the buffer, so buffer adjacency is not trajectory "
                "adjacency"
            )

    if n_chunks == 0:
        empty = torch.empty(0, device=device, dtype=dtype)
        return empty, empty

    offsets = torch.arange(chunk_size, device=device)
    window = starts.unsqueeze(1) + offsets
    gathered = rewards[window.clamp(max=rewards.numel() - 1)]
    valid = offsets.unsqueeze(0) < lengths.unsqueeze(1)
    chunk_rewards = (gathered * valid * (gamma**offsets)).sum(dim=1)

    gamma_h = gamma ** lengths.to(dtype)
    done = terminated | truncated
    injected = truncated & ~terminated
    chunk_rewards = chunk_rewards + gamma_h * injected * q_next_state
    bootstrap = gamma_h * (~done)
    delta = chunk_rewards + bootstrap * q_next_state - q_current_chunk

    if gae_lambda == 0.0:
        return delta, delta + q_current_chunk

    env_id = chunk_env_id.to(device=device, dtype=torch.long).reshape(-1)
    if env_id.numel() != n_chunks:
        raise ValueError(f"chunk_env_id must have {n_chunks} entries, got {env_id.numel()}")

    successor = chunk_successors(env_id)
    trace = gamma_h * (~done) * gae_lambda
    adv = gae_trace(delta, trace, successor)
    return adv, adv + q_current_chunk


def chunk_successors(env_id: torch.Tensor) -> torch.Tensor:
    """The next chunk of each chunk, inside the same env. ``-1`` means none.

    The chunks of different envs interleave in the buffer, so adjacency is
    resolved through the env id, not through the buffer position.
    """
    order = torch.argsort(env_id, stable=True)
    sorted_env = env_id[order]
    same_env = sorted_env[:-1] == sorted_env[1:]
    successor = torch.full_like(order, -1)
    successor[order[:-1][same_env]] = order[1:][same_env]
    return successor


def gae_trace(
    delta: torch.Tensor, trace: torch.Tensor, successor: torch.Tensor
) -> torch.Tensor:
    """The advantage of each chunk, traced back over its successors."""
    adv = delta.clone()
    for i in range(delta.numel() - 1, -1, -1):
        j = int(successor[i].item())
        if j >= 0:
            adv[i] = delta[i] + trace[i] * adv[j]
    return adv

# ---- from collector.py ----
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
# COLLECTION
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
        device: torch.device | str = "cpu",
        pin_memory: bool = False,
    ):
        if max_chunks < 1 or max_steps < 1:
            raise ValueError("max_chunks and max_steps must be positive")
        if obs_steps < 1 or obs_dim < 1 or action_horizon < 1 or action_dim < 1:
            raise ValueError("observation and action dimensions must be positive")
        self.device = torch.device(device)
        # max_chunks owns both the chunk rows and the add_chunk guard. Two
        # capacities once disagreed, so the guard passed and the row write
        # raised IndexError outside collect_chunks_vec's flush handler.
        self.max_chunks = max_chunks
        self.max_steps = max_steps
        self.obs_steps = obs_steps
        self.obs_dim = obs_dim
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.pin_memory = pin_memory

        self.step_frames = self.zeros_for((max_steps, obs_dim), OBS_DTYPE)
        self.step_rewards = self.zeros_for((max_steps,), REWARD_DTYPE)

        self.chunk_state = self.zeros_for(
            (max_chunks, obs_steps, obs_dim), OBS_DTYPE
        )
        self.chunk_next_state = self.zeros_for(
            (max_chunks, obs_steps, obs_dim), OBS_DTYPE
        )
        self.chunk_actions = self.zeros_for(
            (max_chunks, action_horizon, action_dim), OBS_DTYPE
        )
        self.chunk_starts = self.zeros_for((max_chunks,), INDEX_DTYPE)
        self.chunk_lengths = self.zeros_for((max_chunks,), INDEX_DTYPE)
        self.chunk_terminated = self.zeros_for((max_chunks,), torch.bool)
        self.chunk_truncated = self.zeros_for((max_chunks,), torch.bool)
        self.chunk_env_id = self.zeros_for((max_chunks,), torch.int16)
        self.n_chunks = 0
        self.n_steps = 0
        # The env-side state one rollout hands to the next. reset() keeps it.
        self.carry: EnvCarry | None = None
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

        # The copy runs first: a destination too small raises here, and its
        # cursors must not claim rows it does not hold.
        torch._foreach_copy_(
            [prefix(other, name) for name in names],
            [prefix(self, name) for name in names],
            non_blocking=non_blocking,
        )
        other.n_chunks = self.n_chunks
        other.n_steps = self.n_steps

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
    if obs0.dim() == 3:
        if tuple(obs0.shape) != (n, n_obs_steps, obs_dim):
            raise ValueError(
                f"obs0 window shape {tuple(obs0.shape)} != "
                f"{(n, n_obs_steps, obs_dim)}"
            )
        windows = obs0.clone()
    elif obs0.dim() == 2:
        windows = torch.zeros(n, n_obs_steps, obs_dim, dtype=OBS_DTYPE)
        windows.copy_(obs0[:, None, :].expand_as(windows))
    else:
        raise ValueError(
            "obs0 must have shape [n, obs_dim] or [n, n_obs_steps, obs_dim], "
            f"got {tuple(obs0.shape)}"
        )
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
    raw_all = policy(states).detach().to(OBS_DTYPE)
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


def init_collect_state(
    carry, envs, seed, n, n_obs_steps, action_horizon, act_dim, targets
):
    try:
        if carry is None:
            obs, _ = envs.reset(seed=[seed + i for i in range(n)])
            obs0 = torch.from_numpy(np.asarray(obs, dtype=np.float32))
            state = make_state(
                n, n_obs_steps, int(obs0.shape[-1]), action_horizon, act_dim,
                obs0, remaining=sum(targets),
            )
        else:
            state = make_state(
                n, n_obs_steps, int(carry.windows.shape[-1]), action_horizon,
                act_dim, carry.windows, remaining=sum(targets),
            )
            state.awaiting[:] = carry.awaiting
    except (ValueError, RuntimeError, KeyError):
        raise
    return state


def run_collect_loop(
    state, envs, policy, low, high, targets, n_action_steps, buffer,
    n_obs_steps, action_horizon,
):
    while state.remaining:
        envs_needing_a_chunk = [
            i for i in range(len(state.windows))
            if (
                state.in_flight[i] is None
                and not state.awaiting[i]
                and state.executed_total[i] < targets[i]
            )
        ]
        if envs_needing_a_chunk:
            plan_chunks(
                state, envs_needing_a_chunk, policy, low, high,
                targets, n_action_steps,
            )
        obs_t, term_np, trunc_np, done_np = step_envs(
            state, envs, action_horizon
        )
        close_chunks(
            state, obs_t, term_np, trunc_np, done_np,
            buffer, n_obs_steps,
        )

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
    carry: EnvCarry | None = None,
) -> RolloutBuffer:
    """Collect executed action chunks from a vectorised env into ``buffer``.

    ``carry is None`` resets the envs: the first rollout of a run. Any later
    call passes the previous call's ``EnvCarry``, so an episode spans rollout
    boundaries. A carried ``awaiting`` env gets its reset observation through
    the vec env's autoreset step.
    """
    n_obs_steps, action_horizon = resolve_chunk_args(
        n_obs_steps, n_action_steps, buffer.action_horizon
    )

    validate_autoreset_mode(envs)

    n = int(envs.num_envs)
    base_target, remainder = divmod(total_steps, n)
    targets = torch.tensor([base_target + (i < remainder) for i in range(n)], dtype=torch.long)

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

    state = init_collect_state(
        carry, envs, seed, n, n_obs_steps, action_horizon, act_dim, targets
    )
    try:
        run_collect_loop(
            state, envs, policy, low, high, targets, n_action_steps,
            buffer, n_obs_steps, action_horizon,
        )
    except (ValueError, RuntimeError, KeyError):
        if on_error_flush is not None:
            with contextlib.suppress(ValueError, RuntimeError):
                on_error_flush()
        raise
    else:
        buffer.carry = EnvCarry(
            windows=state.windows, awaiting=[bool(x) for x in state.awaiting]
        )
        return buffer


@dataclass
class EnvCarry:
    """The env-side state one rollout hands to the next one."""

    windows: torch.Tensor
    awaiting: list[bool]


@dataclass
class RolloutBuffers:
    """The buffers a rollout reuses between calls."""
    host: RolloutBuffer | None = None
    device: RolloutBuffer | None = None
    h2d_event: torch.cuda.Event | None = None
    carry: EnvCarry | None = None
    env_wrapper: object | None = None


def chunk_advantages(
    cfg,
    active: dict[str, torch.Tensor],
    n_chunks: int,
    q_current_chunk: torch.Tensor,
    q_bootstrap: torch.Tensor,
    storage_dtype: torch.dtype,
):
    """The advantages, the regression targets, and the GAE diagnostics."""
    raw_advantages, raw_returns = compute_chunk_advantages(
        active["step_rewards"],
        q_current_chunk,
        q_bootstrap,
        active["chunk_starts"],
        active["chunk_lengths"],
        chunk_terminated=active["chunk_terminated"],
        chunk_size=cfg.action_horizon,
        gamma=cfg.gamma,
        chunk_truncated=active["chunk_truncated"],
        chunk_env_id=active["chunk_env_id"],
        gae_lambda=cfg.gae_lambda,
    )
    normalised_advantages = raw_advantages
    if cfg.normalize_advantage and raw_advantages.numel() > 0:
        normalised_advantages = (raw_advantages - raw_advantages.mean()) / (
            raw_advantages.std(correction=0).clamp_min(1e-8)
        )
    normalised_advantages = normalised_advantages.clamp(
        -cfg.advantage_clamp_neg, cfg.advantage_clamp_pos
    )
    gae_diagnostics = compute_gae_diagnostics(
        active["chunk_env_id"][:n_chunks],
        active["chunk_terminated"][:n_chunks],
        active["chunk_truncated"][:n_chunks],
        raw_advantages,
        raw_returns,
    )
    return (
        raw_advantages.to(storage_dtype),
        normalised_advantages.to(storage_dtype),
        raw_returns,
        gae_diagnostics,
    )


def bootstrap_values(
    policy,
    active: dict[str, torch.Tensor],
    n_chunks: int,
    device,
    storage_dtype: torch.dtype,
):
    """The bootstrap action, the normalised states, and the two critic reads."""
    cfg = policy.config
    lengths = active["chunk_lengths"]
    actions = active["chunk_actions"]
    states = active["chunk_state"]

    with torch.inference_mode():
        next_states = active["chunk_next_state"]
        boot_actions = actions.clone()
        needs_proposal = ~active["chunk_terminated"]
        if bool(needs_proposal.any()):
            proposed = policy.predict_action_chunk(
                {OBS_STATE: to_device_async(next_states[needs_proposal], device)},
                start_from_zero=False,
            )
            # The backup's action lives in the same space as the stored
            # chunks: the policy's raw action space.
            boot_actions[needs_proposal] = proposed

        # The training pass reloads the batch in the storage dtype before it
        # normalises, so the rollout normalises the same values it will train
        # on. Normalising the fp32 buffer states here would give the rollout
        # and the training pass two different conditions, and the behavior
        # loss would not be the loss the training pass recomputes.
        all_chunk_normalised_states = policy.preprocessor(
            {OBS_STATE: torch.cat([states, next_states], dim=0).to(storage_dtype)}
        )[OBS_STATE]
        normalised_states = all_chunk_normalised_states[:n_chunks]
        normalised_next_states = all_chunk_normalised_states[n_chunks:]
        exec_length_frac = policy.exec_length_frac(lengths)
        fitted_steps = (
            lengths.float().mean()
            if lengths.numel() > 0
            else torch.tensor(
                float(cfg.action_horizon),
                device=lengths.device,
                dtype=torch.float32,
            )
        )
        boot_exec_frac = policy.exec_length_frac(
            torch.full_like(lengths, round(fitted_steps.item()))
        )
        # The critic read is a target, not exploration. A dropout layer in
        # the critic re-rolls the value on every call, so the critic reads in
        # eval mode. The actor proposal above stays in train mode on purpose:
        # its dropout is part of the behaviour noise.
        critic_training = policy.critic.training
        policy.critic.eval()
        try:
            with autocast_context(device):
                q_current_chunk = subsampled_min(
                    policy.critic(normalised_states, actions, exec_length_frac)
                    .float()
                )
                q_bootstrap = subsampled_min(
                    policy.critic(
                        normalised_next_states, boot_actions, boot_exec_frac
                    ).float()
                )
        finally:
            policy.critic.train(critic_training)

    return (
        normalised_states,
        normalised_next_states,
        boot_actions,
        q_current_chunk,
        q_bootstrap,
    )


def collect_chunks(
    policy,
    buffers: RolloutBuffers,
    env,
    total_steps: int,
    action_low: float,
    action_high: float,
    noisy: bool,
    device,
):
    """Run the env, and return the views of the populated buffer."""
    cfg = policy.config

    def propose_batch(states: torch.Tensor) -> torch.Tensor:
        # One chunk is one cudagraph step: the previous chunk's graphs are
        # dead here, so the manager may recycle their memory.
        cudagraph_step_begin()
        chunk = policy.predict_action_chunk(
            {OBS_STATE: to_device_async(states, device)}, start_from_zero=not noisy
        )
        return chunk.cpu()

    if buffers.h2d_event is not None:
        buffers.h2d_event.synchronize()
        buffers.h2d_event = None

    num_envs = int(getattr(env, "num_envs", 1))
    action_dim = int(cfg.action_feature.shape[0])
    obs_dim = int(cfg.robot_state_feature.shape[0])
    # The worst case is one chunk per step, so the rows must cover total_steps.
    # A normal rollout writes total_steps / action_horizon (1024 of 8192 at the
    # defaults). A shorter cap aborts on an off-boundary episode length.
    max_chunks = max(1, total_steps)
    # The field order must match RolloutBuffer.shape_key, which the reuse
    # check below compares against.
    shape_args = dict(
        max_chunks=max_chunks,
        max_steps=total_steps,
        obs_steps=cfg.n_obs_steps,
        obs_dim=obs_dim,
        action_horizon=cfg.action_horizon,
        action_dim=action_dim,
    )
    buffer_key = tuple(shape_args.values())

    def make_buffer(**extra) -> RolloutBuffer:
        """Host and device buffers share every shape argument."""
        return RolloutBuffer(**shape_args, **extra)

    if buffers.host is None or buffers.host.shape_key() != buffer_key:
        buffers.host = make_buffer(pin_memory=device.type == "cuda")
    buffer = buffers.host
    buffer.reset()

    collect_kwargs = dict(
        total_steps=total_steps,
        action_low=action_low,
        action_high=action_high,
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        buffer=buffer,
        carry=buffer.carry,
    )
    if num_envs == 1:
        if buffers.env_wrapper is None:
            import gymnasium as gym

            # The wrapper lives for the whole run. A per-rollout close()
            # walks into close_extras and closes the caller's training env.
            buffers.env_wrapper = gym.vector.SyncVectorEnv([lambda: env])
        envs = buffers.env_wrapper
    else:
        envs = env
    collect_chunks_vec(envs, propose_batch, **collect_kwargs)

    if device.type == "cuda":
        if buffers.device is None or buffers.device.shape_key() != buffer.shape_key():
            buffers.device = make_buffer(device=device)
        buffer.copy_to(buffers.device, non_blocking=True)
        buffers.h2d_event = torch.cuda.Event()
        buffers.h2d_event.record()
        active = buffers.device.active()
    else:
        active = buffer.active()
    return active, buffer.n_chunks


def run_rollout(
    policy,
    buffers: RolloutBuffers,
    env,
    total_steps: int,
    *,
    action_low: float = -1.0,
    action_high: float = 1.0,
    noisy: bool = True,
) -> dict[str, torch.Tensor]:
    """Roll out open-loop chunks, build a training batch."""
    if total_steps <= 0:
        raise ValueError(f"total_steps={total_steps} must be positive")
    cfg = policy.config
    device = next(policy.parameters()).device
    storage_dtype = resolve_storage_dtype(device)

    active, n_chunks = collect_chunks(
        policy,
        buffers,
        env,
        total_steps,
        action_low,
        action_high,
        noisy,
        device,
    )
    (
        normalised_states,
        normalised_next_states,
        boot_actions,
        q_current_chunk,
        q_bootstrap,
    ) = bootstrap_values(policy, active, n_chunks, device, storage_dtype)
    raw_advantages, normalised_advantages, raw_returns, gae_diagnostics = (
        chunk_advantages(
            cfg, active, n_chunks, q_current_chunk, q_bootstrap, storage_dtype
        )
    )

    lengths = active["chunk_lengths"]
    actions = active["chunk_actions"]
    states = active["chunk_state"]

    action_is_pad = torch.arange(cfg.action_horizon, device=device).unsqueeze(
        0
    ) >= lengths.unsqueeze(1)

    t = sample_mc_timesteps(cfg, n_chunks, device)
    eps = sample_eps(cfg, n_chunks, int(cfg.action_feature.shape[0]), device)

    (
        behavior_cfm_loss,
        behavior_cfm_loss_token,
        behavior_velocity_token,
    ) = policy.compute_behavior_cfm_losses(
        normalised_states, actions, t, eps, action_is_pad
    )

    return {
        OBS_STATE: states,
        ACTION: actions,
        "action_is_pad": action_is_pad,
        "raw_advantages": raw_advantages,
        "normalised_advantages": normalised_advantages,
        "raw_returns": raw_returns,
        "normalizer_observations": active["step_frames"].detach(),
        "boot_observations": normalised_next_states.to(storage_dtype),
        "boot_actions": boot_actions.to(storage_dtype),
        "boot_values_start": q_bootstrap.unsqueeze(1),
        "mc_t": t,
        "mc_eps": eps,
        "gae_diagnostics": gae_diagnostics,
        "behavior_cfm_loss": behavior_cfm_loss,
        "behavior_cfm_loss_token": behavior_cfm_loss_token,
        "behavior_velocity_token": behavior_velocity_token,
    }
