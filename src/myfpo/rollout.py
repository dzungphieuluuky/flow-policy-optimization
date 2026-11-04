"""Owns: chunk collection, advantage computation, bootstrap. Key: compute_chunk_advantages, run_rollout.
The rollout procedure, as a module function.
"""

from __future__ import annotations
import torch
import contextlib
import dataclasses
from collections.abc import Callable
import numpy as np
from myfpo.processor import squash_to_env_bounds
from dataclasses import dataclass
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
class TorchEnvAdapter:
    """Wrap a numpy gym env so step() returns torch tensors (Tier 2)."""
    def __init__(self, env):
        self.env = env
    def step_torch(self, actions_t):
        obs, rew, term, trunc, info = self.env.step(actions_t.cpu().numpy())
        device = actions_t.device
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device=device, dtype=OBS_DTYPE)
        rew_t = torch.from_numpy(np.asarray(rew, dtype=np.float32)).to(device=device, dtype=REWARD_DTYPE)
        term_t = torch.from_numpy(np.asarray(term, dtype=bool)).to(device=device)
        trunc_t = torch.from_numpy(np.asarray(trunc, dtype=bool)).to(device=device)
        return obs_t, rew_t, term_t, trunc_t, info
    def reset(self, **kw):
        return self.env.reset(**kw)
    @property
    def observation_space(self): return self.env.observation_space
    @property
    def action_space(self): return self.env.action_space

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
    v_baseline: torch.Tensor | None = None,
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
    baseline = v_baseline.reshape(-1) if v_baseline is not None else q_current_chunk.reshape(-1)
    delta = chunk_rewards + bootstrap * q_next_state - baseline

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


# What one env is executing right now used to live in a per-env named tuple
# (`actions_raw`, `start_state`, `step_limit`). It is now four registers on
# `CollectState` -- `raw_plan`, `start_window`, `step_limits`, `active` -- so a
# closing step can gather every finished chunk with one index, instead of
# touching one Python object per env per step.


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

    def write_chunks(
        self,
        *,
        start_window: torch.Tensor,
        next_window: torch.Tensor,
        actions: torch.Tensor,
        lengths: torch.Tensor,
        frames: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        env_ids: torch.Tensor,
        n_steps: int,
    ) -> None:
        """Append ``k`` executed chunks with one write each, not one per chunk.

        ``add_chunk`` costs ~11 device launches per chunk, which at 1024 envs is
        the collector's ceiling. Every row here is gathered by the caller, so
        the whole batch is ~11 launches per *step*.

        The packing rule matches ``add_chunk`` exactly: chunk ``j`` starts at
        the running offset, and its first ``lengths[j]`` frames are copied
        verbatim. ``n_steps`` is the host-side sum of ``lengths``, passed in
        rather than computed, so the guard does not sync the device.
        """
        k = int(lengths.numel())
        if k == 0:
            return
        if self.n_chunks + k > self.max_chunks:
            raise RuntimeError("RolloutBuffer chunk capacity exceeded")
        if n_steps < 1 or self.n_steps + n_steps > self.max_steps:
            raise RuntimeError("RolloutBuffer step capacity exceeded")
        if start_window.shape != (k, self.obs_steps, self.obs_dim):
            raise ValueError(f"start_window shape {tuple(start_window.shape)} does not fit buffer")
        if next_window.shape != (k, self.obs_steps, self.obs_dim):
            raise ValueError(f"next_window shape {tuple(next_window.shape)} does not fit buffer")
        if actions.shape != (k, self.action_horizon, self.action_dim):
            raise ValueError(f"actions shape {tuple(actions.shape)} does not fit buffer")
        if frames.shape != (k, self.action_horizon, self.obs_dim):
            raise ValueError(f"frames shape {tuple(frames.shape)} does not fit buffer")

        lo = self.n_chunks
        hi = lo + k
        # Chunk blocks are contiguous and in row order, so the step rows they
        # land on are simply the next n_steps rows.
        offsets = self.n_steps + torch.cumsum(lengths, 0) - lengths
        mask = (
            torch.arange(self.action_horizon, device=frames.device)[None, :]
            < lengths[:, None]
        ).reshape(-1)
        step_lo = self.n_steps
        step_hi = self.n_steps + n_steps

        self.chunk_state[lo:hi] = start_window
        self.chunk_next_state[lo:hi] = next_window
        self.chunk_actions[lo:hi] = actions
        self.chunk_starts[lo:hi] = offsets
        self.chunk_lengths[lo:hi] = lengths
        self.chunk_terminated[lo:hi] = terminated
        self.chunk_truncated[lo:hi] = truncated
        self.chunk_env_id[lo:hi] = env_ids
        self.step_frames[step_lo:step_hi] = frames.reshape(-1, self.obs_dim)[mask]
        self.step_rewards[step_lo:step_hi] = rewards.reshape(-1)[mask]
        self.n_chunks = hi
        self.n_steps = step_hi

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
            """Slice one field to the rows this buffer wrote."""
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

    def active_prefix(self) -> dict[str, torch.Tensor]:
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
    """The per-env state of the collection loop.

    The chunk an env is executing is held in registers indexed by env row
    (``raw_plan``, ``start_window``, ``step_limits``), not in one Python object
    per env. That is what lets a closing step gather every finished chunk at
    once.

    The control-flow bookkeeping (``pos_t``, ``active``, ``awaiting``,
    ``executed_total``, ``roll_table``) is host memory: it decides what the
    Python loop does next, and reading it must not sync the device.
    """

    windows: torch.Tensor
    reward_windows: torch.Tensor
    step_frames: torch.Tensor
    active: np.ndarray
    step_limits: np.ndarray
    raw_plan: torch.Tensor
    start_window: torch.Tensor
    pos_t: torch.Tensor
    plan: torch.Tensor
    actions_t: torch.Tensor
    executed_total: np.ndarray
    awaiting: np.ndarray
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
    device: torch.device | str = "cpu",
) -> CollectState:
    """The state the collection loop starts with.

    The observation, reward, plan and action storage live on ``device``: on a
    GPU sim they never leave it. The index bookkeeping (``pos_t``, ``env_rows``,
    ``roll_table``) stays on the host, because it drives Python control flow:
    moving it would turn one sync per env per step into the common case.
    """
    device = torch.device(device)
    if obs0.dim() == 3:
        if tuple(obs0.shape) != (n, n_obs_steps, obs_dim):
            raise ValueError(
                f"obs0 window shape {tuple(obs0.shape)} != "
                f"{(n, n_obs_steps, obs_dim)}"
            )
        windows = obs0.to(device=device, dtype=OBS_DTYPE).clone()
    elif obs0.dim() == 2:
        windows = torch.zeros(n, n_obs_steps, obs_dim, dtype=OBS_DTYPE, device=device)
        windows.copy_(obs0.to(device=device, dtype=OBS_DTYPE)[:, None, :].expand_as(windows))
    else:
        raise ValueError(
            "obs0 must have shape [n, obs_dim] or [n, n_obs_steps, obs_dim], "
            f"got {tuple(obs0.shape)}"
        )
    env_rows = torch.arange(n)
    return CollectState(
        windows=windows,
        reward_windows=torch.zeros(n, action_horizon, dtype=REWARD_DTYPE, device=device),
        step_frames=torch.zeros(
            n, action_horizon, obs_dim, dtype=OBS_DTYPE, device=device
        ),
        active=np.zeros(n, dtype=bool),
        step_limits=np.zeros(n, dtype=np.int64),
        raw_plan=torch.zeros(
            n, action_horizon, act_dim, dtype=torch.float32, device=device
        ),
        start_window=torch.zeros(n, n_obs_steps, obs_dim, dtype=OBS_DTYPE, device=device),
        pos_t=torch.zeros(n, dtype=torch.long),
        plan=torch.zeros(
            n + 1, action_horizon, act_dim, dtype=torch.float32, device=device
        ),
        actions_t=torch.zeros((n, act_dim), dtype=torch.float32, device=device),
        executed_total=np.zeros(n, dtype=np.int64),
        awaiting=np.zeros(n, dtype=bool),
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
    if not envs_needing_a_chunk:
        return
    wavefront = torch.as_tensor(envs_needing_a_chunk, dtype=torch.long)
    states = state.windows[wavefront]
    # The policy returns the raw chunk. The env receives the squash
    # of it, and the buffer keeps the raw chunk: the flow is
    # trained on what it produced, not on the squash of it.
    raw_all = policy(states).detach().to(OBS_DTYPE)
    executed_all = squash_to_env_bounds(raw_all, low, high)
    state.plan[wavefront] = executed_all
    # The chunk is held in registers rather than in one object per env, so a
    # closing step can gather all finished chunks with a single index.
    state.raw_plan[wavefront] = raw_all
    state.start_window[wavefront] = states
    state.pos_t[wavefront] = 0
    rows = np.asarray(envs_needing_a_chunk, dtype=np.int64)
    targets_np = (
        targets.detach().cpu().numpy() if torch.is_tensor(targets)
        else np.asarray(targets, dtype=np.int64)
    )
    state.step_limits[rows] = np.minimum(
        n_action_steps, targets_np[rows] - state.executed_total[rows]
    )
    state.active[rows] = True


def is_torch_env(envs) -> bool:
    """True when the env hands back its own tensors instead of numpy arrays."""
    return callable(getattr(envs, "step_torch", None))


def take_final_observations(info, done_np, reference, device):
    """Replace the done rows of ``reference`` with the true terminal frame.

    A done step must report ``o_T``: the collector stores it as the chunk's
    ``next_state``, and the truncation bootstrap reads it. Gymnasium exposes it
    as ``info["final_obs"]`` and ManiSkill as ``info["final_observation"]``
    when the env resets inside the step and returns the reset frame instead.
    Under NEXT_STEP autoreset the step's own observation is already final and
    neither key is present, so the reference passes through unchanged.
    """
    if not isinstance(info, dict):
        return reference
    final = info.get("final_obs")
    if final is None:
        final = info.get("final_observation")
    if final is None:
        return reference
    rows = np.flatnonzero(done_np)
    if rows.size == 0:
        return reference
    idx = torch.from_numpy(rows).to(device=device)
    out = reference.clone()
    if torch.is_tensor(final):
        out[idx] = final.to(device=device, dtype=reference.dtype)[idx]
        return out
    array = np.asarray(final)
    if array.dtype == object:
        for i in rows:
            if array[i] is None:
                continue
            out[i] = torch.as_tensor(
                np.asarray(array[i], dtype=np.float32), device=device
            )
        return out
    out[idx] = torch.as_tensor(array, dtype=reference.dtype, device=device)[idx]
    return out


def step_envs(state: CollectState, envs, action_horizon: int):
    """Send the next action of every env, and record what came back."""
    stepping = torch.from_numpy(state.active)
    live_idx = state.env_rows[stepping]
    state.actions_t.copy_(
        state.plan[
            torch.where(stepping, state.env_rows, state.idle_row),
            state.pos_t.clamp(max=action_horizon - 1),
        ]
    )
    if is_torch_env(envs):
        # A GPU env keeps its frames on the device. Only the three done flags
        # cross to the host, and only because they drive the chunk-close loop.
        obs_t, rew_t, term_t, trunc_t, info = envs.step_torch(state.actions_t)
        obs_t = obs_t.to(device=state.windows.device, dtype=OBS_DTYPE)
        term_np = term_t.cpu().numpy()
        trunc_np = trunc_t.cpu().numpy()
        rew_t = rew_t.to(device=state.windows.device, dtype=REWARD_DTYPE)
    else:
        obs, rew, term, trunc, info = envs.step(state.actions_t.cpu().numpy())
        term_np = np.asarray(term, dtype=bool)
        trunc_np = np.asarray(trunc, dtype=bool)
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(
            device=state.windows.device
        )
        rew_t = torch.from_numpy(np.asarray(rew, dtype=np.float32)).to(
            device=state.windows.device
        )
    done_np = term_np | trunc_np
    if done_np.any():
        obs_t = take_final_observations(
            info, done_np, obs_t, state.windows.device
        )

    if live_idx.numel():
        p = state.pos_t[live_idx]
        state.reward_windows[live_idx, p] = rew_t[live_idx]
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
    *,
    batched: bool = False,
) -> None:
    """Close every chunk that ran to its limit or hit a terminal.

    ``batched`` picks the writer. A torch env keeps its rows on one device, so
    the closes are gathered and written a launch per column; a numpy env keeps
    the original per-chunk loop, untouched, because its numerics are pinned.
    """
    if batched:
        close_chunks_batched(
            state, obs_t, term_np, trunc_np, done_np, buffer, n_obs_steps
        )
        return
    for i in range(len(state.active)):
        if state.awaiting[i]:
            state.windows[i].copy_(obs_t[i].expand_as(state.windows[i]))
            state.awaiting[i] = False
            continue
        if not state.active[i]:
            continue
        n_exec = int(state.pos_t[i])
        if done_np[i] or n_exec >= state.step_limits[i]:
            raw = state.raw_plan[i]
            start_state = state.start_window[i]
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
            state.active[i] = False
            if done_np[i]:
                state.awaiting[i] = True

def close_chunks_batched(
    state: CollectState,
    obs_t: torch.Tensor,
    term_np: np.ndarray,
    trunc_np: np.ndarray,
    done_np: np.ndarray,
    buffer: RolloutBuffer,
    n_obs_steps: int,
) -> None:
    """``close_chunks`` for a device-resident env: one gather per column.

    The classification stays on the host (it is the loop's control flow), but
    every write is a single index operation over all closing rows at once.
    """
    device = state.windows.device

    # An env that resets this step takes the reset frame for its whole window.
    waiting = np.flatnonzero(state.awaiting)
    if waiting.size:
        w = torch.as_tensor(waiting, device=device)
        state.windows[w] = obs_t[w].unsqueeze(1).expand(-1, n_obs_steps, -1)
        state.awaiting[waiting] = False

    pos = state.pos_t.numpy()
    closing = np.flatnonzero(state.active & (done_np | (pos >= state.step_limits)))
    if closing.size == 0:
        return

    rows = torch.as_tensor(closing, device=device)
    lengths_np = pos[closing]
    n_new = int(lengths_np.sum())
    lengths = torch.as_tensor(lengths_np, device=device)
    # The ring holds the last n_obs_steps frames; `next_state` wants them in
    # chronological order, which is what the roll table permutes them into.
    perm = state.roll_table[
        torch.as_tensor(pos[closing] % n_obs_steps)
    ].to(device)
    next_window = torch.gather(
        state.windows[rows], 1, perm[:, :, None].expand(-1, -1, state.windows.shape[-1])
    )

    buffer.write_chunks(
        start_window=state.start_window[rows],
        next_window=next_window,
        actions=state.raw_plan[rows],
        lengths=lengths,
        frames=state.step_frames[rows],
        rewards=state.reward_windows[rows],
        terminated=torch.from_numpy(term_np).to(device)[rows],
        truncated=torch.from_numpy(trunc_np).to(device)[rows],
        env_ids=rows,
        n_steps=n_new,
    )

    state.windows[rows] = next_window
    state.executed_total[closing] += lengths_np
    state.remaining -= n_new
    state.active[closing] = False
    state.awaiting[closing] |= done_np[closing]



def init_collect_state(
    carry, envs, seed, n, n_obs_steps, action_horizon, act_dim, targets
):
    """Build the collector state. An empty carry resets the env rows. Otherwise rebuild from the carried windows."""
    try:
        if carry is None:
            obs, _ = envs.reset(seed=[seed + i for i in range(n)])
            if torch.is_tensor(obs):
                obs0 = obs.to(dtype=OBS_DTYPE)
            else:
                obs0 = torch.from_numpy(np.asarray(obs, dtype=np.float32))
            state = make_state(
                n, n_obs_steps, int(obs0.shape[-1]), action_horizon, act_dim,
                obs0, remaining=sum(targets), device=obs0.device,
            )
        else:
            state = make_state(
                n, n_obs_steps, int(carry.windows.shape[-1]), action_horizon,
                act_dim, carry.windows, remaining=sum(targets),
                device=carry.windows.device,
            )
            state.awaiting[:] = np.asarray(carry.awaiting, dtype=bool)
    except (ValueError, RuntimeError, KeyError):
        raise
    return state


def run_collect_loop(
    state, envs, policy, low, high, targets, n_action_steps, buffer,
    n_obs_steps, action_horizon,
):
    """Plan chunks, step the envs, and close chunks until every row reaches its step target."""
    # The writer is chosen once per rollout, not once per step.
    batched = is_torch_env(envs)
    targets_np = targets.numpy()
    while state.remaining:
        ready = (~state.active) & (~state.awaiting) & (state.executed_total < targets_np)
        envs_needing_a_chunk = np.flatnonzero(ready).tolist()
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
            buffer, n_obs_steps, batched=batched,
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
    """Collect executed action chunks from a vectorized env into ``buffer``.

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
    reward_normalizer=None,
    ):
    """The advantages, the regression targets, and the GAE diagnostics."""
    if cfg.normalize_rewards and reward_normalizer is not None:
        step_rewards = reward_normalizer(active["step_rewards"].reshape(-1, 1)).reshape(active["step_rewards"].shape)
    else:
        step_rewards = active["step_rewards"]
    raw_advantages, raw_returns = compute_chunk_advantages(
        step_rewards,
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
    normalized_advantages = raw_advantages
    if cfg.normalize_advantage and raw_advantages.numel() > 0:
        normalized_advantages = (raw_advantages - raw_advantages.mean()) / (
            raw_advantages.std(correction=0).clamp_min(1e-8)
        )
    normalized_advantages = normalized_advantages.clamp(
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
        normalized_advantages.to(storage_dtype),
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
    """The bootstrap action, the normalized states, and the two critic reads."""
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
                {"observation.state": to_device_async(next_states[needs_proposal], device)},
                start_from_zero=False,
            )
            # The backup's action lives in the same space as the stored
            # chunks: the policy's raw action space.
            boot_actions[needs_proposal] = proposed

        # The training pass reloads the batch in the storage dtype before it
        # normalizes, so the rollout normalizes the same values it will train
        # on. Normalising the fp32 buffer states here would give the rollout
        # and the training pass two different conditions, and the behavior
        # loss would not be the loss the training pass recomputes.
        all_chunk_normalized_states = policy.obs_normalizer(
            torch.cat([states, next_states], dim=0).to(storage_dtype)
        )
        normalized_states = all_chunk_normalized_states[:n_chunks]
        normalized_next_states = all_chunk_normalized_states[n_chunks:]
        exec_length_frac = policy.exec_length_frac(lengths)
        boot_exec_frac = torch.full(
            (lengths.shape[0],),
            cfg.n_action_steps / float(cfg.action_horizon),
            device=lengths.device,
            dtype=torch.float32,
        )
        # The critic read is a target, not exploration. A dropout layer in
        # the critic re-rolls the value on every call, so the critic reads in
        # eval mode. The actor proposal above stays in train mode on purpose:
        # its dropout is part of the behavior noise.
        critic_training = policy.critic.training
        policy.critic.eval()
        policy.critic_target.eval()  # invariant: target always eval
        try:
            with autocast_context(device):
                actions_squashed = squash_to_env_bounds(actions, policy.action_low, policy.action_high)
                chunk_len_mask = (torch.arange(cfg.action_horizon, device=actions.device).unsqueeze(0) < active["chunk_lengths"].unsqueeze(1)).unsqueeze(-1).float()
                actions_squashed = actions_squashed * chunk_len_mask
                q_current_chunk = subsampled_min(policy.critic(normalized_states, actions_squashed, exec_length_frac).float())
                boot_actions_squashed = squash_to_env_bounds(boot_actions, policy.action_low, policy.action_high)
                boot_mask = (torch.arange(cfg.action_horizon, device=boot_actions.device).unsqueeze(0) < cfg.n_action_steps).unsqueeze(-1).float()
                boot_actions_squashed = boot_actions_squashed * boot_mask
                q_bootstrap = subsampled_min(policy.critic_target(normalized_next_states, boot_actions_squashed, boot_exec_frac).float())
        finally:
            policy.critic.train(critic_training)

    return (
        normalized_states,
        normalized_next_states,
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

    # A torch env keeps its frames on its own device, so the buffer is built
    # there and the rollout never touches host memory. A numpy env collects on
    # the host and pays one H2D after the rollout, overlapped with the update.
    buffer_device = (
        torch.device(getattr(env, "device", "cpu"))
        if is_torch_env(env)
        else torch.device("cpu")
    )

    def propose_batch(states: torch.Tensor) -> torch.Tensor:
        """Predict one raw chunk from the policy and stage it on the buffer device."""
        # One chunk is one cudagraph step: the previous chunk's graphs are
        # dead here, so the manager may recycle their memory.
        cudagraph_step_begin()
        chunk = policy.predict_action_chunk(
            {"observation.state": to_device_async(states, device)}, start_from_zero=not noisy
        )
        # The buffer decides the destination: a GPU rollout keeps the chunk on
        # the device instead of a D2H followed by an H2D.
        return chunk.to(buffer_device)

    if buffers.h2d_event is not None:
        buffers.h2d_event.synchronize()
        buffers.h2d_event = None

    num_envs = int(getattr(env, "num_envs", 1))
    action_dim = int(cfg.action_dim)
    obs_dim = int(cfg.obs_dim)
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
        buffers.host = make_buffer(
            device=buffer_device,
            pin_memory=buffer_device.type == "cpu" and device.type == "cuda",
        )
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
    if num_envs == 1 and not is_torch_env(env):
        if buffers.env_wrapper is None:
            import gymnasium as gym

            # The wrapper lives for the whole run. A per-rollout close()
            # walks into close_extras and closes the caller's training env.
            buffers.env_wrapper = gym.vector.SyncVectorEnv([lambda: env])
        envs = buffers.env_wrapper
    else:
        envs = env
    collect_chunks_vec(envs, propose_batch, **collect_kwargs)

    if buffer.device.type == "cuda":
        # Collected on the device already: the host copy would be a round trip.
        buffers.device = buffer
        active = buffer.active_prefix()
    elif device.type == "cuda":
        if buffers.device is None or buffers.device.shape_key() != buffer.shape_key():
            buffers.device = make_buffer(device=device)
        buffer.copy_to(buffers.device, non_blocking=True)
        buffers.h2d_event = torch.cuda.Event()
        buffers.h2d_event.record()
        active = buffers.device.active_prefix()
    else:
        active = buffer.active_prefix()
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
        normalized_states,
        normalized_next_states,
        boot_actions,
        q_current_chunk,
        q_bootstrap,
    ) = bootstrap_values(policy, active, n_chunks, device, storage_dtype)
    raw_advantages, normalized_advantages, raw_returns, gae_diagnostics = (
        chunk_advantages(
            cfg, active, n_chunks, q_current_chunk, q_bootstrap, storage_dtype,
            reward_normalizer=getattr(policy, "reward_normalizer", None),
        )
    )

    lengths = active["chunk_lengths"]
    actions = active["chunk_actions"]
    states = active["chunk_state"]

    action_is_pad = torch.arange(cfg.action_horizon, device=device).unsqueeze(
        0
    ) >= lengths.unsqueeze(1)

    t = sample_mc_timesteps(cfg, n_chunks, device)
    eps = sample_eps(cfg, n_chunks, int(cfg.action_dim), device)

    (
        behavior_cfm_loss,
        behavior_cfm_loss_token,
        behavior_velocity_token,
    ) = policy.compute_behavior_cfm_losses(
        normalized_states, actions, t, eps, action_is_pad
    )

    return {
        "observation.state": states,
        "action": actions,
        "action_is_pad": action_is_pad,
        "raw_advantages": raw_advantages,
        "normalized_advantages": normalized_advantages,
        "raw_returns": raw_returns,
        "normalizer_observations": active["step_frames"].detach(),
        "boot_observations": normalized_next_states.to(storage_dtype),
        "boot_actions": boot_actions.to(storage_dtype),
        "boot_values_start": q_bootstrap.unsqueeze(1),
        "mc_t": t,
        "mc_eps": eps,
        "gae_diagnostics": gae_diagnostics,
        "behavior_cfm_loss": behavior_cfm_loss,
        "behavior_cfm_loss_token": behavior_cfm_loss_token,
        "behavior_velocity_token": behavior_velocity_token,
        "step_rewards": active.get("step_rewards", torch.tensor([])),
    }
