"""Owns: chunk collection, advantage computation, bootstrap. Key: compute_chunk_advantages, run_rollout.
The rollout procedure, as a module function.
"""

from __future__ import annotations
import torch
import contextlib
import dataclasses
from collections.abc import Callable
import numpy as np
from myfpo.config import validate_num_action_steps
from myfpo.env import resolve_action_bounds
from myfpo.actions import bound_for_storage, to_critic_action, to_env_action
from dataclasses import dataclass
from myfpo.flow import (
    compute_gae_diagnostics,
    sample_eps,
    sample_mc_timesteps,
    subsampled_min,
)
from myfpo.nn import (
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
    v_current: torch.Tensor,
    v_next: torch.Tensor,
    chunk_starts: torch.Tensor,
    chunk_lengths: torch.Tensor,
    *,
    chunk_terminated: torch.Tensor,
    chunk_size: int,
    gamma: float,
    chunk_truncated: torch.Tensor | None = None,
    chunk_prev: torch.Tensor | None = None,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``[n]`` advantages and the ``[n]`` regression target.

    ``v_current``/``v_next`` are state baselines (estimates of V^π, not the
    Q read of the taken chunk). The target is ``advantage + v_current``: a
    TD(lambda) backup whose value equals the lambda-return, an unbiased
    estimate of Q^π(s, a) at the taken chunk. Chunks from different envs
    interleave in the buffer, so adjacency is resolved through ``chunk_prev``
    rather than by buffer position.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")

    device = v_current.device
    dtype = torch.float32

    rewards = step_rewards.to(device=device, dtype=dtype).reshape(-1)
    v_current = v_current.to(device=device, dtype=dtype).reshape(-1)
    v_next = v_next.to(device=device, dtype=dtype).reshape(-1)
    starts = chunk_starts.to(device=device, dtype=torch.long).reshape(-1)
    lengths = chunk_lengths.to(device=device, dtype=torch.long).reshape(-1)
    terminated = chunk_terminated.to(device=device, dtype=torch.bool).reshape(-1)

    n_chunks = starts.numel()
    if any(
        t.numel() != n_chunks
        for t in (lengths, terminated, v_current, v_next)
    ):
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
        if chunk_prev is None:
            raise ValueError(
                "chunk_prev is required when gae_lambda > 0: chunks from different "
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
    chunk_rewards = chunk_rewards + gamma_h * injected * v_next
    bootstrap = gamma_h * (~done)
    delta = chunk_rewards + bootstrap * v_next - v_current

    if gae_lambda == 0.0:
        return delta, delta + v_current

    prev = chunk_prev.to(device=device, dtype=torch.long).reshape(-1)
    if prev.numel() != n_chunks:
        raise ValueError(f"chunk_prev must have {n_chunks} entries, got {prev.numel()}")
    # The trace index is a Python int per node. Keeping the successor vector on
    # the host turns one CUDA sync per node into none.
    linked = prev >= 0
    successor = torch.full((n_chunks,), -1, dtype=torch.long, device=prev.device)
    successor[prev[linked]] = torch.arange(n_chunks, device=prev.device)[linked]
    successor = successor.cpu()
    trace = gamma_h * (~done) * gae_lambda
    adv = gae_trace(delta, trace, successor)
    return adv, adv + v_current


def gae_trace(
    delta: torch.Tensor, trace: torch.Tensor, successor: torch.Tensor
) -> torch.Tensor:
    """The advantage of each chunk, traced back over its successors.

    The recursion is sequential, so it runs on the host. On the GPU it is one
    kernel launch per node, about 6k per 1024-chunk rollout.
    """
    d = delta.detach().cpu()
    t = trace.detach().cpu()
    adv = d.clone()
    for i in range(d.numel() - 1, -1, -1):
        j = int(successor[i])
        if j >= 0:
            adv[i] = d[i] + t[i] * adv[j]
    return adv.to(device=delta.device, dtype=delta.dtype)

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
        # max_chunks owns both the chunk rows and the write guard. Two
        # capacities once disagreed, so the guard passed and the row write
        # raised IndexError outside collect_chunks_vec's flush handler.
        self.max_chunks = max_chunks
        self.max_steps = max_steps
        self.obs_steps = obs_steps
        self.obs_dim = obs_dim
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.pin_memory = pin_memory

        self.normalizer_observations = self.zeros_for((max_steps, obs_dim), OBS_DTYPE)
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
        # Previous chunk in the same env, recorded at write time. GAE adjacency,
        # not a reconstruction from env_id.
        self.chunk_prev = self.zeros_for((max_chunks,), INDEX_DTYPE).fill_(-1)
        self._last_chunk: dict[int, int] = {}
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

    @staticmethod
    def layout_key(
        max_chunks: int,
        max_steps: int,
        obs_steps: int,
        obs_dim: int,
        action_horizon: int,
        action_dim: int,
    ) -> tuple[int, ...]:
        """The buffer layout signature: the six shape fields, and nothing else."""
        return (max_chunks, max_steps, obs_steps, obs_dim, action_horizon, action_dim)

    def shape_key(self) -> tuple[int, ...]:
        """The buffer layout signature: the six shape fields, and nothing else."""
        return RolloutBuffer.layout_key(
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
        self._last_chunk.clear()


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
        env_ids,
        n_steps: int,
    ) -> None:
        """Append ``k`` executed chunks in one gathered write.

        Chunk ``j`` starts at the running offset, and its first ``lengths[j]``
        frames are copied verbatim. ``n_steps`` is the host-side sum of
        ``lengths``, passed in rather than computed, so the guard does not sync
        the device. ``env_ids`` is the chunk's env id per row, and it stays on
        the host: the chunk chain is host bookkeeping, and reading the ids off
        the device would sync the collector once per closing step.
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
        # The ids come in on the host. The chunk chain below reads them there,
        # and only the id column crosses to the device, once.
        ids = torch.as_tensor(env_ids).detach().cpu().tolist()
        self.chunk_env_id[lo:hi] = torch.as_tensor(
            ids, dtype=self.chunk_env_id.dtype
        ).to(self.chunk_env_id.device)
        # Chain each env's chunks in row order. One host pass (env_ids come from
        # Python control flow on the caller side), one device copy.
        prev_list: list[int] = []
        for j, raw_e in enumerate(ids):
            e = int(raw_e)
            prev_list.append(self._last_chunk.get(e, -1))
            self._last_chunk[e] = lo + j
        prev_t = torch.tensor(prev_list, dtype=INDEX_DTYPE)
        if self.chunk_prev.device != prev_t.device:
            prev_t = prev_t.to(device=self.chunk_prev.device, non_blocking=True)
        self.chunk_prev[lo:hi] = prev_t
        self.normalizer_observations[step_lo:step_hi] = frames.reshape(-1, self.obs_dim)[mask]
        self.step_rewards[step_lo:step_hi] = rewards.reshape(-1)[mask]
        self.n_chunks = hi
        self.n_steps = step_hi

    def copy_to(self, other: "RolloutBuffer", *, non_blocking: bool = False) -> None:
        """Copy the active prefix into a compatible buffer, e.g. onto a GPU."""
        names = (
            "normalizer_observations",
            "step_rewards",
            "chunk_state",
            "chunk_next_state",
            "chunk_actions",
            "chunk_starts",
            "chunk_lengths",
            "chunk_terminated",
            "chunk_truncated",
            "chunk_env_id",
            "chunk_prev",
        )
        def prefix(buffer, name):
            """Slice one field to the rows this buffer wrote."""
            n = self.n_steps if name in ("normalizer_observations", "step_rewards") else self.n_chunks
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
            "normalizer_observations": self.normalizer_observations[:n_steps],
            "step_rewards": self.step_rewards[:n_steps],
            "chunk_state": self.chunk_state[:n_chunks],
            "chunk_next_state": self.chunk_next_state[:n_chunks],
            "chunk_actions": self.chunk_actions[:n_chunks],
            "chunk_starts": self.chunk_starts[:n_chunks],
            "chunk_lengths": self.chunk_lengths[:n_chunks],
            "chunk_terminated": self.chunk_terminated[:n_chunks],
            "chunk_truncated": self.chunk_truncated[:n_chunks],
            "chunk_env_id": self.chunk_env_id[:n_chunks],
            "chunk_prev": self.chunk_prev[:n_chunks],
        }


def check_chunk_dims(
    num_obs_steps: int,
    num_action_steps: int,
    action_horizon: int,
) -> tuple[int, int]:
    """Check the window width and the execution span."""
    validate_num_action_steps(num_action_steps, action_horizon)
    return num_obs_steps, action_horizon


def validate_autoreset_mode(envs) -> str | None:
    """Accept the two autoreset contracts the collector implements.

    Returns the mode name ("NEXT_STEP" or "SAME_STEP"), or None when the env
    does not declare one — then the collector assumes NEXT_STEP, as gymnasium
    does. SAME_STEP envs (ManiSkill) reset inside the done step() and hand
    the reset frame back as the done step's observation; the collector
    adopts that frame as the new episode's o₀ natively, with no forced
    extra step.
    """
    mode = getattr(envs, "autoreset_mode", None)
    if mode is None:
        return None
    name = getattr(mode, "name", None) or str(mode)
    if name in ("NEXT_STEP", "SAME_STEP"):
        return name
    raise ValueError(
        f"collect_chunks_vec requires AutoresetMode NEXT_STEP or SAME_STEP "
        f"(gymnasium default NEXT_STEP), got {mode!r}. The collector can only "
        "interpret the reset protocol of these two modes."
    )


@dataclasses.dataclass
class CollectState:
    """The per-env state of the collection loop.

    The chunk an env is executing is held in registers indexed by env row
    (``raw_plan``, ``start_window``, ``step_limits``), not in one Python object
    per env. That is what lets a closing step gather every finished chunk at
    once.

    The control-flow bookkeeping (``chunk_pos``, ``active``, ``awaiting``,
    ``executed_total``, ``roll_table``) is host memory: it decides what the
    Python loop does next, and reading it must not sync the device.
    """

    windows: torch.Tensor
    reward_windows: torch.Tensor
    normalizer_observations: torch.Tensor
    active: np.ndarray
    step_limits: np.ndarray
    raw_plan: torch.Tensor
    start_window: torch.Tensor
    chunk_pos: torch.Tensor
    executed_plan: torch.Tensor
    next_actions: torch.Tensor
    executed_total: np.ndarray
    awaiting: np.ndarray
    remaining: int
    roll_table: torch.Tensor
    action_processing_mode: str
    action_noise_std: float
    autoreset_mode: str | None = None


def make_state(
    n: int,
    num_obs_steps: int,
    obs_dim: int,
    action_horizon: int,
    act_dim: int,
    obs0: torch.Tensor,
    remaining: int,
    device: torch.device | str = "cpu",
    action_processing_mode: str = "tanh_squash",
    action_noise_std: float = 0.0,
    autoreset_mode: str | None = None,
) -> CollectState:
    """The state the collection loop starts with.

    The observation, reward, plan and action storage live on ``device``: on a
    GPU sim they never leave it. The index bookkeeping (``chunk_pos``,
    ``roll_table``) stays on the host, because it drives Python control flow:
    moving it would turn one sync per env per step into the common case.
    """
    device = torch.device(device)
    if obs0.dim() == 3:
        if tuple(obs0.shape) != (n, num_obs_steps, obs_dim):
            raise ValueError(
                f"obs0 window shape {tuple(obs0.shape)} != "
                f"{(n, num_obs_steps, obs_dim)}"
            )
        windows = obs0.to(device=device, dtype=OBS_DTYPE).clone()
    elif obs0.dim() == 2:
        windows = torch.zeros(n, num_obs_steps, obs_dim, dtype=OBS_DTYPE, device=device)
        windows.copy_(obs0.to(device=device, dtype=OBS_DTYPE)[:, None, :].expand_as(windows))
    else:
        raise ValueError(
            "obs0 must have shape [n, obs_dim] or [n, num_obs_steps, obs_dim], "
            f"got {tuple(obs0.shape)}"
        )
    return CollectState(
        windows=windows,
        reward_windows=torch.zeros(n, action_horizon, dtype=REWARD_DTYPE, device=device),
        normalizer_observations=torch.zeros(
            n, action_horizon, obs_dim, dtype=OBS_DTYPE, device=device
        ),
        active=np.zeros(n, dtype=bool),
        step_limits=np.zeros(n, dtype=np.int64),
        raw_plan=torch.zeros(
            n, action_horizon, act_dim, dtype=torch.float32, device=device
        ),
        start_window=torch.zeros(n, num_obs_steps, obs_dim, dtype=OBS_DTYPE, device=device),
        chunk_pos=torch.zeros(n, dtype=torch.long),
        executed_plan=torch.zeros(
            n + 1, action_horizon, act_dim, dtype=torch.float32, device=device
        ),
        next_actions=torch.zeros((n, act_dim), dtype=torch.float32, device=device),
        executed_total=np.zeros(n, dtype=np.int64),
        awaiting=np.zeros(n, dtype=bool),
        remaining=remaining,
        roll_table=(
            torch.arange(num_obs_steps)[:, None]
            + torch.arange(num_obs_steps)[None, :]
        )
        % num_obs_steps,
        action_processing_mode=action_processing_mode,
        action_noise_std=action_noise_std,
        autoreset_mode=autoreset_mode,
    )


def plan_chunks(
    state: CollectState,
    envs_needing_a_chunk: list[int],
    policy: Callable[[torch.Tensor], torch.Tensor],
    low: torch.Tensor,
    high: torch.Tensor,
    targets: list[int],
    num_action_steps: int,
) -> None:
    """Give every idle env a new chunk, and hold it until it runs out."""
    if not envs_needing_a_chunk:
        return
    wavefront = torch.as_tensor(envs_needing_a_chunk, dtype=torch.long)
    states = state.windows[wavefront]
    # The policy returns the chunk in policy space. The env receives it
    # mapped to env bounds, and the buffer keeps the policy-space chunk:
    # the flow is trained on what it produced, not on the env mapping.
    raw_all = policy(states).detach().to(OBS_DTYPE)
    executed_all = to_env_action(raw_all, low, high, mode=state.action_processing_mode)
    state.executed_plan[wavefront] = executed_all
    # The chunk is held in registers rather than in one object per env, so a
    # closing step can gather all finished chunks with a single index.
    state.raw_plan[wavefront] = raw_all
    state.start_window[wavefront] = states
    state.chunk_pos[wavefront] = 0
    rows = np.asarray(envs_needing_a_chunk, dtype=np.int64)
    targets_np = (
        targets.detach().cpu().numpy() if torch.is_tensor(targets)
        else np.asarray(targets, dtype=np.int64)
    )
    state.step_limits[rows] = np.minimum(
        num_action_steps, targets_np[rows] - state.executed_total[rows]
    )
    state.active[rows] = True


def is_torch_env(envs) -> bool:
    """True when the env hands back its own tensors instead of numpy arrays."""
    return callable(getattr(envs, "step_torch", None))


def take_final_observations(info, done_np, reference, device, strict: bool = False):
    """Replace the done rows of ``reference`` with the true terminal frame.

    A done step must report ``o_T``: the collector stores it as the chunk's
    ``next_state``, and the truncation bootstrap reads it. Gymnasium exposes it
    as ``info["final_obs"]`` and ManiSkill as ``info["final_observation"]``
    when the env resets inside the step and returns the reset frame instead.
    Under NEXT_STEP autoreset the step's own observation is already final and
    neither key is present, so the reference passes through unchanged. When
    ``strict`` (SAME_STEP envs), a done row without a final frame raises:
    storing the reset frame as ``o_T`` corrupts the truncation bootstrap.

    The frame may carry one row per env or one row per done row. A shape the
    collector cannot map onto the done rows raises.
    """
    rows = np.flatnonzero(done_np)
    if rows.size == 0:
        return reference
    if not isinstance(info, dict) or (info.get("final_obs") is None and info.get("final_observation") is None):
        if strict:
            raise ValueError(
                f"SAME_STEP env reset inside step() but supplied no final "
                f"observation for done rows {rows.tolist()}; refusing to store the "
                "reset frame as o_T"
            )
        return reference
    final = info.get("final_obs")
    if final is None:
        final = info.get("final_observation")
    idx = torch.from_numpy(rows).to(device=device)
    out = reference.clone()
    n_envs = reference.shape[0]

    def select(values: torch.Tensor) -> torch.Tensor:
        """Pick the done rows out of ``values``.

        The env may report one row per env, or one row per done row. Any
        other shape is a frame the collector cannot read.
        """
        if values.shape[0] == n_envs:
            return values[idx]
        if values.shape[0] == rows.size:
            return values
        raise ValueError(
            f"the final observation has {values.shape[0]} rows; the collector "
            f"needs {n_envs} (one per env) or {rows.size} (one per done row)"
        )

    if torch.is_tensor(final):
        out[idx] = select(final.to(device=device, dtype=reference.dtype))
        return out
    array = np.asarray(final)
    if array.dtype == object:
        for k, i in enumerate(rows):
            value = array[i] if array.shape[0] == n_envs else array[k]
            if value is None:
                if strict:
                    raise ValueError(
                        f"SAME_STEP env reset inside step() but the final "
                        f"observation for done row {i} is None; refusing to "
                        "store the reset frame as o_T"
                    )
                continue
            out[i] = torch.as_tensor(
                np.asarray(value, dtype=np.float32), device=device
            )
        return out
    out[idx] = select(torch.as_tensor(array, dtype=reference.dtype, device=device))
    return out


def step_envs(state: CollectState, envs, action_horizon: int):
    """Send the next action of every env, and record what came back.

    Returns ``(obs_t, term_np, trunc_np, done_np, reset_frames)``.
    ``reset_frames`` is the raw returned observation of the done rows under
    SAME_STEP autoreset (the env resets inside the done step, so it is the
    first frame of the new episode); ``None`` under NEXT_STEP/undeclared.
    """
    n = len(state.active)
    stepping = torch.from_numpy(state.active)
    rows = torch.arange(n)
    live_idx = rows[stepping]
    state.next_actions.copy_(
        state.executed_plan[
            torch.where(stepping, rows, torch.full_like(rows, n)),
            state.chunk_pos.clamp(max=action_horizon - 1),
        ]
    )
    if is_torch_env(envs):
        # A GPU env keeps its frames on the device. Only the three done flags
        # cross to the host, and only because they drive the chunk-close loop.
        obs_t, rew_t, term_t, trunc_t, info = envs.step_torch(state.next_actions)
        raw_obs = obs_t.to(device=state.windows.device, dtype=OBS_DTYPE)
        obs_t = raw_obs
        term_np = term_t.cpu().numpy()
        trunc_np = trunc_t.cpu().numpy()
        rew_t = rew_t.to(device=state.windows.device, dtype=REWARD_DTYPE)
    else:
        obs, rew, term, trunc, info = envs.step(state.next_actions.cpu().numpy())
        term_np = np.asarray(term, dtype=bool)
        trunc_np = np.asarray(trunc, dtype=bool)
        raw_obs = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(
            device=state.windows.device
        )
        obs_t = raw_obs
        rew_t = torch.from_numpy(np.asarray(rew, dtype=np.float32)).to(
            device=state.windows.device
        )
    done_np = term_np | trunc_np
    strict = state.autoreset_mode == "SAME_STEP"
    reset_frames = raw_obs if (strict and done_np.any()) else None
    if done_np.any():
        obs_t = take_final_observations(
            info, done_np, obs_t, state.windows.device, strict=strict
        )

    if live_idx.numel():
        p = state.chunk_pos[live_idx]
        state.reward_windows[live_idx, p] = rew_t[live_idx]
        state.normalizer_observations[live_idx, p] = obs_t[live_idx]
        state.windows[live_idx, p] = obs_t[live_idx]
        state.chunk_pos[live_idx] = p + 1
    return obs_t, term_np, trunc_np, done_np, reset_frames



def close_chunks(
    state: CollectState,
    obs_t: torch.Tensor,
    term_np: np.ndarray,
    trunc_np: np.ndarray,
    done_np: np.ndarray,
    buffer: RolloutBuffer,
    num_obs_steps: int,
    *,
    reset_frames: torch.Tensor | None = None,
    autoreset_mode: str | None = None,
) -> None:
    """Close every chunk that ran to its limit or hit a terminal.

    The classification stays on the host (it is the loop's control flow), but
    every write is a single index operation over all closing rows at once.
    """
    device = state.windows.device
    same_step = autoreset_mode == "SAME_STEP"

    def fill_window(rows_t: torch.Tensor) -> None:
        """Seed a fresh episode's window from the done step's returned frame."""
        if reset_frames is None:
            raise RuntimeError(
                "SAME_STEP env reported a done row but the step carried no "
                "reset frame to seed the next episode's window"
            )
        state.windows[rows_t] = reset_frames[rows_t].unsqueeze(1).expand(
            -1, num_obs_steps, -1
        )

    # An env that resets this step takes the reset frame for its whole window.
    # This is the NEXT_STEP contract only. A SAME_STEP env never waits: its
    # done step already carries the new episode's o₀, and obs_t holds o_T for a
    # done row, so seeding a window from it would store the previous episode's
    # terminal frame as the new episode's first observation.
    if same_step:
        if state.awaiting.any():
            raise RuntimeError(
                f"SAME_STEP env has awaiting rows "
                f"{np.flatnonzero(state.awaiting).tolist()}; the state machine "
                "and the carry disagree with the autoreset mode."
            )
    else:
        waiting = np.flatnonzero(state.awaiting)
        if waiting.size:
            w = torch.as_tensor(waiting, device=device)
            state.windows[w] = obs_t[w].unsqueeze(1).expand(-1, num_obs_steps, -1)
            state.awaiting[waiting] = False

    # A finished row still steps until every row hits its target. Treat each
    # extra step as a one-step chunk, or the carried window goes stale; a done
    # step resets the env, so it waits for the awaiting branch next step.
    idle = np.flatnonzero(~(state.active | state.awaiting))
    if idle.size:
        done_idle = done_np[idle]
        moved = idle[~done_idle]
        if moved.size:
            rows = torch.as_tensor(moved, device=device)
            w = state.windows[rows]
            w[:, :-1] = w[:, 1:].clone()
            w[:, -1] = obs_t[rows]
            state.windows[rows] = w
        if same_step:
            done_rows = idle[done_idle]
            if done_rows.size:
                fill_window(torch.as_tensor(done_rows, device=device))
        else:
            state.awaiting[idle[done_idle]] = True

    pos = state.chunk_pos.numpy()
    closing = np.flatnonzero(state.active & (done_np | (pos >= state.step_limits)))
    if closing.size == 0:
        return

    rows = torch.as_tensor(closing, device=device)
    lengths_np = pos[closing]
    n_new = int(lengths_np.sum())
    lengths = torch.as_tensor(lengths_np, device=device)
    # The ring holds the last num_obs_steps frames; `next_state` wants them in
    # chronological order, which is what the roll table permutes them into.
    perm = state.roll_table[
        torch.as_tensor(pos[closing] % num_obs_steps)
    ].to(device)
    next_window = torch.gather(
        state.windows[rows], 1, perm[:, :, None].expand(-1, -1, state.windows.shape[-1])
    )

    stored_actions = state.raw_plan[rows]
    if state.action_noise_std > 0:
        # Perturb only the stored copy feeding the CFM ratio; the executed plan
        # is untouched (ratio-level entropy regularization). The bound applies
        # after the noise: the stored chunk must stay in the flow's domain.
        stored_actions = stored_actions + state.action_noise_std * torch.randn_like(stored_actions)
        stored_actions = bound_for_storage(stored_actions, state.action_processing_mode)
    buffer.write_chunks(
        start_window=state.start_window[rows],
        next_window=next_window,
        actions=stored_actions,
        lengths=lengths,
        frames=state.normalizer_observations[rows],
        rewards=state.reward_windows[rows],
        terminated=torch.from_numpy(term_np).to(device)[rows],
        truncated=torch.from_numpy(trunc_np).to(device)[rows],
        env_ids=closing,
        n_steps=n_new,
    )

    state.windows[rows] = next_window
    state.executed_total[closing] += lengths_np
    state.remaining -= n_new
    state.active[closing] = False
    if same_step:
        # The done step already carried the new episode's o₀ (the env resets
        # inside the done step). Adopt it instead of idling one step.
        closing_done = closing[done_np[closing]]
        if closing_done.size:
            rows = torch.as_tensor(closing_done, device=device)
            fill_window(rows)
    else:
        state.awaiting[closing] |= done_np[closing]



def init_collect_state(
    carry, envs, seed, n, num_obs_steps, action_horizon, act_dim, targets,
    action_processing_mode: str = "tanh_squash",
    action_noise_std: float = 0.0,
    autoreset_mode: str | None = None,
):
    """Build the collector state. An empty carry resets the env rows. Otherwise rebuild from the carried windows."""
    if carry is None:
        obs, _ = envs.reset(seed=[seed + i for i in range(n)])
        if torch.is_tensor(obs):
            obs0 = obs.to(dtype=OBS_DTYPE)
        else:
            obs0 = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        state = make_state(
            n, num_obs_steps, int(obs0.shape[-1]), action_horizon, act_dim,
            obs0, remaining=sum(targets), device=obs0.device,
            action_processing_mode=action_processing_mode,
            action_noise_std=action_noise_std,
            autoreset_mode=autoreset_mode,
        )
    else:
        state = make_state(
            n, num_obs_steps, int(carry.windows.shape[-1]), action_horizon,
            act_dim, carry.windows, remaining=sum(targets),
            device=carry.windows.device,
            action_processing_mode=action_processing_mode,
            action_noise_std=action_noise_std,
            autoreset_mode=autoreset_mode,
        )
        awaiting = np.asarray(carry.awaiting, dtype=bool)
        # A SAME_STEP env never parks a row on a reset. A carry that claims one
        # comes from a run whose autoreset mode changed, or from an edited
        # checkpoint. Taking it would seed the new episode's window from the
        # previous episode's terminal frame.
        if autoreset_mode == "SAME_STEP" and awaiting.any():
            raise RuntimeError(
                f"SAME_STEP env carries awaiting rows "
                f"{np.flatnonzero(awaiting).tolist()}; the carry and the "
                "autoreset mode disagree."
            )
        state.awaiting[:] = awaiting
    return state


def run_collect_loop(
    state, envs, policy, low, high, targets, num_action_steps, buffer,
    num_obs_steps, action_horizon,
    autoreset_mode: str | None = None,
):
    """Plan chunks, step the envs, and close chunks until every row reaches its step target."""
    targets_np = targets.numpy()
    while state.remaining:
        ready = (~state.active) & (~state.awaiting) & (state.executed_total < targets_np)
        envs_needing_a_chunk = np.flatnonzero(ready).tolist()
        if envs_needing_a_chunk:
            plan_chunks(
                state, envs_needing_a_chunk, policy, low, high,
                targets, num_action_steps,
            )
        obs_t, term_np, trunc_np, done_np, reset_frames = step_envs(
            state, envs, action_horizon
        )
        close_chunks(
            state, obs_t, term_np, trunc_np, done_np, buffer, num_obs_steps,
            reset_frames=reset_frames, autoreset_mode=autoreset_mode,
        )

def collect_chunks_vec(
    envs,
    policy: Callable[[torch.Tensor], torch.Tensor],
    total_steps: int,
    action_low: float | np.ndarray,
    action_high: float | np.ndarray,
    *,
    num_obs_steps: int,
    num_action_steps: int,
    seed: int | None = None,
    on_error_flush: Callable[[], None] | None = None,
    buffer: RolloutBuffer,
    carry: EnvCarry | None = None,
    action_processing_mode: str = "tanh_squash",
    action_noise_std: float = 0.0,
) -> RolloutBuffer:
    """Collect executed action chunks from a vectorized env into ``buffer``.

    ``carry is None`` resets the envs: the first rollout of a run. Any later
    call passes the previous call's ``EnvCarry``, so an episode spans rollout
    boundaries. A carried ``awaiting`` env gets its reset observation through
    the vec env's autoreset step.
    """
    num_obs_steps, action_horizon = check_chunk_dims(
        num_obs_steps, num_action_steps, buffer.action_horizon
    )

    mode = validate_autoreset_mode(envs)

    n = int(envs.num_envs)
    base_target, remainder = divmod(total_steps, n)
    targets = torch.tensor([base_target + (i < remainder) for i in range(n)], dtype=torch.long)

    low, high = resolve_action_bounds(action_low, action_high)

    if num_action_steps > num_obs_steps:
        raise ValueError(
            f"num_action_steps={num_action_steps} exceeds num_obs_steps={num_obs_steps}: "
            "the vec obs window is a ring of num_obs_steps rows and cannot hold a "
            "chunk longer than itself"
        )
    if seed is None:
        seed = int(torch.randint(0, 2**31, (1,)).item())
    act_dim = int(np.asarray(envs.single_action_space.shape).prod())

    state = init_collect_state(
        carry, envs, seed, n, num_obs_steps, action_horizon, act_dim, targets,
        action_processing_mode=action_processing_mode,
        action_noise_std=action_noise_std,
        autoreset_mode=mode,
    )
    try:
        run_collect_loop(
            state, envs, policy, low, high, targets, num_action_steps,
            buffer, num_obs_steps, action_horizon,
            autoreset_mode=mode,
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
    """The buffers a rollout reuses between calls.

    The env carry is not held here: it lives on ``host``, the buffer the
    collector reads it from, so one object owns it.
    """

    host: RolloutBuffer | None = None
    device: RolloutBuffer | None = None
    h2d_event: torch.cuda.Event | None = None
    env_wrapper: object | None = None


def chunk_advantages(
    cfg,
    active: dict[str, torch.Tensor],
    n_chunks: int,
    v_current: torch.Tensor,
    v_next: torch.Tensor,
    storage_dtype: torch.dtype,
    reward_normalizer=None,
    ):
    """The advantages, the regression targets, and the GAE diagnostics.

    ``v_current``/``v_next`` are the state baselines (mean of K fresh policy
    chunks per state). The regression target is the lambda-return against
    those baselines; it remains an unbiased estimate of Q^π(s, a).
    """
    if cfg.normalize_rewards and reward_normalizer is not None:
        step_rewards = reward_normalizer(active["step_rewards"].reshape(-1, 1)).reshape(active["step_rewards"].shape)
    else:
        step_rewards = active["step_rewards"]
    raw_advantages, raw_returns = compute_chunk_advantages(
        step_rewards,
        v_current,
        v_next,
        active["chunk_starts"],
        active["chunk_lengths"],
        chunk_terminated=active["chunk_terminated"],
        chunk_size=cfg.action_horizon,
        gamma=cfg.gamma,
        chunk_truncated=active["chunk_truncated"],
        chunk_prev=active["chunk_prev"],
        gae_lambda=cfg.gae_lambda,
    )
    normalized_advantages = raw_advantages
    if cfg.normalize_advantage and raw_advantages.numel() > 0:
        normalized_advantages = (raw_advantages - raw_advantages.mean()) / (
            raw_advantages.std(correction=0).clamp_min(1e-8)
        )
    gae_diagnostics = compute_gae_diagnostics(
        active["chunk_env_id"][:n_chunks],
        active["chunk_terminated"][:n_chunks],
        active["chunk_truncated"][:n_chunks],
        raw_advantages,
        raw_returns,
    )
    return (
        # raw_advantages stays fp32; a bf16 standard deviation is only good to
        # about 0.4%.
        raw_advantages,
        normalized_advantages.to(storage_dtype),
        raw_returns,
        gae_diagnostics,
    )


def estimate_state_values(
    policy,
    raw_states: torch.Tensor,
    normalized_states: torch.Tensor,
    *,
    exec_length_frac: torch.Tensor,
    mask: torch.Tensor,
    critic,
) -> torch.Tensor:
    """V̂(s): the subsampled-min critic averaged over fresh policy chunks.

    The advantage needs a state baseline. A taken-chunk baseline makes the
    advantage zero-mean given (s, a) under a calibrated critic, so the actor
    trains on critic error; here each state's baseline averages K fresh
    policy samples, estimating V^π(s) with the same critic reduction the Q
    reads use. ``raw_states`` are pre-normalizer (encode_states normalizes
    internally); ``normalized_states`` must be the same tensors the critic
    reads for the taken chunk.

    The baseline is drawn from the behaviour distribution, not from the bare
    flow: the proposal carries the exploration perturbation, and the stored
    chunks carry the same perturbation. The baseline and the data therefore
    come from the same distribution, which is what makes the advantage
    centred. The min over the critic members is the REDQ pessimism; it biases
    the baseline down, on purpose.
    """
    cfg = policy.config
    # The encoder runs once for the whole sample set: the window is the same
    # for every sample, so re-encoding it per sample is a repeated read of
    # identical data.
    condition = policy.encode_states(raw_states)
    qs = []
    for _ in range(cfg.num_baseline_samples):
        proposed = policy.sample_chunk(condition, start_from_zero=False)
        proposed = to_critic_action(
            proposed, policy.action_low, policy.action_high,
            mode=cfg.action_processing_mode, mask=mask,
        )
        qs.append(
            subsampled_min(critic(normalized_states, proposed, exec_length_frac).float())
        )
    return torch.stack(qs).mean(0)


def bootstrap_values(
    policy,
    active: dict[str, torch.Tensor],
    n_chunks: int,
    device,
    storage_dtype: torch.dtype,
    *,
    need_bootstrap: bool = False,
):
    """The bootstrap action, the normalized states, and the V(s) baseline reads.

    The advantage baseline is a state value, not Q(s, a_taken): each read
    averages ``cfg.num_baseline_samples`` fresh policy chunks.

    ``need_bootstrap`` turns on the diagnostic-only bootstrap: the fresh
    proposal for the next state and its Q snapshot. Only
    ``measure_value_drift`` reads them, and the proposal costs one flow
    integration over every chunk. Without the flag both come back as None.
    """
    cfg = policy.config
    lengths = active["chunk_lengths"]
    actions = active["chunk_actions"]
    states = active["chunk_state"]

    with torch.inference_mode():
        next_states = active["chunk_next_state"]
        boot_actions = None
        if need_bootstrap:
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
            cfg.num_action_steps / float(cfg.action_horizon),
            device=lengths.device,
            dtype=torch.float32,
        )
        # The critic read is a target, not exploration, so the critic reads in
        # eval mode. The actor proposal above stays in train mode on purpose.
        critic_training = policy.critic.training
        policy.critic.eval()
        policy.critic_target.eval()  # invariant: target always eval
        try:
            with autocast_context(device):
                chunk_len_mask = (torch.arange(cfg.action_horizon, device=actions.device).unsqueeze(0) < active["chunk_lengths"].unsqueeze(1)).unsqueeze(-1).float()
                # The next state's proposal runs for num_action_steps, so its
                # mask is the executed prefix, not the chunk length.
                boot_mask = (torch.arange(cfg.action_horizon, device=actions.device).unsqueeze(0) < cfg.num_action_steps).unsqueeze(-1).float()
                q_bootstrap = None
                if need_bootstrap:
                    boot_actions_for_critic = to_critic_action(
                        boot_actions, policy.action_low, policy.action_high,
                        mode=cfg.action_processing_mode, mask=boot_mask,
                    )
                    # Diagnostic-only snapshot for measure_value_drift
                    # (log_drift_diagnostics); never feeds the advantage.
                    q_bootstrap = subsampled_min(policy.critic_target(normalized_next_states, boot_actions_for_critic, boot_exec_frac).float())
                v_current = estimate_state_values(
                    policy, states, normalized_states,
                    exec_length_frac=exec_length_frac, mask=chunk_len_mask,
                    critic=policy.critic,
                )
                v_next = estimate_state_values(
                    policy, next_states, normalized_next_states,
                    exec_length_frac=boot_exec_frac, mask=boot_mask,
                    critic=policy.critic_target,
                )
        finally:
            policy.critic.train(critic_training)

    return (
        normalized_states,
        normalized_next_states,
        boot_actions,
        v_current,
        v_next,
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
    shape_args = dict(
        max_chunks=max_chunks,
        max_steps=total_steps,
        obs_steps=cfg.num_obs_steps,
        obs_dim=obs_dim,
        action_horizon=cfg.action_horizon,
        action_dim=action_dim,
    )
    buffer_key = RolloutBuffer.layout_key(**shape_args)

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
        num_obs_steps=cfg.num_obs_steps,
        num_action_steps=cfg.num_action_steps,
        buffer=buffer,
        carry=buffer.carry,
        action_processing_mode=cfg.action_processing_mode,
        action_noise_std=cfg.storage_action_noise_std,
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
    need_bootstrap: bool = False,
) -> dict[str, torch.Tensor]:
    """Roll out open-loop chunks, build a training batch.

    ``need_bootstrap`` adds the diagnostic-only bootstrap values to the batch.
    Without them the batch carries no ``boot_actions`` and no
    ``boot_values_start``, so the drift diagnostic cannot run.
    """
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
        v_current,
        v_next,
        q_bootstrap,
    ) = bootstrap_values(
        policy, active, n_chunks, device, storage_dtype,
        need_bootstrap=need_bootstrap,
    )
    raw_advantages, normalized_advantages, raw_returns, gae_diagnostics = (
        chunk_advantages(
            cfg, active, n_chunks, v_current, v_next, storage_dtype,
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

    behavior_cfm_loss, behavior_cfm_loss_token = policy.compute_behavior_cfm_losses(
        normalized_states, actions, t, eps, action_is_pad
    )

    batch = {
        "observation.state": states,
        "action": actions,
        "action_is_pad": action_is_pad,
        "chunk_truncated": active["chunk_truncated"],
        "chunk_env_id": active["chunk_env_id"],
        "raw_advantages": raw_advantages,
        "normalized_advantages": normalized_advantages,
        "raw_returns": raw_returns,
        "normalizer_observations": active["normalizer_observations"].detach(),
        "boot_observations": normalized_next_states.to(storage_dtype),
        "mc_t": t,
        "mc_eps": eps,
        "gae_diagnostics": gae_diagnostics,
        "behavior_cfm_loss": behavior_cfm_loss,
        "behavior_cfm_loss_token": behavior_cfm_loss_token,
        "step_rewards": active["step_rewards"],
    }
    if need_bootstrap:
        # The diagnostic reads these two. They cost one flow integration and
        # one critic read, so they are only built when the diagnostic runs.
        batch["boot_actions"] = boot_actions.to(storage_dtype)
        batch["boot_values_start"] = q_bootstrap.unsqueeze(1)
    return batch
