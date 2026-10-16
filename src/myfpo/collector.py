"""Chunked open-loop rollout collector.

Autoreset
---------
``collect_chunks_vec`` runs the vec-env reset as a state machine. An env that
returns ``done=True`` is flagged ``awaiting``. The function picks up its reset
observation on the *next* call to ``envs.step()``.

This is correct only under the default of gymnasium,
``AutoresetMode.NEXT_STEP``. Under ``SAME_STEP``, the reset observation arrives
with ``done=True``. The state machine would then step a fresh env with a zero
action. The function validates the mode before the run.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from myfpo.configuration_sjfpo import SJFPOConfig

__all__ = ["RolloutBuffer", "collect_chunks_vec"]


# Lazily-constructed config (import cost only on first use).
_DEFAULT_CFG: SJFPOConfig | None = None


def get_default_config() -> SJFPOConfig:
    """Default config singleton with feature specs for smoke/training."""
    global _DEFAULT_CFG
    if _DEFAULT_CFG is None:
        # Include minimal feature specs so validation passes.
        from lerobot.configs.types import FeatureType, PolicyFeature
        _DEFAULT_CFG = SJFPOConfig(
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,))
            },
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        )
    return _DEFAULT_CFG


# Dtypes: observations and rewards are fp32 end-to-end. Only policy outputs
# and policy data use the execution-device dtype policy.
_OBS_DTYPE = torch.float32
_REWARD_DTYPE = torch.float32
_INDEX_DTYPE = torch.int64


class RolloutBuffer:
    """Fixed-capacity CPU storage for chunk and executed-step rollout data.

    The buffer has two axes. The chunk axis holds one obs window per chunk. So
    ``total_steps // action_horizon`` rows would be enough. The callers pass
    ``total_steps`` anyway. That gives headroom when chunks end early.

    The step axis holds one frame per executed step. So ``total_steps`` rows is
    the right count there.
    """

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

        # One frame per executed step. The obs windows live on the chunk axis.
        # The envs interleave. So the obs_steps frames of a chunk are not
        # adjacent rows. They cannot be gathered back out of a step timeline.
        #
        # There is deliberately no step_terminated or step_truncated pair. The
        # GAE backup consumes the per-chunk flags.
        self.step_frames = self.zeros_for((max_steps, obs_dim), _OBS_DTYPE)
        self.step_rewards = self.zeros_for((max_steps,), _REWARD_DTYPE)

        self.chunk_state = self.zeros_for(
            (chunk_capacity, obs_steps, obs_dim), _OBS_DTYPE
        )
        self.chunk_next_state = self.zeros_for(
            (chunk_capacity, obs_steps, obs_dim), _OBS_DTYPE
        )
        self.chunk_actions = self.zeros_for(
            (chunk_capacity, action_horizon, action_dim), _OBS_DTYPE
        )
        self.chunk_starts = self.zeros_for((chunk_capacity,), _INDEX_DTYPE)
        self.chunk_lengths = self.zeros_for((chunk_capacity,), _INDEX_DTYPE)
        self.chunk_terminated = self.zeros_for((chunk_capacity,), torch.bool)
        self.chunk_truncated = self.zeros_for((chunk_capacity,), torch.bool)
        self.chunk_clip_counts = self.zeros_for((chunk_capacity,), _INDEX_DTYPE)
        self.chunk_env_id = self.zeros_for((chunk_capacity,), torch.int16)
        self.n_chunks = 0
        self.n_steps = 0

    # Pinning costs one cudaHostAlloc for each tensor. So it pays only when the
    # tensor is large enough for a DMA. Three of the fourteen tensors qualify.
    # The rest are index and flag vectors under 300 KiB. Pinning those costs
    # time and gains nothing.
    _PIN_MIN_BYTES = 1 << 20

    def zeros_for(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """One zero tensor in this buffer's layout, dtype and device."""
        nbytes = dtype.itemsize
        for dim in shape:
            nbytes *= dim
        return torch.zeros(
            *shape,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory and nbytes >= self._PIN_MIN_BYTES,
        )

    def capacity_key(self) -> tuple[int, ...]:
        """Buffer layout signature, device excluded."""
        return (
            self.max_chunks,
            self.max_steps,
            self.obs_steps,
            self.obs_dim,
            self.action_horizon,
            self.action_dim,
            self.pin_memory,
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
        clip_count: int,
        env_id: int = 0,
    ) -> int:
        """Append one executed chunk.

        The argument ``state`` is the obs window that the policy saw. The
        argument ``next_state`` is the window that the chunk ran to. The
        argument ``step_frames`` holds the raw observation after each executed
        step.

        The window chains across the chunk. So the collector supplies the two
        ends and one frame per step.

        The value ``env_id`` is the vec slot. Appends follow wavefront order. So
        consecutive entries usually belong to different envs. The chunk backup
        uses this to find the trajectory boundaries.
        """
        if self.n_chunks >= self.max_chunks:
            raise RuntimeError("RolloutBuffer chunk capacity exceeded")
        reward_t = torch.as_tensor(rewards, dtype=_REWARD_DTYPE, device=self.device).reshape(-1)
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
        # Pre-allocated with exact capacity. The buffer never grows.
        self.chunk_state[chunk_index].copy_(state)
        self.chunk_next_state[chunk_index].copy_(next_state)
        self.step_frames[offset : offset + n_exec].copy_(step_frames)
        self.step_rewards[offset : offset + n_exec].copy_(reward_t)
        self.chunk_actions[chunk_index].copy_(action)
        self.chunk_starts[chunk_index] = offset
        self.chunk_lengths[chunk_index] = n_exec
        self.chunk_terminated[chunk_index] = terminated
        self.chunk_truncated[chunk_index] = truncated
        # Written unconditionally: reset() clears cursors, not storage.
        self.chunk_clip_counts[chunk_index] = int(clip_count)
        self.chunk_env_id[chunk_index] = int(env_id)
        self.n_chunks += 1
        self.n_steps += n_exec
        return chunk_index

    def copy_to(self, other: "RolloutBuffer", *, non_blocking: bool = False) -> None:
        """Copy the active prefix into a compatible buffer, e.g. onto a GPU.

        No capacity or dim check here. A mismatch surfaces as a shape error
        from ``_foreach_copy_``. A caller keys its own buffer off
        :meth:`capacity_key`. That gives this pair a match by construction.
        """
        # Pre-allocated with exact capacity. The buffer never grows.
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
            "chunk_clip_counts",
            "chunk_env_id",
        )
        # One multi-tensor copy rather than 14 launches. Only the active prefix
        # moves, and step_* tensors are indexed by n_steps while the rest are
        # indexed by n_chunks.
        def prefix(buffer, name):
            n = self.n_steps if name.startswith("step_") else self.n_chunks
            return getattr(buffer, name)[:n]

        torch._foreach_copy_(
            [prefix(other, name) for name in names],
            [prefix(self, name) for name in names],
            non_blocking=non_blocking,
        )

    def mark_incomplete(self, chunk_index: int) -> None:
        """No-op, and deliberately so.

        Called when the step budget cuts a chunk short. The episode continues, so
        this must NOT set ``chunk_truncated``: that flag is the env's own time
        limit and drives the GAE bootstrap. Budget cuts therefore stay invisible
        -- there is no flag for them -- which is why
        ``budget_truncated_chunk_frac`` reports NaN rather than 0.0.
        """
        if not 0 <= chunk_index < self.n_chunks:
            raise IndexError(f"chunk index {chunk_index} out of range")

    def active(self) -> dict[str, torch.Tensor]:
        """Contiguous views of the populated prefix.

        Returns buffer name -> view covering only the written region.
        """
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
            "chunk_clip_counts": self.chunk_clip_counts[:n_chunks],
            "chunk_env_id": self.chunk_env_id[:n_chunks],
        }


def as_float_tensor(x: object) -> torch.Tensor:
    """fp32 CPU/GPU tensor, without forcing a device hop.

    A ``torch.Tensor`` keeps its device (no implicit ``.cpu()``). numpy and
    Python inputs lift to a fresh CPU fp32 tensor.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().to(_OBS_DTYPE)
    return torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=_OBS_DTYPE)


def resolve_action_bounds(
    low: float | np.ndarray | torch.Tensor,
    high: float | np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coerce env action bounds into fp32 tensors, strict ``high > low``.

    :meth:`SJFPOPolicy.set_action_bounds` must use the same convention, else a
    policy can build a degenerate dim that only fails on the first rollout.
    """
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clamp a chunk to env bounds, then rescale to ``[-1, 1]``.

    ``proposed`` is ``[H, action_dim]`` or ``[B, H, action_dim]``, so the vec
    collector scales once per wavefront instead of once per env.
    """
    high = low + range_
    clip_mask = (proposed < low) | (proposed > high)
    executed = torch.clamp(proposed, min=low, max=high)
    scaled = 2.0 * (executed - low) / range_ - 1.0
    return executed, scaled, clip_mask.sum(dim=(-2, -1))


def resolve_chunk_args(
    n_obs_steps: int | None,
    n_action_steps: int,
    chunk_size: int | None,
) -> tuple[int, int]:
    """Fill the chunk defaults from the config, then check the execution span.

    The caller adds one further check, that ``n_action_steps`` fits inside the
    obs ring, because the vec window is a ring rather than a slide.
    """
    cfg = get_default_config()
    n_obs_steps = cfg.n_obs_steps if n_obs_steps is None else n_obs_steps
    chunk_size = cfg.action_horizon if chunk_size is None else int(chunk_size)
    if not 1 <= n_action_steps <= chunk_size:
        raise ValueError(
            f"n_action_steps={n_action_steps} must satisfy "
            f"1 <= n <= chunk_size={chunk_size}"
        )
    return n_obs_steps, chunk_size


def flush_on_collection_error(
    on_error_flush: Callable[[], None] | None,
) -> None:
    """Call the error flush hook, swallowing any raise.

    Best effort: a failing flush must not mask the collection error that
    triggered it.
    """
    if on_error_flush is not None:
        try:
            on_error_flush()
        except (ValueError, RuntimeError):
            pass


def validate_autoreset_mode(envs) -> None:
    """Reject vec envs whose autoreset mode breaks the awaiting machine.

    ``collect_chunks_vec`` resets in a two-step handshake: flag the env on the
    ``done=True`` step, take the reset obs on the following ``step()``. Only
    works under ``NEXT_STEP``.
    """
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


def collect_chunks_vec(
    envs,
    policy: Callable[[torch.Tensor], torch.Tensor],
    total_steps: int,
    action_low: float | np.ndarray,
    action_high: float | np.ndarray,
    *,
    n_obs_steps: int | None = None,
    n_action_steps: int,
    chunk_size: int | None = None,
    seed: int | None = None,
    on_error_flush: Callable[[], None] | None = None,
    buffer: RolloutBuffer,
) -> RolloutBuffer:
    """Collect executed action chunks from a vectorised env into ``buffer``.

    The step budget divides evenly across the envs. The remainder goes to the
    first envs. All actions that are still active gather into one tensor per
    wavefront. So the loop costs one device-to-host transfer per chunk, not one
    per step.

    Each env has one cursor, named ``pos_t``. It counts steps into the chunk that
    the env executes now. It rewinds when a new chunk is planned. It also acts
    as the write slot of the obs window, because both are the same number.

    The window is a ring. The function rolls it back into chronological order
    once per closing chunk.

    Needs ``AutoresetMode.NEXT_STEP``, see :func:`validate_autoreset_mode`.
    """
    n_obs_steps, chunk_size = resolve_chunk_args(
        n_obs_steps, n_action_steps, chunk_size
    )

    validate_autoreset_mode(envs)

    n = int(envs.num_envs)
    base_target, remainder = divmod(total_steps, n)
    targets = [base_target + int(i < remainder) for i in range(n)]

    low, range_ = resolve_action_bounds(action_low, action_high)

    # The obs window is a ring of n_obs_steps rows. A chunk writes one row per
    # step. So a chunk deeper than the window has nowhere to put its frames.
    # Single-env collection slides the window instead. It has no such limit.
    # That is why the check is not in the config.
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
        # envs.step hands back a [n, obs] array. Everything below indexes it whole
        # rather than row by row, and the obs window is a ring cursor, not a slide.
        obs0 = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        windows = torch.zeros(n, n_obs_steps, obs_dim, dtype=_OBS_DTYPE)
        windows.copy_(obs0[:, None, :].expand_as(windows))
        # Reused per chunk: slot ``pos`` is overwritten on every chunk of env i, so
        # ``add_chunk`` must consume ``reward_windows[i, :n_exec]`` first.
        reward_windows = torch.zeros(n, chunk_size, dtype=_REWARD_DTYPE)
        # One frame per step, not one window per step. The window the chunk
        # ends on is already in hand when the chunk closes.
        step_frames = torch.zeros(n, chunk_size, obs_dim, dtype=_OBS_DTYPE)
        cur: list = [None] * n
        pos_t = torch.zeros(n, dtype=torch.long)
        # Action chunks for the wavefront, indexed by env. Row n is a zero row so an
        # env with no chunk reads zeros.
        plan = torch.zeros(n + 1, chunk_size, act_dim, dtype=torch.float32)
        env_rows = torch.arange(n)
        idle_row = torch.full_like(env_rows, n)
        # The table satisfies roll_table[w][j] == (w + j) % n_obs_steps. So
        # restoring the chronological order of a window costs one row lookup.
        # The wrapping rows dominate when the chunks are short.
        roll_table = (
            torch.arange(n_obs_steps)[:, None]
            + torch.arange(n_obs_steps)[None, :]
        ) % n_obs_steps
        executed_total = [0] * n
        awaiting = [False] * n
        prev_done = ["continue"] * n
        prev_chunk_indices: list[int | None] = [None] * n

        actions_t = torch.zeros((n, act_dim), dtype=torch.float32)
        remaining = sum(targets)
        while remaining:
            need = [
                i
                for i in range(n)
                if (
                    cur[i] is None
                    and not awaiting[i]
                    and executed_total[i] < targets[i]
                )
            ]
            if need:
                # pos_t is 0 here: rewound when this env's previous chunk closed.
                # Advanced indexing already returns a copy, so no clone.
                states = windows[need]
                proposed = as_float_tensor(policy(states))
                # One clamp-and-rescale for the whole wavefront, not one per env.
                executed_all, scaled_all, clip_counts_all = scale_and_clip_chunk(
                    proposed, low, range_
                )
                plan[torch.as_tensor(need)] = executed_all
                pos_t[torch.as_tensor(need)] = 0
                for k, i in enumerate(need):
                    cur[i] = (
                        executed_all[k],
                        scaled_all[k],
                        int(clip_counts_all[k]),
                        states[k],
                        min(n_action_steps, targets[i] - executed_total[i]),
                    )

            # An env is stepping exactly when it holds a chunk. So the code
            # reads that off ``cur``. It does not track it with a second flag.
            # Such a flag would have to stay in step with ``cur``. An idle env
            # and an awaiting env read the zero row n.
            stepping = torch.tensor([c is not None for c in cur], dtype=torch.bool)
            live_idx = env_rows[stepping]
            actions_t.copy_(
                plan[
                    torch.where(stepping, env_rows, idle_row),
                    pos_t.clamp(max=chunk_size - 1),
                ]
            )
            # No early-out: envs flagged ``awaiting`` still need this step() to take
            # their reset observation.
            actions_np = actions_t.cpu().numpy()

            obs, rew, term, trunc, _ = envs.step(actions_np)
            term_np = np.asarray(term, dtype=bool)
            trunc_np = np.asarray(trunc, dtype=bool)
            done_np = term_np | trunc_np
            rew_np = np.asarray(rew, dtype=np.float32)
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32))

            if live_idx.numel():
                # Three scatters for the whole vec step, over the batch envs.step
                # already handed us. One cursor, because the reward row, the frame
                # row and the window row all sit at the step index in the chunk.
                p = pos_t[live_idx]
                reward_windows[live_idx, p] = torch.from_numpy(rew_np)[live_idx]
                step_frames[live_idx, p] = obs_t[live_idx]
                windows[live_idx, p] = obs_t[live_idx]
                pos_t[live_idx] = p + 1

            for i in range(n):
                if awaiting[i]:
                    windows[i].copy_(obs_t[i].expand_as(windows[i]))
                    awaiting[i] = False
                    continue
                if cur[i] is None:
                    continue
                n_exec = int(pos_t[i])
                if done_np[i] or n_exec >= cur[i][4]:
                    _executed, scaled, clip_count, replan, _limit = cur[i]
                    # Restore chronological order once per closing chunk. Frames
                    # went in at slot 0, 1, ... n_exec, so a left roll by the cursor
                    # puts the oldest frame back at row 0. The modulo is load-bearing
                    # at n_action_steps == n_obs_steps, where the cursor reaches
                    # n_obs_steps on the chunk's last step.
                    windows[i].copy_(
                        windows[i][roll_table[int(pos_t[i]) % n_obs_steps]]
                    )
                    executed_total[i] += n_exec
                    remaining -= n_exec
                    prev_chunk_indices[i] = buffer.add_chunk(
                        state=replan,
                        next_state=windows[i],
                        action=scaled,
                        rewards=reward_windows[i, :n_exec],
                        step_frames=step_frames[i, :n_exec],
                        terminated=bool(term_np[i]),
                        truncated=bool(trunc_np[i]),
                        clip_count=clip_count,
                        env_id=i,
                    )
                    prev_done[i] = (
                        "terminated" if term_np[i]
                        else "truncated" if trunc_np[i]
                        else "continue"
                    )
                    cur[i] = None
                    if done_np[i]:
                        awaiting[i] = True

        for i in range(n):
            if prev_done[i] == "continue" and prev_chunk_indices[i] is not None:
                buffer.mark_incomplete(prev_chunk_indices[i])
        return buffer

    except (ValueError, RuntimeError, KeyError):
        flush_on_collection_error(on_error_flush)
        raise
