"""Chunked open-loop rollout collector."""

from __future__ import annotations

import contextlib

from typing import NamedTuple

from collections.abc import Callable

import numpy as np
import torch

from myfpo.configuration_sjfpo import SJFPOConfig

__all__ = ["RolloutBuffer", "collect_chunks_vec"]


_DEFAULT_CFG: SJFPOConfig | None = None


def get_default_config() -> SJFPOConfig:
    """Default config singleton, with feature specs for smoke and training."""
    global _DEFAULT_CFG
    if _DEFAULT_CFG is None:
        from lerobot.configs.types import FeatureType, PolicyFeature
        _DEFAULT_CFG = SJFPOConfig(
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,))
            },
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        )
    return _DEFAULT_CFG


_OBS_DTYPE = torch.float32
_REWARD_DTYPE = torch.float32
_INDEX_DTYPE = torch.int64


class InFlightChunk(NamedTuple):
    """What one env is executing right now, while it holds a chunk."""

    actions_scaled: torch.Tensor
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
        self.chunk_env_id = self.zeros_for((chunk_capacity,), torch.int16)
        self.n_chunks = 0
        self.n_steps = 0

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

    def assert_budget_cut_is_not_a_timeout(self, chunk_index: int) -> None:
        """Check the bounds of a budget-cut chunk, and deliberately record nothing."""
        if not 0 <= chunk_index < self.n_chunks:
            raise IndexError(f"chunk index {chunk_index} out of range")

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
        return x.detach().to(_OBS_DTYPE)
    return torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=_OBS_DTYPE)


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
    rng = hi - lo
    if not bool((rng > 0).all()):
        raise ValueError("action bounds must satisfy high > low in every dim")
    return lo, rng


def scale_action_chunk(
    proposed: torch.Tensor, low: torch.Tensor, range_: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rescale an in-bounds chunk to ``[-1, 1]``."""
    scaled = 2.0 * (proposed - low) / range_ - 1.0
    return proposed, scaled


def resolve_chunk_args(
    n_obs_steps: int | None,
    n_action_steps: int,
    action_horizon: int,
) -> tuple[int, int]:
    """Fill the obs-step default, then check the execution span."""
    cfg = get_default_config()
    n_obs_steps = cfg.n_obs_steps if n_obs_steps is None else n_obs_steps
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


_CLOSE_CONTINUE = "continue"
_CLOSE_TERMINATED = "terminated"
_CLOSE_TRUNCATED = "truncated"


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

    low, range_ = resolve_action_bounds(action_low, action_high)

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
        windows = torch.zeros(n, n_obs_steps, obs_dim, dtype=_OBS_DTYPE)
        windows.copy_(obs0[:, None, :].expand_as(windows))
        reward_windows = torch.zeros(n, action_horizon, dtype=_REWARD_DTYPE)
        step_frames = torch.zeros(n, action_horizon, obs_dim, dtype=_OBS_DTYPE)
        in_flight: list[InFlightChunk | None] = [None] * n
        pos_t = torch.zeros(n, dtype=torch.long)
        plan = torch.zeros(n + 1, action_horizon, act_dim, dtype=torch.float32)
        env_rows = torch.arange(n)
        idle_row = torch.full_like(env_rows, n)
        roll_table = (
            torch.arange(n_obs_steps)[:, None]
            + torch.arange(n_obs_steps)[None, :]
        ) % n_obs_steps
        executed_total = [0] * n
        awaiting = [False] * n
        prev_close_reason = [_CLOSE_CONTINUE] * n
        prev_chunk_indices: list[int | None] = [None] * n

        actions_t = torch.zeros((n, act_dim), dtype=torch.float32)
        remaining = sum(targets)
        while remaining:
            envs_needing_a_chunk = [
                i
                for i in range(n)
                if (
                    in_flight[i] is None
                    and not awaiting[i]
                    and executed_total[i] < targets[i]
                )
            ]
            if envs_needing_a_chunk:
                states = windows[envs_needing_a_chunk]
                proposed = as_float_tensor(policy(states))
                executed_all, scaled_all = scale_action_chunk(
                    proposed, low, range_
                )
                wavefront = torch.as_tensor(envs_needing_a_chunk)
                plan[wavefront] = executed_all
                pos_t[wavefront] = 0
                for k, i in enumerate(envs_needing_a_chunk):
                    in_flight[i] = InFlightChunk(
                        actions_scaled=scaled_all[k],
                        start_state=states[k],
                        step_limit=min(n_action_steps,
                                       targets[i] - executed_total[i]),
                    )

            stepping = torch.tensor(
                [held is not None for held in in_flight], dtype=torch.bool
            )
            live_idx = env_rows[stepping]
            actions_t.copy_(
                plan[
                    torch.where(stepping, env_rows, idle_row),
                    pos_t.clamp(max=action_horizon - 1),
                ]
            )
            actions_np = actions_t.cpu().numpy()

            obs, rew, term, trunc, _ = envs.step(actions_np)
            term_np = np.asarray(term, dtype=bool)
            trunc_np = np.asarray(trunc, dtype=bool)
            done_np = term_np | trunc_np
            rew_np = np.asarray(rew, dtype=np.float32)
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32))

            if live_idx.numel():
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
                if in_flight[i] is None:
                    continue
                n_exec = int(pos_t[i])
                held = in_flight[i]
                if done_np[i] or n_exec >= held.step_limit:
                    scaled = held.actions_scaled
                    start_state = held.start_state
                    windows[i].copy_(
                        windows[i][roll_table[int(pos_t[i]) % n_obs_steps]]
                    )
                    executed_total[i] += n_exec
                    remaining -= n_exec
                    prev_chunk_indices[i] = buffer.add_chunk(
                        state=start_state,
                        next_state=windows[i],
                        action=scaled,
                        rewards=reward_windows[i, :n_exec],
                        step_frames=step_frames[i, :n_exec],
                        terminated=bool(term_np[i]),
                        truncated=bool(trunc_np[i]),
                        env_id=i,
                    )
                    prev_close_reason[i] = (
                        _CLOSE_TERMINATED if term_np[i]
                        else _CLOSE_TRUNCATED if trunc_np[i]
                        else _CLOSE_CONTINUE
                    )
                    in_flight[i] = None
                    if done_np[i]:
                        awaiting[i] = True

        for i in range(n):
            if prev_close_reason[i] == _CLOSE_CONTINUE and prev_chunk_indices[i] is not None:
                buffer.assert_budget_cut_is_not_a_timeout(prev_chunk_indices[i])
        return buffer

    except (ValueError, RuntimeError, KeyError):
        flush_on_collection_error(on_error_flush)
        raise
