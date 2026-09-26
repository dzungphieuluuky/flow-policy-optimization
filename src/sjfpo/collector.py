"""Chunked open-loop rollout collector.

Performance notes
-----------------
* Actions are transferred host-side **once per step** (batched over envs) rather
  than once per env per step. That per-step `.numpy()` sync is the single
  biggest cost in the vec collector.
* Observation windows and rewards stay in **fp32** end-to-end; only policy data
  follows the execution-device dtype policy. bf16 rewards corrupt the GAE
  target.
* Bounds are validated once per collector call and reduced to
  ``(low, range)``; ``scale_and_clip_chunk`` does not re-validate.
* Warmup windows share one clone (the deque holds N references to the same
  tensor) rather than cloning ``n_obs_steps`` copies; safe because the window
  is only read, never mutated in place.
* Tail truncation writes into the existing reward/done tensors rather than
  constructing a fresh scalar per tail.

Autoreset contract
------------------
``collect_chunks_vec`` implements the vec-env reset as a state machine: an env
that returns ``done=True`` is flagged ``awaiting``, and the reset observation is
picked up on the *next* ``envs.step()``. This is only correct under gymnasium's
default ``AutoresetMode.NEXT_STEP``. Under ``SAME_STEP`` the reset observation
arrives together with ``done=True`` and the state machine would step a
freshly-reset env with a zero action. The collector validates the mode up front
and refuses to run otherwise.
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
    """Return the lazily built default config singleton.

    Constructed on first use so importing this module stays cheap.

    Returns:
        The shared default :class:`SJFPOConfig`.
    """
    global _DEFAULT_CFG
    if _DEFAULT_CFG is None:
        _DEFAULT_CFG = SJFPOConfig()
    return _DEFAULT_CFG


# Dtypes: observations and rewards are fp32 end-to-end. Only policy outputs
# and policy data use the execution-device dtype policy.
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
    """Transition dict with post-step windows stored outside public keys.

    The extra ``step_next_states`` attribute is a runtime-only convenience for
    callers that consume the record list directly; it is deliberately not part
    of the eight-key :data:`Transition` contract and is not preserved by
    ``copy`` / serialization. The buffer path does not use it — the per-step
    windows live in :class:`RolloutBuffer` instead.
    """

    def __init__(self, record: Transition, step_next_states: torch.Tensor):
        super().__init__(record)
        # Keep the public eight-key Transition contract while retaining the
        # state window after every executed action for step-level GAE.
        self.step_next_states = step_next_states


class RolloutBuffer:
    """Fixed-capacity CPU storage for chunk and executed-step rollout data.

    The active training path reuses one instance per capacity/device. Chunks
    are written by the collector; ``SJFPOPolicy.generate_rollout`` reads views
    of the used prefix and transfers those contiguous tensors to the device.
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

        def zeros(shape, dtype):
            return torch.zeros(
                *shape, dtype=dtype, device=self.device, pin_memory=self.pin_memory
            )

        self.step_states = zeros((max_steps, obs_steps, obs_dim), _OBS_DTYPE)
        self.step_next_states = zeros(
            (max_steps, obs_steps, obs_dim), _OBS_DTYPE
        )
        self.step_rewards = zeros((max_steps,), _REWARD_DTYPE)
        self.step_terminated = zeros((max_steps,), torch.bool)
        self.step_truncated = zeros((max_steps,), torch.bool)

        self.chunk_actions = zeros(
            (max_chunks, action_horizon, action_dim), _OBS_DTYPE
        )
        self.chunk_starts = zeros((max_chunks,), _INDEX_DTYPE)
        self.chunk_lengths = zeros((max_chunks,), _INDEX_DTYPE)
        self.chunk_terminated = zeros((max_chunks,), torch.bool)
        self.chunk_truncated = zeros((max_chunks,), torch.bool)
        self.chunk_clip_counts = zeros((max_chunks,), _INDEX_DTYPE)
        self.chunk_env_id = zeros((max_chunks,), torch.int16)
        self.n_chunks = 0
        self.n_steps = 0

    def capacity_key(self) -> tuple[int, ...]:
        """Return the buffer's layout signature, excluding its device.

        Two buffers may exchange data only when this matches, so the device is
        deliberately left out.

        Returns:
            Tuple of capacities, dimensions, and the pin-memory flag.
        """
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
        """Reset cursors without clearing the large storage tensors."""
        self.n_chunks = 0
        self.n_steps = 0

    def add_chunk(
        self,
        *,
        state: torch.Tensor,
        action: torch.Tensor,
        rewards: list[float] | np.ndarray | torch.Tensor,
        step_next_states: torch.Tensor,
        terminated: bool,
        truncated: bool,
        clip_count: int,
        env_id: int = 0,
    ) -> int:
        """Append one executed chunk to the buffer.

        The step-level observation window is chained across the chunk, so
        ``step_next_states`` must be supplied for every executed step. ``env_id`` is
        the vec-env slot that produced the chunk: the buffer appends in wavefront
        order, so consecutive entries usually belong to different envs, and the
        chunk-level backup uses this to find trajectory boundaries.

        Raises ``RuntimeError`` if the chunk or step capacity is exhausted, and
        ``ValueError`` if any tensor shape does not match the buffer layout.
        """
        if self.n_chunks >= self.max_chunks:
            raise RuntimeError("RolloutBuffer chunk capacity exceeded")
        reward_t = torch.as_tensor(rewards, dtype=_REWARD_DTYPE, device=self.device).reshape(-1)
        n_exec = int(reward_t.numel())
        if n_exec < 1 or self.n_steps + n_exec > self.max_steps:
            raise RuntimeError("RolloutBuffer step capacity exceeded")
        if state.shape != (self.obs_steps, self.obs_dim):
            raise ValueError(f"state shape {tuple(state.shape)} does not fit buffer")
        if action.shape != (self.action_horizon, self.action_dim):
            raise ValueError(f"action shape {tuple(action.shape)} does not fit buffer")
        if tuple(step_next_states.shape) != (n_exec, self.obs_steps, self.obs_dim):
            raise ValueError(
                f"step_next_states shape {tuple(step_next_states.shape)} does not fit buffer"
            )

        chunk_index = self.n_chunks
        offset = self.n_steps
        self.step_states[offset].copy_(state)
        if n_exec > 1:
            self.step_states[offset + 1 : offset + n_exec].copy_(step_next_states[:-1])
        self.step_next_states[offset : offset + n_exec].copy_(step_next_states)
        self.step_rewards[offset : offset + n_exec].copy_(reward_t)
        self.chunk_actions[chunk_index].copy_(action)
        self.chunk_starts[chunk_index] = offset
        self.chunk_lengths[chunk_index] = n_exec
        self.chunk_terminated[chunk_index] = terminated
        self.chunk_truncated[chunk_index] = truncated
        self.chunk_clip_counts[chunk_index] = int(clip_count)
        self.chunk_env_id[chunk_index] = int(env_id)
        self.step_terminated[offset + n_exec - 1] = terminated
        self.step_truncated[offset + n_exec - 1] = truncated
        self.n_chunks += 1
        self.n_steps += n_exec
        return chunk_index

    def copy_to(self, other: "RolloutBuffer", *, non_blocking: bool = False) -> None:
        """Copy the active prefix into a compatible buffer, e.g. onto a GPU.

        Raises ``ValueError`` if the two buffers have different capacities or
        dimensions.
        """
        if other.capacity_key()[:6] != self.capacity_key()[:6]:
            raise ValueError("RolloutBuffer capacities do not match")
        other.n_chunks = self.n_chunks
        other.n_steps = self.n_steps
        for name in (
            "step_states",
            "step_next_states",
            "step_rewards",
            "step_terminated",
            "step_truncated",
            "chunk_actions",
            "chunk_starts",
            "chunk_lengths",
            "chunk_terminated",
            "chunk_truncated",
            "chunk_clip_counts",
            "chunk_env_id",
        ):
            getattr(other, name)[: self.n_steps if name.startswith("step_") else self.n_chunks].copy_(
                getattr(self, name)[: self.n_steps if name.startswith("step_") else self.n_chunks],
                non_blocking=non_blocking,
            )

    def mark_truncated(self, chunk_index: int) -> None:
        """Flag a chunk, and its final executed step, as truncated.

        Used when the step budget cuts a chunk short, so GAE bootstraps the cut
        instead of treating it as a terminal transition. Raises ``IndexError`` if
        ``chunk_index`` is outside the populated prefix.
        """
        if not 0 <= chunk_index < self.n_chunks:
            raise IndexError(f"chunk index {chunk_index} out of range")
        self.chunk_truncated[chunk_index] = True
        start = int(self.chunk_starts[chunk_index])
        length = int(self.chunk_lengths[chunk_index])
        self.step_truncated[start + length - 1] = True

    def active(self) -> dict[str, torch.Tensor]:
        """Return contiguous views of the populated buffer prefix.

        Returns:
            Dict of buffer name to a view covering only the written region.
        """
        n_chunks, n_steps = self.n_chunks, self.n_steps
        return {
            "step_states": self.step_states[:n_steps],
            "step_next_states": self.step_next_states[:n_steps],
            "step_rewards": self.step_rewards[:n_steps],
            "step_terminated": self.step_terminated[:n_steps],
            "step_truncated": self.step_truncated[:n_steps],
            "chunk_actions": self.chunk_actions[:n_chunks],
            "chunk_starts": self.chunk_starts[:n_chunks],
            "chunk_lengths": self.chunk_lengths[:n_chunks],
            "chunk_terminated": self.chunk_terminated[:n_chunks],
            "chunk_truncated": self.chunk_truncated[:n_chunks],
            "chunk_clip_counts": self.chunk_clip_counts[:n_chunks],
            "chunk_env_id": self.chunk_env_id[:n_chunks],
        }


def as_float_tensor(x: object) -> torch.Tensor:
    """Coerce to an fp32 CPU/GPU tensor without forcing a device hop.

    A ``torch.Tensor`` input keeps its device (no implicit ``.cpu()``); numpy /
    Python inputs are lifted to a fresh CPU fp32 tensor.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().to(_OBS_DTYPE)
    return torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=_OBS_DTYPE)


def _bounds_to_tensor(
    low: float | np.ndarray | torch.Tensor,
    high: float | np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coerce and validate env action bounds into fp32 tensors.

    Enforces the strict ``high > low`` convention, matching the collector's
    action-scaling contract. ``SJFPOPolicy.set_action_bounds`` must use the same
    convention, otherwise a policy can be built with a degenerate dimension that
    only fails on the first rollout.

    Returns ``(low, high - low)``. Raises ``ValueError`` if the bounds disagree in
    shape or any dimension fails the strict ``high > low`` check.
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
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Clamp a proposed chunk to the env bounds and rescale it to ``[-1, 1]``.

    ``range_`` is the precomputed ``high - low`` (see :func:`_bounds_to_tensor`).
    Returns the clipped actions, their rescaling, and the number of clipped
    elements.
    """
    clip_mask = (proposed < low) | (proposed > low + range_)
    clip_count = int(clip_mask.sum())
    executed = torch.clamp(proposed, min=low, max=low + range_)
    scaled = 2.0 * (executed - low) / range_ - 1.0
    return executed, scaled, clip_count


def build_warmup_window(
    first_obs: torch.Tensor, n_obs_steps: int
) -> deque[torch.Tensor]:
    """Build a warmup deque of ``n_obs_steps`` identical frames.

    Uses a single clone referenced N times, safe because the window is only read,
    never mutated in place. Appends past the initial fill replace the oldest
    reference, not the shared tensor.
    """
    shared = first_obs.detach().clone()
    return deque((shared for _ in range(n_obs_steps)), maxlen=n_obs_steps)


def reset_chunk_window(env, n_obs_steps: int) -> deque[torch.Tensor]:
    """Reset an environment and build its warmup observation window.
    """
    obs, _ = env.reset()
    return build_warmup_window(as_float_tensor(obs), n_obs_steps)


def _bool_tensor(v: bool) -> torch.Tensor:
    """Wrap a Python bool as a scalar bool tensor.
    """
    return torch.tensor(v, dtype=torch.bool)


def _int_tensor(v: int) -> torch.Tensor:
    """Wrap a Python int as a scalar index tensor.
    """
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
    """Assemble the transition record for one executed action chunk.

    Raises ``ValueError`` if the chunk, ``n_action_steps``, or ``step_next_states``
    shape is inconsistent.
    """
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


def execute_open_loop_steps(
    env,
    executed_np: np.ndarray,
    window: deque[torch.Tensor],
    executed_total: int,
    total_steps: int,
    n_action_steps: int,
) -> tuple[list[float], str, torch.Tensor]:
    """Execute the executed prefix of a chunk one env step at a time.

    ``executed_np`` is kept on the host so stepping never forces a
    device-to-host sync per step; the observation window is advanced after every
    step and stopped at the step budget.
    """
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


def execute_open_loop_steps_into(
    env,
    executed_np: np.ndarray,
    window: torch.Tensor,
    step_windows: torch.Tensor,
    rewards: torch.Tensor,
    executed_total: int,
    total_steps: int,
    n_action_steps: int,
) -> tuple[int, str]:
    """Execute one chunk into preallocated window and reward storage.
    """
    n_exec = 0
    done_kind = "continue"
    for i in range(n_action_steps):
        if executed_total + n_exec >= total_steps:
            break
        obs, reward, terminated, truncated, _info = env.step(executed_np[i])
        n_exec += 1
        rewards[n_exec - 1] = reward
        window[:-1].copy_(window[1:].clone())
        window[-1].copy_(torch.as_tensor(np.asarray(obs, dtype=np.float32)))
        step_windows[n_exec - 1].copy_(window)
        if terminated:
            done_kind = "terminated"
            break
        if truncated:
            done_kind = "truncated"
            break
    return n_exec, done_kind


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
    """Propose one chunk from the current window and execute it open loop.
    """
    state = torch.stack(tuple(window), dim=0)
    proposed = as_float_tensor(policy(state))
    executed, scaled, clip_count = scale_and_clip_chunk(proposed, low, range_)
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
    """Invoke the error flush hook, swallowing any failure it raises.

    The hook is best effort: a failing flush must not mask the collection error
    that triggered it.
    """
    if on_error_flush is not None:
        try:
            on_error_flush()
        except Exception:
            pass


def _validate_autoreset_mode(envs) -> None:
    """Reject vec envs whose autoreset mode breaks the awaiting state machine.

    ``collect_chunks_vec`` handles the reset as a two-step handshake: it flags the
    env on the ``done=True`` step and picks up the reset observation on the
    following ``step()``. That only works under ``NEXT_STEP`` autoreset.
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
    buffer: RolloutBuffer | None = None,
) -> list[Transition] | RolloutBuffer:
    """Vectorised twin of :func:`collect_chunks` -- same records, same chaining.

    The step budget is split as evenly as possible across the environments, with
    the remainder handed to the first environments. Budget-exhausted tails are
    marked truncated so GAE bootstraps the cut. Each chunk is transferred to the
    host in one copy before being indexed, and all still-active actions are
    gathered into a single tensor per wavefront, so the loop performs one
    device-to-host transfer per chunk rather than one per step.

    Requires gymnasium ``AutoresetMode.NEXT_STEP`` on ``envs`` (the default); see
    :func:`_validate_autoreset_mode`. ``n_obs_steps`` and ``chunk_size`` fall back
    to the default config when omitted.
    """
    cfg = _defaults()
    n_obs_steps = cfg.n_obs_steps if n_obs_steps is None else n_obs_steps
    chunk_size = cfg.action_horizon if chunk_size is None else int(chunk_size)

    _validate_autoreset_mode(envs)

    n = int(envs.num_envs)
    base_target, remainder = divmod(total_steps, n)
    targets = [base_target + int(i < remainder) for i in range(n)]

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
        if buffer is not None:
            obs_dim = int(obs.shape[-1])
            windows = torch.zeros(n, n_obs_steps, obs_dim, dtype=_OBS_DTYPE)
            for i in range(n):
                windows[i].copy_(as_float_tensor(obs[i]).expand_as(windows[i]))
            # Reused per chunk: slot ``pos`` is overwritten on every chunk of
            # env i, so ``add_chunk`` must consume ``reward_windows[i,
            # :n_exec]`` before env i's next chunk.
            reward_windows = torch.zeros(n, chunk_size, dtype=_REWARD_DTYPE)
            step_windows = torch.zeros(n, chunk_size, n_obs_steps, obs_dim, dtype=_OBS_DTYPE)
        else:
            windows = [build_warmup_window(as_float_tensor(o), n_obs_steps) for o in obs]
            reward_windows = None
            step_windows = None
        cur: list = [None] * n
        pos = [0] * n
        rewards: list[list[float]] | None = None if buffer is not None else [[] for _ in range(n)]
        step_states: list[list[torch.Tensor]] | None = (
            None if buffer is not None else [[] for _ in range(n)]
        )
        prev: list[Transition | None] = [None] * n
        prev_done = ["continue"] * n
        executed_total = [0] * n
        awaiting = [False] * n
        records: list[list[Transition]] | None = (
            None if buffer is not None else [[] for _ in range(n)]
        )
        prev_chunk_indices: list[int | None] = [None] * n

        actions_t = torch.zeros((n, act_dim), dtype=torch.float32)
        while any(executed_total[i] < targets[i] for i in range(n)):
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
                if buffer is not None:
                    # Advanced indexing already returns a copy, so no clone.
                    states = windows[need]
                else:
                    states = torch.stack(
                        [torch.stack(tuple(windows[i]), dim=0) for i in need]
                    )
                proposed = as_float_tensor(policy(states))
                for k, i in enumerate(need):
                    executed, scaled, clip_count = scale_and_clip_chunk(
                        proposed[k], low, range_
                    )
                    execution_limit = min(
                        n_action_steps, targets[i] - executed_total[i]
                    )
                    cur[i] = (
                        executed,
                        scaled,
                        clip_count,
                        states[k],
                        execution_limit,
                    )
                    pos[i] = 0
                    if rewards is not None:
                        rewards[i] = []
                    if step_states is not None:
                        step_states[i] = []

            actions_t.zero_()
            for i in range(n):
                if cur[i] is not None:
                    actions_t[i] = cur[i][0][pos[i]]
            # No early-out: envs flagged ``awaiting`` have ``cur[i] is None``
            # but still need this step() to pick up their reset observation
            # (NEXT_STEP autoreset). The while condition exits once every env
            # is done and no resets are pending.
            actions_np = actions_t.cpu().numpy()

            obs, rew, term, trunc, _ = envs.step(actions_np)
            term_np = np.asarray(term, dtype=bool)
            trunc_np = np.asarray(trunc, dtype=bool)
            done_np = term_np | trunc_np
            rew_np = np.asarray(rew, dtype=np.float32)

            for i in range(n):
                if awaiting[i]:
                    if buffer is not None:
                        windows[i].copy_(as_float_tensor(obs[i]).expand_as(windows[i]))
                    else:
                        windows[i] = build_warmup_window(
                            as_float_tensor(obs[i]), n_obs_steps
                        )
                    awaiting[i] = False
                    continue
                if cur[i] is None:
                    continue
                if buffer is not None:
                    reward_windows[i, pos[i]] = float(rew_np[i])
                    windows[i, :-1].copy_(windows[i, 1:].clone())
                    windows[i, -1].copy_(as_float_tensor(obs[i]))
                    step_windows[i, pos[i]].copy_(windows[i])
                else:
                    rewards[i].append(float(rew_np[i]))
                    windows[i].append(as_float_tensor(obs[i]))
                    step_states[i].append(torch.stack(tuple(windows[i]), dim=0))
                pos[i] += 1
                execution_limit = cur[i][4]
                if done_np[i] or pos[i] >= execution_limit:
                    _executed, scaled, clip_count, replan, _limit = cur[i]
                    n_exec = pos[i]
                    if term_np[i]:
                        done_kind = "terminated"
                    elif trunc_np[i]:
                        done_kind = "truncated"
                    else:
                        done_kind = "continue"
                    next_state = (
                        windows[i]
                        if buffer is not None
                        else torch.stack(tuple(windows[i]), dim=0)
                    )
                    executed_total[i] += n_exec
                    if buffer is not None:
                        prev_chunk_indices[i] = buffer.add_chunk(
                            state=replan,
                            action=scaled,
                            rewards=reward_windows[i, :n_exec],
                            step_next_states=step_windows[i, :n_exec],
                            terminated=bool(term_np[i]),
                            truncated=bool(trunc_np[i]),
                            clip_count=clip_count,
                            env_id=i,
                        )
                    else:
                        record = finalize_transition(
                            replan,
                            next_state,
                            scaled,
                            rewards[i],
                            done_kind,
                            clip_count,
                            chunk_size,
                            execution_limit,
                            torch.stack(step_states[i], dim=0),
                        )
                        records[i].append(record)
                        prev[i] = record
                    prev_done[i] = done_kind
                    cur[i] = None
                    if done_np[i]:
                        awaiting[i] = True

        for i in range(n):
            if prev_done[i] == "continue" and (
                prev[i] is not None or prev_chunk_indices[i] is not None
            ):
                if buffer is not None:
                    buffer.mark_truncated(prev_chunk_indices[i])  # type: ignore[arg-type]
                else:
                    prev[i][TRUNCATED].fill_(True)  # type: ignore[index]

        if buffer is not None:
            return buffer
        return [r for traj in records for r in traj]  # type: ignore[union-attr]

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
    n_obs_steps: int | None = None,
    n_action_steps: int,
    chunk_size: int | None = None,
    on_error_flush: Callable[[], None] | None = None,
    buffer: RolloutBuffer | None = None,
) -> list[Transition] | RolloutBuffer:
    """Collect executed action chunks from a single environment.

    Chunks are proposed from the current observation window and executed open
    loop, with the window chained across the steps of each chunk. ``n_obs_steps``
    and ``chunk_size`` fall back to the default config when omitted.
    """
    cfg = _defaults()
    n_obs_steps = cfg.n_obs_steps if n_obs_steps is None else n_obs_steps
    chunk_size = cfg.action_horizon if chunk_size is None else int(chunk_size)

    low, range_ = _bounds_to_tensor(action_low, action_high)

    if not 1 <= n_action_steps <= chunk_size:
        raise ValueError(
            f"n_action_steps={n_action_steps} must satisfy "
            f"1 <= n <= chunk_size={chunk_size}"
        )
    try:
        obs, _ = env.reset()
        window_deque = build_warmup_window(as_float_tensor(obs), n_obs_steps)
        if buffer is not None:
            window = torch.stack(tuple(window_deque), dim=0)
            step_windows = torch.zeros(chunk_size, n_obs_steps, int(window.shape[-1]))
            reward_window = torch.zeros(chunk_size, dtype=_REWARD_DTYPE)
        records: list[Transition] | None = [] if buffer is None else None
        executed_total = 0
        while executed_total < total_steps:
            if buffer is not None:
                state = window.clone()
                proposed = as_float_tensor(policy(state))
                executed, scaled, clip_count = scale_and_clip_chunk(proposed, low, range_)
                executed_np = executed.detach().cpu().numpy()
                n_exec, done_kind = execute_open_loop_steps_into(
                    env,
                    executed_np,
                    window,
                    step_windows,
                    reward_window,
                    executed_total,
                    total_steps,
                    n_action_steps,
                )
                if n_exec < 1:
                    raise RuntimeError("buffer collection executed zero steps")
                chunk_index = buffer.add_chunk(
                    state=state,
                    action=scaled,
                    rewards=reward_window[:n_exec],
                    step_next_states=step_windows[:n_exec],
                    terminated=done_kind == "terminated",
                    truncated=done_kind == "truncated",
                    clip_count=clip_count,
                )
                if done_kind == "continue":
                    buffer.mark_truncated(chunk_index)
                executed_total += n_exec
                if done_kind != "continue":
                    obs, _ = env.reset()
                    window.copy_(as_float_tensor(obs).expand_as(window))
            else:
                record, done_kind = propose_and_execute_chunk(
                    policy,
                    window_deque,
                    low,
                    range_,
                    env,
                    executed_total,
                    total_steps,
                    chunk_size,
                    n_action_steps,
                )
                executed_total += int(record["executed_n_action_steps"])
                records.append(record)  # type: ignore[union-attr]
                if done_kind != "continue":
                    window_deque = reset_chunk_window(env, n_obs_steps)
        if buffer is not None:
            return buffer
        return records  # type: ignore[return-value]
    except Exception:
        flush_on_collection_error(on_error_flush)
        raise