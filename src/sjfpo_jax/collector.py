"""Chunked open-loop rollout collector.

Port of ``collect_chunks_vec`` and ``RolloutBuffer`` from
``spectral_joint_policy/collector.py``. Two things are deliberately unchanged:

* **The Python loop stays.** ``envs.step`` is a Python call and the awaiting/reset
  state machine is a correctness contract, not an optimisation. Only gymnasium's
  ``AutoresetMode.NEXT_STEP`` is correct here (``validate_autoreset_mode``).
* **The window ring stays a ring.** ``wpos`` is a per-env cursor into a fixed
  ``[n, n_obs_steps, obs]`` array, and chronological order is restored once per
  closing chunk through ``roll_table``. A per-step slide would be both slower and
  harder to read.

What changed: the storage is ``jnp`` and preallocated at max capacity, so the
growth path and the capacity guess are gone -- the chunk count is a Python int
that slices the prefix at use time. ``active()`` returns a dict under the same
keys torch uses.

``collect_chunks`` (the single-env path) is not ported: ``num_envs > 1`` is the
default and the only path anything exercises.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# The chunk axis holds one obs window per chunk; the step axis holds one raw
# frame per executed step.
STEP_KEYS = ("step_frames", "step_rewards")
CHUNK_KEYS = (
    "chunk_state",
    "chunk_next_state",
    "chunk_actions",
    "chunk_starts",
    "chunk_lengths",
    "chunk_terminated",
    "chunk_truncated",
    "chunk_budget_truncated",
    "chunk_clip_counts",
    "chunk_env_id",
)


def validate_autoreset_mode(envs) -> None:
    """Reject vec envs whose autoreset mode breaks the awaiting machine.

    The collector resets in a two-step handshake: flag the env on the
    ``done=True`` step, take the reset obs on the following ``step()``. Only
    works under ``NEXT_STEP``; under ``SAME_STEP`` this would step a fresh env
    with a zero action.
    """
    mode = getattr(envs, "autoreset_mode", None)
    if mode is None:
        return
    name = getattr(mode, "name", None) or str(mode)
    if name != "NEXT_STEP":
        raise ValueError(
            "collect_chunks_vec requires AutoresetMode.NEXT_STEP (the "
            f"gymnasium default), got {mode!r}. The awaiting/reset state "
            "machine assumes the reset observation arrives on the step() "
            "call after done=True."
        )


def resolve_action_bounds(action_low, action_high):
    """Coerce and validate env action bounds. Returns ``(low, range)``.

    Enforces strict ``high > low``, matching the collector's scaling contract.
    """
    low = jnp.asarray(np.asarray(action_low, dtype=np.float32), dtype=jnp.float32).reshape(-1)
    high = jnp.asarray(np.asarray(action_high, dtype=np.float32), dtype=jnp.float32).reshape(-1)
    if low.shape != high.shape:
        raise ValueError(
            f"action_low shape {low.shape} != action_high shape {high.shape}"
        )
    rng = high - low
    if not bool(jnp.all(rng > 0)):
        raise ValueError("action bounds must satisfy high > low in every dim")
    return low, rng


def scale_and_clip_chunk(proposed, low, range_):
    """Clamp a chunk to env bounds, then rescale to ``[-1, 1]``.

    ``proposed`` is ``[H, action_dim]`` for one chunk or ``[B, H, action_dim]``
    for a whole wavefront, so the vec collector scales once per wavefront rather
    than once per env. Returns the clipped actions, their rescaling, and the
    clipped element count per leading row.
    """
    clip_mask = (proposed < low) | (proposed > low + range_)
    executed = jnp.clip(proposed, low, low + range_)
    scaled = 2.0 * (executed - low) / range_ - 1.0
    return executed, scaled, jnp.sum(clip_mask, axis=(-2, -1))


class RolloutBuffer:
    """Fixed-capacity rollout storage, allocated once and reused.

    The chunk count is data-dependent -- a chunk cut short by a terminal is one
    more chunk -- but every array here is allocated at ``chunk_capacity`` and the
    used prefix is sliced at read time, so nothing is ever reallocated.
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
    ):
        if max_chunks < 1 or max_steps < 1:
            raise ValueError("max_chunks and max_steps must be positive")
        if obs_steps < 1 or obs_dim < 1 or action_horizon < 1 or action_dim < 1:
            raise ValueError("observation and action dimensions must be positive")
        self.max_chunks = max_chunks
        self.max_steps = max_steps
        self.obs_steps = obs_steps
        self.obs_dim = obs_dim
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        # One chunk per action_horizon steps, plus one per episode end, because a
        # chunk cut by a terminal is short. The +8 absorbs that.
        self.chunk_capacity = chunk_capacity or -(-max_steps // action_horizon) + 8
        if self.chunk_capacity < 1:
            raise ValueError("chunk_capacity must be positive")

        # One frame per executed step. The obs windows live on the chunk axis:
        # envs interleave, so a chunk's obs_steps frames are not obs_steps
        # adjacent rows and cannot be gathered back out of a step timeline.
        f32 = jnp.float32
        self.step_frames = jnp.zeros((max_steps, obs_dim), dtype=f32)
        self.step_rewards = jnp.zeros((max_steps,), dtype=f32)
        c = self.chunk_capacity
        self.chunk_state = jnp.zeros((c, obs_steps, obs_dim), dtype=f32)
        self.chunk_next_state = jnp.zeros((c, obs_steps, obs_dim), dtype=f32)
        self.chunk_actions = jnp.zeros((c, action_horizon, action_dim), dtype=f32)
        self.chunk_starts = jnp.zeros((c,), dtype=jnp.int32)
        self.chunk_lengths = jnp.zeros((c,), dtype=jnp.int32)
        self.chunk_terminated = jnp.zeros((c,), dtype=jnp.bool_)
        self.chunk_truncated = jnp.zeros((c,), dtype=jnp.bool_)
        # Cut short by the rollout budget, not by an episode end: the next chunk
        # in this env is this one's successor and GAE must chain to it. Kept
        # apart from chunk_truncated, the env's time-limit signal, which does
        # end the episode.
        self.chunk_budget_truncated = jnp.zeros((c,), dtype=jnp.bool_)
        self.chunk_clip_counts = jnp.zeros((c,), dtype=jnp.int32)
        self.chunk_env_id = jnp.zeros((c,), dtype=jnp.int32)
        self.n_chunks = 0
        self.n_steps = 0

    def reset(self) -> None:
        """Reset the cursors. The storage is overwritten by the next writes."""
        self.n_chunks = 0
        self.n_steps = 0

    def _write(self, name: str, index, value) -> None:
        setattr(self, name, getattr(self, name).at[index].set(value))

    def add_chunk(
        self,
        *,
        state,
        next_state,
        action,
        rewards,
        step_frames,
        terminated: bool,
        truncated: bool,
        clip_count: int,
        env_id: int = 0,
    ) -> int:
        """Append one executed chunk. Returns its row.

        ``state`` is the obs window the policy saw and ``next_state`` the window
        the chunk ran to; ``step_frames`` carries one raw frame per executed
        step, which is what the normalizer updates from. ``env_id`` is the vec
        slot, because the buffer appends in wavefront order and the backup uses it
        to find trajectory boundaries.
        """
        if self.n_chunks >= self.max_chunks:
            raise RuntimeError("RolloutBuffer chunk capacity exceeded")
        n_exec = int(np.asarray(rewards).reshape(-1).shape[0])
        if n_exec < 1 or self.n_steps + n_exec > self.max_steps:
            raise RuntimeError("RolloutBuffer step capacity exceeded")
        if state.shape != (self.obs_steps, self.obs_dim):
            raise ValueError(f"state shape {state.shape} does not fit buffer")
        if next_state.shape != (self.obs_steps, self.obs_dim):
            raise ValueError(f"next_state shape {next_state.shape} does not fit buffer")
        if action.shape != (self.action_horizon, self.action_dim):
            raise ValueError(f"action shape {action.shape} does not fit buffer")
        if step_frames.shape != (n_exec, self.obs_dim):
            raise ValueError(
                f"step_frames shape {step_frames.shape} != {(n_exec, self.obs_dim)}"
            )

        index, offset = self.n_chunks, self.n_steps
        self._write("chunk_state", index, state)
        self._write("chunk_next_state", index, next_state)
        self._write("chunk_actions", index, action)
        self._write("step_frames", jnp.arange(offset, offset + n_exec), step_frames)
        self._write("step_rewards", jnp.arange(offset, offset + n_exec), jnp.asarray(rewards).reshape(-1))
        self._write("chunk_starts", index, offset)
        self._write("chunk_lengths", index, n_exec)
        self._write("chunk_terminated", index, terminated)
        self._write("chunk_truncated", index, truncated)
        # Written unconditionally: reset() clears cursors, not storage, so a stale
        # mark would make the metric a running OR over rollouts.
        self._write("chunk_budget_truncated", index, False)
        self._write("chunk_clip_counts", index, clip_count)
        self._write("chunk_env_id", index, env_id)
        self.n_chunks += 1
        self.n_steps += n_exec
        return index

    def mark_incomplete(self, chunk_index: int) -> None:
        """Flag a chunk the rollout budget cut short. Not ``chunk_truncated``."""
        if not 0 <= chunk_index < self.n_chunks:
            raise IndexError(f"chunk index {chunk_index} out of range")
        self._write("chunk_budget_truncated", chunk_index, True)

    def active(self) -> dict:
        """The populated prefix, as ``{name: array}`` under torch's keys."""
        n_chunks, n_steps = self.n_chunks, self.n_steps
        out = {
            "step_frames": self.step_frames[:n_steps],
            "step_rewards": self.step_rewards[:n_steps],
        }
        for name in CHUNK_KEYS:
            if name == "chunk_state" or name == "chunk_next_state":
                out[name] = getattr(self, name)[:n_chunks]
            else:
                out[name] = getattr(self, name)[:n_chunks]
        return out


def collect_chunks_vec(
    envs,
    policy,
    total_steps: int,
    action_low,
    action_high,
    *,
    n_obs_steps: int,
    n_action_steps: int,
    chunk_size: int,
    buffer: RolloutBuffer,
    seed: int | None = None,
) -> RolloutBuffer:
    """Vectorised open-loop chunk collection. Returns the filled ``buffer``.

    The step budget splits evenly across envs, remainder to the first. All
    still-active chunks gather into one tensor per wavefront, so the loop costs
    one device-to-host transfer per vec step rather than one per env.

    Needs ``AutoresetMode.NEXT_STEP``, see :func:`validate_autoreset_mode`.
    """
    validate_autoreset_mode(envs)
    n = int(envs.num_envs)
    base_target, remainder = divmod(total_steps, n)
    targets = [base_target + int(i < remainder) for i in range(n)]
    low, range_ = resolve_action_bounds(action_low, action_high)
    if not 1 <= n_action_steps <= chunk_size:
        raise ValueError(
            f"n_action_steps={n_action_steps} must satisfy 1 <= n <= chunk_size={chunk_size}"
        )
    if seed is None:
        seed = int(jax.random.randint(jax.random.key(0), (), 0, 2**31 - 1))
    act_dim = int(np.prod(envs.single_action_space.shape))
    obs_dim = int(buffer.obs_dim)
    buffer.reset()

    obs, _ = envs.reset(seed=[seed + i for i in range(n)])
    obs_j = jnp.asarray(np.asarray(obs, dtype=np.float32))
    # The window is kept chronological by rolling it left by wpos[i] when its
    # chunk closes, so a write is O(1) rather than a slide of n_obs rows.
    windows = jnp.broadcast_to(obs_j[:, None, :], (n, n_obs_steps, obs_dim))
    wpos = jnp.zeros((n,), dtype=jnp.int32)
    # Reused per chunk: slot `pos` is overwritten on every chunk of env i, so
    # add_chunk must consume reward_windows[i, :n_exec] before env i's next chunk.
    reward_windows = jnp.zeros((n, chunk_size), dtype=jnp.float32)
    # One frame per step, not one window per step: the window the chunk ends on
    # is already in hand when the chunk closes.
    step_frames = jnp.zeros((n, chunk_size, obs_dim), dtype=jnp.float32)
    # Row n is a zero row so an env with no chunk reads zeros.
    plan = jnp.zeros((n + 1, chunk_size, act_dim), dtype=jnp.float32)
    env_rows = jnp.arange(n)
    live = jnp.zeros((n,), dtype=jnp.bool_)
    # roll_table[w][j] == (w + j) % n_obs_steps, so restoring a window's
    # chronological order is one row lookup instead of an arange, an add and a
    # modulo per closing chunk.
    steps_i = jnp.arange(n_obs_steps)
    roll_table = (steps_i[:, None] + jnp.arange(n_obs_steps)[None, :]) % n_obs_steps
    actions = jnp.zeros((n, act_dim), dtype=jnp.float32)

    cur: list = [None] * n
    pos = jnp.zeros((n,), dtype=jnp.int32)
    live_idx = None
    prev_done = ["continue"] * n
    executed_total = [0] * n
    awaiting = [False] * n
    prev_chunk = [None] * n
    remaining = sum(targets)

    while remaining:
        need = [
            i
            for i in range(n)
            if cur[i] is None and not awaiting[i] and executed_total[i] < targets[i]
        ]
        if need:
            # Advanced indexing already returns a copy, so no clone. wpos is 0
            # here: the window was rolled when the previous chunk of this env
            # closed, or refilled on autoreset.
            idx = jnp.asarray(need, dtype=jnp.int32)
            states = windows[idx]
            proposed = policy(states)
            # One clamp-and-rescale for the whole wavefront, not one per env.
            executed_all, scaled_all, clip_counts_all = scale_and_clip_chunk(
                proposed, low, range_
            )
            plan = plan.at[idx].set(executed_all)
            pos = pos.at[idx].set(0)
            for k, i in enumerate(need):
                cur[i] = (
                    executed_all[k],
                    scaled_all[k],
                    int(clip_counts_all[k]),
                    states[k],
                    min(n_action_steps, targets[i] - executed_total[i]),
                )
                live = live.at[i].set(True)
            live_idx = None  # the stepping set just changed

        # One gather replaces the n select+copy pairs. Envs with no chunk read
        # row n, the zero row.
        row = jnp.where(live, env_rows, jnp.full_like(env_rows, n))
        actions = plan[row, jnp.clip(pos, max=chunk_size - 1)]
        # No early-out: envs flagged `awaiting` have cur[i] is None but still need
        # this step() to take their reset observation (NEXT_STEP).
        obs, rew, term, trunc, _ = envs.step(np.asarray(actions))
        term_np = np.asarray(term, dtype=bool)
        trunc_np = np.asarray(trunc, dtype=bool)
        done_np = term_np | trunc_np
        obs_j = jnp.asarray(np.asarray(obs, dtype=np.float32))
        rew_j = jnp.asarray(np.asarray(rew, dtype=np.float32))

        if live_idx is None:
            live_idx = env_rows[live]
        if live_idx is not None and live_idx.shape[0]:
            # Three scatters for the whole vec step.
            p = pos[live_idx]
            reward_windows = reward_windows.at[live_idx, p].set(rew_j[live_idx])
            step_frames = step_frames.at[live_idx, p].set(obs_j[live_idx])
            windows = windows.at[live_idx, wpos[live_idx]].set(obs_j[live_idx])
            pos = pos.at[live_idx].set(p + 1)
            wpos = wpos.at[live_idx].add(1)

        for i in range(n):
            if awaiting[i]:
                # Refill the window with the reset observation.
                windows = windows.at[i].set(jnp.broadcast_to(obs_j[i], (n_obs_steps, obs_dim)))
                wpos = wpos.at[i].set(0)
                awaiting[i] = False
                live_idx = None
                continue
            if cur[i] is None:
                continue
            n_exec = int(pos[i])
            execution_limit = cur[i][4]
            if done_np[i] or n_exec >= execution_limit:
                _executed, scaled, clip_count, replan, _limit = cur[i]
                done_kind = "terminated" if term_np[i] else ("truncated" if trunc_np[i] else "continue")
                # Restore chronological order once per closing chunk: frames went
                # in at slot 0, 1, ... n_exec, so a left roll by the cursor puts
                # the oldest frame back at row 0.
                windows = windows.at[i].set(windows[i][roll_table[int(wpos[i]) % n_obs_steps]])
                wpos = wpos.at[i].set(0)
                live = live.at[i].set(False)
                live_idx = None
                next_state = windows[i]
                executed_total[i] += n_exec
                remaining -= n_exec
                prev_chunk[i] = buffer.add_chunk(
                    state=replan,
                    next_state=next_state,
                    action=scaled,
                    rewards=reward_windows[i, :n_exec],
                    step_frames=step_frames[i, :n_exec],
                    terminated=bool(term_np[i]),
                    truncated=bool(trunc_np[i]),
                    clip_count=clip_count,
                    env_id=i,
                )
                prev_done[i] = done_kind
                cur[i] = None
                if done_np[i]:
                    awaiting[i] = True

    for i in range(n):
        if prev_done[i] == "continue" and prev_chunk[i] is not None:
            buffer.mark_incomplete(prev_chunk[i])
    return buffer
