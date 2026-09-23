"""Chunk-level rollout storage (JAX)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["RolloutStorage", "mini_batch_indices"]


class RolloutStorage:
    """Chunk-level storage. Immutable-free; ``add_chunk`` returns a new self."""

    def __init__(
        self,
        num_envs: int,
        max_chunks: int,
        n_obs_steps: int,
        action_horizon: int,
        obs_dim: int,
        action_dim: int,
        dtype=jnp.float32,
    ):
        if num_envs < 1 or max_chunks < 1:
            raise ValueError("num_envs and max_chunks must be >= 1")
        self.num_envs = int(num_envs)
        self.max_chunks = int(max_chunks)
        self.n_obs_steps = int(n_obs_steps)
        self.action_horizon = int(action_horizon)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.dtype = dtype

        self.states = jnp.zeros(
            (max_chunks, num_envs, n_obs_steps, obs_dim), dtype=dtype
        )
        self.next_states = jnp.zeros_like(self.states)
        self.actions = jnp.zeros(
            (max_chunks, num_envs, action_horizon, action_dim), dtype=dtype
        )
        self.rewards = jnp.zeros(
            (max_chunks, num_envs, action_horizon), dtype=dtype
        )
        self.lengths = jnp.zeros((max_chunks, num_envs), dtype=jnp.int32)
        self.dones = jnp.zeros((max_chunks, num_envs), dtype=bool)
        self.truncated = jnp.zeros((max_chunks, num_envs), dtype=bool)
        self.values = jnp.zeros((max_chunks, num_envs), dtype=dtype)
        self.returns = jnp.zeros((max_chunks, num_envs), dtype=dtype)
        self.advantages = jnp.zeros((max_chunks, num_envs), dtype=dtype)
        self.cursor = jnp.zeros(num_envs, dtype=jnp.int32)
        self.count = 0

    def num_chunks(self) -> int:
        return self.count

    def clear(self) -> None:
        self.cursor = jnp.zeros(self.num_envs, dtype=jnp.int32)
        self.count = 0

    def add_chunk(
        self,
        env_idx: int,
        state,
        action,
        next_state,
        rewards_1d,
        done: bool,
        truncated: bool = False,
    ) -> None:
        """Mutate the buffer in place (Python-level mutation, jnp arrays)."""
        c = int(self.cursor[env_idx])
        if c >= self.max_chunks:
            raise OverflowError("Rollout buffer overflow: call clear() before reuse")
        r = jnp.asarray(rewards_1d, dtype=self.dtype).reshape(-1)
        length = int(r.shape[0])
        if not 1 <= length <= self.action_horizon:
            raise ValueError(f"chunk must execute 1..H steps, got {length}")

        self.states = self.states.at[c, env_idx].set(
            jnp.asarray(state, dtype=self.dtype)
        )
        self.actions = self.actions.at[c, env_idx].set(
            jnp.asarray(action, dtype=self.dtype)
        )
        self.next_states = self.next_states.at[c, env_idx].set(
            jnp.asarray(next_state, dtype=self.dtype)
        )
        self.rewards = self.rewards.at[c, env_idx].set(
            jnp.zeros((self.action_horizon,), dtype=self.dtype).at[:length].set(r)
        )
        self.lengths = self.lengths.at[c, env_idx].set(length)
        self.dones = self.dones.at[c, env_idx].set(bool(done))
        self.truncated = self.truncated.at[c, env_idx].set(bool(truncated))
        self.cursor = self.cursor.at[env_idx].add(1)
        self.count += 1

    def to_transition_list(self) -> list[dict]:
        from lerobot.utils.constants import DONE, OBS_STATE, REWARD, TRUNCATED

        out: list[dict] = []
        for e in range(self.num_envs):
            for c in range(int(self.cursor[e])):
                length = int(self.lengths[c, e])
                out.append(
                    {
                        OBS_STATE: self.states[c, e],
                        "action": self.actions[c, e],
                        "next.observation.state": self.next_states[c, e],
                        REWARD: self.rewards[c, e, :length],
                        DONE: jnp.asarray(bool(self.dones[c, e])),
                        TRUNCATED: jnp.asarray(bool(self.truncated[c, e])),
                        "executed_n_action_steps": jnp.asarray(length),
                        "clip_count": jnp.asarray(0, dtype=jnp.int32),
                    }
                )
        return out


def mini_batch_indices(n: int, num_minibatches: int, key, num_epochs: int = 1):
    """Yield index slices for minibatch training (explicit PRNG)."""
    if n < 1:
        raise ValueError(f"n={n} must be >= 1")
    n_mb = max(1, min(int(num_minibatches), n))
    for _ in range(max(1, int(num_epochs))):
        key, sub = jax.random.split(key)
        perm = jax.random.permutation(sub, n)
        for i in range(n_mb):
            start = (i * n) // n_mb
            stop = ((i + 1) * n) // n_mb
            yield perm[start:stop]