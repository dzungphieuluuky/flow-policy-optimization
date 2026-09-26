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


    def clear(self) -> None:
        self.cursor = jnp.zeros(self.num_envs, dtype=jnp.int32)
        self.count = 0




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