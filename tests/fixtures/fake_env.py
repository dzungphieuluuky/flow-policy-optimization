"""Canonical FakeGymEnv fixture — minimal gym.Env for testing without MuJoCo.

Shared by the CPU-pipeline, CLI, and wandb-logging tests: obs_dim/action_dim/
max_steps defaults plus seed support in both constructor and reset.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class FakeGymEnv(gym.Env):
    metadata: dict = {}

    def __init__(
        self, obs_dim: int = 4, action_dim: int = 2, max_steps: int = 20, seed: int = 0
    ):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.step_count = 0
        self.rng = np.random.RandomState(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.step_count = 0
        return self.rng.randn(self.obs_dim).astype(np.float32), {}

    def step(self, action):
        self.step_count += 1
        obs = self.rng.randn(self.obs_dim).astype(np.float32)
        reward = float(self.rng.randn() * 0.1)
        terminated = False
        truncated = self.step_count >= self.max_steps
        return obs, reward, terminated, truncated, {}
