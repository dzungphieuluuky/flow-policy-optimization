"""Minimal persistent rollout storage matching rsl-rl design (fixed arrays,
chunk-level batch indexing, multi-env [envs, chunk_size, ...] tensors).
No Python dict per transition overhead; arrays updated in-place."""
from __future__ import annotations

import torch


class RolloutStorage:
    """Persistent array rollout storage (rsl-rl style)."""

    def __init__(self, num_envs: int, chunk_size: int, obs_dim: int,
                 action_dim: int, hidden_dim: int = 0, max_steps: int = 10000):
        self.num_envs = num_envs
        self.chunk_size = chunk_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        # Persistent arrays [envs, max_steps, ...]
        self.obs = torch.zeros(num_envs, max_steps, obs_dim)
        self.action = torch.zeros(num_envs, max_steps, action_dim)
        self.reward = torch.zeros(num_envs, max_steps)
        self.done = torch.zeros(num_envs, max_steps, dtype=torch.bool)
        self.truncated = torch.zeros(num_envs, max_steps, dtype=torch.bool)
        self.advantage = torch.zeros(num_envs, max_steps)
        self.returnss = torch.zeros(num_envs, max_steps)
        self.value = torch.zeros(num_envs, max_steps)
        if hidden_dim > 0:
            self.hidden_state = torch.zeros(num_envs, hidden_dim)
        else:
            self.hidden_state = None
        self.step = torch.zeros(num_envs, dtype=torch.long)
        self.total_steps = 0

    def add_chunk(self, env_idx: int, step_offset: int,
                  obs: torch.Tensor, action: torch.Tensor,
                  reward: float, done: bool, truncated: bool = False):
        s = self.step[env_idx].item()
        self.obs[env_idx, s] = obs.squeeze(0)
        self.action[env_idx, s] = action.squeeze(0)
        self.reward[env_idx, s] = float(reward)
        self.done[env_idx, s] = bool(done)
        self.truncated[env_idx, s] = bool(truncated)
        self.step[env_idx] += 1
        self.total_steps += 1

    def get_batch(self, batch_size: int = 256, start: int = 0) -> dict:
        # Minimal batch indexing: return slice of stored arrays
        end = min(start + batch_size, self.total_steps)
        s = torch.arange(start, end)
        return {
            "obs": self.obs[:, s],
            "action": self.action[:, s],
            "reward": self.reward[:, s],
            "done": self.done[:, s],
            "truncated": self.truncated[:, s],
            "advantage": self.advantage[:, s],
            "return": self.returns[:, s],
            "value": self.value[:, s],
        }

    def clear_done_envs(self, done_mask: torch.Tensor):
        for i in range(self.num_envs):
            if done_mask[i].item():
                self.step[i] = 0
                self.obs[i] = 0
                self.action[i] = 0
                self.reward[i] = 0
                self.done[i] = False
                self.truncated[i] = False
