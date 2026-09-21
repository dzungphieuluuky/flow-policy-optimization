"""Chunk-level rollout storage (rsl-rl style, preallocated device buffers).

One entry = one open-loop chunk: window [T, obs], full proposed chunk
[H, act], padded executed rewards [H] + length, done/truncated flags.
Buffers are allocated once per collection round on the policy device and
written with in-place ``copy_`` (no per-chunk allocs, no list-of-dicts in
the hot path). ``to_transition_list`` rebuilds the legacy ``Transition``
dicts for tests / the stacking path in ``generate_rollout``.
"""
from __future__ import annotations

import torch

__all__ = ["RolloutStorage", "mini_batch_indices"]


class RolloutStorage:
    """Preallocated [C, N, ...] chunk buffers (C = max chunks per env)."""

    def __init__(
        self,
        num_envs: int,
        max_chunks: int,
        n_obs_steps: int,
        action_horizon: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if num_envs < 1 or max_chunks < 1:
            raise ValueError("num_envs and max_chunks must be >= 1")
        self.num_envs = int(num_envs)
        self.max_chunks = int(max_chunks)
        self.n_obs_steps = int(n_obs_steps)
        self.action_horizon = int(action_horizon)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.dtype = dtype
        # Rollout fields (floats on device/dtype; flags as bool).
        self.states = torch.zeros(
            max_chunks, num_envs, n_obs_steps, obs_dim, device=self.device, dtype=dtype
        )
        self.next_states = torch.zeros(
            max_chunks, num_envs, n_obs_steps, obs_dim, device=self.device, dtype=dtype
        )
        self.actions = torch.zeros(
            max_chunks, num_envs, action_horizon, action_dim, device=self.device, dtype=dtype
        )
        self.rewards = torch.zeros(
            max_chunks, num_envs, action_horizon, device=self.device, dtype=dtype
        )
        self.lengths = torch.zeros(max_chunks, num_envs, device=self.device, dtype=torch.long)
        self.dones = torch.zeros(max_chunks, num_envs, device=self.device, dtype=torch.bool)
        self.truncated = torch.zeros(max_chunks, num_envs, device=self.device, dtype=torch.bool)
        # Post-collection targets (filled by generate_rollout, same layout).
        self.values = torch.zeros(max_chunks, num_envs, device=self.device, dtype=dtype)
        self.returns = torch.zeros(max_chunks, num_envs, device=self.device, dtype=dtype)
        self.advantages = torch.zeros(max_chunks, num_envs, device=self.device, dtype=dtype)
        self.cursor = torch.zeros(num_envs, dtype=torch.long)
        self.count = 0

    def num_chunks(self) -> int:
        """Total chunks stored across envs."""
        return self.count

    def clear(self) -> None:
        """Reset cursors for the next collection round (buffers reused)."""
        self.cursor.zero_()
        self.count = 0

    def add_chunk(
        self,
        env_idx: int,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        rewards_1d: torch.Tensor | list[float],
        done: bool,
        truncated: bool = False,
    ) -> None:
        """Copy one chunk into ``[cursor, env_idx]`` (rsl-rl ``add_transition``)."""
        c = int(self.cursor[env_idx].item())
        if c >= self.max_chunks:
            raise OverflowError("Rollout buffer overflow: call clear() before reuse")
        if isinstance(rewards_1d, torch.Tensor):
            r = rewards_1d.detach().to(device=self.device, dtype=self.dtype).reshape(-1)
        else:
            r = torch.as_tensor(rewards_1d, device=self.device, dtype=self.dtype).reshape(-1)
        length = int(r.numel())
        if not 1 <= length <= self.action_horizon:
            raise ValueError(f"chunk must execute 1..H steps, got {length}")
        self.states[c, env_idx].copy_(state.to(device=self.device, dtype=self.dtype))
        self.actions[c, env_idx].copy_(action.to(device=self.device, dtype=self.dtype))
        self.next_states[c, env_idx].copy_(next_state.to(device=self.device, dtype=self.dtype))
        self.rewards[c, env_idx].zero_()
        self.rewards[c, env_idx, :length].copy_(r)
        self.lengths[c, env_idx] = length
        self.dones[c, env_idx] = bool(done)
        self.truncated[c, env_idx] = bool(truncated)
        self.cursor[env_idx] += 1
        self.count += 1

    def to_transition_list(self) -> list[dict]:
        """Legacy ``Transition`` dicts in [env, chunk] order (tests / stack path)."""
        from lerobot.processor.converters import to_tensor
        from lerobot.utils.constants import DONE, OBS_STATE, REWARD, TRUNCATED

        out: list[dict] = []
        for e in range(self.num_envs):
            for c in range(int(self.cursor[e].item())):
                length = int(self.lengths[c, e].item())
                out.append(
                    {
                        OBS_STATE: to_tensor(self.states[c, e].detach().cpu()),
                        "action": to_tensor(self.actions[c, e].detach().cpu()),
                        "next.observation.state": to_tensor(
                            self.next_states[c, e].detach().cpu()
                        ),
                        REWARD: to_tensor(
                            self.rewards[c, e, :length].detach().cpu(), dtype=self.dtype
                        ),
                        DONE: to_tensor(bool(self.dones[c, e].item()), dtype=torch.bool),
                        TRUNCATED: to_tensor(
                            bool(self.truncated[c, e].item()), dtype=torch.bool
                        ),
                        "executed_n_action_steps": to_tensor(length, dtype=torch.int64),
                        "clip_count": to_tensor(0, dtype=torch.int64),
                    }
                )
        return out


def mini_batch_indices(
    n: int, num_minibatches: int, device: torch.device | str, num_epochs: int = 1
):
    """Yield one GPU ``randperm`` split per epoch (rsl-rl ``mini_batch_generator``).

    Single permutation per epoch, contiguous splits (last batch takes the
    remainder) — no dropped samples,unlike ``split(n // n_mb)``.
    """
    if n < 1:
        raise ValueError(f"n={n} must be >= 1")
    n_mb = max(1, min(int(num_minibatches), n))
    device = torch.device(device)
    for _ in range(max(1, int(num_epochs))):
        perm = torch.randperm(n, device=device)
        for i in range(n_mb):
            start = (i * n) // n_mb
            stop = ((i + 1) * n) // n_mb
            yield perm[start:stop]
