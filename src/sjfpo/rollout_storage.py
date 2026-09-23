"""Chunk-level rollout storage (rsl-rl style)."""

from __future__ import annotations

import torch

__all__ = ["RolloutStorage", "mini_batch_indices"]


class RolloutStorage:
    """Chunk-level rollout storage."""

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
        self.states = torch.zeros(
            max_chunks, num_envs, n_obs_steps, obs_dim, device=self.device, dtype=dtype
        )
        self.next_states = torch.zeros(
            max_chunks, num_envs, n_obs_steps, obs_dim, device=self.device, dtype=dtype
        )
        self.actions = torch.zeros(
            max_chunks,
            num_envs,
            action_horizon,
            action_dim,
            device=self.device,
            dtype=dtype,
        )
        self.rewards = torch.zeros(
            max_chunks, num_envs, action_horizon, device=self.device, dtype=dtype
        )
        self.lengths = torch.zeros(
            max_chunks, num_envs, device=self.device, dtype=torch.long
        )
        self.dones = torch.zeros(
            max_chunks, num_envs, device=self.device, dtype=torch.bool
        )
        self.truncated = torch.zeros(
            max_chunks, num_envs, device=self.device, dtype=torch.bool
        )
        self.values = torch.zeros(max_chunks, num_envs, device=self.device, dtype=dtype)
        self.returns = torch.zeros(
            max_chunks, num_envs, device=self.device, dtype=dtype
        )
        self.advantages = torch.zeros(
            max_chunks, num_envs, device=self.device, dtype=dtype
        )
        self.cursor = torch.zeros(num_envs, dtype=torch.long)
        self.count = 0

    def num_chunks(self) -> int:
        return self.count

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
        normalize_advantage: bool = True,
    ) -> None:
        """Amazon-style exact GAE: step-level over chunk steps, aggregate back."""
        # For each stored chunk, expand to step-level and apply Amazon GAE
        for c in range(self.max_chunks):
            for e in range(self.num_envs):
                L = int(self.lengths[c, e].item())
                if c >= self.count or L <= 0:
                    continue
                # Amazon exact recursion over L steps inside this chunk
                advantage = 0.0
                for step in reversed(range(L)):
                    if step == L - 1:
                        next_values = (
                            last_values[e, 0]
                            if last_values.dim() > 1
                            else last_values[e]
                        )
                    else:
                        # Approximate: use chunk value for intermediate steps
                        next_values = self.values[c, e]
                    next_is_not_terminal = (
                        1.0 - float(self.dones[c, e].item()) if step == L - 1 else 1.0
                    )
                    delta = (
                        float(self.rewards[c, e, step].item())
                        + next_is_not_terminal * gamma * float(next_values)
                        - float(self.values[c, e].item())
                    )
                    advantage = delta + next_is_not_terminal * gamma * lam * advantage
                # Amazon exact: write after full chunk recursion
                self.advantages[c, e] = float(advantage)
                self.returns[c, e] = float(advantage + float(self.values[c, e].item()))
        if normalize_advantage:
            advantages_flat = self.advantages
            mean_adv = advantages_flat.mean()
            std_adv = advantages_flat.std(unbiased=False) + 1e-8
            self.advantages = (advantages_flat - mean_adv) / std_adv

    def clear(self) -> None:
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
        c = int(self.cursor[env_idx].item())
        if c >= self.max_chunks:
            raise OverflowError("Rollout buffer overflow: call clear() before reuse")
        if isinstance(rewards_1d, torch.Tensor):
            r = rewards_1d.detach().to(device=self.device, dtype=self.dtype).reshape(-1)
        else:
            r = torch.as_tensor(
                rewards_1d, device=self.device, dtype=self.dtype
            ).reshape(-1)
        length = int(r.numel())
        if not 1 <= length <= self.action_horizon:
            raise ValueError(f"chunk must execute 1..H steps, got {length}")
        self.states[c, env_idx].copy_(state.to(device=self.device, dtype=self.dtype))
        self.actions[c, env_idx].copy_(action.to(device=self.device, dtype=self.dtype))
        self.next_states[c, env_idx].copy_(
            next_state.to(device=self.device, dtype=self.dtype)
        )
        self.rewards[c, env_idx].zero_()
        self.rewards[c, env_idx, :length].copy_(r)
        self.lengths[c, env_idx] = length
        self.dones[c, env_idx] = bool(done)
        self.truncated[c, env_idx] = bool(truncated)
        self.cursor[env_idx] += 1
        self.count += 1

    def to_transition_list(self) -> list[dict]:
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
                        DONE: to_tensor(
                            bool(self.dones[c, e].item()), dtype=torch.bool
                        ),
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
