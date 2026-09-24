"""Step-level GAE over executed rollout transitions.

The Torch path computes GAE per executed environment step and then selects
the first-step target of each action chunk.
"""

from __future__ import annotations

import torch




@torch.no_grad()
def compute_step_advantages_torch(
    rewards_1d: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
    truncated: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute step-level GAE over a flat stream of executed transitions.

    ``values[t]`` and ``next_values[t]`` are the critic values before and
    after executed transition ``t``. The returned advantages and returns are
    both one-dimensional and aligned with the flat reward stream. A
    terminated transition has no bootstrap and cuts recursion; a truncated
    transition bootstraps from ``next_values[t]`` but also cuts recursion.
    """
    device = values.device
    dtype = torch.float32
    rewards = rewards_1d.to(device=device, dtype=dtype).reshape(-1)
    values = values.to(device=device, dtype=dtype).reshape(-1)
    next_values = next_values.to(device=device, dtype=dtype).reshape(-1)
    terminated = terminated.to(device=device, dtype=torch.bool).reshape(-1)
    if truncated is None:
        truncated = torch.zeros_like(terminated)
    else:
        truncated = truncated.to(device=device, dtype=torch.bool).reshape(-1)

    total_steps = rewards.numel()
    if not (
        values.numel() == next_values.numel() == terminated.numel() == total_steps
        and truncated.numel() == total_steps
    ):
        raise ValueError(
            "step GAE inputs must all have the same length as rewards_1d"
        )
    if total_steps == 0:
        return values, values

    delta = rewards - values + gamma * torch.where(
        terminated, torch.zeros_like(next_values), next_values
    )
    advantages = torch.empty_like(delta)
    next_advantage = torch.zeros((), device=device, dtype=dtype)
    continuation_factor = torch.as_tensor(gamma * lam, device=device, dtype=dtype)
    cut_recursion = terminated | truncated
    for index in range(total_steps - 1, -1, -1):
        continuation = torch.where(
            cut_recursion[index],
            torch.zeros((), device=device, dtype=dtype),
            continuation_factor,
        )
        next_advantage = delta[index] + continuation * next_advantage
        advantages[index] = next_advantage

    return advantages, advantages + values
