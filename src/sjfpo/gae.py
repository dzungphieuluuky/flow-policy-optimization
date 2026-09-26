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

    Inputs must be ordered by trajectory. A trajectory boundary must be
    indicated by either ``terminated`` or ``truncated``.

    For transition t, the FPO++ timeout handling adjusts the reward before
    computing the TD residual:

        r_adjusted = r_t + gamma * V(s_t)  when truncated

    Then:

        delta_t = r_adjusted + gamma * bootstrap_t - V(s_t)

    where ``bootstrap_t`` is zero for terminated transitions and
    ``V(s_(t+1))`` otherwise. GAE recursion continues across ordinary
    transitions and stops at both terminated and truncated transitions.

    Returns:
        advantages: GAE advantage for every executed step.
        returns: ``values + advantages``, i.e. the GAE value target.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    if rewards_1d.ndim != 1:
        raise ValueError(
            f"rewards_1d must be 1D, got shape {tuple(rewards_1d.shape)}"
        )

    device = values.device
    dtype = torch.float32

    rewards = rewards_1d.to(device=device, dtype=dtype)
    values = values.to(device=device, dtype=dtype).reshape(-1)
    next_values = next_values.to(device=device, dtype=dtype).reshape(-1)
    terminated = terminated.to(device=device, dtype=torch.bool).reshape(-1)

    if truncated is None:
        truncated = torch.zeros_like(terminated)
    else:
        truncated = truncated.to(device=device, dtype=torch.bool).reshape(-1)

    total_steps = rewards.numel()

    if not (
        values.numel()
        == next_values.numel()
        == terminated.numel()
        == truncated.numel()
        == total_steps
    ):
        raise ValueError(
            "step GAE inputs must all have the same length as rewards_1d"
        )

    if total_steps == 0:
        empty = torch.empty(0, device=device, dtype=dtype)
        return empty, empty

    # Match rsl_rl/FPO++: timeouts are handled by adding gamma * V(s_t) to
    # the reward, then the transition is treated as terminal for GAE.
    effective_rewards = torch.where(
        truncated,
        rewards + gamma * values,
        rewards,
    )
    boundary = terminated | truncated
    bootstrap = torch.where(
        boundary,
        torch.zeros_like(next_values),
        next_values,
    )
    delta = effective_rewards + gamma * bootstrap - values

    advantages = torch.empty_like(delta)
    next_advantage = torch.zeros((), device=device, dtype=dtype)

    continuation_factor = torch.as_tensor(
        gamma * lam,
        device=device,
        dtype=dtype,
    )
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
