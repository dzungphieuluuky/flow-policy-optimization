"""Chunk-level GAE: one chunk is one macro-transition.

A chunk is ``h`` open-loop steps, so backup at chunk granularity, not per step.
Chunk ``i`` at step ``s_i`` running ``h_i`` steps:

    R_i     = sum_{k < h_i} gamma^k * r_{s_i + k}
    delta_i = R_i + gamma^{h_i} * (1 - terminated_i) * V(s_{s_i + h_i}) - V(s_{s_i})
    A_i     = delta_i + gamma^{h_i} * lam * continue_i * A_{i+1}
    return_i = A_i + V(s_{s_i})

Timeout keeps the bootstrap. Only real termination zeroes it. The collector
always ends a chunk on a terminal, so ``~terminated`` matches QC's
``masks[..., -1]``. Keep that true.

The vec collector appends in wavefront order, so adjacent rows are different
envs. ``chunk_env_id`` is what tells the recursion ``i`` and ``i+1`` share a
trajectory.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def compute_chunk_advantages(
    step_rewards: torch.Tensor,
    chunk_value: torch.Tensor,
    chunk_next_value: torch.Tensor,
    chunk_starts: torch.Tensor,
    chunk_lengths: torch.Tensor,
    *,
    chunk_terminated: torch.Tensor,
    chunk_truncated: torch.Tensor,
    chunk_env_id: torch.Tensor,
    chunk_size: int,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk-level GAE over a rollout's executed chunks.

    All chunk tensors must be the same length. Returns per-chunk advantages and
    ``values + advantages``, the value target. ``ValueError`` if ``gamma`` or
    ``lam`` leaves ``[0, 1]``, or lengths disagree.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    device = chunk_value.device
    dtype = torch.float32

    rewards = step_rewards.to(device=device, dtype=dtype).reshape(-1)
    chunk_value = chunk_value.to(device=device, dtype=dtype).reshape(-1)
    chunk_next_value = chunk_next_value.to(device=device, dtype=dtype).reshape(-1)
    starts = chunk_starts.to(device=device, dtype=torch.long).reshape(-1)
    lengths = chunk_lengths.to(device=device, dtype=torch.long).reshape(-1)
    terminated = chunk_terminated.to(device=device, dtype=torch.bool).reshape(-1)
    truncated = chunk_truncated.to(device=device, dtype=torch.bool).reshape(-1)
    env_id = chunk_env_id.to(device=device, dtype=torch.long).reshape(-1)

    n_chunks = starts.numel()
    for name, tensor in (
        ("chunk_lengths", lengths),
        ("chunk_terminated", terminated),
        ("chunk_truncated", truncated),
        ("chunk_env_id", env_id),
        ("chunk_value", chunk_value),
        ("chunk_next_value", chunk_next_value),
    ):
        if tensor.numel() != n_chunks:
            raise ValueError(f"{name} must have {n_chunks} entries, got {tensor.numel()}")

    if n_chunks == 0:
        empty = torch.empty(0, device=device, dtype=dtype)
        return empty, empty

    # One shot gather, then discount by gamma^k. A cumsum diff would need
    # gamma**-s_i, which loses all fp32 precision once s_i hits a few thousand.
    offsets = torch.arange(chunk_size, device=device)
    window = starts.unsqueeze(1) + offsets                      # [B, chunk_size]
    gathered = rewards[window.clamp(max=rewards.numel() - 1)]  # [B, chunk_size]
    valid = offsets.unsqueeze(0) < lengths.unsqueeze(1)        # [B, chunk_size]
    chunk_rewards = (gathered * valid * (gamma**offsets)).sum(dim=1)   # [B]

    # Wavefront order puts one env's chunks n_envs apart, never adjacent, so an
    # adjacency mask finds no links. A stable sort groups each env and keeps
    # its temporal order.
    order = torch.argsort(env_id, stable=True)
    unsort = torch.argsort(order)
    lengths, env_id = lengths[order], env_id[order]
    terminated, truncated = terminated[order], truncated[order]
    chunk_rewards = chunk_rewards[order]
    chunk_value, chunk_next_value = chunk_value[order], chunk_next_value[order]
    same_env_sorted = env_id[1:] == env_id[:-1]

    gamma_h = gamma ** lengths.to(dtype)                        # [B]
    # Only real termination kills the bootstrap. A timeout keeps it.
    delta = (
        chunk_rewards
        + gamma_h * (~terminated) * chunk_next_value
        - chunk_value
    )                                                           # [B]

    continuation_factor = torch.zeros(n_chunks, device=device, dtype=dtype)
    if n_chunks > 1:
        continues = same_env_sorted & ~(terminated[:-1] | truncated[:-1])
        continuation_factor[:-1] = gamma_h[:-1] * lam * continues.to(dtype)

    advantages = torch.empty(n_chunks, device=device, dtype=dtype)
    next_advantage = torch.zeros((), device=device, dtype=dtype)
    for index in range(n_chunks - 1, -1, -1):
        next_advantage = delta[index] + continuation_factor[index] * next_advantage
        advantages[index] = next_advantage

    advantages = advantages[unsort]
    return advantages, advantages + chunk_value[unsort]
