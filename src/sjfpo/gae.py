"""Chunk-level GAE: one action chunk is one macro-transition.

The rollout action is a chunk of ``h`` open-loop steps, so the Bellman
backup is taken at chunk granularity (QC-style, ``discount ** horizon_length``)
rather than per executed step. For chunk ``i`` starting at executed step
``s_i`` and running ``h_i`` steps:

    R_i     = sum_{k < h_i} gamma^k * r_{s_i + k}
    delta_i = R_i + gamma^{h_i} * (1 - terminated_i) * V(s_{s_i + h_i}) - V(s_{s_i})
    A_i     = delta_i + gamma^{h_i} * lam * continue_i * A_{i+1}
    return_i = A_i + V(s_{s_i})

``R_i`` is the discounted sum, which is also what QC's ``rewards[..., -1]``
holds (``utils/datasets.py`` accumulates it over the sequence). A truncated
(time-limit) chunk still bootstraps from the value after its last executed
step; only a true termination zeroes the bootstrap. The collector always ends
a chunk on a terminal, so ``~terminated`` coincides with QC's
``masks[..., -1]``; that invariant must be preserved.

The vec collector appends chunks in wavefront order, so consecutive buffer
entries belong to different envs. ``chunk_env_id`` is what lets the recursion
know that ``i`` and ``i+1`` are on the same trajectory.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def compute_chunk_advantages(
    step_rewards: torch.Tensor,
    step_values: torch.Tensor,
    step_next_values: torch.Tensor,
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
    """Compute chunk-level GAE over the executed chunks of a rollout.

    The chunk tensors must all have the same length. Returns advantages per chunk,
    and ``values + advantages``, the chunk-level value target. Raises
    ``ValueError`` if ``gamma`` or ``lam`` fall outside ``[0, 1]``, or if the
    chunk tensors are not the same length.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    device = step_values.device
    dtype = torch.float32

    rewards = step_rewards.to(device=device, dtype=dtype).reshape(-1)
    values = step_values.to(device=device, dtype=dtype).reshape(-1)
    next_values = step_next_values.to(device=device, dtype=dtype).reshape(-1)
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
    ):
        if tensor.numel() != n_chunks:
            raise ValueError(f"{name} must have {n_chunks} entries, got {tensor.numel()}")

    if n_chunks == 0:
        empty = torch.empty(0, device=device, dtype=dtype)
        return empty, empty

    # Gather each chunk's executed rewards in one shot and discount them by
    # gamma^k. A cumsum difference would need gamma**-s_i, which loses all
    # precision in fp32 once s_i reaches a few thousand steps.
    offsets = torch.arange(chunk_size, device=device)
    window = starts.unsqueeze(1) + offsets                      # [B, chunk_size]
    gathered = rewards[window.clamp(max=rewards.numel() - 1)]  # [B, chunk_size]
    valid = offsets.unsqueeze(0) < lengths.unsqueeze(1)        # [B, chunk_size]
    chunk_rewards = (gathered * valid * (gamma**offsets)).sum(dim=1)   # [B]

    last_step = starts + lengths - 1                            # [B]
    chunk_value = values[starts]                                 # [B]
    chunk_next_value = next_values[last_step]                   # [B]

    # Chunks are stored in wavefront order, so one env's chunks sit n_envs
    # apart and never adjacent: an adjacency mask finds no links at all. A
    # stable sort groups each env contiguously and keeps its temporal order.
    order = torch.argsort(env_id, stable=True)
    unsort = torch.argsort(order)
    lengths, env_id = lengths[order], env_id[order]
    terminated, truncated = terminated[order], truncated[order]
    chunk_rewards = chunk_rewards[order]
    chunk_value, chunk_next_value = chunk_value[order], chunk_next_value[order]
    same_env_sorted = env_id[1:] == env_id[:-1]

    gamma_h = gamma ** lengths.to(dtype)                        # [B]
    # Only a true termination removes the bootstrap; a timeout keeps it.
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
