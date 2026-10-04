"""Chunked Q-learning: one chunk is one macro-transition.

A chunk is ``h`` open-loop steps, so backup at chunk granularity:

    R_i      = sum_{k < h_i} gamma^k * r_{s_i + k}
    return_i = R_i + gamma^{h_i} * (1 - terminated_i) * Q(s_{s_i + h_i}, a')

Conditioning both sides on the chunk makes this an unbiased ``h``-step target
(Li, Zhou, Levine 2025): the Q that produced the rewards is the Q being backed
up from.

No lambda argument: QC backs up exactly one chunk. Timeout keeps the bootstrap,
only real termination zeroes it, and the collector always ends a chunk on a
terminal, so ``~terminated`` matches QC's ``masks[..., -1]``. ``gamma^h`` uses
the executed length.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def compute_chunk_advantages(
    step_rewards: torch.Tensor,
    chunk_q: torch.Tensor,
    chunk_next_q: torch.Tensor,
    chunk_starts: torch.Tensor,
    chunk_lengths: torch.Tensor,
    *,
    chunk_terminated: torch.Tensor,
    chunk_size: int,
    gamma: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[n]`` advantages and regression target; the target is ``chunk_q +
    advantages``, so it reduces exactly to QC's ``R_i + gamma^{h_i}
    (1 - term_i) Q(s_{i+1}, a')``. ``chunk_q[i]`` is Q(s_i, a_i),
    ``chunk_next_q[i]`` is Q(s_{i+1}, a' ~ pi(s_{i+1}))."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")

    device = chunk_q.device
    dtype = torch.float32

    rewards = step_rewards.to(device=device, dtype=dtype).reshape(-1)
    chunk_q = chunk_q.to(device=device, dtype=dtype).reshape(-1)
    chunk_next_q = chunk_next_q.to(device=device, dtype=dtype).reshape(-1)
    starts = chunk_starts.to(device=device, dtype=torch.long).reshape(-1)
    lengths = chunk_lengths.to(device=device, dtype=torch.long).reshape(-1)
    terminated = chunk_terminated.to(device=device, dtype=torch.bool).reshape(-1)

    n_chunks = starts.numel()
    for name, tensor in (
        ("chunk_lengths", lengths),
        ("chunk_terminated", terminated),
        ("chunk_q", chunk_q),
        ("chunk_next_q", chunk_next_q),
    ):
        if tensor.numel() != n_chunks:
            raise ValueError(f"{name} must have {n_chunks} entries, got {tensor.numel()}")

    if n_chunks == 0:
        empty = torch.empty(0, device=device, dtype=dtype)
        return empty, empty

    # One shot gather, then discount by gamma^k. A cumsum diff would need
    # gamma**-s_i, which loses all fp32 precision once s_i hits a few thousand.
    offsets = torch.arange(chunk_size, device=device)
    window = starts.unsqueeze(1) + offsets
    gathered = rewards[window.clamp(max=rewards.numel() - 1)]
    valid = offsets.unsqueeze(0) < lengths.unsqueeze(1)
    chunk_rewards = (gathered * valid * (gamma**offsets)).sum(dim=1)

    gamma_h = gamma ** lengths.to(dtype)
    delta = chunk_rewards + gamma_h * (~terminated) * chunk_next_q - chunk_q
    return delta, delta + chunk_q
