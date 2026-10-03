"""Chunked Q-learning: one chunk is one macro-transition.

A chunk is ``h`` open-loop steps, so backup at chunk granularity, not per step.
Chunk ``i`` at step ``s_i`` running ``h_i`` steps, with ``a_i`` the chunk it
executed and ``a'`` drawn from the policy at ``s_{s_i + h_i}``:

    R_i      = sum_{k < h_i} gamma^k * r_{s_i + k}
    delta_i  = R_i + gamma^{h_i} * (1 - terminated_i) * Q(s_{s_i + h_i}, a') - Q(s_{s_i}, a_i)
    return_i = R_i + gamma^{h_i} * (1 - terminated_i) * Q(s_{s_i + h_i}, a')

Both sides condition on the chunk, which is what makes the backup an unbiased
``h``-step target (Li, Zhou, Levine 2025) rather than an n-step return: the
Q that produced the rewards is the Q being backed up from.

One backup per chunk, no GAE tail. A lambda-GAE recursion over chunks used to
sit on top of this, giving an effective horizon of ``1 / (1 - gamma^h * lam)``
chunks -- about 5 at gamma=0.99, h=16, lam=0.95 -- where QC backs up exactly
one chunk. There is no lambda argument here on purpose.

Timeout keeps the bootstrap. Only real termination zeroes it. The collector
always ends a chunk on a terminal, so ``~terminated`` matches QC's
``masks[..., -1]``. Keep that true.

``gamma^h`` uses the executed length, not the configured horizon: a chunk cut
short by termination discounts by what actually ran.
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
    """One chunked TD backup per chunk of a rollout.

    ``chunk_q[i]`` is Q(s_i, a_i) and ``chunk_next_q[i]`` is
    Q(s_{i+1}, a' ~ pi(s_{i+1})). Returns the advantages and the regression
    target, which is ``chunk_q + advantages`` and so reduces exactly to QC's
    ``R_i + gamma^{h_i} (1 - term_i) Q(s_{i+1}, a')``. ``ValueError`` if ``gamma``
    leaves ``[0, 1]``, or lengths disagree.
    """
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
    # Only real termination kills the bootstrap. A timeout keeps it.
    delta = chunk_rewards + gamma_h * (~terminated) * chunk_next_q - chunk_q
    return delta, delta + chunk_q
