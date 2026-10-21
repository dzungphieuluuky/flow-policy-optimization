"""Chunked Q-learning. One chunk is one macro-transition."""
from __future__ import annotations

import torch


@torch.no_grad()
def compute_chunk_advantages(
    step_rewards: torch.Tensor,
    q_current_chunk: torch.Tensor,
    q_next_state: torch.Tensor,
    chunk_starts: torch.Tensor,
    chunk_lengths: torch.Tensor,
    *,
    chunk_terminated: torch.Tensor,
    chunk_size: int,
    gamma: float = 0.99,
    chunk_truncated: torch.Tensor | None = None,
    chunk_env_id: torch.Tensor | None = None,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``[n]`` advantages and the ``[n]`` regression target.

    The target is ``advantage + q_current_chunk``: a TD(lambda) backup over the
    macro-transitions defined by ``chunk_starts`` / ``chunk_lengths``. Chunks
    from different envs interleave in the buffer, so adjacency is resolved
    through ``chunk_env_id`` rather than by buffer position.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")

    device = q_current_chunk.device
    dtype = torch.float32

    rewards = step_rewards.to(device=device, dtype=dtype).reshape(-1)
    q_current_chunk = q_current_chunk.to(device=device, dtype=dtype).reshape(-1)
    q_next_state = q_next_state.to(device=device, dtype=dtype).reshape(-1)
    starts = chunk_starts.to(device=device, dtype=torch.long).reshape(-1)
    lengths = chunk_lengths.to(device=device, dtype=torch.long).reshape(-1)
    terminated = chunk_terminated.to(device=device, dtype=torch.bool).reshape(-1)

    n_chunks = starts.numel()
    for name, tensor in (
        ("chunk_lengths", lengths),
        ("chunk_terminated", terminated),
        ("q_current_chunk", q_current_chunk),
        ("q_next_state", q_next_state),
    ):
        if tensor.numel() != n_chunks:
            raise ValueError(f"{name} must have {n_chunks} entries, got {tensor.numel()}")

    if chunk_truncated is None:
        truncated = torch.zeros(n_chunks, device=device, dtype=torch.bool)
    else:
        truncated = chunk_truncated.to(device=device, dtype=torch.bool).reshape(-1)
        if truncated.numel() != n_chunks:
            raise ValueError(
                f"chunk_truncated must have {n_chunks} entries, got {truncated.numel()}"
            )

    if gae_lambda > 0.0:
        if chunk_truncated is None:
            raise ValueError(
                "chunk_truncated is required when gae_lambda > 0: without it the "
                "advantage trace crosses an episode reset"
            )
        if chunk_env_id is None:
            raise ValueError(
                "chunk_env_id is required when gae_lambda > 0: chunks from different "
                "envs interleave in the buffer, so buffer adjacency is not trajectory "
                "adjacency"
            )

    if n_chunks == 0:
        empty = torch.empty(0, device=device, dtype=dtype)
        return empty, empty

    offsets = torch.arange(chunk_size, device=device)
    window = starts.unsqueeze(1) + offsets
    gathered = rewards[window.clamp(max=rewards.numel() - 1)]
    valid = offsets.unsqueeze(0) < lengths.unsqueeze(1)
    chunk_rewards = (gathered * valid * (gamma**offsets)).sum(dim=1)

    gamma_h = gamma ** lengths.to(dtype)
    done = terminated | truncated
    injected = truncated & ~terminated
    chunk_rewards = chunk_rewards + gamma_h * injected * q_next_state
    bootstrap = gamma_h * (~done)
    delta = chunk_rewards + bootstrap * q_next_state - q_current_chunk

    if gae_lambda == 0.0:
        return delta, delta + q_current_chunk

    env_id = chunk_env_id.to(device=device, dtype=torch.long).reshape(-1)
    if env_id.numel() != n_chunks:
        raise ValueError(f"chunk_env_id must have {n_chunks} entries, got {env_id.numel()}")

    successor = chunk_successors(env_id)
    trace = gamma_h * (~done) * gae_lambda
    adv = gae_trace(delta, trace, successor)
    return adv, adv + q_current_chunk


def chunk_successors(env_id: torch.Tensor) -> torch.Tensor:
    """The next chunk of each chunk, inside the same env. ``-1`` means none.

    The chunks of different envs interleave in the buffer, so adjacency is
    resolved through the env id, not through the buffer position.
    """
    order = torch.argsort(env_id, stable=True)
    sorted_env = env_id[order]
    same_env = sorted_env[:-1] == sorted_env[1:]
    successor = torch.full_like(order, -1)
    successor[order[:-1][same_env]] = order[1:][same_env]
    return successor


def gae_trace(
    delta: torch.Tensor, trace: torch.Tensor, successor: torch.Tensor
) -> torch.Tensor:
    """The advantage of each chunk, traced back over its successors."""
    successor_host = successor.tolist()
    adv = delta.clone()
    for i in range(delta.numel() - 1, -1, -1):
        j = successor_host[i]
        if j >= 0:
            adv[i] = delta[i] + trace[i] * adv[j]
    return adv