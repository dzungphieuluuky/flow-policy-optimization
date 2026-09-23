"""Step-level GAE over chunk-level rollout (Amazon-style).

Chunk-level semi-MDP GAE: intra-chunk rewards are discounted by γ^j, the
chunk advantage is δ_k + (γλ)^{L_k} · A_{k+1}, and the bootstrap uses
γ^{L_k} · V_{k+1} unless the chunk is terminal or truncated.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def compute_advantages_torch(
    rewards_1d: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    lengths: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    normalize: bool = True,
    truncated: torch.Tensor | None = None,
    clamp: float | None = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk-level semi-MDP GAE.

    Parameters
    ----------
    rewards_1d : [sum(lengths)]  flat rewards across every chunk, in order.
    values     : [N]             V(s_k) at the start of each chunk.
    next_values: [N]             V(s_{k+1}) at the end of each chunk.
    lengths    : [N]             number of steps in each chunk (>= 1).
    terminated : [N]             bool, True when the chunk ended the episode.
    truncated  : [N] or None     bool, True when the chunk was cut short.
    """
    device = values.device
    dtype = torch.float32

    rewards = rewards_1d.to(device=device, dtype=dtype).reshape(-1)
    values = values.to(device=device, dtype=dtype).reshape(-1)
    next_values = next_values.to(device=device, dtype=dtype).reshape(-1)
    lengths = lengths.to(device=device, dtype=torch.long).reshape(-1)
    terminated = terminated.to(device=device, dtype=torch.bool).reshape(-1)

    n = values.numel()
    if n == 0:
        empty = torch.empty(0, device=device, dtype=dtype)
        return empty, empty

    if truncated is None:
        truncated = torch.zeros(n, device=device, dtype=torch.bool)
    else:
        truncated = truncated.to(device=device, dtype=torch.bool).reshape(-1)

    # Validate the flat rewards stream without a sync.
    total_steps = int(lengths.sum().item())
    if total_steps != rewards.numel():
        raise ValueError(
            f"rewards length {rewards.numel()} != sum(lengths) {total_steps}"
        )
    if (lengths <= 0).any():
        raise ValueError("every chunk must have length >= 1")

    # --- Intra-chunk discounted rewards via vectorized segment sum ---
    # chunk_ids[j] = chunk index of step j.
    chunk_ids = torch.repeat_interleave(
        torch.arange(n, device=device, dtype=torch.long), lengths
    )
    # offsets[k] = start index of chunk k in the flat rewards stream.
    offsets = torch.zeros(n + 1, device=device, dtype=torch.long)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    pos = (
        torch.arange(rewards.numel(), device=device, dtype=torch.long)
        - offsets[chunk_ids]
    ).to(dtype)  # [total_steps]

    # γ^pos = exp(pos · log γ); log γ is a Python float (compile-time).
    import math

    log_gamma = math.log(gamma)
    r_chunk = torch.zeros(n, device=device, dtype=dtype)
    r_chunk.index_add_(0, chunk_ids, rewards * torch.exp(pos * log_gamma))

    # --- Precompute per-chunk continuation factors ---
    len_f = lengths.to(dtype)
    gamma_len = torch.exp(len_f * log_gamma)  # γ^{L_k}
    gamma_lam_log = math.log(gamma * lam)
    gamma_lam_len = torch.exp(len_f * gamma_lam_log)  # (γλ)^{L_k}

    done = terminated | truncated  # [N]

    # --- Backward recursion over chunks (still a Python loop, but no syncs) ---
    # lengths / done / gamma_len / gamma_lam_len are all on-device tensors;
    # the loop only carries a scalar, and no .item() is called inside it.
    # For deep streams, replace with a compiled scan.
    advantages = torch.empty(n, device=device, dtype=dtype)
    next_adv = torch.zeros((), device=device, dtype=dtype)
    # Reverse iteration; we can index tensors without sync because we use
    # precomputed CUDA-side vectors and only touch Python on the loop counter.
    r_rev = r_chunk.flip(0)
    v_rev = values.flip(0)
    nv_rev = next_values.flip(0)
    gl_rev = gamma_len.flip(0)
    gll_rev = gamma_lam_len.flip(0)
    done_rev = done.flip(0)
    for k in range(n):
        delta = (
            r_rev[k]
            - v_rev[k]
            + torch.where(
                done_rev[k],
                torch.zeros((), device=device, dtype=dtype),
                gl_rev[k] * nv_rev[k],
            )
        )
        continuation = torch.where(
            done_rev[k], torch.zeros((), device=device, dtype=dtype), gll_rev[k]
        )
        next_adv = delta + continuation * next_adv
        advantages[n - 1 - k] = next_adv

    returns = advantages + values

    if normalize:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        if clamp is not None:
            advantages = advantages.clamp(-clamp, clamp)

    return advantages, returns
