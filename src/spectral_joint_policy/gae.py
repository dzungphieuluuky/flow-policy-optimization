"""Chunked Q-learning: one chunk is one macro-transition.

A chunk is ``h`` open-loop steps, so backup at chunk granularity:

    R_i      = sum_{k < h_i} gamma^k * r_{s_i + k}
    return_i = R_i + gamma^{h_i} * (1 - terminated_i) * Q(s_{s_i + h_i}, a')

Conditioning both sides on the chunk makes this an unbiased ``h``-step target
(Li, Zhou, Levine 2025): the Q that produced the rewards is the Q being backed
up from.

``gae_lambda=0`` (the default) is exactly that, and is what the paper does: QC
backs up exactly one chunk, an effective horizon of one. ``gae_lambda > 0``
re-adds the chunk-level GAE trace, which trades that fidelity for a longer
effective horizon of ``1 / (1 - gamma^h * lambda)`` chunks -- about 5 at
gamma=0.99, h=16, lambda=0.95. It was removed in 7202519 to match QC and is
back here as opt-in only, so the QC identity below stays the default and the
sweeps that assume it stay valid.

Which convention is right depends on the paper being reproduced, and QC and
FPO++ differ here: FPO++ reports a GAE lambda of 0.95 (0.99 on manipulation),
while QC specifies a one-chunk backup. Deciding between them is a paper-level
call, not a config default -- hence 0.0.

Two masks, and they are not the same mask:

* the **bootstrap** mask ``1 - terminated`` zeroes ``Q(s', a')``. Only a real
  termination zeroes it; a timeout keeps the bootstrap, and the collector always
  ends a chunk on a terminal, so ``~terminated`` matches QC's ``masks[..., -1]``.
* the **trace** mask ``1 - truncated`` stops the advantage from flowing across
  an episode reset. It is only read when ``gae_lambda > 0``, since with a
  one-chunk backup no credit crosses a chunk boundary and the flag is inert.

``gamma^h`` uses the executed length throughout, and the trace discount is
``gamma^h * lambda`` -- lambda once per chunk, matching the macro-MDP this
module reasons in. (verl-vla and RLinf instead use ``(gamma * lambda)^h``, which
decays lambda per primitive env step; the two differ for ``h > 1``.)
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
    chunk_truncated: torch.Tensor | None = None,
    chunk_env_id: torch.Tensor | None = None,
    gae_lambda: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[n]`` advantages and regression target; the target is ``chunk_q +
    advantages``, so it reduces exactly to QC's ``R_i + gamma^{h_i}
    (1 - term_i) Q(s_{i+1}, a')``. ``chunk_q[i]`` is Q(s_i, a_i),
    ``chunk_next_q[i]`` is Q(s_{i+1}, a' ~ pi(s_{i+1})).

    With ``gae_lambda > 0`` the advantage instead follows the chunk-level trace
    ``A_i = delta_i + gamma^{h_i} * lambda * (1 - term_i) * (1 - trunc_i) *
    A_j``, where ``j`` is the next chunk of the *same env*. Buffer order is time
    order, so ``j > i`` always and one reverse sweep is exact.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")

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

    if gae_lambda > 0.0:
        # Without these the trace would run across an episode reset, and across
        # environments -- the buffer interleaves envs, so buffer adjacency is not
        # trajectory adjacency.
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

    # One shot gather, then discount by gamma^k. A cumsum diff would need
    # gamma**-s_i, which loses all fp32 precision once s_i hits a few thousand.
    offsets = torch.arange(chunk_size, device=device)
    window = starts.unsqueeze(1) + offsets
    gathered = rewards[window.clamp(max=rewards.numel() - 1)]
    valid = offsets.unsqueeze(0) < lengths.unsqueeze(1)
    chunk_rewards = (gathered * valid * (gamma**offsets)).sum(dim=1)

    gamma_h = gamma ** lengths.to(dtype)
    bootstrap = gamma_h * (~terminated)
    delta = chunk_rewards + bootstrap * chunk_next_q - chunk_q

    if gae_lambda == 0.0:
        # QC. Returns early so the default path is untouched by the recursion
        # below -- not merely equal to it, but not running it at all.
        return delta, delta + chunk_q

    truncated = chunk_truncated.to(device=device, dtype=torch.bool).reshape(-1)
    env_id = chunk_env_id.to(device=device, dtype=torch.long).reshape(-1)
    if truncated.numel() != n_chunks:
        raise ValueError(
            f"chunk_truncated must have {n_chunks} entries, got {truncated.numel()}"
        )
    if env_id.numel() != n_chunks:
        raise ValueError(f"chunk_env_id must have {n_chunks} entries, got {env_id.numel()}")

    # Stable, so ties keep buffer order, which is already time order. The next
    # same-env chunk in this order is therefore always the next one in time, and
    # so always at a strictly larger buffer index -- which is what makes the
    # single reverse sweep below correct rather than merely plausible.
    order = torch.argsort(env_id, stable=True)
    sorted_env = env_id[order]
    same_env = sorted_env[:-1] == sorted_env[1:]
    succ = torch.full_like(order, -1)
    succ[order[:-1][same_env]] = order[1:][same_env]

    trace = bootstrap * (~truncated) * gae_lambda
    # One device->host transfer for the whole successor map; the loop then does
    # no further syncs, so this is n_chunks cheap kernels rather than n_chunks
    # synchronisations.
    succ_host = succ.tolist()
    adv = delta.clone()
    for i in range(n_chunks - 1, -1, -1):
        j = succ_host[i]
        if j >= 0:
            adv[i] = delta[i] + trace[i] * adv[j]
    return adv, adv + chunk_q