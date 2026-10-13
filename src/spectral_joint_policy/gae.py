"""Chunked Q-learning: one chunk is one macro-transition.

A chunk is ``h`` open-loop steps, so backup at chunk granularity:

    R_i      = sum_{k < h_i} gamma^k * r_{s_i + k}
    return_i = R_i + gamma^{h_i} * (1 - terminated_i) * Q(s_{s_i + h_i}, a')

Conditioning both sides on the chunk makes this an unbiased ``h``-step target
(Li, Zhou, Levine 2025): the Q that produced the rewards is the Q being backed
up from.

With ``gae_lambda=0.95`` (the default) the advantage follows the chunk-level
trace

    A_i = delta_i + gamma^{h_i} * lambda * (1 - term_i) * (1 - trunc_i) * A_j

over ``j``, the next chunk of the *same env*. This is the GAE of the macro-MDP
this module reasons in, with lambda applied once per chunk: an effective horizon
of ``1 / (1 - gamma^h * lambda)`` chunks -- about 11 chunks / 46 env steps at
gamma=0.99, h=4, lambda=0.95.

``gae_lambda=0`` is the opt-out. It is exactly one backup per chunk, an
effective horizon of one, which is what QC specifies. It is reachable by config
rather than being the default, so the QC identity stays testable rather than
being the only thing a run can do.

The recursion was deleted in 7202519 to match QC and is back here. QC and FPO++
disagree on this (QC specifies a one-chunk backup, FPO++ reports lambda 0.95);
the default tracks the estimator actually being implemented rather than picking
a side by omission.

Two masks, and they are not the same mask:

* the **bootstrap** mask ``1 - terminated`` zeroes ``Q(s', a')``. Only a real
  termination zeroes it; a timeout keeps the bootstrap, and the collector always
  ends a chunk on a terminal, so ``~terminated`` matches QC's ``masks[..., -1]``.
* the **trace** mask ``1 - truncated`` stops the advantage from flowing across
  an episode reset. Read whenever ``gae_lambda > 0``, which is the default, so
  ``chunk_truncated`` is required rather than optional.

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
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[n]`` advantages and regression target; the target is ``chunk_q +
    advantages``. ``chunk_q[i]`` is Q(s_i, a_i), ``chunk_next_q[i]`` is
    Q(s_{i+1}, a' ~ pi(s_{i+1})).

    At ``gae_lambda == 0`` the target reduces exactly to QC's ``R_i +
    gamma^{h_i} (1 - term_i) Q(s_{i+1}, a')``. Above 0 the advantage follows the
    chunk-level trace ``A_i = delta_i + gamma^{h_i} * lambda * (1 - term_i) *
    (1 - trunc_i) * A_j``, where ``j`` is the next chunk of the *same env*. Buffer
    order is time order, so ``j > i`` always and one reverse sweep is exact.

    The default matches ``SJFPOConfig.gae_lambda`` and is deliberately > 0, so
    ``chunk_truncated`` and ``chunk_env_id`` are required here rather than
    optional. Callers wanting the QC backup pass ``gae_lambda=0.0`` explicitly.
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

    if chunk_truncated is None:
        # Absent means "no timeouts in this rollout", which is exactly the QC
        # backup: nothing to inject, and `done` collapses to `terminated`.
        truncated = torch.zeros(n_chunks, device=device, dtype=torch.bool)
    else:
        truncated = chunk_truncated.to(device=device, dtype=torch.bool).reshape(-1)
        if truncated.numel() != n_chunks:
            raise ValueError(
                f"chunk_truncated must have {n_chunks} entries, got {truncated.numel()}"
            )

    if gae_lambda > 0.0:
        # Without the truncation mask the trace would run across an episode
        # reset; without the ids it would run across environments too, since the
        # buffer interleaves envs and buffer adjacency is not trajectory
        # adjacency.
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
    # rsl-rl / FPO++ encoding (fpo.py:process_env_step): the value of the state
    # a timeout cut us off in enters through the *reward*, and the bootstrap is
    # masked by `done` so that correction is counted exactly once. Arithmetically
    # identical to masking the bootstrap by (1 - terminated) alone -- equal in all
    # four combinations of the two masks.
    done = terminated | truncated
    # A chunk that both terminated and truncated keeps the terminal reading:
    # injecting there would resurrect a value for an episode that really ended.
    injected = truncated & ~terminated
    chunk_rewards = chunk_rewards + gamma_h * injected * chunk_next_q
    bootstrap = gamma_h * (~done)
    delta = chunk_rewards + bootstrap * chunk_next_q - chunk_q

    if gae_lambda == 0.0:
        # QC. Returns early so the default path is untouched by the recursion
        # below -- not merely equal to it, but not running it at all.
        return delta, delta + chunk_q

    env_id = chunk_env_id.to(device=device, dtype=torch.long).reshape(-1)
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

    trace = gamma_h * (~done) * gae_lambda
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