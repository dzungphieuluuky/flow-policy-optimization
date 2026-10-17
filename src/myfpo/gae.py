"""Chunked Q-learning. One chunk is one macro-transition.

A chunk is ``h`` open-loop steps. The backup therefore runs at chunk
granularity:

    R_i      = sum_{k < h_i} gamma^k * r_{s_i + k}
    return_i = R_i + gamma^{h_i} * (1 - terminated_i) * Q(s_{s_i + h_i}, a')

Conditioning both sides on the chunk makes this an unbiased ``h``-step target
(Li, Zhou, Levine 2025). The Q that produced the rewards is the Q that the
backup reads.

With ``gae_lambda=0.95``, which is the default, the advantage follows the
chunk-level trace:

    A_i = delta_i + gamma^{h_i} * lambda * (1 - term_i) * (1 - trunc_i) * A_j

The index ``j`` is the next chunk of the *same env*. This is the GAE of the
macro-MDP that this module describes. Lambda applies once per chunk. The
effective horizon is ``1 / (1 - gamma^h * lambda)`` chunks. At gamma=0.99,
h=4 and lambda=0.95, that is about 11 chunks, or 46 env steps.

The value ``gae_lambda=0`` is the opt-out. It gives exactly one backup per
chunk and an effective horizon of one. This is what QC specifies. A config value
reaches it, so the QC identity stays testable. The default is not this case, so
a run is not limited to it.

Commit 7202519 deleted the recursion to match QC. The recursion is back. QC and
FPO++ disagree here. QC specifies a one-chunk backup. FPO++ reports lambda 0.95.
The default follows the estimator that this code implements. It does not pick a
side by omission.

The module uses two masks. They are not the same mask.

* The **bootstrap** mask is ``1 - terminated``. It zeroes ``Q(s', a')``. Only a
  real termination zeroes it. A timeout keeps the bootstrap. The collector
  always ends a chunk on a terminal. So ``~terminated`` matches the
  ``masks[..., -1]`` of QC.
* The **trace** mask is ``1 - truncated``. It stops the advantage from crossing
  an episode reset. The function reads it whenever ``gae_lambda > 0``. That is
  the default. So ``chunk_truncated`` is required and not optional.

The term ``gamma^h`` uses the executed length everywhere. The trace discount is
``gamma^h * lambda``, so lambda applies once per chunk. This matches the
macro-MDP that this module describes. Note that verl-vla and RLinf use
``(gamma * lambda)^h`` instead. Their form decays lambda per primitive env
step. The two forms differ for ``h > 1``.
"""
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
    """Return the ``[n]`` advantages and the regression target. The target is
    ``q_current_chunk + advantages``.

    ``q_current_chunk[i]`` is Q(s_i, a_i). The value ``q_next_state[i]`` is
    Q(s_{i+1}, a' ~ pi(s_{i+1})), the bootstrap at the chunk end.

    The two Q arguments were named ``chunk_q`` and ``chunk_next_q``. The prefixes
    ordered by position in the chunk, not by role, and the caller also held a
    ``boot_q``. The names now say which chunk each value belongs to.

    At ``gae_lambda == 0`` the target reduces exactly to the QC target
    ``R_i + gamma^{h_i} (1 - term_i) Q(s_{i+1}, a')``.

    Above 0 the advantage follows the chunk-level trace:

        A_i = delta_i + gamma^{h_i} * lambda * (1 - term_i) * (1 - trunc_i) * A_j

    The index ``j`` is the next chunk of the *same env*. Buffer order is time
    order, so ``j > i`` always holds. One reverse sweep is therefore exact.

    The default matches ``SJFPOConfig.gae_lambda`` and is above 0. So
    ``chunk_truncated`` and ``chunk_env_id`` are required here. A caller that
    wants the QC backup passes ``gae_lambda=0.0``.
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
        # An absent mask means "no timeouts in this rollout". That is exactly the
        # QC backup. There is nothing to inject, and `done` collapses to
        # `terminated`.
        truncated = torch.zeros(n_chunks, device=device, dtype=torch.bool)
    else:
        truncated = chunk_truncated.to(device=device, dtype=torch.bool).reshape(-1)
        if truncated.numel() != n_chunks:
            raise ValueError(
                f"chunk_truncated must have {n_chunks} entries, got {truncated.numel()}"
            )

    if gae_lambda > 0.0:
        # Without the truncation mask the trace runs across an episode reset.
        # Without the ids the trace runs across environments too. The buffer
        # interleaves the envs, so adjacency in the buffer is not adjacency in
        # a trajectory.
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
    # This is the encoding of rsl-rl and FPO++ (fpo.py:process_env_step).
    # The value of the state where a timeout cut the episode enters through the
    # *reward*. The mask `done` covers the bootstrap. So the correction counts
    # exactly once.
    # The arithmetic matches a bootstrap masked by (1 - terminated) alone. The
    # two agree in all four combinations of the two masks.
    done = terminated | truncated
    # A chunk that both terminated and truncated keeps the terminal reading:
    # injecting there would resurrect a value for an episode that really ended.
    injected = truncated & ~terminated
    chunk_rewards = chunk_rewards + gamma_h * injected * q_next_state
    bootstrap = gamma_h * (~done)
    delta = chunk_rewards + bootstrap * q_next_state - q_current_chunk

    if gae_lambda == 0.0:
        # QC. The function returns early, so the default path never runs the
        # recursion below. It is not merely equal to that path. It does not run
        # it at all.
        return delta, delta + q_current_chunk

    env_id = chunk_env_id.to(device=device, dtype=torch.long).reshape(-1)
    if env_id.numel() != n_chunks:
        raise ValueError(f"chunk_env_id must have {n_chunks} entries, got {env_id.numel()}")

    # The sort is stable. So a tie keeps the buffer order. The buffer order is
    # already the time order. The next same-env chunk in this order is therefore
    # always the next chunk in time. Its buffer index is therefore strictly
    # larger. That fact is what makes the single reverse sweep below correct
    # rather than merely plausible.
    order = torch.argsort(env_id, stable=True)
    sorted_env = env_id[order]
    same_env = sorted_env[:-1] == sorted_env[1:]
    successor = torch.full_like(order, -1)
    successor[order[:-1][same_env]] = order[1:][same_env]

    trace = gamma_h * (~done) * gae_lambda
    # One device-to-host transfer covers the whole successor map. The loop then
    # makes no further sync. The cost is n_chunks cheap kernels instead of
    # n_chunks synchronisations.
    successor_host = successor.tolist()
    adv = delta.clone()
    for i in range(n_chunks - 1, -1, -1):
        j = successor_host[i]
        if j >= 0:
            adv[i] = delta[i] + trace[i] * adv[j]
    return adv, adv + q_current_chunk