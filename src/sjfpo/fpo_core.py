# %% [markdown]
# # FPO++ core math: CFM losses, per-sample ratios, ASPO trust region, Euler sampler
#
# Plan task 5. Implements the BINDING conventions from plan Scope §Exact FPO++
# math (arXiv 2602.02481 Eq.3-13):
#   - interpolation ``x_t = t·ε + (1−t)·a`` with t=1 pure noise,
#   - target velocity ``v* = ε − a``,
#   - τ ~ U[0.005, 0.995] (config tau_range), num_mc_samples = 16 (τ, ε) pairs per chunk,
#   - CFM losses clamped ±cfm_clamp (20) symmetric,
#   - negative-advantage clamp ``ℓ_new ← min(ℓ_new, ℓ_old + cfm_clamp)`` iff Â<0,
#   - per-sample ratios ``ρ_i = exp(min(ℓ_old,i − ℓ_new,i, log_ratio_cap))``
#     with a straight-through clamp (forward value clamped, gradient flows),
#   - ASPO trust region ε_clip=0.05 piecewise: Â≥0 → PPO-clip
#     ``min(ρÂ, clip(ρ, 1±ε)Â)``; Â<0 → SPO ``ρÂ − (|Â|/(2ε))(ρ−1)²``,
#   - objective = MEAN over i of ψ per chunk (paper Eq.13; NEVER exp-of-mean),
#   - Euler integration t:1→0 with ``x ← x − δ·v̂``.
#
# The whole pipeline operates on tensors whose TRAILING dims are the flow
# coordinates (action-chunk DCT coefficients per Scope §Flow space); leading
# dim is the chunk batch B and a middle dim of size num_mc_samples carries the Monte-
# Carlo (τ, ε) draws. fp32 only — no autocast/bf16/fp16 anywhere.
# CELLORDER: 6
#
# Tensor Shape Conventions
# ------------------------
# - B: chunk batch size (number of action chunks)
# - num_mc_samples: Monte Carlo draws per chunk (default 16, config.SJFPO_CFG.num_mc_samples)
# - flow_shape: trailing dimensions of the flow space (e.g., DCT coefficients per chunk)
#   Typically a 1D shape like (D,) where D = chunk_size * action_dim, but kept generic.
# - t (tau): interpolation time in [0.005, 0.995] (config.SJFPO_CFG.tau_range)
#   Can be scalar, [num_mc_samples], or [B, num_mc_samples] — broadcast to [B, num_mc_samples] internally.
#
# Common shapes used throughout:
# - a (anchor): [B, *flow_shape]
# - eps (noise): [B, num_mc_samples, *flow_shape]
# - x_t (interpolated): [B * num_mc_samples, *flow_shape] after flattening MC axis
# - tau_bn: [B * num_mc_samples] — tau broadcast and flattened along MC axis
# - v_hat, v_star: [B * num_mc_samples, *flow_shape] — predicted and target velocities
# - per-sample losses ℓ: [B, num_mc_samples] — mean squared error over flow coords, clamped
# - ratios ρ: [B, num_mc_samples] — importance weights
# - advantages Â: [B, num_mc_samples] — per-sample advantages
# - ASPO surrogate ψ: [B, num_mc_samples] — per-sample trust-region objective
# - policy_loss: scalar — mean of ψ over all (B * num_mc_samples) samples
#
# Binding Notes (from plan Scope §Exact FPO++ math):
# - t=1 is pure noise (ε), t=0 is data (a); interpolation x_t = t·ε + (1−t)·a
# - Target velocity v* = ε − a (independent of t)
# - CFM loss: MSE(v_hat, v*) clamped symmetrically to ±cfm_clamp (default 20)
# - Negative-advantage clamp: if Â < 0, ℓ_new ← min(ℓ_new, ℓ_old + cfm_clamp)
# - Straight-through estimator (STE) on log-ratio cap: forward uses capped Δℓ,
#   backward passes gradient unchanged (detach trick)
# - ASPO: Â ≥ 0 uses PPO clip; Â < 0 uses SPO (soft penalty)
# - Objective is MEAN of ψ over samples (Eq.13), NOT exp(mean(log ρ))
# - Euler sampler: t grid from 1→0, x ← x − δ·v_hat with δ = 1/steps
#

# %%
from __future__ import annotations

import einx
import torch

from sjfpo import config


def pair_tensor(a: torch.Tensor, eps: torch.Tensor, tau):
    """Broadcast (a, eps, tau) onto the shared [B*num_mc_samples] MC axis.

    Args:
        a: Anchor chunks (DCT coefficients), shape [B, *flow_shape].
        eps: Noise draws, shape [B, num_mc_samples, *flow_shape].
        tau: Interpolation times. Can be scalar, [num_mc_samples], or [B, num_mc_samples].
            Values in [0.005, 0.995] per config.SJFPO_CFG.tau_range.

    Returns:
        tuple: (tau_bn, eps_flat, a_rep, flow_shape, num_mc) where
            - tau_bn: [B * num_mc_samples] — tau broadcast and flattened along MC axis.
            - eps_flat: [B * num_mc_samples, D] — eps reshaped, D = prod(flow_shape).
            - a_rep: [B * num_mc_samples, D] — a repeated num_mc_samples times per batch item.
            - flow_shape: tuple[int, ...] — trailing dims of flow space.
            - num_mc: int — number of Monte Carlo draws per chunk.

    Raises:
        ValueError: If eps shape mismatches (B, num_mc_samples, *flow_shape).
        ValueError: If tau shape is not scalar, [num_mc_samples], or [B, num_mc_samples].
        ValueError: If tau vector length != num_mc_samples.
    """
    b, num_mc = eps.shape[0], eps.shape[1]  # [B], [num_mc_samples]
    flow_shape = tuple(a.shape[1:])  # [*flow_shape]
    if tuple(eps.shape) != (b, num_mc, *flow_shape):
        raise ValueError(
            f"eps shape {tuple(eps.shape)} != {(b, num_mc, *flow_shape)} (= B,num_mc_samples,+a.shape[1:])"
        )
    t = torch.as_tensor(
        tau, dtype=torch.float32
    )  # scalar | [num_mc_samples] | [B, num_mc_samples]
    if t.dim() == 0:
        t = einx.id(" -> b m", t, b=b, m=num_mc)  # [B, num_mc_samples]
    elif t.dim() == 1:
        if t.numel() != num_mc:
            raise ValueError(
                f"tau vector must have num_mc_samples={num_mc} entries, got {t.numel()}"
            )
        t = einx.id("m -> b m", t, b=b)  # [B, num_mc_samples]
    elif tuple(t.shape) != (b, num_mc):
        raise ValueError(
            f"tau shape {tuple(t.shape)} must be scalar, [num_mc_samples] or [B, num_mc_samples]"
        )
    d = a.numel() // b  # D = prod(flow_shape)
    a_flat = a.reshape(
        b, d
    )  # [B, D] -- keep torch: dynamic prod(flow_shape) with variable rank has no clean einx axis name
    return (
        einx.id("b m -> (b m)", t),  # tau_bn [B * num_mc_samples]
        eps.reshape(
            b * num_mc, d
        ),  # eps_flat [B * num_mc_samples, D] -- keep torch: dynamic prod(flow_shape) flatten
        einx.id("b d -> (b m) d", a_flat, m=num_mc),  # a_rep [B * num_mc_samples, D]
        flow_shape,
        num_mc,
    )


def repeat_conditioning(cond, num_mc: int):
    """Repeat per-chunk conditioning along the MC axis (None passes through).

    Args:
        cond: Conditioning tensor [B, K, C] or tuple of such tensors, or None.
        num_mc: Number of Monte Carlo draws per chunk.

    Returns:
        Conditioning repeated along batch dim to [B * num_mc_samples, K, C] (or tuple
        thereof), or None if input was None.
    """
    if cond is None:
        return None
    if isinstance(cond, torch.Tensor):
        return einx.id(
            "b k d -> (b m) k d", cond, m=num_mc
        )  # [B * num_mc_samples, K, C]
    return tuple(repeat_conditioning(c, num_mc) for c in cond)


def squared_error(net, a: torch.Tensor, eps: torch.Tensor, tau, cond) -> torch.Tensor:
    """Per-(chunk, mc-draw) squared velocity errors.

    Args:
        net: Velocity network callable (x_t, t, cond) -> v_hat.
            x_t: [B * num_mc_samples, *flow_shape], t: [B * num_mc_samples], cond: repeated.
        a: Anchor chunks, shape [B, *flow_shape].
        eps: Noise draws, shape [B, num_mc_samples, *flow_shape].
        tau: Interpolation times, scalar | [num_mc_samples] | [B, num_mc_samples].
        cond: Conditioning [B, K, C] or tuple, or None.

    Returns:
        Squared errors, shape [B * num_mc_samples, *flow_shape].

    Binding:
        - x_t = t·ε + (1−t)·a with t=1 pure noise, t=0 data.
        - Target velocity v* = ε − a (independent of t).
    """
    tau_bn, eps_flat, a_rep, flow_shape, num_mc = pair_tensor(a, eps, tau)
    # tau_bn [B * num_mc_samples], eps_flat [B * num_mc_samples, D], a_rep [B * num_mc_samples, D]
    x_t = (
        tau_bn[:, None] * eps_flat + (1.0 - tau_bn[:, None]) * a_rep
    )  # [B * num_mc_samples, D]  t·ε+(1−t)·a
    x_t = x_t.reshape(
        -1, *flow_shape
    )  # [B * num_mc_samples, *flow_shape] -- keep torch: dynamic *flow_shape unpack has no einx equivalent
    v_hat = net(
        x_t, tau_bn, repeat_conditioning(cond, num_mc)
    )  # [B * num_mc_samples, *flow_shape]
    v_star = eps_flat - a_rep  # [B * num_mc_samples, D]  binding target velocity ε − a
    return (
        v_hat - v_star.reshape(v_hat.shape)
    ) ** 2  # [B * num_mc_samples, *flow_shape]

    # %%


def cfm_loss(net, a: torch.Tensor, eps: torch.Tensor, tau, cond) -> torch.Tensor:
    """Scalar CFM training loss: mean squared error against v* = ε − a.

    Args:
        net: Velocity network callable (x_t, t, cond) -> v_hat.
            x_t: [B * num_mc_samples, *flow_shape], t: [B * num_mc_samples], cond: repeated.
        a: Anchor chunks (DCT coefficients), shape [B, *flow_shape].
        eps: Noise draws, shape [B, num_mc_samples, *flow_shape].
        tau: Interpolation times, scalar | [num_mc_samples] | [B, num_mc_samples].
            Values in [0.005, 0.995] per config.SJFPO_CFG.tau_range.
        cond: Conditioning [B, K, C] or tuple, or None.

    Returns:
        Scalar tensor — mean squared velocity error over all (B * num_mc_samples * D) elements.

    Binding:
        - Training objective for the CFM velocity network.
        - No clamping applied here (clamping is in flow_loglik for ratio math).
    """
    return squared_error(net, a, eps, tau, cond).mean()  # scalar

    # retained: test-only public API — exercised directly by tests/test_fpo_core.py
    # (ratio/loss unit tests); not called by the src train path. Keep per fn-consolidation D1.


def flow_loglik(
    net,
    a: torch.Tensor,
    eps: torch.Tensor,
    tau,
    cond,
    clamp: float | None = None,
) -> torch.Tensor:
    """Per-sample flow 'log-likelihood' losses ℓ_i, shape [B, num_mc_samples].

    ℓ_i = mean over the flow coords of the squared velocity error, clamped to
    ±``clamp`` (symmetric cfm_clamp, default config.SJFPO_CFG.cfm_clamp). These are
    the quantities consumed by the ratio math (NOT the training objective).

    Args:
        net: Velocity network callable (x_t, t, cond) -> v_hat.
            x_t: [B * num_mc_samples, *flow_shape], t: [B * num_mc_samples], cond: repeated.
        a: Anchor chunks, shape [B, *flow_shape].
        eps: Noise draws, shape [B, num_mc_samples, *flow_shape].
        tau: Interpolation times, scalar | [num_mc_samples] | [B, num_mc_samples].
        cond: Conditioning [B, K, C] or tuple, or None.
        clamp: Symmetric clamp value. Defaults to config.SJFPO_CFG.cfm_clamp (20.0).

    Returns:
        Per-sample losses ℓ_i, shape [B, num_mc_samples]. Each entry is the mean squared
        velocity error over flow coordinates, clamped to [-clamp, clamp].

    Binding:
        - These ℓ_i values feed into flow_ratio for importance weights.
        - Clamping prevents extreme ratios from outlier velocity errors.
    """
    clamp = config.SJFPO_CFG.cfm_clamp if clamp is None else clamp
    _, _, _, flow_shape, num_mc = pair_tensor(a, eps, tau)
    b = a.shape[0]  # B
    per = (
        squared_error(net, a, eps, tau, cond).flatten(1).mean(dim=1)
    )  # [B * num_mc_samples]
    return torch.clamp(
        einx.id("(b m) -> b m", per, b=b, m=num_mc), min=-clamp, max=clamp
    )  # [B, num_mc_samples]

    # %%


def flow_ratio(
    l_old: torch.Tensor,
    l_new: torch.Tensor,
    cap: float | None = None,
) -> torch.Tensor:
    """Per-sample importance ratios ρ_i = exp(min(ℓ_old,i − ℓ_new,i, cap)).

    Straight-through clamp: the FORWARD value uses the capped Δℓ while the
    backward pass sees the identity, so gradients keep flowing to ℓ_new even
    when the cap engages.

    Args:
        l_old: Old per-sample losses ℓ_old, shape [B, num_mc_samples].
        l_new: New per-sample losses ℓ_new, shape [B, num_mc_samples].
        cap: Log-ratio cap. Defaults to config.SJFPO_CFG.log_ratio_cap.

    Returns:
        Importance ratios ρ_i, shape [B, num_mc_samples]. Always positive.

    Binding:
        - Δℓ = ℓ_old − ℓ_new; positive Δℓ means new policy fits better.
        - STE clamp: forward = clamp(Δℓ, max=cap), backward grad = 1.
        - This prevents ratio explosion while preserving gradient flow.
    """
    cap = config.SJFPO_CFG.log_ratio_cap if cap is None else cap
    delta = l_old - l_new  # [B, num_mc_samples]
    # Straight-through: forward value = clamp(delta, max=cap), backward grad = 1.
    delta_st = (
        delta + (torch.clamp(delta, max=cap) - delta).detach()
    )  # [B, num_mc_samples]
    return torch.exp(delta_st)  # [B, num_mc_samples]


def aspo_surrogate(
    rho: torch.Tensor, adv: torch.Tensor, clip_range: float
) -> torch.Tensor:
    """ASPO piecewise trust region ψ(ρ, Â) (paper Eq.12).

    Â ≥ 0: PPO-clip  ψ = min(ρÂ, clip(ρ, 1−ε, 1+ε)·Â);
    Â < 0: SPO       ψ = ρÂ − (|Â| / (2ε)) · (ρ−1)².
    Elementwise; scalars and tensors both supported.

    Args:
        rho: Importance ratios ρ_i, shape [B, num_mc_samples] or broadcastable.
        adv: Advantages Â_i, shape [B, num_mc_samples] or broadcastable.
        clip_range: PPO/SPO clip epsilon (default config.SJFPO_CFG.clip_range = 0.05).

    Returns:
        ASPO surrogate values ψ_i, shape [B, num_mc_samples] (or broadcast result).

    Binding:
        - For Â ≥ 0: standard PPO clipping prevents over-optimistic updates.
        - For Â < 0: SPO (Simple Policy Optimization) applies quadratic penalty
          around ρ=1, preventing catastrophic degradation on negative-adv samples.
        - ε_clip = 0.05 per plan binding convention.
    """
    pos = torch.min(
        rho * adv,  # [B, num_mc_samples]
        torch.clamp(rho, 1.0 - clip_range, 1.0 + clip_range)
        * adv,  # [B, num_mc_samples]
    )  # [B, num_mc_samples]
    neg = (
        rho * adv
        - (adv.abs() / (2.0 * clip_range)) * (rho - 1.0) ** 2  # [B, num_mc_samples]
    )  # [B, num_mc_samples]
    return torch.where(adv >= 0, pos, neg)  # [B, num_mc_samples]


def policy_ratio(
    l_old: torch.Tensor,
    l_new: torch.Tensor,
    adv: torch.Tensor,
    cfm_clamp: float | None = None,
    cap: float | None = None,
) -> torch.Tensor:
    """Ratios after the negative-advantage trust clamp.

    Iff Â < 0: ℓ_new ← min(ℓ_new, ℓ_old + cfm_clamp) before the STE-capped
    exponential, bounding how much worse the new policy may fit a chunk whose
    advantage is negative.

    Args:
        l_old: Old per-sample losses ℓ_old, shape [B, num_mc_samples].
        l_new: New per-sample losses ℓ_new, shape [B, num_mc_samples].
        adv: Advantages Â_i, shape [B, num_mc_samples].
        cfm_clamp: Clamp for negative-advantage ℓ_new adjustment.
            Defaults to config.SJFPO_CFG.cfm_clamp (20.0).
        cap: Log-ratio cap for STE. Defaults to config.SJFPO_CFG.log_ratio_cap.

    Returns:
        Adjusted importance ratios ρ_i, shape [B, num_mc_samples].

    Binding:
        - Negative-advantage clamp: if Â < 0, ℓ_new_eff = min(ℓ_new, ℓ_old + cfm_clamp).
        - This limits how much the new policy can degrade fit on bad-advantage chunks.
        - Then applies flow_ratio with STE-capped exponential.
    """
    cfm_clamp = config.SJFPO_CFG.cfm_clamp if cfm_clamp is None else cfm_clamp
    l_new_eff = torch.where(
        adv < 0, torch.minimum(l_new, l_old + cfm_clamp), l_new
    )  # [B, num_mc_samples]
    return flow_ratio(l_old, l_new_eff, cap)  # [B, num_mc_samples]

    # retained: test-only public API — exercised directly by tests/test_fpo_core.py
    # (finite-difference gradient test); not called by the src train path. Keep per fn-consolidation D1.


def policy_loss(
    l_old: torch.Tensor,
    l_new: torch.Tensor,
    adv: torch.Tensor,
    clip_range: float | None = None,
    cfm_clamp: float | None = None,
    cap: float | None = None,
) -> torch.Tensor:
    """Scalar FPO++ policy objective: MEAN over i of ψ per chunk (Eq.13).

    Per-sample ratios enter the mean directly — never an exp-of-mean estimator.

    Args:
        l_old: Old per-sample losses ℓ_old, shape [B, num_mc_samples].
        l_new: New per-sample losses ℓ_new, shape [B, num_mc_samples].
        adv: Advantages Â_i, shape [B, num_mc_samples].
        clip_range: ASPO clip epsilon. Defaults to config.SJFPO_CFG.clip_range (0.05).
        cfm_clamp: Negative-advantage clamp. Defaults to config.SJFPO_CFG.cfm_clamp (20.0).
        cap: Log-ratio cap for STE. Defaults to config.SJFPO_CFG.log_ratio_cap.

    Returns:
        Scalar tensor — mean of ASPO surrogate ψ over all (B * num_mc_samples) samples.

    Binding:
        - Objective = mean(ψ_i) per paper Eq.13, NOT exp(mean(log ρ)).
        - This is the policy gradient objective maximized during PPO-style updates.
    """
    clip_range = config.SJFPO_CFG.clip_range if clip_range is None else clip_range
    rho = policy_ratio(
        l_old, l_new, adv, cfm_clamp=cfm_clamp, cap=cap
    )  # [B, num_mc_samples]
    return aspo_surrogate(rho, adv, clip_range).mean()  # scalar

    # %%


def euler_sample(
    net,
    shape,
    cond,
    steps: int,
    eps: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    device=None,
    state=None,
) -> torch.Tensor:
    """Integrate the learned ODE from t=1 (noise) to t=0 (data).

    t-grid = linspace(1, 0, steps+1) DESCENDING; each step updates
    ``x ← x − δ·v̂`` with δ = 1/steps using the left-endpoint t. ``eps=None``
    draws N(0, I) via ``generator``; passing explicit ``eps`` (e.g. zeros for
    the deterministic zero-sampling eval path) bypasses randomness entirely.
    Runs under no_grad — sampling never needs gradients.

    Args:
        net: Velocity network callable (x_t, t, cond, state) -> v_hat.
            x_t: [*shape], t: scalar, cond/state: conditioning.
        shape: Output tensor shape (e.g., [B, *flow_shape]).
        cond: Conditioning for the network.
        steps: Number of Euler integration steps.
        eps: Initial noise tensor, shape [*shape]. If None, draws N(0, I).
        generator: Optional torch.Generator for reproducible sampling.
        device: Target device for the initial noise (if eps is None).
        state: Raw obs window passed through to net (SJFPO DiT prefix tokens).

    Returns:
        Sampled tensor x_0, shape [*shape] — the denoised output at t=0.

    Binding:
        - t=1 is pure noise (ε), t=0 is data (a).
        - Euler step: x ← x − δ·v̂ with δ = grid[i] - grid[i+1] = 1/steps.
        - Uses left-endpoint t (grid[i]) for velocity evaluation.
        - No gradients — sampling is inference only.
    """
    grid = torch.linspace(1.0, 0.0, steps + 1)  # [steps + 1] descending
    if eps is None:
        eps = torch.randn(*shape, generator=generator)  # [*shape]
        if device is not None:
            eps = eps.to(device)
    x = eps  # [*shape]
    with torch.no_grad():
        for i in range(steps):
            v_hat = net(x, grid[i], cond, state)  # [*shape]
            x = x - (grid[i] - grid[i + 1]) * v_hat  # [*shape]  δ = 1/steps
    return x  # [*shape]

    # %%


def total_update_loss(
    policy_loss,
    value_loss,
    sigreg_loss,
    *,
    vf_coef: float,
    sigreg_coef: float,
):
    """Sum the three update terms under caller-supplied weights (single weight home)."""
    return policy_loss + vf_coef * value_loss + sigreg_coef * sigreg_loss
