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
from torch import nn
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


def compute_cfm_loss(
    v_hat: torch.Tensor,
    v_star: torch.Tensor,
    reduce_mode: str = "scalar",
    clamp: float | None = None,
) -> torch.Tensor:
    """CFM squared-velocity loss with selectable reduction.

    Pure reduction over precomputed velocities: interpolation, target
    construction, and network calls happen before this (see
    SJFPOPolicy.compute_target_velocity / compute_predicted_velocity).

    Args:
        v_hat: Predicted velocities, shape [B, num_mc_samples, *flow_shape].
        v_star: Target velocities, same shape as v_hat.
        reduce_mode: Reduction to apply over the squared errors.
            "scalar" — mean over everything (training objective, unclamped).
            "sample" — mean over flow coords only, shape [B, num_mc_samples],
            symmetrically clamped (per-sample ℓ_i for the ratio math).
        clamp: Symmetric clamp for "sample" only.
            Defaults to config.SJFPO_CFG.cfm_clamp (20.0).

    Returns:
        "scalar": 0-dim mean squared velocity error.
        "sample": per-sample losses ℓ_i, shape [B, num_mc_samples] (NOT the
        training objective).

    Binding:
        - The ℓ_i values feed into compute_policy_ratio for importance weights;
          clamping prevents extreme ratios from outlier velocity errors.
    """
    if v_hat.shape != v_star.shape:
        raise ValueError(
            f"v_hat shape {tuple(v_hat.shape)} != v_star shape {tuple(v_star.shape)}"
        )
    se = (v_hat - v_star).pow(2)  # [B, num_mc_samples, *flow_shape]
    if reduce_mode == "scalar":
        if clamp is not None:
            raise ValueError("clamp applies to reduce_mode='sample' only")
        return se.mean()  # scalar
    elif reduce_mode == "sample":
        per = se.flatten(2).mean(dim=-1)  # [B, num_mc_samples]
    else:
        raise ValueError(f"reduce_mode={reduce_mode!r} must be 'scalar' or 'sample'")

    clamp = config.SJFPO_CFG.cfm_clamp if clamp is None else clamp
    return torch.clamp(per, min=-clamp, max=clamp)  # [B, num_mc_samples]

    # retained: test-only public API — exercised directly by tests/test_fpo_core.py
    # (ratio/loss unit tests); not called by the src train path. Keep per fn-consolidation D1.

    # %%


def compute_aspo_surrogate(
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


def compute_policy_ratio(
    l_old: torch.Tensor,
    l_new: torch.Tensor,
    adv: torch.Tensor,
    cfm_clamp: float | None = None,
    cap: float | None = None,
) -> torch.Tensor:
    """Per-sample importance ratios: negative-advantage trust clamp + STE cap.

    Iff Â < 0: ℓ_new ← min(ℓ_new, ℓ_old + cfm_clamp) before the STE-capped
    exponential, bounding how much worse the new policy may fit a chunk whose
    advantage is negative. Then ρ_i = exp(min(ℓ_old,i − ℓ_new,i, cap)) with a
    straight-through clamp (forward value clamped, gradient flows).

    Args:
        l_old: Old per-sample losses ℓ_old, shape [B, num_mc_samples].
        l_new: New per-sample losses ℓ_new, shape [B, num_mc_samples].
        adv: Advantages Â_i, shape [B, num_mc_samples].
        cfm_clamp: Clamp for negative-advantage ℓ_new adjustment.
            Defaults to config.SJFPO_CFG.cfm_clamp (20.0).
        cap: Log-ratio cap for STE. Defaults to config.SJFPO_CFG.log_ratio_cap.

    Returns:
        Importance ratios ρ_i, shape [B, num_mc_samples]. Always positive.

    Binding:
        - Negative-advantage clamp: if Â < 0, ℓ_new_eff = min(ℓ_new, ℓ_old + cfm_clamp).
        - Δℓ = ℓ_old − ℓ_new_eff; positive Δℓ means new policy fits better.
        - STE clamp: forward = clamp(Δℓ, max=cap), backward grad = 1.
    """
    cfm_clamp = config.SJFPO_CFG.cfm_clamp if cfm_clamp is None else cfm_clamp
    l_new_eff = torch.where(
        adv < 0, torch.minimum(l_new, l_old + cfm_clamp), l_new
    )  # [B, num_mc_samples]
    cap = config.SJFPO_CFG.log_ratio_cap if cap is None else cap
    delta = l_old - l_new_eff  # [B, num_mc_samples]
    # Straight-through: forward value = clamp(delta, max=cap), backward grad = 1.
    delta_st = (
        delta + (torch.clamp(delta, max=cap) - delta).detach()
    )  # [B, num_mc_samples]
    return torch.exp(delta_st)  # [B, num_mc_samples]


def compute_policy_loss(
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
    rho = compute_policy_ratio(
        l_old, l_new, adv, cfm_clamp=cfm_clamp, cap=cap
    )  # [B, num_mc_samples]
    return compute_aspo_surrogate(rho, adv, clip_range).mean()  # scalar

    # %%


def sample(
    net: nn.Module,
    shape,
    condition,
    steps: int,
    eps: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    device=None,
    state=None,
    mode: str = "euler",
) -> torch.Tensor:
    """Sample the learned flow from t=1 (noise) to t=0 (data).

    ``mode="euler"`` integrates the ODE with ``steps`` Euler steps
    (``x ← x − δ·v̂`` on the descending linspace(1, 0, steps+1) grid);
    other samplers plug in here under new mode names.

    t-grid = linspace(1, 0, steps+1) DESCENDING; each step updates
    ``x ← x − δ·v̂`` with δ = 1/steps using the left-endpoint t. ``eps=None``
    draws N(0, I) via ``generator``; passing explicit ``eps`` (e.g. zeros for
    the deterministic zero-sampling eval path) bypasses randomness entirely.
    Runs under inference_mode — sampling never needs gradients.

    Args:
        net: Velocity network callable (x_t, t, condition, state) -> v_hat.
            x_t: [*shape], t: scalar, condition/state: conditioning.
        shape: Output tensor shape (e.g., [B, *flow_shape]).
        condition: Conditioning for the network.
        steps: Number of Euler integration steps.
        eps: Initial noise tensor, shape [*shape]. If None, draws N(0, I).
        generator: Optional torch.Generator for reproducible sampling.
        device: Target device for the initial noise (if eps is None).
        state: Raw obs window passed through to net (SJFPO DiT prefix tokens).
        mode: Sampler name. Only "euler" exists; raise, never silently fall back.

    Returns:
        Sampled tensor x_0, shape [*shape] — the denoised output at t=0.

    Binding:
        - t=1 is pure noise (ε), t=0 is data (a).
        - Euler step: x ← x − δ·v̂ with δ = grid[i] - grid[i+1] = 1/steps.
        - Uses left-endpoint t (grid[i]) for velocity evaluation.
        - No gradients — sampling is inference only.
    """
    if mode != "euler":
        raise ValueError(f"mode={mode!r} unknown; expected 'euler'")
    grid = torch.linspace(1.0, 0.0, steps + 1)  # [steps + 1] descending
    if eps is None:
        eps = torch.randn(*shape, generator=generator)  # [*shape]
        if device is not None:
            eps = eps.to(device)
    x = eps  # [*shape]
    with torch.inference_mode():
        for i in range(steps):
            v_hat = net(x, grid[i], condition, state)  # [*shape]
            x = x - (grid[i] - grid[i + 1]) * v_hat  # [*shape]  δ = 1/steps
    return x  # [*shape]
