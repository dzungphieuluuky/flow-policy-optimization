"""FPO++ core math: CFM losses, per-sample ratios, ASPO trust region, Euler sampler (arXiv 2602.02481 Eq.3-13)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from myfpo.configuration_sjfpo import SJFPOConfig
from myfpo.utils import resolve_storage_dtype


def sample_timesteps(
    num_samples: int,
    cfg: SJFPOConfig,
    device: torch.device | str,
    generator: torch.Generator | None,
):
    """Flow timesteps in ``[min_timestep, max_timestep]``. ``[num_samples]``."""
    lo, hi = cfg.min_timestep, cfg.max_timestep
    mode = cfg.timestep_sampling_mode
    if mode == "uniform":
        return (
            torch.rand(num_samples, device=device, generator=generator) * (hi - lo) + lo
        )
    if mode == "normal":
        mid = 0.5 * (lo + hi)
        std = 0.25 * (hi - lo)
        z = torch.randn(num_samples, device=device, generator=generator)
        return (z * std + mid).clamp(lo, hi)
    if mode == "fpo_beta":
        u = torch.rand(num_samples, device=device, generator=generator)
        beta = cfg.timestep_inverse_cdf_beta
        return lo + (hi - lo) * (
            1.0 - torch.pow(1.0 - u, 1.0 / beta)
        )
    if mode == "beta":
        from lerobot.policies.common.flow_matching import sample_time_beta

        return sample_time_beta(
            num_samples,
            device,
            alpha=cfg.timestep_beta_alpha,
            beta=cfg.timestep_beta_beta,
            scale=hi - lo,
            offset=lo,
        )
    raise ValueError(
        f"mode={mode!r} unknown; expected 'uniform', 'normal', 'beta', or 'fpo_beta'"
    )


def sample_noise(
    shape,
    cfg: SJFPOConfig,
    device: torch.device | str,
    generator: torch.Generator | None,
):
    """MC noise. Marginal from ``noise_sampling_mode``.

    Only definition of the base sample. The CFM path reads it twice --
    ``compute_mc_xt`` builds ``lerp(noise, action, t)`` and
    ``compute_target_velocity`` returns ``a - eps``.
    """
    mode = cfg.noise_sampling_mode
    if mode == "normal":
        return torch.randn(*shape, device=device, generator=generator)
    if mode == "uniform":
        return (torch.rand(*shape, device=device, generator=generator) * 2.0 - 1.0) * math.sqrt(3.0)
    raise ValueError(f"mode={mode!r} unknown; expected 'normal' or 'uniform'")


def run_monte_carlo_sampling(
    a: torch.Tensor, eps: torch.Tensor, tau
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], int]:
    """Broadcast action, noise, per-token timestep over MC -> ``[B*M, ...]``.

    ``tau`` is ``[B, M, K, 1]``: one time per action step, which is what keeps
    a token's velocity and its target on the same ``tau``. The scalar and
    per-chunk shapes this used to accept were never reachable from any caller
    and are gone. Cast onto ``eps``'s device/dtype so a scalar cannot hop to CPU
    mid-graph.
    """
    b, num_mc = eps.shape[0], eps.shape[1]
    flow_shape = tuple(a.shape[1:])
    if tuple(eps.shape) != (b, num_mc, *flow_shape):
        raise ValueError(f"eps shape {tuple(eps.shape)} != {(b, num_mc, *flow_shape)}")

    storage_dtype = resolve_storage_dtype(eps.device)
    if isinstance(tau, torch.Tensor):
        t = tau.to(device=eps.device, dtype=storage_dtype)
    else:
        t = torch.as_tensor(tau, dtype=storage_dtype, device=eps.device)

    if t.dim() != 4 or tuple(t.shape) != (b, num_mc, flow_shape[0], 1):
        raise ValueError(
            f"tau shape {tuple(t.shape)} must be [B, num_mc_samples, K, 1] "
            f"with B={b}, num_mc_samples={num_mc}, K={flow_shape[0]}"
        )

    d = a.numel() // b
    a_flat = a.reshape(b, 1, d).expand(b, num_mc, d).reshape(b * num_mc, d)
    flatten_mc_timestep = t.reshape(b * num_mc, flow_shape[0], 1)
    return (
        flatten_mc_timestep,
        eps.reshape(b * num_mc, d),
        a_flat,
        flow_shape,
        num_mc,
    )


def clip_transported(x: torch.Tensor, clip: float | None) -> torch.Tensor:
    """Bound the actor's transported output. FPO++ ``transported_clip_value``.

    Applied at every site the raw velocity is produced -- the training loss,
    the frozen ``cfm_loss_old`` pass, and the Euler sampler -- so all three see the
    same map and the ratio stays consistent. ``None`` is a no-op.
    """
    if clip is None:
        return x
    return x.clamp(-clip, clip)


def compute_cfm_loss(
    v_hat: torch.Tensor,
    v_star: torch.Tensor,
    cfg: SJFPOConfig,
    reduce_mode: str,
    action_is_pad: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """CFM squared-velocity loss, FPO++ Eq. 7-8.

    ``weight`` multiplies the squared error before any reduction, which is where
    FPO++ applies its t-dependent schedule (``flow_model.py:631-634``). Applied
    after the padding mask, so padded rows stay zero.

    ``sample``: flatten tokens, reduce per ``cfg.cfm_loss_reduction``.
    ``token_mean``: divide by action_dim only, so masking cannot move it.
    ``both``: pair off one masked square. ``scalar``: one number.

    ``action_is_pad`` ``[n, horizon]`` True where padded: zero those rows and,
    for every reduction but ``sum``, renormalise by the executed count, so a
    short chunk keeps a full chunk's magnitude. ``sum`` deliberately does not
    renormalise -- it is Eq. 8's squared norm over the whole chunk -- so a short
    chunk's loss, and the trust region it carries, scale with its length.
    """
    if v_hat.shape != v_star.shape:
        raise ValueError(
            f"v_hat shape {tuple(v_hat.shape)} != v_star shape {tuple(v_star.shape)}"
        )
    # F.mse_loss with reduction="none" is exactly square(v_hat - v_star); the
    # default reduction would divide, which the masking below already does.
    diff = v_hat - v_star
    if cfg.cfm_loss_use_huber:
        # FPO++ flow_model.py:_compute_squared_error. The quadratic branch
        # carries no 0.5 factor, so this equals MSE for |diff| <= delta and
        # only the tail turns linear.
        delta = cfg.cfm_loss_huber_delta
        abs_diff = diff.abs()
        sq = torch.where(
            abs_diff <= delta,
            diff.square(),
            2.0 * delta * abs_diff - delta * delta,
        )
    else:
        sq = diff.square()
    n_valid = None
    if action_is_pad is not None:
        # action_is_pad marks what to drop, so flip it: keep is 1 on executed
        # positions, 0 on padded ones.
        keep = (~action_is_pad).to(sq.dtype)[:, None, :, None]
        # Scaled by action_dim so the divisor counts scalar elements, not tokens.
        n_valid = (
            (~action_is_pad).sum(dim=-1, keepdim=True).to(sq.dtype) * sq.shape[-1]
        )
        n_valid = n_valid.clamp(min=1.0)
        sq = sq * keep

    if weight is not None:
        sq = sq * weight

    if reduce_mode == "scalar":
        return sq.mean() if n_valid is None else sq.sum() / n_valid.sum()

    if reduce_mode in ("sample", "both"):
        flat = sq.flatten(2)
        if cfg.cfm_loss_reduction == "mean":
            sample = flat.mean(dim=-1) if n_valid is None else flat.sum(dim=-1) / n_valid
        elif cfg.cfm_loss_reduction == "sum":
            sample = flat.sum(dim=-1)
        else:
            total = flat.sum(dim=-1)
            sample = (
                total / math.sqrt(flat.shape[-1])
                if n_valid is None
                else total / n_valid.sqrt()
            )
        if reduce_mode == "sample":
            return sample
        return sample, sq.mean(dim=-1)
    if reduce_mode == "token_mean":
        # Per-token (or per-frequency-coefficient when DCT is on) diagnostic.
        # A token aggregates only action_dim, so dividing by that keeps the
        # value independent of token count. mask_sq already zeroed padded rows.
        return sq.mean(dim=-1)
    raise ValueError(
        f"reduce_mode={reduce_mode!r} must be 'scalar', 'sample', 'token_mean', "
        "or 'both'"
    )


def group_cfm_loss(loss: torch.Tensor, group_size: int) -> torch.Tensor:
    """Average a CFM loss over groups of MC samples, ``[n, num_mc, ...]``.

    FPO++ groups the Monte-Carlo axis before exponentiating
    (``finetune_online_rl.py:803-806``): ``-1`` puts every sample in one group,
    which is FPO Eq. 9's "mean the MC losses, then exponentiate". Grouping is
    on the loss, not on ``old - new``; the two agree because
    ``exp(mean(old) - mean(new)) == exp(mean(old - new))``.

    ``1`` returns the input unchanged, so per-sample ratios cost nothing.
    """
    if group_size == 1:
        return loss
    n, num_mc = loss.shape[0], loss.shape[1]
    size = num_mc if group_size == -1 else group_size
    if num_mc % size != 0:
        raise ValueError(
            f"num_mc_samples={num_mc} must divide evenly by group size {size}"
        )
    return loss.reshape(n, num_mc // size, size, *loss.shape[2:]).mean(dim=2)


def cfm_loss_weight(t: torch.Tensor, cfg: SJFPOConfig) -> torch.Tensor:
    """Per-token CFM loss weight, broadcast over the action dimension.

    FPO++'s two ramps are ``1 - 0.9t`` and ``1 - 0.99t`` on *its* time
    (``flow_model.py:648-661``), where ``t=0`` is the action and ``t=1`` the
    noise. This repo runs time the other way, so those same two schedules are
    ``0.1 + 0.9t`` and ``0.01 + 0.99t`` here: full weight at the action end,
    least at noise. Copying the formula verbatim would weight noise hardest,
    which is the opposite of the intent.

    ``t`` is ``[..., 1]``, so the weight broadcasts over action dims untouched.
    """
    mode = cfg.cfm_loss_weight_from_t
    if mode == "constant":
        return torch.ones_like(t)
    if mode == "linear_1_to_0.1":
        return 0.1 + 0.9 * t
    return 0.01 + 0.99 * t


def cfm_prediction(
    v: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, cfg: SJFPOConfig
) -> torch.Tensor:
    """Map a velocity prediction to whatever ``cfg.cfm_loss_target`` compares.

    The network always predicts the velocity, so only the loss moves. On the
    straight path ``x_t = (1 - t) * eps + t * a`` and ``v = a - eps``, so at the
    true velocity every mode below returns its target exactly:
    ``x_t + (1 - t) * v == a`` and ``x_t - t * v == eps``. That identity is the
    test.

    FPO++ reaches the same three endpoints by also varying what the network
    predicts (``flow_network_output_param``); holding that at ``u`` here is the
    smaller change and leaves ``transported_clip_value`` clipping one thing.
    """
    mode = cfg.cfm_loss_target
    if mode == "velocity":
        return v
    if mode == "action":
        return x_t + (1.0 - t) * v
    return x_t - t * v


def cfm_target(
    a: torch.Tensor, eps: torch.Tensor, cfg: SJFPOConfig
) -> torch.Tensor:
    """The regression target for ``cfg.cfm_loss_target``. ``[B, *flow]``."""
    mode = cfg.cfm_loss_target
    if mode == "velocity":
        return a.unsqueeze(1) - eps
    if mode == "action":
        return a.unsqueeze(1).expand_as(eps)
    return eps


def compute_aspo_surrogate(
    ratio: torch.Tensor,
    advantage: torch.Tensor,
    cfg: SJFPOConfig,
) -> torch.Tensor:
    """FPO++ trust region. ASPO = PPO on A>0, SPO on A<0. Elementwise.

    PPO clips at ``clip_param``, the SPO quadratic at ``spo_clip_param``. Two
    widths, not one: FPO++'s manipulation loop carries them as separate
    coefficients (``finetune_online_rl.py:117-118``).
    """
    ratio_adv = ratio * advantage
    ratio_clipped = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param)
    ppo = torch.min(ratio_adv, ratio_clipped * advantage)
    spo = ratio_adv - (advantage.abs() / (2.0 * cfg.spo_clip_param)) * torch.square(
        ratio - 1.0
    )

    if cfg.trust_region_mode == "ppo":
        return ppo
    if cfg.trust_region_mode == "spo":
        return spo
    return torch.where(advantage > 0, ppo, spo)


def compute_policy_ratio(
    cfm_loss_old: torch.Tensor,
    cfm_loss_new: torch.Tensor,
    cfg: SJFPOConfig,
    advantage: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ratio exp(cfm_loss_old - cfm_loss_new), standing in for a log-prob ratio.

    Elementwise, so the caller's reduction fixes the granularity: ``per_chunk``
    passes the MC means (one ratio per chunk), ``per_action`` the per-pair
    losses, ``per_sample`` the per-token losses.

    Clamp both losses, optionally clamp cfm_loss_new harder on A<0, then cap the
    log-ratio one-sided with an STE: capped forward, original backward.
    """
    old = torch.clamp(cfm_loss_old, max=cfg.cfm_clamp)
    new = torch.clamp(cfm_loss_new, max=cfg.cfm_clamp)
    if cfg.cfm_loss_clamp_negative_advantages and advantage is not None:
        new = torch.where(
            advantage < 0,
            torch.clamp(new, max=cfg.cfm_loss_clamp_negative_advantages_max),
            new,
        )
    delta = old - new
    safe_delta = delta.clamp(max=cfg.log_ratio_cap)
    delta_st = safe_delta.detach() + delta - delta.detach()
    return torch.exp(delta_st)


def trajectory_straightness(velocities) -> float:
    """Mean cosine between consecutive velocities of an Euler trajectory.

    Lower means a curvier path. Return is not a substitute: it can hold while
    the sampling cost moves, or the reverse.

    ``velocities`` is the sequence of ``v_hat`` the sampler used, in order.
    Returns NaN for fewer than two steps, which is what a 1-step sampler gets.
    """
    if len(velocities) < 2:
        return float("nan")
    cosines = []
    for a, b in zip(velocities[:-1], velocities[1:]):
        num = (a * b).sum()
        den = a.norm() * b.norm()
        if float(den) > 0:
            cosines.append(float(num / den))
    if not cosines:
        return float("nan")
    return sum(cosines) / len(cosines)


def sample_action(
    predict_velocity: Callable[[torch.Tensor, torch.Tensor, Any], torch.Tensor],
    shape,
    condition,
    num_inference_steps: int,
    eps: torch.Tensor,
    device: torch.device | str | None,
    *,
    return_velocities: bool = False,
):
    """Euler-sample the flow t=0 (noise) -> t=1 (data). Caller draws ``eps``.

    ``device`` has no default on purpose, and ``None`` is rejected below rather
    than silently meaning CPU: the sampler is on the rollout path, where the
    device is the one the policy runs on, not something to infer.

    ``return_velocities`` also hands back the ``v_hat`` sequence for
    :func:`trajectory_straightness`. Off by default: it keeps every velocity
    alive for the whole solve, which the inference path does not need. Nothing
    in the training or rollout path sets it -- the straightness metric has no
    caller outside tests.
    """
    if device is None:
        raise ValueError("device must be provided explicitly")
    if tuple(eps.shape) != tuple(shape):
        raise ValueError(f"eps shape {tuple(eps.shape)} != requested {tuple(shape)}")

    grid = torch.linspace(
        0.0,
        1.0,
        num_inference_steps + 1,
        device=device,
        dtype=torch.float32,
    )

    if eps.device != torch.device(device):
        eps = eps.to(device)

    x = eps
    velocities = [] if return_velocities else None
    with torch.inference_mode():
        dts = grid[1:] - grid[:-1]
        for i in range(num_inference_steps):
            t_current = grid[i].reshape(1)
            v_hat = predict_velocity(x, t_current, condition)
            if velocities is not None:
                velocities.append(v_hat.detach().float())
            # addcmul, not x + dt*v: same value, one kernel instead of two, and
            # this loop runs num_euler_steps times per policy call.
            x = torch.addcmul(x, v_hat, dts[i])
    return (x, velocities) if return_velocities else x
