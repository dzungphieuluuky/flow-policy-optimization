"""FPO++ core maths. The module holds the CFM loss, the ratio, the ASPO trust
region, and the Euler sampler.

Source: arXiv 2602.02481, equations 3 to 13.
"""

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
    """MC noise. The marginal comes from ``noise_sampling_mode``.

    This is the only definition of the base sample. The CFM path reads it
    twice. ``compute_mc_xt`` builds ``lerp(noise, action, t)``.
    ``cfm_target`` returns ``a - eps``.
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
    """Broadcast the action, the noise, and the per-token timestep over the MC
    axis. The function returns ``[B*M, ...]``.

    ``tau`` has the shape ``[B, M, K, 1]``. Each action step gets its own time.
    That step keeps the velocity of a token and the target of that token on the
    same ``tau``.

    This function no longer accepts a scalar or a per-chunk shape. No caller
    could reach either one.

    The cast onto the device and dtype of ``eps`` stops a scalar from moving to
    the CPU in the middle of the graph.
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

    The bound applies at every site that produces a raw velocity. Those sites
    are the training loss, the frozen ``cfm_loss_old`` pass, and the Euler
    sampler. All three see the same map, so the ratio stays consistent.

    ``None`` does nothing.
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
    """CFM squared-velocity loss. FPO++ Eq. 7-8.

    ``weight`` multiplies the squared error before the reduction. This is where
    FPO++ applies its schedule over t (``flow_model.py:631-634``). The weight
    applies after the padding mask, so padded rows stay zero.

    ``sample``: flatten the tokens, then reduce with ``cfg.cfm_loss_reduction``.
    ``token_mean``: divide by action_dim only. Masking cannot move this value.
    ``both``: return the masked sample and one masked square.
    ``scalar``: return one number.

    ``action_is_pad`` has the shape ``[n, horizon]`` and is True where padded.
    The function zeroes those rows. For every reduction except ``sum``, it then
    divides by the executed count, so a short chunk keeps the magnitude of a
    full chunk.

    ``sum`` does not divide. Equation 8 takes the squared norm over the whole
    chunk. So the loss of a short chunk, and the trust region it carries, both
    scale with the chunk length.

    The reduction sets the level of the returned loss. ``mean`` stays flat as
    the chunk grows. ``sqrt`` grows as the square root of the element count.
    ``sum`` grows as the element count. ``cfm_clamp`` is compared against that
    level. So one clamp value means different things under the three
    reductions, and the meaning also moves with ``action_horizon``. Under
    ``sum`` at a horizon above 4, the level exceeds the default clamp, and
    :func:`clamp_saturation` then reports a saturated share.
    """
    if v_hat.shape != v_star.shape:
        raise ValueError(
            f"v_hat shape {tuple(v_hat.shape)} != v_star shape {tuple(v_star.shape)}"
        )
    # F.mse_loss with reduction="none" is square(v_hat - v_star). The default
    # reduction would divide, and the masking below already divides.
    diff = v_hat - v_star
    if cfg.cfm_loss_use_huber:
        # FPO++ flow_model.py:_compute_squared_error. The quadratic branch has no
        # 0.5 factor. So this equals MSE for |diff| <= delta. Only the tail is
        # linear.
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
        # action_is_pad marks the rows to drop, so the function flips it. Keep is 1 on
        # an executed position and 0 on a padded position.
        keep = (~action_is_pad).to(sq.dtype)[:, None, :, None]
        # The scale factor is action_dim, so the divisor counts scalar elements
        # and not tokens.
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
        # Per-token diagnostic. Dividing by action_dim keeps the value independent of
        # the token count. The mask above already zeroed the padded rows.
        return sq.mean(dim=-1)
    raise ValueError(
        f"reduce_mode={reduce_mode!r} must be 'scalar', 'sample', 'token_mean', "
        "or 'both'"
    )


def group_cfm_loss(loss: torch.Tensor, group_size: int) -> torch.Tensor:
    """Average a CFM loss over groups of MC samples. Input ``[n, num_mc, ...]``.

    FPO++ groups the Monte-Carlo axis before it exponentiates
    (``finetune_online_rl.py:803-806``). The value ``-1`` puts every sample in
    one group. This is FPO Eq. 9: mean the MC losses, then exponentiate.

    The function groups the loss. It does not group ``old - new``. The two
    agree because ``exp(mean(old) - mean(new)) == exp(mean(old - new))``.

    The value ``1`` returns the input unchanged. So per-sample ratios cost
    nothing.
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


def _sigreg_sketch(z: torch.Tensor, sketch_dim: int) -> torch.Tensor:
    """Project rows to at most ``sketch_dim`` columns with a random map."""
    n, c = z.shape
    s = min(sketch_dim, n, c)
    if c <= s:
        return z
    mat = torch.randn(s, c, device=z.device, dtype=z.dtype) / (c**0.5)
    return z @ mat.T


def sigreg_weak_loss(z: torch.Tensor, sketch_dim: int = 64) -> torch.Tensor:
    """Weak SIGReg. The function matches the row covariance to identity
    (LeWorldModel).

    The penalty is ``||Cov - I||_F`` over sketched, centered rows. One row
    returns zero, because no covariance exists. The input casts to float32, so
    bf16 callers stay valid.
    """
    z = z.float()
    if z.shape[0] < 2:
        return torch.zeros((), device=z.device)
    z = _sigreg_sketch(z, sketch_dim)
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / (z.shape[0] - 1)
    eye = torch.eye(z.shape[1], device=z.device, dtype=z.dtype)
    return torch.norm(cov - eye, p="fro")


def sigreg_decorr_loss(z: torch.Tensor, sketch_dim: int = 64) -> torch.Tensor:
    """Decorrelation-only SIGReg. The function standardises the columns, then
    removes the off-diagonal.

    The column scales survive. Only redundancy pays. The function uses the same
    row-count guard and the same float32 cast as :func:`sigreg_weak_loss`.
    """
    z = z.float()
    if z.shape[0] < 2:
        return torch.zeros((), device=z.device)
    z = _sigreg_sketch(z, sketch_dim)
    z = z - z.mean(dim=0, keepdim=True)
    std = z.std(dim=0, keepdim=True).clamp_min(1e-6)
    corr = ((z / std).T @ (z / std)) / (z.shape[0] - 1)
    off = corr - torch.diag(torch.diag(corr))
    return torch.norm(off, p="fro")


def cfm_loss_weight(t: torch.Tensor, cfg: SJFPOConfig) -> torch.Tensor:
    """Return the per-token CFM loss weight. The value broadcasts over the action
    dimension.

    FPO++ has two ramps. They are ``1 - 0.9t`` and ``1 - 0.99t`` on the time of
    the reference (``flow_model.py:648-661``). In the reference, ``t=0`` is the
    action and ``t=1`` is the noise.

    This repo runs time in the other direction. So the same two schedules read
    ``0.1 + 0.9t`` and ``0.01 + 0.99t`` here. The weight is full at the action
    end and least at the noise end.

    A verbatim copy of the formula gives the noise end the largest weight. That
    is the opposite of the intent.

    ``t`` has the shape ``[..., 1]``. The weight therefore broadcasts over the
    action dimensions without a change.
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
    """Map a velocity prediction to the quantity that ``cfg.cfm_loss_target``
    compares.

    The network always predicts the velocity. Only the loss changes.

    On the straight path ``x_t = (1 - t) * eps + t * a`` and ``v = a - eps``.
    At the true velocity, every mode below returns its own target exactly.
    ``x_t + (1 - t) * v == a`` and ``x_t - t * v == eps``. That identity is
    what the test checks.

    FPO++ reaches the same three endpoints by also changing what the network
    predicts (``flow_network_output_param``). Holding that at ``u`` here is the
    smaller change. It also leaves ``transported_clip_value`` clipping one
    thing.
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
    """FPO++ trust region. ASPO is PPO for A>0 and SPO for A<0. It runs on each
    element.

    PPO clips at ``clip_param``. The SPO quadratic uses ``spo_clip_param``.
    These are two widths, not one. The manipulation loop of FPO++ carries them
    as two coefficients (``finetune_online_rl.py:117-118``).
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


def clamp_saturation(
    cfm_loss_old: torch.Tensor,
    cfm_loss_new: torch.Tensor,
    cfg: SJFPOConfig,
) -> torch.Tensor:
    """The share of elements whose two losses both sit above ``cfm_clamp``.

    ``compute_policy_ratio`` clamps both losses at ``cfm_clamp``. When both
    exceed it, both clamp to the same bound. The delta becomes zero. The ratio
    becomes exactly 1.0. The surrogate returns the advantage, so the policy
    receives no gradient. The critic keeps training.

    This function reports that share. It is the direct cause, not the level.
    A run can sit above ``cfm_clamp`` and still keep a gradient, when the old
    loss is below it. So ``cfm_above_clamp_frac`` alone does not show the
    failure.

    The input shapes must match ``compute_policy_ratio``'s, so that a caller
    measures the pair that the ratio clamps.
    """
    if cfm_loss_old.shape != cfm_loss_new.shape:
        raise ValueError(
            f"cfm_loss_old shape {tuple(cfm_loss_old.shape)} != "
            f"{tuple(cfm_loss_new.shape)}"
        )
    bound = cfg.cfm_clamp
    both = (cfm_loss_old > bound) & (cfm_loss_new > bound)
    return both.to(torch.float32).mean()


def compute_policy_ratio(
    cfm_loss_old: torch.Tensor,
    cfm_loss_new: torch.Tensor,
    cfg: SJFPOConfig,
    advantage: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return exp(cfm_loss_old - cfm_loss_new). This ratio takes the place of a
    log-probability ratio.

    The function runs on each element. So the caller sets the granularity
    through the shape it passes in. Per-chunk granularity passes the MC means,
    so one ratio covers a whole chunk. Per-pair granularity passes one loss per
    pair. Per-token granularity passes one loss per token.

    The function clamps both losses. It then clamps ``cfm_loss_new`` harder on
    A<0, if the configuration asks for it. Then it caps the log-ratio on one
    side with a straight-through estimator. The forward pass sees the cap. The
    backward pass sees the original value.
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


def sample_action(
    predict_velocity: Callable[[torch.Tensor, torch.Tensor, Any], torch.Tensor],
    shape,
    condition,
    num_inference_steps: int,
    eps: torch.Tensor,
    device: torch.device | str | None,
):
    """Euler-sample the flow from t=0 (noise) to t=1 (data). The caller draws
    ``eps``.

    ``device`` has no default, on purpose. The function rejects ``None`` below.
    It does not treat ``None`` as a request for the CPU. The sampler runs on the
    rollout path. There, the device is the device the policy runs on. It is not
    something to infer.
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
    with torch.inference_mode():
        dts = grid[1:] - grid[:-1]
        for i in range(num_inference_steps):
            t_current = grid[i].reshape(1)
            v_hat = predict_velocity(x, t_current, condition)
            # The code uses addcmul, not x + dt*v. Both give the same value, and
            # addcmul uses one kernel, not two. This loop runs num_euler_steps
            # times per policy call.
            x = torch.addcmul(x, v_hat, dts[i])
    return x
