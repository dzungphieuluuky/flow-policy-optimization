"""The single home of the action-processing contract.

The flow regresses raw policy-space chunks; stored actions are that flow
output, bounded only by :func:`bound_for_storage`, with nothing else applied.
This module is the only place the ``action_processing_mode`` branch lives:

* :func:`bound_for_storage` — the bound applied to the stored action at collection.
* :func:`to_env_action` — the env-facing map (linear: unscale to env bounds; tanh: squash).
* :func:`to_critic_action` — the action space the critic reads (linear: scaled
  space with the pad mask; tanh: env bounds).

Every collector, eval, loss and critic consumer calls one of these; no other
file may branch on the mode.
"""

from __future__ import annotations
import torch


def env_squash_terms(
    low: torch.Tensor, high: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The scale and bias of the env's action bounds.

    ``scale = (high - low) / 2`` and ``bias = low + scale``. The bias is
    the low end plus the half width, so the two terms come from one
    difference and cannot drift apart.
    """
    scale = 0.5 * (high - low)
    return scale, low + scale


def _squash(
    raw: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Core tanh map: ``bias + scale * tanh(raw)``.

    Private. ``scale`` and ``bias`` are the affine terms of the env's
    bounds, from :func:`env_squash_terms`. The ``tanh`` is the only thing
    that bounds the action, and it is applied exactly once, on the way to
    the env. The flow regresses on raw policy-space targets; it never
    trains on the env-facing squash of its own output.
    """
    scale = scale.to(device=raw.device, dtype=raw.dtype)
    bias = bias.to(device=raw.device, dtype=raw.dtype)
    return torch.addcmul(bias, scale, torch.tanh(raw))


def squash_to_env_bounds(
    raw: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map a raw policy output into the env's action bounds.

    The single env-facing squash: derive the affine terms once, apply the
    tanh, and optionally zero padded steps. Collector, training diagnostics
    and critic all read through here, so the bound invariant has one home.
    """
    scale, bias = env_squash_terms(low, high)
    out = _squash(raw, scale, bias)
    if mask is not None:
        out = out * mask
    return out


def bound_to_unit(x: torch.Tensor) -> torch.Tensor:
    """Clamp to the unit hypercube ``[-1, 1]`` (the linear-mode stored space).

    The stored action under linear processing is bounded in ``[-1, 1]``; this
    only bounds outliers, it does not rescale. Bounding lives here, not in the
    policy, so there is one home for the bound invariant.
    """
    return x.clamp(-1.0, 1.0)


def unscale_to_env(
    scaled: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map a bounded ``[-1, 1]`` action back to env scale.

    ``low + (scaled + 1) / 2 * (high - low)``.
    """
    scale = (high - low).to(device=scaled.device, dtype=scaled.dtype)
    low_t = low.to(device=scaled.device, dtype=scaled.dtype)
    out = low_t + (scaled + 1.0) * 0.5 * scale
    if mask is not None:
        out = out * mask
    return out


def bound_for_storage(x: torch.Tensor, mode: str) -> torch.Tensor:
    """The bound applied to the stored action at collection.

    ``linear_rescale``: clamp the stored chunk to the unit hypercube (outliers
    only; the flow's domain is the unit box). ``tanh_squash``: no bound — the
    raw flow output is stored as-is.
    """
    if mode == "linear_rescale":
        return bound_to_unit(x)
    return x


def to_env_action(
    x: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Map the raw flow output into the env's action bounds.

    ``linear_rescale``: unscale the unit box to the bounds. ``tanh_squash``:
    squash through ``tanh`` into the bounds.
    """
    if mode == "linear_rescale":
        return unscale_to_env(x, low, high)
    return squash_to_env_bounds(x, low, high)


def to_critic_action(
    x: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    mode: str,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map the raw flow output into the action space the critic reads.

    ``linear_rescale`` keeps the stored (unit-box) space — only the pad mask
    applies. ``tanh_squash`` reads the squashed env-bound space.
    """
    if mode == "linear_rescale":
        return x if mask is None else x * mask
    return squash_to_env_bounds(x, low, high, mask=mask)
