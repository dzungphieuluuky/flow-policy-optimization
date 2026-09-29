"""DCT-based Fourier coefficient utilities for FAFM."""

from __future__ import annotations

import math
from functools import lru_cache

import torch

def _normalize_device(device: torch.device | str | None) -> torch.device:
    target = torch.device("cpu") if device is None else torch.device(device)
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    return target


def _dct_norm(count: int, sequence_length: int, device: torch.device) -> torch.Tensor:
    if count < 1 or sequence_length < 1:
        raise ValueError("count and sequence_length must be positive")
    norm = torch.full(
        (count,), math.sqrt(2.0 / sequence_length), dtype=torch.float32, device=device
    )
    # DCT-II: k=0 gets sqrt(1/K), the rest sqrt(2/K).
    norm[0] *= 1.0 / math.sqrt(2.0)
    return norm


@lru_cache(maxsize=64)
def _get_analysis_basis(
    K: int, M: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if K < 1:
        raise ValueError(f"K must be positive, got {K}")
    if M < 0 or M >= K:
        raise ValueError(f"M must satisfy 0 <= M < K, got M={M}, K={K}")
    j = torch.arange(K, dtype=torch.float32, device=device)
    k = torch.arange(M + 1, dtype=torch.float32, device=device)
    basis = torch.cos(math.pi * k[:, None] * (j[None, :] + 0.5) / K)
    return basis, _dct_norm(M + 1, K, device)


def extract_dct_coeffs(v_star: torch.Tensor, M: int) -> torch.Tensor:
    """Extract the first M+1 DCT-II coefficients of a velocity sequence."""
    squeeze = False
    if v_star.dim() == 3:
        B, K, d = v_star.shape
        if B < 1:
            raise ValueError(f"v_star batch size must be positive, got {B}")
    elif v_star.dim() == 2:
        K, d = v_star.shape
        v_star = v_star.unsqueeze(0)
        B = 1
        squeeze = True
    else:
        raise ValueError(
            f"v_star must be 2D (K, d) or 3D (B, K, d), got shape {v_star.shape}"
        )
    if d < 1:
        raise ValueError(f"v_star feature dimension must be positive, got {d}")

    basis, norm = _get_analysis_basis(K, M, _normalize_device(v_star.device))
    with torch.autocast(device_type=v_star.device.type, enabled=False):
        coeffs = torch.einsum("mk,bkd->bmd", basis, v_star.float())
    coeffs = (coeffs * norm[None, :, None]).to(v_star.dtype)
    return coeffs.squeeze(0) if squeeze else coeffs


def velocity_to_action_steps(vel: torch.Tensor, num_rows: int) -> torch.Tensor:
    """Expand DCT coeffs back onto the half-open grid ``t_j = j*H/H``.

    Takes any leading batch axes, so ``[n, mc, K, d]`` and ``[n, K, d]`` both
    work. Inverse of :func:`extract_dct_coeffs` for a full-length signal: read
    coeffs as time samples, velocity in row space, so a per-row execution mask
    can index it.
    """
    lead, k, d = vel.shape[:-2], vel.shape[-2], vel.shape[-1]
    tau = torch.arange(num_rows, device=vel.device, dtype=torch.float32) / num_rows
    with torch.autocast(device_type=vel.device.type, enabled=False):
        rows = decode_coeffs_to_velocity(
            vel.reshape(-1, k, d).to(torch.float32), tau, 1.0, num_rows
        )
    return rows.reshape(*lead, num_rows, d).to(vel.dtype)


def decode_coeffs_to_velocity(
    coeffs: torch.Tensor,
    tau: torch.Tensor,
    T: float,
    K_original: int,
) -> torch.Tensor:
    """Decode DCT coeffs at query times.

    Times use ``t_j = j*T/K``, so a query maps to ``j = tau*K/T``. ``tau = T``
    gives the DCT-II even-symmetric extension (mirror of the last sample), not
    a periodic wrap.
    """
    squeeze = False
    if coeffs.dim() == 3:
        B, M_plus_1, d = coeffs.shape
    elif coeffs.dim() == 2:
        M_plus_1, d = coeffs.shape
        coeffs = coeffs.unsqueeze(0)
        B = 1
        squeeze = True
    else:
        raise ValueError(
            f"coeffs must be 2D (M+1, d) or 3D (B, M+1, d), got shape {coeffs.shape}"
        )

    if B < 1 or M_plus_1 < 1 or d < 1:
        raise ValueError(
            "coeffs must have positive batch, frequency, and feature dimensions"
        )
    if K_original < 1:
        raise ValueError(f"K_original must be positive, got {K_original}")
    if M_plus_1 > K_original:
        raise ValueError(
            f"number of coefficients {M_plus_1} cannot exceed K_original={K_original}"
        )
    if not math.isfinite(T) or T <= 0:
        raise ValueError(f"T must be finite and positive, got {T}")

    M = M_plus_1 - 1
    device = _normalize_device(coeffs.device)
    norm = _dct_norm(M + 1, K_original, device)
    k = torch.arange(M + 1, dtype=torch.float32, device=device)

    if tau.dim() == 1:
        tau_q = tau.to(device=device, dtype=torch.float32)
        if tau_q.numel() == 0:
            raise ValueError("tau must contain at least one query time")
        if not bool(torch.all((tau_q >= 0) & (tau_q <= T))):
            raise ValueError("tau must lie in [0, T]")
        j = tau_q * (K_original / T)
        basis = torch.cos(math.pi * k[:, None] * (j[None, :] + 0.5) / K_original)
        with torch.autocast(device_type=device.type, enabled=False):
            v = torch.einsum("bkd,kn,k->bnd", coeffs.to(torch.float32), basis, norm)
    elif tau.dim() == 2:
        if tau.shape[0] != B:
            raise ValueError(
                f"tau batch size {tau.shape[0]} must match coeffs batch size {B}"
            )
        tau_q = tau.to(device=device, dtype=torch.float32)
        if tau_q.numel() == 0:
            raise ValueError("tau must contain at least one query time")
        if not bool(torch.all((tau_q >= 0) & (tau_q <= T))):
            raise ValueError("tau must lie in [0, T]")
        j = tau_q * (K_original / T)
        basis = torch.cos(
            math.pi * k[None, :, None] * (j[:, None, :] + 0.5) / K_original
        )
        with torch.autocast(device_type=device.type, enabled=False):
            v = torch.einsum("bkd,bkn,k->bnd", coeffs.to(torch.float32), basis, norm)
    else:
        raise ValueError(f"tau must be 1D (N,) or 2D (B, N), got shape {tau.shape}")

    v = v.to(coeffs.dtype)
    return v.squeeze(0) if squeeze else v