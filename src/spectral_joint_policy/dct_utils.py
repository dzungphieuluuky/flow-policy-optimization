"""DCT-based Fourier coefficient utilities for FAFM."""

from __future__ import annotations

import math
from functools import lru_cache

import torch

def normalize_device(device: torch.device | str | None) -> torch.device:
    """Resolve a device, filling in the current CUDA index."""
    target = torch.device("cpu") if device is None else torch.device(device)
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    return target


def compute_dct_norm(count: int, sequence_length: int, device: torch.device) -> torch.Tensor:
    """Orthonormal DCT-II norms: sqrt(2/K), halved at k=0."""
    if count < 1 or sequence_length < 1:
        raise ValueError("count and sequence_length must be positive")
    norm = torch.full(
        (count,), math.sqrt(2.0 / sequence_length), dtype=torch.float32, device=device
    )
    # DCT-II: k=0 gets sqrt(1/K), the rest sqrt(2/K).
    norm[0] *= 1.0 / math.sqrt(2.0)
    return norm


@lru_cache(maxsize=64)
def get_analysis_basis(
    K: int, num_modes: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached ``[num_modes, K]`` DCT-II basis and its norms."""
    if K < 1:
        raise ValueError(f"K must be positive, got {K}")
    if not 1 <= num_modes <= K:
        raise ValueError(
            f"num_modes must satisfy 1 <= num_modes <= K, "
            f"got num_modes={num_modes}, K={K}"
        )
    j = torch.arange(K, dtype=torch.float32, device=device)
    k = torch.arange(num_modes, dtype=torch.float32, device=device)
    basis = torch.cos(math.pi * k[:, None] * (j[None, :] + 0.5) / K)
    return basis, compute_dct_norm(num_modes, K, device)


def extract_dct_coeffs(v_star: torch.Tensor, num_modes: int) -> torch.Tensor:
    """Extract the first ``num_modes`` DCT-II coefficients of a velocity sequence.

    ``num_modes`` is a count, not a highest index, so ``num_modes == K`` keeps
    every frequency and the transform is the full orthonormal one.
    """
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

    basis, norm = get_analysis_basis(K, num_modes, normalize_device(v_star.device))
    with torch.autocast(device_type=v_star.device.type, enabled=False):
        coeffs = torch.einsum("mk,bkd->bmd", basis, v_star.float())
    coeffs = (coeffs * norm[None, :, None]).to(v_star.dtype)
    return coeffs.squeeze(0) if squeeze else coeffs


def compute_spectral_energy_retained(v: torch.Tensor, num_modes: int) -> torch.Tensor:
    """Share of a chunk's energy the kept ``num_modes`` carry. Orthonormal
    basis, so Parseval makes this 1.0 at ``num_modes == v.shape[1]``; below
    that it is what the truncation actually discards."""
    coeffs = extract_dct_coeffs(v, num_modes=num_modes)
    # square().sum() rather than an rms: this is a Parseval share, so the
    # divisor must be the total energy, not a per-element average.
    return coeffs.float().square().sum() / v.float().square().sum().clamp_min(1e-12)


def velocity_to_action_steps(vel: torch.Tensor, num_rows: int) -> torch.Tensor:
    """Expand DCT coeffs onto the half-open grid ``t_j = j*H/H``, any leading
    batch axes. Inverse of :func:`extract_dct_coeffs` for a full-length signal,
    in row space, so a per-row execution mask can index it."""
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
    """Decode DCT coeffs at query times ``t_j = j*T/K``, so a query maps to
    ``j = tau*K/T``. ``tau = T`` gives the DCT-II even-symmetric extension, not
    a periodic wrap."""
    squeeze = False
    if coeffs.dim() == 3:
        B, num_modes, d = coeffs.shape
    elif coeffs.dim() == 2:
        num_modes, d = coeffs.shape
        coeffs = coeffs.unsqueeze(0)
        B = 1
        squeeze = True
    else:
        raise ValueError(
            f"coeffs must be 2D (num_modes, d) or 3D (B, num_modes, d), "
            f"got shape {coeffs.shape}"
        )

    if B < 1 or num_modes < 1 or d < 1:
        raise ValueError(
            "coeffs must have positive batch, frequency, and feature dimensions"
        )
    if K_original < 1:
        raise ValueError(f"K_original must be positive, got {K_original}")
    if num_modes > K_original:
        raise ValueError(
            f"number of coefficients {num_modes} cannot exceed K_original={K_original}"
        )
    if not math.isfinite(T) or T <= 0:
        raise ValueError(f"T must be finite and positive, got {T}")

    device = normalize_device(coeffs.device)
    norm = compute_dct_norm(num_modes, K_original, device)
    k = torch.arange(num_modes, dtype=torch.float32, device=device)

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