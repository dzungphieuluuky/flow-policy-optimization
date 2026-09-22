"""DCT-based Fourier coefficient utilities for FAFM."""

from __future__ import annotations

import math

import numpy as np
import torch
from sjfpo.utils import resolve_storage_dtype

STORAGE_DTYPE = resolve_storage_dtype()


def extract_dct_coeffs(
    v_star: torch.Tensor,
    M: int,
) -> torch.Tensor:
    """Extract DCT-II coefficients from velocity sequence."""
    if v_star.dim() == 3:
        B, K, d = v_star.shape
        batched = True
    elif v_star.dim() == 2:
        K, d = v_star.shape
        batched = False
        v_star = v_star.unsqueeze(0)
        B = 1
    else:
        raise ValueError(f"v_star must be 2D (K, d) or 3D (B, K, d), got shape {v_star.shape}")

    j = torch.arange(K, dtype=torch.float32, device=v_star.device)
    k = torch.arange(M + 1, dtype=torch.float32, device=v_star.device)

    basis = torch.cos(math.pi * k[:, None] * (j[None, :] + 0.5) / K)
    norm = torch.ones(M + 1, dtype=torch.float32, device=v_star.device)
    norm = norm.clone()
    norm[0] = 1.0 / math.sqrt(2.0)
    norm = norm * math.sqrt(2.0 / K)

    coeffs = torch.einsum('mk,bkd->bmd', basis.to(v_star.dtype), v_star) * norm[:, None].to(v_star.dtype)
    if not batched:
        coeffs = coeffs.squeeze(0)
    return coeffs


def decode_coeffs_to_velocity(
    coeffs: torch.Tensor,
    tau: torch.Tensor,
    T: float,
    K_original: int,
) -> torch.Tensor:
    """Decode DCT coefficients to velocities at arbitrary physical times."""
    if coeffs.dim() == 3:
        B, M_plus_1, d = coeffs.shape
        batched = True
    else:
        M_plus_1, d = coeffs.shape
        batched = False
        coeffs = coeffs.unsqueeze(0)
        B = 1

    M = M_plus_1 - 1
    k = torch.arange(0, M + 1, dtype=torch.float32, device=coeffs.device)

    norm = torch.ones(M + 1, dtype=torch.float32, device=coeffs.device)
    norm[0] = 1.0 / math.sqrt(2.0)
    norm = norm * math.sqrt(2.0 / K_original)

    if tau.dim() == 1:
        tau_query = tau.unsqueeze(0)
    elif tau.dim() == 2:
        tau_query = tau
    else:
        raise ValueError(f"tau must be 1D (N,) or 2D (B, N), got shape {tau.shape}")

    j_indices = tau_query.to(torch.float32) * K_original / T
    basis = torch.cos(math.pi * k[None, :, None] * (j_indices[:, None, :] + 0.5) / K_original)

    norm_idct = torch.ones(M + 1, dtype=torch.float32, device=coeffs.device)
    norm_idct[0] = 1.0 / math.sqrt(2.0)
    norm_idct = norm_idct * math.sqrt(2.0 / K_original)

    v_decoded = torch.einsum('bkd,bkn,k->bnd', coeffs.to(torch.float32), basis.to(torch.float32), norm_idct)
    v_decoded = v_decoded.to(coeffs.dtype)
    if not batched:
        v_decoded = v_decoded.squeeze(0)
    return v_decoded


def sample_freq_weighted_noise(
    M: int,
    d: int,
    T: float,
    B: int = 1,
    sigma: float = 1.0,
    key: int | None = None,
) -> torch.Tensor:
    if key is not None:
        torch.manual_seed(key)
    device = torch.device('cpu') if STORAGE_DTYPE == torch.float32 else torch.device('cpu')
    k_idx = torch.arange(0, M + 1, dtype=torch.float32, device=device)
    omega_k = k_idx * math.pi / T
    std_k = sigma / torch.sqrt(1.0 + omega_k ** 2)
    std_k = std_k.to(device)
    xi = torch.randn(B, M + 1, d, dtype=torch.float32, device=device)
    noise = xi * std_k[None, :, None].to(xi.dtype)
    return noise


def estimate_M_from_dataset(dataset, energy_threshold: float = 0.95) -> int:
    cumulative_energies = []
    for item in dataset:
        if isinstance(item, dict):
            v_star = np.asarray(item.get('actions', item.get('action', item)))
        else:
            v_star = np.asarray(item)
        if v_star.ndim == 2:
            K, d = v_star.shape
            coeffs = np.fft.dct(np.transpose(v_star, (1, 0)), norm='ortho', axis=-1)
        else:
            raise ValueError(f"Expected 2D, got {v_star.shape}")
        energy = (coeffs ** 2).sum(axis=0)
        cumsum = np.cumsum(energy) / energy.sum()
        cumulative_energies.append(cumsum)
    mean_cumsum = np.stack(cumulative_energies).mean(axis=0)
    M = int(np.argmax(mean_cumsum >= energy_threshold))
    M = max(1, M)
    return M


def dct_basis_matrix(n: int) -> torch.Tensor:
    k = torch.arange(n, dtype=STORAGE_DTYPE)
    t = torch.arange(n, dtype=STORAGE_DTYPE).unsqueeze(0)
    m = math.sqrt(2.0 / n) * torch.cos(math.pi * (2.0 * t + 1.0) * k.unsqueeze(1) / (2.0 * n))
    m[0].mul_(torch.rsqrt(torch.tensor(2.0)))
    return m


def apply_matrix_along_dim(x: torch.Tensor, dim: int, mat: torch.Tensor) -> torch.Tensor:
    dim = dim % x.dim()
    if dim != x.dim() - 1:
        x = x.movedim(dim, -1)
    out = x @ mat.to(x.device, dtype=x.dtype)
    if dim != out.dim() - 1:
        out = out.movedim(-1, dim)
    return out


def dct(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return apply_matrix_along_dim(x, dim, dct_basis_matrix(x.shape[dim % x.dim()]).t())


def idct(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return apply_matrix_along_dim(x, dim, dct_basis_matrix(x.shape[dim % x.dim()]))


def discounted_chunk_reward(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    n = rewards.shape[-1]
    weights = gamma ** torch.arange(n, dtype=rewards.dtype, device=rewards.device)
    return (rewards * weights).sum(dim=-1)
