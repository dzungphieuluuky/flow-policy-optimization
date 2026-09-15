"""Orthonormal DCT-II / DCT-III utilities (pure torch matmuls, no scipy).

M[k, n] = sqrt(2/N)·c_k·cos(π(2n+1)k/2N), c_0 = 1/√2. Orthonormality (MᵀM = I)
preserves Euclidean measure, so Gaussians and log-densities survive the
change of variables (log-det = 0). ``dct(x) = x @ Mᵀ``, ``idct(y) = y @ M``.
"""

from __future__ import annotations

import math

import einx
import torch

__all__ = [
    "dct",
    "idct",
    "mask_actions",
    "discounted_chunk_reward",
]

BASIS_CACHE: dict[int, torch.Tensor] = {}


def dct_basis_matrix(n: int) -> torch.Tensor:
    """[n, n] orthonormal DCT-II basis M (MᵀM = I)."""
    m = BASIS_CACHE.get(n)
    if m is None:
        k = einx.id("n -> n 1", torch.arange(n, dtype=torch.float32))
        t = einx.id("n -> 1 n", torch.arange(n, dtype=torch.float32))
        m = math.sqrt(2.0 / n) * torch.cos(math.pi * (2.0 * t + 1.0) * k / (2.0 * n))
        m[0].mul_(torch.rsqrt(torch.tensor(2.0)))
        BASIS_CACHE[n] = m
    return m


def apply_matrix_along_dim(
    x: torch.Tensor, dim: int, mat: torch.Tensor
) -> torch.Tensor:
    """Right-multiply the ``dim`` slice of ``x`` by ``mat`` (same shape out)."""
    dim = dim % x.dim()
    if dim != x.dim() - 1:
        x = x.movedim(dim, -1)
    out = x @ mat.to(x.device)
    if dim != out.dim() - 1:
        out = out.movedim(-1, dim)
    return out


def dct(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return apply_matrix_along_dim(x, dim, dct_basis_matrix(x.shape[dim % x.dim()]).t())


def idct(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Inverse (DCT-III) along ``dim``: matmul with M (M⁻¹ = Mᵀ)."""
    return apply_matrix_along_dim(x, dim, dct_basis_matrix(x.shape[dim % x.dim()]))


def mask_actions(actions: torch.Tensor, executed_n_action_steps, dim: int = 1) -> torch.Tensor:
    """Zero-pad unexecuted tail rows of a real-space chunk; return same space.

    Stored chunks are real-space with time == action_horizon, so masking is a plain
    row multiply. Executed counts [N] (1 <= L <= H) along ``dim`` (default 1).
    """
    h = actions.shape[dim % actions.dim()]
    lens = torch.as_tensor(list(executed_n_action_steps), device=actions.device)
    row = torch.arange(h, device=actions.device).unsqueeze(0)
    mask = (row < lens.unsqueeze(1)).to(actions.dtype).unsqueeze(-1)
    return actions * mask


def discounted_chunk_reward(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    n = rewards.shape[-1]
    weights = gamma ** torch.arange(n, dtype=rewards.dtype, device=rewards.device)
    return (rewards * weights).sum(dim=-1)
