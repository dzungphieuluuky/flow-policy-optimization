"""Orthonormal DCT-II / DCT-III utilities (pure torch, fp32).

Plan task 3. The FPO++ flow pipeline operates in ORTHONORMAL DCT-II
coefficient space of the scaled action chunk (plan Scope §Flow space,
BINDING): ``a_dct = DCT(a_chunk)``, noise/velocity/Euler all live in
coefficient space, and execution maps back via ``IDCT``. Orthonormality
(MᵀM = I) is required so the transform preserves Euclidean measure — a
Gaussian in coefficient space is a Gaussian in action space and log-densities
are invariant under an orthogonal change of variables (log-det = 0).

Implementation constraint (plan Must-NOT-do): NO scipy at runtime — the
transform is a plain matmul against a precomputed orthonormal DCT-II cosine
basis matrix held as a buffer-less plain tensor (module-level cache), fp32
throughout.

Definitions (orthonormal DCT-II), N = basis size:
    M[k, n] = sqrt(2/N) · c_k · cos(π·(2n+1)·k / (2N)),
    c_0 = 1/sqrt(2), c_k = 1 for k > 0.
Rows k are frequencies, columns n are time samples, so for a tensor with the
transform axis last: ``dct(x) = x @ Mᵀ`` and the inverse (DCT-III, the
transpose of an orthogonal matrix is its inverse) is ``idct(y) = y @ M``.

Tensor conventions:
    B: batch size
    H: action-chunk horizon (default 8)
    Da: action dimension (env-dependent)
    D: state dimension (default 128)
    N: DCT basis size (typically H=8 or Da)

All tensors are fp32.
"""

from __future__ import annotations

import math

import einx
import torch

__all__ = [
    "dct",
    "idct",
    "mask_unexecuted_tail",
    "discounted_chunk_reward",
]

# Module-level cache of precomputed orthonormal bases, keyed by size N.
# Plain tensors (buffer-less per plan); read-only after construction.
BASIS_CACHE: dict[int, torch.Tensor] = {}


def dct_basis_matrix(n: int) -> torch.Tensor:
    """Return the [n, n] orthonormal DCT-II basis M with MᵀM = I (fp32).

    Args:
        n: Basis size (sequence length).

    Returns:
        Orthonormal DCT-II basis matrix of shape [n, n].
    """
    m = BASIS_CACHE.get(n)
    if m is None:
        k = einx.id(
            "n -> n 1", torch.arange(n, dtype=torch.float32)
        )  # [n, 1] frequency rows
        t = einx.id(
            "n -> 1 n", torch.arange(n, dtype=torch.float32)
        )  # [1, n] time columns
        m = math.sqrt(2.0 / n) * torch.cos(
            math.pi * (2.0 * t + 1.0) * k / (2.0 * n)
        )  # [n, n]
        m[0].mul_(1.0 / math.sqrt(2.0))  # c_0 normalization → orthonormal rows
        BASIS_CACHE[n] = m
    return m


def apply_matrix_along_dim(
    x: torch.Tensor, dim: int, mat: torch.Tensor
) -> torch.Tensor:
    """Right-multiply the slice of ``x`` along ``dim`` by ``mat``.

    Args:
        x: Input tensor of shape [..., N, ...] where N is the dimension to transform.
        dim: Dimension index to apply transform along.
        mat: Transform matrix of shape [N, N].

    Returns:
        Transformed tensor with same shape as x.
    """
    dim = dim % x.dim()
    if dim != x.dim() - 1:
        x = x.movedim(dim, -1)
    out = x @ mat.to(x.device)  # [..., N] @ [N, N] -> [..., N]
    if dim != out.dim() - 1:
        out = out.movedim(-1, dim)
    return out


def dct(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Orthonormal DCT-II of ``x`` along ``dim`` via matmul (fp32).

    For chunk tensors [B, H=8, A] use ``dim=1``; default transforms the last
    axis.

    Args:
        x: Input tensor of shape [..., N, ...].
        dim: Dimension to transform. Default -1 (last dimension).

    Returns:
        DCT-II coefficients of same shape as x.
    """
    return apply_matrix_along_dim(x, dim, dct_basis_matrix(x.shape[dim % x.dim()]).t())


def idct(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Inverse transform (DCT-III) along ``dim``: matmul with M itself.

    M is orthogonal so M⁻¹ = Mᵀ; applying DCT-III to coefficients returns the
    original signal exactly up to float round-off.

    Args:
        x: DCT coefficients of shape [..., N, ...].
        dim: Dimension to transform. Default -1 (last dimension).

    Returns:
        Reconstructed signal of same shape as x.
    """
    return apply_matrix_along_dim(x, dim, dct_basis_matrix(x.shape[dim % x.dim()]))


def mask_unexecuted_tail(
    actions_dct: torch.Tensor, lengths, dim: int = 1
) -> torch.Tensor:
    """Zero-pad unexecuted tail rows in real space; return real-space actions.

    The critic and the executed-step anchor must see only executed steps:
    IDCT back to real space, zero rows past each row's executed length,
    leave full-length rows bit-close to the roundtrip. Stored rows are DCT
    frequency bins (NOT time steps) — masking must happen post-IDCT.

    Args:
        actions_dct: DCT coefficients of shape [N, H, Da].
        lengths: Executed steps per row ([N] ints, 1 <= L <= H).
        dim: Time axis. Default 1.

    Returns:
        Real-space actions [N, H, Da] with unexecuted tails zeroed.
    """
    real = idct(actions_dct, dim=dim)  # [N, H, Da]
    h = actions_dct.shape[dim % actions_dct.dim()]
    lens = torch.as_tensor(list(lengths), device=real.device)  # [N]
    row = torch.arange(h, device=real.device).unsqueeze(0)  # [1, H]
    mask = (row < lens.unsqueeze(1)).to(real.dtype).unsqueeze(-1)  # [N, H, 1]
    return real * mask  # [N, H, Da]

    # retained: test-only public API — exercised directly by tests/test_dct.py;
    # not called by the src train path. Keep per fn-consolidation D1.


def discounted_chunk_reward(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Σ_t γ^t · r_t along the last axis (chunk length 8).

    Supports batched input [..., 8]; returns a tensor with the batch dims
    preserved (scalar for a bare [8] input).

    Args:
        rewards: Reward tensor of shape [..., 8].
        gamma: Discount factor.

    Returns:
        Discounted sum of shape [...] (last dimension reduced).
    """
    n = rewards.shape[-1]
    weights = gamma ** torch.arange(
        n, dtype=rewards.dtype, device=rewards.device
    )  # [8]
    return (rewards * weights).sum(dim=-1)  # [...]
