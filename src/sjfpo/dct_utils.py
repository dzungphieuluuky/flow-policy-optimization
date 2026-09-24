"""DCT-based Fourier coefficient utilities for FAFM.

Performance notes
-----------------
* All DCT basis matrices are cached per ``(n, device)`` in fp32 and cast at
  the use site, so repeated calls hit a dict lookup instead of rebuilding
  ``cos`` tables.
* ``extract_dct_coeffs`` caches its ``[M+1, K]`` analysis basis and the
  ``[M+1]`` normalization vector.
* ``decode_coeffs_to_velocity`` shares one ``k``/``norm`` computation across
  the two normalization vectors and avoids the redundant fp32 round-trips.
* ``sample_freq_weighted_noise`` uses a local ``torch.Generator`` so it does
  not mutate global RNG state, and accepts a ``device`` argument.
* ``estimate_M_from_dataset`` stacks once and batch-DCTs via a cached basis
  instead of looping through Python and calling ``np.fft.dct`` per item.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from sjfpo.utils import resolve_storage_dtype

# sqrt(0.5), used to orthonormalize the DC row of every basis matrix.
_INV_SQRT2 = math.sqrt(0.5)


# ---------------------------------------------------------------------------
# Cached DCT-II basis (fp32) keyed by (n, device)
# ---------------------------------------------------------------------------

_DCT_BASIS_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _build_dct_basis_f32(n: int, device: torch.device) -> torch.Tensor:
    """Orthonormal DCT-II matrix ``D[k, j]`` of shape ``[n, n]``, fp32.

    ``D[k, j] = c_k · cos(π k (2j+1) / (2n))`` with ``c_0 = sqrt(1/n)`` and
    ``c_k = sqrt(2/n)`` for ``k > 0``. Forward DCT is ``D @ x``; inverse is
    ``Dᵀ @ x``.
    """
    k = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)
    t = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(0)
    m = math.sqrt(2.0 / n) * torch.cos(math.pi * (2.0 * t + 1.0) * k / (2.0 * n))
    m[0] *= _INV_SQRT2  # sqrt(2/n) · 1/sqrt(2) = sqrt(1/n)
    return m


def _get_dct_basis_f32(n: int, device: torch.device) -> torch.Tensor:
    key = (n, str(device))
    mat = _DCT_BASIS_CACHE.get(key)
    if mat is None:
        mat = _build_dct_basis_f32(n, device)
        _DCT_BASIS_CACHE[key] = mat
    return mat


def dct_basis_matrix(
    n: int, device: torch.device | str | None = None
) -> torch.Tensor:
    """Orthonormal DCT-II matrix for ``device`` in its storage dtype.

    Returns a cached tensor; callers must not mutate it in place.
    """
    target = torch.device("cpu") if device is None else torch.device(device)
    return _get_dct_basis_f32(n, target).to(resolve_storage_dtype(target))


# ---------------------------------------------------------------------------
# Analysis / synthesis bases for FAFM
# ---------------------------------------------------------------------------

# Analysis: [M+1, K] basis + [M+1] norm, keyed by (K, M, device).
_ANALYSIS_BASIS_CACHE: dict[
    tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]
] = {}


def _get_analysis_basis(
    K: int, M: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (K, M, str(device))
    cached = _ANALYSIS_BASIS_CACHE.get(key)
    if cached is not None:
        return cached
    j = torch.arange(K, dtype=torch.float32, device=device)
    k = torch.arange(M + 1, dtype=torch.float32, device=device)
    basis = torch.cos(math.pi * k[:, None] * (j[None, :] + 0.5) / K)
    norm = torch.full((M + 1,), math.sqrt(2.0 / K), dtype=torch.float32, device=device)
    norm[0] *= _INV_SQRT2
    cached = (basis, norm)
    _ANALYSIS_BASIS_CACHE[key] = cached
    return cached


def extract_dct_coeffs(v_star: torch.Tensor, M: int) -> torch.Tensor:
    """Extract the first ``M+1`` DCT-II coefficients of a velocity sequence.

    ``v_star`` is either ``[K, d]`` or ``[B, K, d]``; returns ``[M+1, d]``
    or ``[B, M+1, d]`` respectively.
    """
    squeeze = False
    if v_star.dim() == 3:
        B, K, d = v_star.shape
    elif v_star.dim() == 2:
        K, d = v_star.shape
        v_star = v_star.unsqueeze(0)
        B = 1
        squeeze = True
    else:
        raise ValueError(
            f"v_star must be 2D (K, d) or 3D (B, K, d), got shape {v_star.shape}"
        )

    basis, norm = _get_analysis_basis(K, M, v_star.device)
    # Cast once; the einsum runs in v_star.dtype.
    basis = basis.to(v_star.dtype)
    norm = norm.to(v_star.dtype)

    coeffs = torch.einsum("mk,bkd->bmd", basis, v_star)
    coeffs = coeffs * norm[None, :, None]

    if squeeze:
        coeffs = coeffs.squeeze(0)
    return coeffs


# Synthesis: the k-vector and norm depend only on (M, K_original, device).
_SYNTH_NORM_CACHE: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_synth_norm(
    M: int, K_original: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (M, K_original, str(device))
    cached = _SYNTH_NORM_CACHE.get(key)
    if cached is not None:
        return cached
    k = torch.arange(M + 1, dtype=torch.float32, device=device)
    norm = torch.full(
        (M + 1,), math.sqrt(2.0 / K_original), dtype=torch.float32, device=device
    )
    norm[0] *= _INV_SQRT2
    cached = (k, norm)
    _SYNTH_NORM_CACHE[key] = cached
    return cached


def decode_coeffs_to_velocity(
    coeffs: torch.Tensor,
    tau: torch.Tensor,
    T: float,
    K_original: int,
) -> torch.Tensor:
    """Decode DCT coefficients to velocities at arbitrary physical times."""
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

    M = M_plus_1 - 1
    k, norm = _get_synth_norm(M, K_original, coeffs.device)

    if tau.dim() == 1:
        tau_query = tau.unsqueeze(0)
    elif tau.dim() == 2:
        tau_query = tau
    else:
        raise ValueError(f"tau must be 1D (N,) or 2D (B, N), got shape {tau.shape}")

    j_indices = tau_query.to(torch.float32) * (K_original / T)
    # [B, M+1, N] basis — depends on tau, cannot be cached.
    basis = torch.cos(
        math.pi * k[None, :, None] * (j_indices[:, None, :] + 0.5) / K_original
    )

    # Do the contraction in fp32 for accuracy; cast back at the end.
    v = torch.einsum(
        "bkd,bkn,k->bnd",
        coeffs.to(torch.float32),
        basis,
        norm,
    )
    v = v.to(coeffs.dtype)

    if squeeze:
        v = v.squeeze(0)
    return v


# ---------------------------------------------------------------------------
# Noise sampling
# ---------------------------------------------------------------------------


def sample_freq_weighted_noise(
    M: int,
    d: int,
    T: float,
    B: int = 1,
    sigma: float = 1.0,
    key: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Frequency-weighted Gaussian noise of shape ``[B, M+1, d]``.

    Spectral envelope ``std_k = sigma / sqrt(1 + (k π / T)²)``. Uses a local
    ``torch.Generator`` when ``key`` is provided, so global RNG state is
    untouched.
    """
    device = torch.device(device) if device is not None else torch.device("cpu")
    k_idx = torch.arange(M + 1, dtype=torch.float32, device=device)
    omega_k = k_idx * (math.pi / T)
    std_k = sigma / torch.sqrt(1.0 + omega_k**2)  # [M+1]

    if key is not None:
        gen = torch.Generator(device=device).manual_seed(int(key))
        xi = torch.randn(B, M + 1, d, dtype=torch.float32, device=device, generator=gen)
    else:
        xi = torch.randn(B, M + 1, d, dtype=torch.float32, device=device)

    return xi * std_k[None, :, None]


# ---------------------------------------------------------------------------
# Dataset-driven M selection
# ---------------------------------------------------------------------------


def estimate_M_from_dataset(dataset, energy_threshold: float = 0.95) -> int:
    """Smallest ``M`` whose first ``M+1`` DCT coefficients capture
    ``energy_threshold`` of the mean cumulative spectral energy."""
    arrays: list[np.ndarray] = []
    for item in dataset:
        if isinstance(item, dict):
            v_star = np.asarray(item.get("actions", item.get("action", item)))
        else:
            v_star = np.asarray(item)
        if v_star.ndim != 2:
            raise ValueError(f"Expected 2D, got {v_star.shape}")
        arrays.append(v_star)

    if not arrays:
        raise ValueError("estimate_M_from_dataset received an empty dataset")

    K = arrays[0].shape[0]
    if any(a.shape[0] != K for a in arrays):
        raise ValueError("All dataset items must share the same time dimension")

    stacked = np.stack(arrays, axis=0)  # [N, K, d]
    # Orthonormal DCT along the time axis via the cached basis.
    basis = _get_dct_basis_f32(K, torch.device("cpu")).numpy()  # [K, K]
    coeffs = np.einsum("kj,njd->nkd", basis, stacked)  # [N, K, d]

    energy = (coeffs**2).sum(axis=-1)  # [N, K]
    total = energy.sum(axis=-1, keepdims=True)  # [N, 1]
    cumsum = np.cumsum(energy, axis=-1) / total  # [N, K]
    mean_cumsum = cumsum.mean(axis=0)  # [K]

    # First index where the mean cumulative energy reaches the threshold.
    idx = np.searchsorted(mean_cumsum, energy_threshold, side="left")
    if idx >= K:
        idx = K - 1
    M = max(1, int(idx))
    return M


# ---------------------------------------------------------------------------
# Matrix application / DCT / IDCT
# ---------------------------------------------------------------------------


def apply_matrix_along_dim(
    x: torch.Tensor, dim: int, mat: torch.Tensor
) -> torch.Tensor:
    """Apply ``mat`` along ``dim`` of ``x``; output dtype matches ``x``."""
    dim = dim % x.dim()
    moved = dim != x.dim() - 1
    if moved:
        x = x.movedim(dim, -1)
    out = x @ mat.to(x.device, dtype=x.dtype)
    if moved:
        out = out.movedim(-1, dim)
    return out


def dct(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Orthonormal DCT-II along ``dim`` (forward, ``D @ x``)."""
    n = x.shape[dim % x.dim()]
    basis = _get_dct_basis_f32(n, x.device)
    return apply_matrix_along_dim(x, dim, basis.t())


def idct(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Orthonormal inverse DCT-II along ``dim`` (``Dᵀ @ x``)."""
    n = x.shape[dim % x.dim()]
    basis = _get_dct_basis_f32(n, x.device)
    return apply_matrix_along_dim(x, dim, basis)


# ---------------------------------------------------------------------------
# Chunked reward discounting
# ---------------------------------------------------------------------------

_LOG_GAMMA_CACHE: dict[float, float] = {}


def discounted_chunk_reward(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Discounted sum of a chunk's rewards, then summed over the time axis."""
    n = rewards.shape[-1]
    log_gamma = _LOG_GAMMA_CACHE.get(gamma)
    if log_gamma is None:
        log_gamma = math.log(gamma)
        _LOG_GAMMA_CACHE[gamma] = log_gamma
    pos = torch.arange(n, dtype=torch.float32, device=rewards.device)
    weights = torch.exp(pos * log_gamma).to(rewards.dtype)
    return (rewards * weights).sum(dim=-1)
