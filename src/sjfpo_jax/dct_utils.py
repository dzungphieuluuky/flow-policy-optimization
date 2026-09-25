"""DCT-II utilities for the JAX SJFPO port.

Basis matrices are constructed from concrete dimensions and remain pure JAX
operations, so they are safe to call inside transformed functions.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from sjfpo_jax.utils import resolve_storage_dtype

STORAGE_DTYPE = resolve_storage_dtype()
_INV_SQRT2 = math.sqrt(0.5)


# ---------------------------------------------------------------------------
# Orthonormal DCT-II basis
# ---------------------------------------------------------------------------


def _build_dct_basis_f32(n: int) -> jnp.ndarray:
    """Orthonormal DCT-II matrix ``D[k, j]`` of shape ``[n, n]``, fp32."""
    k = jnp.arange(n, dtype=jnp.float32)[:, None]
    t = jnp.arange(n, dtype=jnp.float32)[None, :]
    m = math.sqrt(2.0 / n) * jnp.cos(math.pi * (2.0 * t + 1.0) * k / (2.0 * n))
    return m.at[0].mul(_INV_SQRT2)


def _dct_basis_f32(n: int) -> jnp.ndarray:
    """Concrete-``n`` getter for the orthonormal DCT-II matrix."""
    return _build_dct_basis_f32(n)


def dct_basis_matrix(n: int) -> jnp.ndarray:
    return _dct_basis_f32(n).astype(STORAGE_DTYPE)

# ---------------------------------------------------------------------------
# Analysis / synthesis bases for FAFM
# ---------------------------------------------------------------------------


def _analysis_basis(K: int, M: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    j = jnp.arange(K, dtype=jnp.float32)
    k = jnp.arange(M + 1, dtype=jnp.float32)
    basis = jnp.cos(math.pi * k[:, None] * (j[None, :] + 0.5) / K)
    norm = jnp.full((M + 1,), math.sqrt(2.0 / K), dtype=jnp.float32)
    norm = norm.at[0].mul(_INV_SQRT2)
    return basis, norm


def extract_dct_coeffs(v_star, M: int) -> jnp.ndarray:
    """First ``M+1`` DCT-II coefficients of ``[K,d]`` or ``[B,K,d]``."""
    squeeze = False
    if v_star.ndim == 3:
        B, K, d = v_star.shape
    elif v_star.ndim == 2:
        K, d = v_star.shape
        v_star = v_star[None]
        squeeze = True
    else:
        raise ValueError(
            f"v_star must be 2D (K, d) or 3D (B, K, d), got shape {v_star.shape}"
        )
    basis, norm = _analysis_basis(K, M)
    basis = basis.astype(v_star.dtype)
    norm = norm.astype(v_star.dtype)
    coeffs = jnp.einsum("mk,bkd->bmd", basis, v_star) * norm[None, :, None]
    return coeffs[0] if squeeze else coeffs


def _synth_norm(M: int, K_original: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    k = jnp.arange(M + 1, dtype=jnp.float32)
    norm = jnp.full((M + 1,), math.sqrt(2.0 / K_original), dtype=jnp.float32)
    norm = norm.at[0].mul(_INV_SQRT2)
    return k, norm


def decode_coeffs_to_velocity(coeffs, tau, T: float, K_original: int) -> jnp.ndarray:
    """Decode DCT coeffs at arbitrary physical times."""
    squeeze = False
    if coeffs.ndim == 3:
        B, M_plus_1, d = coeffs.shape
    elif coeffs.ndim == 2:
        M_plus_1, d = coeffs.shape
        coeffs = coeffs[None]
        squeeze = True
    else:
        raise ValueError(f"coeffs must be 2D or 3D, got {coeffs.shape}")

    M = M_plus_1 - 1
    k, norm = _synth_norm(M, K_original)

    if tau.ndim == 1:
        tau_q = tau[None]
    elif tau.ndim == 2:
        tau_q = tau
    else:
        raise ValueError(f"tau must be 1D or 2D, got {tau.shape}")

    j_idx = tau_q.astype(jnp.float32) * (K_original / T)
    basis = jnp.cos(
        math.pi * k[None, :, None] * (j_idx[:, None, :] + 0.5) / K_original
    )
    v = jnp.einsum(
        "bkd,bkn,k->bnd", coeffs.astype(jnp.float32), basis, norm
    ).astype(coeffs.dtype)
    return v[0] if squeeze else v


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


def sample_freq_weighted_noise(
    M: int, d: int, T: float, B: int = 1, sigma: float = 1.0,
    key=None, dtype=jnp.float32,
) -> jnp.ndarray:
    """``[B, M+1, d]`` with envelope ``sigma/sqrt(1+(kπ/T)^2)``."""
    if key is None:
        key = jax.random.PRNGKey(0)
    k_idx = jnp.arange(M + 1, dtype=jnp.float32)
    std_k = sigma / jnp.sqrt(1.0 + (k_idx * (math.pi / T)) ** 2)
    xi = jax.random.normal(key, (B, M + 1, d), dtype=dtype)
    return xi * std_k[None, :, None]


# ---------------------------------------------------------------------------
# Matrix application / DCT / IDCT
# ---------------------------------------------------------------------------


def apply_matrix_along_dim(x: jnp.ndarray, dim: int, mat: jnp.ndarray) -> jnp.ndarray:
    dim = dim % x.ndim
    moved = dim != x.ndim - 1
    if moved:
        x = jnp.moveaxis(x, dim, -1)
    out = x @ mat.astype(x.dtype)
    if moved:
        out = jnp.moveaxis(out, -1, dim)
    return out


def dct(x: jnp.ndarray, dim: int = -1) -> jnp.ndarray:
    """Orthonormal DCT-II along ``dim`` (forward, ``D @ x``)."""
    n = x.shape[dim % x.ndim]
    return apply_matrix_along_dim(x, dim, _dct_basis_f32(n).T)


def idct(x: jnp.ndarray, dim: int = -1) -> jnp.ndarray:
    """Orthonormal inverse DCT-II along ``dim`` (``Dᵀ @ x``)."""
    n = x.shape[dim % x.ndim]
    return apply_matrix_along_dim(x, dim, _dct_basis_f32(n))


# ---------------------------------------------------------------------------
# Chunked reward discounting
# ---------------------------------------------------------------------------


_LOG_GAMMA_CACHE: dict[float, float] = {}


def discounted_chunk_reward(rewards: jnp.ndarray, gamma: float) -> jnp.ndarray:
    n = rewards.shape[-1]
    log_gamma = _LOG_GAMMA_CACHE.get(gamma)
    if log_gamma is None:
        log_gamma = math.log(gamma)
        _LOG_GAMMA_CACHE[gamma] = log_gamma
    pos = jnp.arange(n, dtype=jnp.float32)
    weights = jnp.exp(pos * log_gamma).astype(rewards.dtype)
    return (rewards * weights).sum(-1)
