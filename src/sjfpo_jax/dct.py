"""Orthonormal DCT-II / DCT-III utilities (pure ``jnp`` matmuls).

Same basis as ``sjfpo.dct_utils``: ``M[k, n] = sqrt(2/N)·c_k·cos(...)``,
``dct(x) = x @ Mᵀ``, ``idct(y) = y @ M``. Basis cached host-side per ``n``.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

__all__ = ["dct", "dct_basis_matrix", "discounted_chunk_reward", "idct"]

BASIS_CACHE: dict[int, jnp.ndarray] = {}


def dct_basis_matrix(n: int) -> jnp.ndarray:
    """[n, n] orthonormal DCT-II basis M (MᵀM = I), fp32."""
    m = BASIS_CACHE.get(n)
    if m is None:
        k = jnp.arange(n, dtype=jnp.float32).reshape(n, 1)
        t = jnp.arange(n, dtype=jnp.float32).reshape(1, n)
        m = math.sqrt(2.0 / n) * jnp.cos(math.pi * (2.0 * t + 1.0) * k / (2.0 * n))
        m = m.at[0].multiply(1.0 / math.sqrt(2.0))
        BASIS_CACHE[n] = m
    return m


def _apply_matrix_along_dim(x: jnp.ndarray, dim: int, mat: jnp.ndarray) -> jnp.ndarray:
    dim = dim % x.ndim
    if dim != x.ndim - 1:
        x = jnp.moveaxis(x, dim, -1)
    out = x @ mat
    if dim != out.ndim - 1:
        out = jnp.moveaxis(out, -1, dim)
    return out


def dct(x: jnp.ndarray, dim: int = -1) -> jnp.ndarray:
    return _apply_matrix_along_dim(x, dim, dct_basis_matrix(x.shape[dim % x.ndim]).T)


def idct(x: jnp.ndarray, dim: int = -1) -> jnp.ndarray:
    """Inverse (DCT-III) along ``dim``: matmul with M (M⁻¹ = Mᵀ)."""
    return _apply_matrix_along_dim(x, dim, dct_basis_matrix(x.shape[dim % x.ndim]))


def discounted_chunk_reward(rewards: jnp.ndarray, gamma: float) -> jnp.ndarray:
    n = rewards.shape[-1]
    weights = jnp.power(gamma, jnp.arange(n, dtype=rewards.dtype))
    return (rewards * weights).sum(axis=-1)
