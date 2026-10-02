"""Orthonormal DCT-II and its inverse in JAX, mirroring ``dct_utils.py``.

``get_analysis_basis`` keeps no cache, unlike torch: a Python-int-keyed cache over
``jnp`` values stores a tracer from whichever trace filled it first, and the next
trace to read it fails. Inside ``jit`` the basis is a constant fold anyway.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def compute_dct_norm(count: int, sequence_length: int) -> jax.Array:
    """``[count]`` DCT-II scaling: k=0 gets sqrt(1/K), the rest sqrt(2/K)."""
    if count < 1 or sequence_length < 1:
        raise ValueError("count and sequence_length must be positive")
    norm = jnp.full((count,), math.sqrt(2.0 / sequence_length), dtype=jnp.float32)
    return norm.at[0].multiply(1.0 / math.sqrt(2.0))


def get_analysis_basis(K: int, num_modes: int) -> tuple[jax.Array, jax.Array]:
    """``[num_modes, K]`` analysis basis and its ``[num_modes]`` norm.

    Deliberately **not** memoised, unlike the torch original. An ``lru_cache``
    keyed on Python ints stores whichever trace populated it first, and a tracer
    handed to a later trace is rejected as an unexpected tracer. ``K`` and
    ``num_modes`` are static shapes, so inside ``jit`` XLA folds this to a constant
    anyway and the cache bought nothing.
    """
    if K < 1:
        raise ValueError(f"K must be positive, got {K}")
    if not 1 <= num_modes <= K:
        raise ValueError(
            f"num_modes must satisfy 1 <= num_modes <= K, "
            f"got num_modes={num_modes}, K={K}"
        )
    j = jnp.arange(K, dtype=jnp.float32)
    k = jnp.arange(num_modes, dtype=jnp.float32)
    basis = jnp.cos(math.pi * k[:, None] * (j[None, :] + 0.5) / K)
    return basis, compute_dct_norm(num_modes, K)


def extract_dct_coeffs(v_star: jax.Array, num_modes: int) -> jax.Array:
    """Extract the first ``num_modes`` DCT-II coefficients of a velocity sequence.

    ``num_modes`` is a count, not a highest index, so ``num_modes == K`` keeps
    every frequency and the transform is the full orthonormal one. Accepts 2D
    ``(K, d)`` or 3D ``(B, K, d)`` and squeezes the batch off on the way out.
    """
    squeeze = False
    if v_star.ndim == 3:
        b = v_star.shape[0]
        if b < 1:
            raise ValueError(f"v_star batch size must be positive, got {b}")
    elif v_star.ndim == 2:
        v_star = v_star[None]
        squeeze = True
    else:
        raise ValueError(
            f"v_star must be 2D (K, d) or 3D (B, K, d), got shape {v_star.shape}"
        )
    if v_star.shape[-1] < 1:
        raise ValueError(
            f"v_star feature dimension must be positive, got {v_star.shape[-1]}"
        )
    K, num_kept = v_star.shape[1], num_modes
    basis, norm = get_analysis_basis(K, num_kept)
    coeffs = jnp.einsum("mk,bkd->bmd", basis, v_star.astype(jnp.float32))
    coeffs = coeffs * norm[None, :, None]
    return coeffs[0] if squeeze else coeffs


def compute_spectral_energy_retained(v: jax.Array, num_modes: int) -> jax.Array:
    """Fraction of a chunk's energy carried by the kept ``num_modes``.

    The analysis basis is orthonormal, so by Parseval the coefficient energy
    equals the signal energy and this is 1.0 at ``num_modes == v.shape[1]``.
    Below that it is the share of the chunk the truncation actually discards,
    which is the number that says whether a smaller ``num_kept_tokens`` is
    doing anything.
    """
    coeffs = extract_dct_coeffs(v, num_modes=num_modes)
    # square().sum() rather than an rms: this is a Parseval share, so the
    # divisor must be the total energy, not a per-element average.
    return jnp.sum(jnp.square(coeffs)) / jnp.clip(jnp.sum(jnp.square(v)), min=1e-12)


def velocity_to_action_steps(vel: jax.Array, num_rows: int) -> jax.Array:
    """Expand DCT coeffs back onto the half-open grid ``t_j = j*H/H``.

    Takes any leading batch axes, so ``[n, mc, K, d]`` and ``[n, K, d]`` both
    work. Inverse of :func:`extract_dct_coeffs` for a full-length signal: read
    coeffs as time samples, velocity in row space, so a per-row execution mask
    can index it.
    """
    lead, k, d = vel.shape[:-2], vel.shape[-2], vel.shape[-1]
    tau = jnp.arange(num_rows, dtype=jnp.float32) / num_rows
    rows = decode_coeffs_to_velocity(
        vel.reshape(-1, k, d).astype(jnp.float32), tau, 1.0, num_rows
    )
    return rows.reshape(*lead, num_rows, d)


def check_query_range(tau, T):
    """Every query time lies in ``[0, T]``.

    A no-op inside ``jit``: the comparison is a tracer there, and forcing it to a
    Python bool would break the program. Shape checks stay unconditional, because
    shapes are static under tracing.
    """
    in_range = jnp.all((tau >= 0) & (tau <= T))
    if isinstance(in_range, jax.core.Tracer):
        return
    if not bool(in_range):
        raise ValueError("tau must lie in [0, T]")


def decode_coeffs_to_velocity(
    coeffs: jax.Array, tau: jax.Array, T: float, K_original: int
) -> jax.Array:
    """Decode DCT coeffs at query times.

    Times use ``t_j = j*T/K``, so a query maps to ``j = tau*K/T``. ``tau = T``
    gives the DCT-II even-symmetric extension (mirror of the last sample), not
    a periodic wrap. ``tau`` is 1D ``(N,)`` or 2D ``(B, N)``.
    """
    squeeze = False
    if coeffs.ndim == 3:
        b = coeffs.shape[0]
    elif coeffs.ndim == 2:
        coeffs = coeffs[None]
        b = 1
        squeeze = True
    else:
        raise ValueError(
            f"coeffs must be 2D (num_modes, d) or 3D (B, num_modes, d), "
            f"got shape {coeffs.shape}"
        )

    if b < 1 or coeffs.shape[1] < 1 or coeffs.shape[2] < 1:
        raise ValueError(
            "coeffs must have positive batch, frequency, and feature dimensions"
        )
    if K_original < 1:
        raise ValueError(f"K_original must be positive, got {K_original}")
    if coeffs.shape[1] > K_original:
        raise ValueError(
            f"number of coefficients {coeffs.shape[1]} cannot exceed "
            f"K_original={K_original}"
        )
    if not math.isfinite(T) or T <= 0:
        raise ValueError(f"T must be finite and positive, got {T}")

    num_modes = coeffs.shape[1]
    norm = compute_dct_norm(num_modes, K_original)
    k = jnp.arange(num_modes, dtype=jnp.float32)
    tau_q = tau.astype(jnp.float32)
    if tau_q.size == 0:
        raise ValueError("tau must contain at least one query time")
    check_query_range(tau_q, T)
    j = tau_q * (K_original / T)

    if tau.ndim == 1:
        basis = jnp.cos(math.pi * k[:, None] * (j[None, :] + 0.5) / K_original)
        v = jnp.einsum("bkd,kn,k->bnd", coeffs.astype(jnp.float32), basis, norm)
    elif tau.ndim == 2:
        if tau.shape[0] != b:
            raise ValueError(
                f"tau batch size {tau.shape[0]} must match coeffs batch size {b}"
            )
        basis = jnp.cos(math.pi * k[None, :, None] * (j[:, None, :] + 0.5) / K_original)
        v = jnp.einsum("bkd,bkn,k->bnd", coeffs.astype(jnp.float32), basis, norm)
    else:
        raise ValueError(f"tau must be 1D (N,) or 2D (B, N), got shape {tau.shape}")

    return v[0] if squeeze else v
