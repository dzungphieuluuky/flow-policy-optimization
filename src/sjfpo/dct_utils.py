"""DCT-based Fourier coefficient utilities for FAFM.

Performance notes
-----------------
* DCT bases are bounded LRU caches keyed by ``(shape, device)`` and stored in
  fp32. Low-precision DCT/IDCT contractions run in fp32 before casting back.
* ``extract_dct_coeffs`` caches its ``[M+1, K]`` analysis basis and the
  ``[M+1]`` normalization vector.
* ``decode_coeffs_to_velocity`` shares its cached normalization vector across
  calls and performs the coefficient contraction in fp32.
* ``sample_freq_weighted_noise`` uses a local ``torch.Generator`` when ``key``
  is provided; otherwise it samples from the global RNG.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from sjfpo.utils import resolve_storage_dtype

# sqrt(0.5), used to scale the DC row so each basis is orthonormal.
_INV_SQRT2 = math.sqrt(0.5)


def _normalize_device(device: torch.device | str | None) -> torch.device:
    """Resolve a device argument to a concrete device with an index.

    A bare ``"cuda"`` is pinned to the current device so cached tensors are
    never keyed to an ambiguous device.

    Args:
        device: Device, device string, or None for CPU.

    Returns:
        A device with a resolved index.
    """
    target = torch.device("cpu") if device is None else torch.device(device)
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    return target


def _dct_norm(count: int, sequence_length: int, device: torch.device) -> torch.Tensor:
    """Build the orthonormal DCT-II normalization for a frequency count.

    Args:
        count: Number of frequency bins.
        sequence_length: Length of the sequence being transformed.
        device: Device the tensor is allocated on.

    Returns:
        Per-bin normalization vector of shape ``[count]``.

    Raises:
        ValueError: If either argument is below one.
    """
    if count < 1 or sequence_length < 1:
        raise ValueError("count and sequence_length must be positive")
    norm = torch.full(
        (count,), math.sqrt(2.0 / sequence_length), dtype=torch.float32, device=device
    )
    norm[0] *= _INV_SQRT2
    return norm


def _build_dct_basis_f32(n: int, device: torch.device) -> torch.Tensor:
    """Orthonormal DCT-II matrix ``D[k, j]`` of shape ``[n, n]``, fp32.

    ``D[k, j] = c_k · cos(π k (2j+1) / (2n))`` with ``c_0 = sqrt(1/n)`` and
    ``c_k = sqrt(2/n)`` for ``k > 0``. Forward DCT is ``D @ x``; inverse is
    ``Dᵀ @ x``.

    Args:
        n: Sequence length; the basis is square.
        device: Device the basis is allocated on.

    Returns:
        The ``[n, n]`` fp32 basis matrix.

    Raises:
        ValueError: If ``n`` is below one.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    k = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)
    t = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(0)
    norm = _dct_norm(n, n, device)
    return norm[:, None] * torch.cos(
        math.pi * (2.0 * t + 1.0) * k / (2.0 * n)
    )


@lru_cache(maxsize=32)
def _get_dct_basis_f32(n: int, device: torch.device) -> torch.Tensor:
    """Return the cached fp32 DCT-II basis for a length and device.

    Args:
        n: Sequence length.
        device: Device the basis is allocated on.

    Returns:
        The cached ``[n, n]`` basis matrix. Callers must copy before mutating.
    """
    return _build_dct_basis_f32(n, device)


def dct_basis_matrix(
    n: int, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return a mutable copy of the orthonormal DCT-II basis matrix.

    The returned copy uses the device's resolved storage dtype.

    Args:
        n: Sequence length.
        device: Device the basis is allocated on; resolved when omitted.

    Returns:
        A fresh ``[n, n]`` basis matrix the caller may modify in place.

    Raises:
        ValueError: If ``n`` is below one.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    target = _normalize_device(device)
    return _get_dct_basis_f32(n, target).to(
        target, dtype=resolve_storage_dtype(target), copy=True
    )


@lru_cache(maxsize=64)
def _get_analysis_basis(
    K: int, M: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the cached analysis basis and its normalization.

    Args:
        K: Full sequence length.
        M: Highest retained frequency; ``M+1`` bins are produced.
        device: Device the tensors are allocated on.

    Returns:
        The ``[M+1, K]`` basis matrix and the ``[M+1]`` normalization vector.

    Raises:
        ValueError: If ``K`` is below one, or if ``M`` is outside ``[0, K)``.
    """
    if K < 1:
        raise ValueError(f"K must be positive, got {K}")
    if M < 0 or M >= K:
        raise ValueError(f"M must satisfy 0 <= M < K, got M={M}, K={K}")
    j = torch.arange(K, dtype=torch.float32, device=device)
    k = torch.arange(M + 1, dtype=torch.float32, device=device)
    basis = torch.cos(math.pi * k[:, None] * (j[None, :] + 0.5) / K)
    return basis, _dct_norm(M + 1, K, device)


def extract_dct_coeffs(v_star: torch.Tensor, M: int) -> torch.Tensor:
    """Extract the first ``M+1`` DCT-II coefficients of a velocity sequence.

    ``v_star`` is either ``[K, d]`` or ``[B, K, d]``; returns ``[M+1, d]``
    or ``[B, M+1, d]`` respectively. The contraction accumulates in fp32 even
    under autocast, then casts back to the input dtype.

    Args:
        v_star: Velocity sequence or batch of sequences.
        M: Highest frequency retained.

    Returns:
        Coefficients of shape ``[M+1, d]``, or ``[B, M+1, d]`` for a batch.

    Raises:
        ValueError: If the rank is not 2 or 3, or if the batch or feature
            dimension is not positive.
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

    basis, norm = _get_analysis_basis(K, M, _normalize_device(v_star.device))
    with torch.autocast(device_type=v_star.device.type, enabled=False):
        coeffs = torch.einsum("mk,bkd->bmd", basis, v_star.float())
    coeffs = (coeffs * norm[None, :, None]).to(v_star.dtype)

    if squeeze:
        coeffs = coeffs.squeeze(0)
    return coeffs


@lru_cache(maxsize=64)
def _get_synth_norm(M: int, K_original: int, device: torch.device) -> torch.Tensor:
    """Return the cached synthesis normalization.

    The synthesis norm depends only on the retained frequency count and the
    original sequence length, not on any query times.

    Args:
        M: Highest retained frequency.
        K_original: Length of the sequence the coefficients came from.
        device: Device the tensor is allocated on.

    Returns:
        The ``[M+1]`` normalization vector.
    """
    return _dct_norm(M + 1, K_original, device)


def decode_coeffs_to_velocity(
    coeffs: torch.Tensor,
    tau: torch.Tensor,
    T: float,
    K_original: int,
) -> torch.Tensor:
    """Decode DCT coefficients to velocities at arbitrary physical times.

    ``coeffs`` is ``[M+1, d]`` or ``[B, M+1, d]``. A 1D ``tau`` with shape
    ``[N]`` is shared across the coefficient batch; a 2D ``tau`` with shape
    ``[B, N]`` must use the same batch size as ``coeffs``. The original K
    samples use the periodic grid ``t_j = j*T/K``; queries use the same
    continuous convention and must lie in ``[0, T]``. Because those samples
    sit on the periodic half-open grid ``j*T/K`` rather than an
    endpoint-inclusive ``j*T/(K-1)`` grid, a continuous query ``tau`` maps to
    ``j = tau*K/T``. The contraction runs in fp32 even under autocast, because
    CUDA would otherwise lower it to bf16.

    Args:
        coeffs: Coefficients of shape ``[M+1, d]`` or ``[B, M+1, d]``.
        tau: Query times, shape ``[N]`` shared across the batch or ``[B, N]``
            per batch element.
        T: Physical duration of the chunk.
        K_original: Length of the sequence the coefficients came from.

    Returns:
        Velocities at the query times, shaped ``[N, d]`` or ``[B, N, d]``.

    Raises:
        ValueError: If the ranks disagree, if ``T`` is not finite and
            positive, if ``tau`` does not match the batch, or if any query
            time falls outside ``[0, T]``.
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
    if not math.isfinite(T) or T <= 0:
        raise ValueError(f"T must be finite and positive, got {T}")

    M = M_plus_1 - 1
    device = _normalize_device(coeffs.device)
    norm = _get_synth_norm(M, K_original, device)
    k = torch.arange(M + 1, dtype=torch.float32, device=device)

    if tau.dim() == 1:
        tau_query = tau.to(device=device, dtype=torch.float32).unsqueeze(0)
        tau_query = tau_query.expand(B, -1)
    elif tau.dim() == 2:
        if tau.shape[0] != B:
            raise ValueError(
                f"tau batch size {tau.shape[0]} must match coeffs batch size {B}"
            )
        tau_query = tau.to(device=device, dtype=torch.float32)
    else:
        raise ValueError(f"tau must be 1D (N,) or 2D (B, N), got shape {tau.shape}")

    if tau_query.numel() == 0:
        raise ValueError("tau must contain at least one query time")
    if not bool(torch.all((tau_query >= 0) & (tau_query <= T))):
        raise ValueError("tau must lie in [0, T]")

    j_indices = tau_query * (K_original / T)
    basis = torch.cos(
        math.pi * k[None, :, None] * (j_indices[:, None, :] + 0.5) / K_original
    )

    with torch.autocast(device_type=device.type, enabled=False):
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


def sample_freq_weighted_noise(
    M: int,
    d: int,
    T: float,
    B: int = 1,
    sigma: float = 1.0,
    key: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Frequency-weighted Gaussian noise of shape ``[B, M+1, d]``.

    Spectral envelope ``std_k = sigma / sqrt(1 + (k π / T)²)``. Uses a local
    ``torch.Generator`` when ``key`` is provided, leaving global RNG state
    untouched; otherwise it samples from the global RNG. Reproducibility with
    a key is guaranteed on the same device, not across CPU/CUDA generators.

    Args:
        M: Highest frequency; ``M+1`` bins are sampled.
        d: Feature dimension, one per action dimension.
        T: Physical duration of the chunk, setting the spectral envelope.
        B: Batch size.
        sigma: Scale of the frequency-0 component.
        key: Optional seed for a local generator; the global RNG is used when
            omitted.
        device: Device the noise is allocated on; resolved when omitted.
        dtype: Output dtype; the device's storage dtype when omitted.

    Returns:
        Frequency-weighted noise of shape ``[B, M+1, d]``.

    Raises:
        ValueError: If ``M`` is negative, if ``d`` or ``B`` is not positive,
            or if ``T`` is not finite and positive.
    """
    if M < 0 or d < 1 or B < 1:
        raise ValueError("M must be nonnegative; d and B must be positive")
    if not math.isfinite(T) or T <= 0:
        raise ValueError(f"T must be finite and positive, got {T}")
    device = _normalize_device(device)
    out_dtype = resolve_storage_dtype(device) if dtype is None else dtype
    k_idx = torch.arange(M + 1, dtype=torch.float32, device=device)
    omega_k = k_idx * (math.pi / T)
    std_k = sigma / torch.sqrt(1.0 + omega_k**2)

    if key is not None:
        gen = torch.Generator(device=device).manual_seed(int(key))
        xi = torch.randn(
            B, M + 1, d, dtype=out_dtype, device=device, generator=gen
        )
    else:
        xi = torch.randn(B, M + 1, d, dtype=out_dtype, device=device)

    return (xi.float() * std_k[None, :, None]).to(out_dtype)


def apply_matrix_along_dim(
    x: torch.Tensor, dim: int, mat: torch.Tensor
) -> torch.Tensor:
    """Apply a 2D matrix along one dimension, preserving the input dtype.

    Half and bfloat16 contractions run in fp32 for accuracy; float64 input
    stays in float64. Autocast is disabled for the same reason.

    Args:
        x: Input tensor.
        dim: Dimension to contract over.
        mat: Matrix whose first dimension matches ``x.shape[dim]``.

    Returns:
        The contracted tensor, cast back to ``x``'s dtype.

    Raises:
        ValueError: If ``mat`` is not 2D, or if its first dimension does not
            match ``x.shape[dim]``.
    """
    dim = dim % x.dim()
    moved = dim != x.dim() - 1
    if moved:
        x = x.movedim(dim, -1)
    if mat.ndim != 2:
        raise ValueError(f"mat must be 2D, got shape {mat.shape}")
    if mat.shape[0] != x.shape[-1]:
        raise ValueError(
            f"mat.shape[0] must match x.shape[{dim}]={x.shape[-1]}, got {mat.shape[0]}"
        )

    compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    x_compute = x if x.dtype == compute_dtype else x.to(compute_dtype)
    if mat.device != x_compute.device or mat.dtype != compute_dtype:
        mat = mat.to(x_compute.device, dtype=compute_dtype)
    with torch.autocast(device_type=x_compute.device.type, enabled=False):
        out = x_compute @ mat
    if moved:
        out = out.movedim(-1, dim)
    return out.to(x.dtype)


def dct(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply the orthonormal forward DCT-II along a dimension.

    Args:
        x: Input tensor.
        dim: Dimension holding the sequence.

    Returns:
        The transformed tensor, keeping the input dtype.

    Raises:
        ValueError: If ``x`` has no dimensions, or the sequence length is not
            positive.
    """
    if x.dim() < 1:
        raise ValueError("DCT input must have at least one dimension")
    n = x.shape[dim % x.dim()]
    if n < 1:
        raise ValueError(f"DCT length must be positive, got {n}")
    basis = _get_dct_basis_f32(n, x.device)
    return apply_matrix_along_dim(x, dim, basis.t())


def idct(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply the orthonormal inverse DCT-II along a dimension.

    Args:
        x: Frequency-domain tensor.
        dim: Dimension holding the spectrum.

    Returns:
        The reconstructed tensor, keeping the input dtype.

    Raises:
        ValueError: If ``x`` has no dimensions, or the spectrum length is not
            positive.
    """
    if x.dim() < 1:
        raise ValueError("DCT input must have at least one dimension")
    n = x.shape[dim % x.dim()]
    if n < 1:
        raise ValueError(f"DCT length must be positive, got {n}")
    basis = _get_dct_basis_f32(n, x.device)
    return apply_matrix_along_dim(x, dim, basis)


@lru_cache(maxsize=128)
def _discount_weights(n: int, gamma: float, device: torch.device) -> torch.Tensor:
    """Return the cached geometric discount weights for a chunk length.

    Args:
        n: Chunk length.
        gamma: Per-step discount factor in ``[0, 1]``.
        device: Device the tensor is allocated on.

    Returns:
        Weights of shape ``[n]``, with the degenerate cases ``gamma == 0`` and
        ``gamma == 1`` handled exactly. RL chunks sit far below ``2**24``,
        where an fp32 ``arange`` is still exact.

    Raises:
        ValueError: If ``gamma`` falls outside ``[0, 1]``.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if n == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    if gamma == 0.0:
        weights = torch.zeros(n, dtype=torch.float32, device=device)
        weights[0] = 1.0
        return weights
    if gamma == 1.0:
        return torch.ones(n, dtype=torch.float32, device=device)
    pos = torch.arange(n, dtype=torch.float32, device=device)
    return torch.exp(pos * math.log(gamma))


def discounted_chunk_reward(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Discount a chunk's rewards along the time axis and sum them.

    Args:
        rewards: Per-step rewards, discounted over the last axis.
        gamma: Per-step discount factor in ``[0, 1]``.

    Returns:
        Discounted chunk rewards, with the time axis summed out.
    """
    weights = _discount_weights(
        rewards.shape[-1], float(gamma), _normalize_device(rewards.device)
    ).to(rewards.dtype)
    return (rewards * weights).sum(dim=-1)
