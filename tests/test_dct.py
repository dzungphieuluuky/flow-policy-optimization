"""Unit tests for src/sjfpo/dct_utils.py.

``extract_dct_coeffs`` and ``decode_coeffs_to_velocity`` are the public DCT
entry points, so these tests cover the analysis/synthesis pair, the raw basis
matrix accessor, and frequency-weighted noise. The roundtrip, Parseval,
DC-identity, axis-generality, gradient, linearity, and discounted-reward tests
were removed together with the ``dct``/``idct``/
``discounted_chunk_reward`` helpers they exercised.

All checks are pure torch / fp32; scipy is deliberately NOT used anywhere
(runtime constraint: torch matmul against a precomputed basis only).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.dct_utils import (  # noqa: E402
    _get_analysis_basis,
    decode_coeffs_to_velocity,
    dct_basis_matrix,
    extract_dct_coeffs,
    sample_freq_weighted_noise,
)


def rand(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    """Seeded fp32 random tensor (deterministic across runs)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def test_public_basis_matrix_is_mutable_copy():
    first = dct_basis_matrix(8)
    first[0, 0] = 123.0
    second = dct_basis_matrix(8)
    assert second[0, 0].item() != 123.0


# ---------------------------------------------- FAFM coefficient/noise helpers


def test_decode_batched_coeffs_broadcasts_1d_tau():
    coeffs = rand((3, 4, 2), seed=21)
    tau = torch.tensor([0.0, 0.2, 0.7])
    actual = decode_coeffs_to_velocity(coeffs, tau, T=1.0, K_original=8)
    repeated = decode_coeffs_to_velocity(
        coeffs, tau.unsqueeze(0).expand(3, -1), T=1.0, K_original=8
    )
    assert actual.shape == (3, 3, 2)
    assert torch.equal(actual, repeated)


def test_decode_rejects_mismatched_2d_tau_batch():
    coeffs = rand((3, 4, 2), seed=22)
    with pytest.raises(ValueError, match="must match coeffs batch size"):
        decode_coeffs_to_velocity(
            coeffs, torch.zeros(2, 5), T=1.0, K_original=8
        )


def test_decode_rejects_invalid_query_time():
    with pytest.raises(ValueError, match=r"tau must lie in \[0, T\]"):
        decode_coeffs_to_velocity(
            torch.ones(2, 3), torch.tensor([-0.1, 0.5]), T=1.0, K_original=8
        )


def test_extract_coeffs_keeps_output_dtype_and_validates_modes():
    values = torch.randn(2, 8, 3, dtype=torch.float16)
    actual = extract_dct_coeffs(values, M=3)
    assert actual.shape == (2, 4, 3)
    assert actual.dtype == torch.float16
    with pytest.raises(ValueError, match="M must satisfy"):
        extract_dct_coeffs(values, M=8)


def test_extract_coeffs_full_window_matches_orthonormal_transform():
    """M = K-1 keeps every bin, so the result is the full orthonormal transform.

    This is the contract the state conditioner relies on now that ``dct`` is
    gone: it passes M = T-1 over a length-T observation window.
    """
    for T, obs in ((2, 3), (4, 11), (16, 17)):
        values = torch.randn(2, T, obs, generator=torch.Generator().manual_seed(T))
        coeffs = extract_dct_coeffs(values, M=T - 1)
        assert coeffs.shape == (2, T, obs)
        # The analysis basis is unnormalized; the returned norm vector makes
        # it orthonormal, so scaling by it must give an identity Gram matrix.
        basis, norm = _get_analysis_basis(T, T - 1, values.device)
        scaled = basis * norm[:, None]
        assert torch.allclose(scaled @ scaled.T, torch.eye(T), atol=1e-5)


def test_frequency_noise_dtype_and_key_are_deterministic():
    kwargs = dict(M=3, d=2, T=1.0, B=4, key=123, dtype=torch.float16)
    first = sample_freq_weighted_noise(**kwargs)
    second = sample_freq_weighted_noise(**kwargs)
    assert first.shape == (4, 4, 2)
    assert first.dtype == torch.float16
    assert torch.equal(first, second)

    state_before = torch.random.get_rng_state()
    sample_freq_weighted_noise(**kwargs)
    assert torch.equal(state_before, torch.random.get_rng_state())
