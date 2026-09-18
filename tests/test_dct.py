"""Unit tests for src/sjfpo/dct_utils.py.

The whole FPO++ flow pipeline operates in ORTHONORMAL DCT-II coefficient
space of the scaled action chunk (plan Scope §Flow space, BINDING). An
orthonormal transform preserves the Euclidean measure, so a Gaussian drawn
in coefficient space maps to a Gaussian in action space and log-densities
stay consistent (log-det of an orthogonal map is 0). These tests pin that
property via Parseval energy preservation and the DC-coefficient identity.

All checks are pure torch / fp32; scipy is deliberately NOT used anywhere
(runtime constraint: torch matmul against a precomputed basis only).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.dct_utils import (  # noqa: E402
    dct,
    discounted_chunk_reward,
    idct,
)

# Chunk length H=8 per plan; A_dim varies. Both plan-mandated shapes.
SHAPES = [(4, 8, 3), (7, 8, 11)]
CHUNK_DIM = 1


def rand(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    """Seeded fp32 random tensor (deterministic across runs)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32)

    # ---------------------------------------------------------------- (a) roundtrip


@pytest.mark.parametrize("shape", SHAPES)
def test_roundtrip_random_middle_chunk_axis(shape):
    """idct(dct(x)) reconstructs x along the middle (chunk) axis, ‖·‖∞<1e-5."""
    x = rand(shape)
    recon = idct(dct(x, dim=CHUNK_DIM), dim=CHUNK_DIM)
    err = (recon - x).abs().max().item()
    assert err < 1e-5, f"roundtrip max-abs err {err:.3e} for shape {shape}"


def test_roundtrip_random_last_axis():
    """Same guarantee with the default (last) axis on shape [7,8,11]."""
    x = rand((7, 8, 11))
    recon = idct(dct(x))
    err = (recon - x).abs().max().item()
    assert err < 1e-5, f"roundtrip max-abs err {err:.3e}"


def test_roundtrip_adversarial_constant_and_alternating():
    """Adversarial inputs: constant chunks and alternating ±1 chunks still
    reconstruct exactly (DC-only and Nyquist-only spectra)."""
    x = torch.zeros(4, 8, 3, dtype=torch.float32)
    x[:, :, 0] = 5.0  # constant chunks
    x[:, :, 1] = torch.tensor(
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    )  # ±1 alternating
    x[:, :, 2] = torch.tensor([-2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0])
    recon = idct(dct(x, dim=CHUNK_DIM), dim=CHUNK_DIM)
    err = (recon - x).abs().max().item()
    assert err < 1e-5, f"adversarial roundtrip max-abs err {err:.3e}"

    # ------------------------------------------------------- (b) Parseval / energy


@pytest.mark.parametrize("shape", SHAPES)
def test_parseval_energy_preserved(shape):
    """Orthonormal scaling ⇒ sum(coeffs²) == sum(x²), relative error <1e-5."""
    x = rand(shape)
    c = dct(x, dim=CHUNK_DIM)
    e_time = (x**2).sum().item()
    e_freq = (c**2).sum().item()
    rel_err = abs(e_freq - e_time) / e_time
    assert rel_err < 1e-5, f"Parseval relative energy error {rel_err:.3e}"

    # ------------------------------------------------------------ (c) DC identity


def test_dc_coefficient_is_mean_times_sqrt_n():
    """Orthonormal DCT-II: coeff[...,0,:] == mean(chunk axis) * sqrt(H).

    Row k=0 of the orthonormal basis is 1/sqrt(N) everywhere, so the DC
    coefficient equals sum(x)/sqrt(N) = mean(x)*sqrt(N). Verified against
    direct computation (<1e-6) — no scipy dependency.
    """
    x = rand((5, 8, 4), seed=7)
    c = dct(x, dim=CHUNK_DIM)
    expected = x.mean(dim=CHUNK_DIM) * math.sqrt(x.shape[CHUNK_DIM])
    err = (c.select(CHUNK_DIM, 0) - expected).abs().max().item()
    assert err < 1e-6, f"DC coefficient max-abs err {err:.3e}"


def test_dc_coefficient_hand_computed_1d():
    """Plan fallback case: hand-computed 1D input [1..8].

    mean = 4.5, sqrt(8) ≈ 2.8284271 ⇒ DC ≈ 12.7279221.
    """
    x = torch.tensor([float(i) for i in range(1, 9)])
    dc = dct(x)[0].item()
    expected = 4.5 * math.sqrt(8.0)
    assert abs(dc - expected) < 1e-6, f"DC {dc!r} vs hand-computed {expected!r}"

    # --------------------------------------------------------- (d) axis generality


def test_transform_works_on_last_and_middle_axes():
    """Axis argument generality: roundtrip holds for BOTH last and middle dims
    on the same tensor, and results differ from each other (real transforms,
    not accidental identity)."""
    x = rand((3, 8, 6), seed=3)

    last = idct(dct(x, dim=-1), dim=-1)
    mid = idct(dct(x, dim=1), dim=1)
    assert (last - x).abs().max().item() < 1e-5
    assert (mid - x).abs().max().item() < 1e-5
    # Sanity: transforming different axes gives genuinely different spectra.
    assert not torch.allclose(dct(x, dim=-1), dct(x, dim=1))

    # ------------------------------------------------------------- (e) gradients


def test_gradients_flow_through_dct():
    """x.requires_grad → y=dct(x); y.sum().backward() ⇒ finite non-zero x.grad.

    Sum over ALL coefficients: every input element contributes to many
    coefficients, so the full gradient should be strictly non-zero.
    """
    x = rand((4, 8, 3)).requires_grad_(True)
    y = dct(x, dim=CHUNK_DIM)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), "gradient contains non-finite values"
    assert x.grad.abs().sum().item() > 0.0, "gradient is all-zero"


def test_gradients_flow_through_idct():
    """Mirror check for the inverse transform."""
    c = rand((4, 8, 3)).requires_grad_(True)
    out = idct(c, dim=CHUNK_DIM)
    out.sum().backward()
    assert c.grad is not None
    assert torch.isfinite(c.grad).all()
    assert c.grad.abs().sum().item() > 0.0

    # ---------------------------------------------------------------- (f) linearity


def test_linearity():
    """dct(a·x + b·y) == a·dct(x) + b·dct(y) within 1e-5 (matmul is linear)."""
    x = rand((4, 8, 3), seed=11)
    y = rand((4, 8, 3), seed=12)
    a, b = 2.5, -0.75
    lhs = dct(a * x + b * y, dim=CHUNK_DIM)
    rhs = a * dct(x, dim=CHUNK_DIM) + b * dct(y, dim=CHUNK_DIM)
    err = (lhs - rhs).abs().max().item()
    assert err < 1e-5, f"linearity max-abs err {err:.3e}"

    # ------------------------------------------- helper: discounted_chunk_reward


def test_discounted_chunk_reward_hand_computed():
    """Σ γ^t·r_t over an 8-step chunk, hand-checked for r≡1, γ=0.9:
    (1 − 0.9⁸)/0.1 ≈ 5.695312..."""
    rewards = torch.ones(8, dtype=torch.float32)
    got = discounted_chunk_reward(rewards, gamma=0.9).item()
    expected = sum(0.9**t for t in range(8))
    assert abs(got - expected) < 1e-5, f"{got!r} vs hand-computed {expected!r}"


def test_discounted_chunk_reward_batched_shape():
    """Batched input [B,8] reduces to [B]; weights applied along last dim."""
    g = torch.Generator().manual_seed(5)
    r = torch.rand(6, 8, generator=g)
    out = discounted_chunk_reward(r, gamma=0.99)
    assert out.shape == (6,)
    manual_first = sum((0.99**t) * r[0, t].item() for t in range(8))
    assert abs(out[0].item() - manual_first) < 1e-5
