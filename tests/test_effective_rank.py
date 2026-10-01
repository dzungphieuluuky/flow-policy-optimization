"""``effective_rank`` is read off a chart, so it must be pinned to numbers.

The metric exists only as a W&B trace (Dohare et al., arXiv:2306.13812, rank
loss as the correlate of plasticity loss). A broken normalisation here would
still log a plausible-looking curve, which is why the ranks are checked against
inputs of known rank rather than against a reference implementation.
"""

import pytest
import torch

from spectral_joint_policy.networks import effective_rank

N, HIDDEN = 128, 128


def _rank(matrix: torch.Tensor) -> torch.Tensor:
    """A ``[N, hidden]`` matrix whose rank is ``matrix``'s row count."""
    return torch.randn(N, matrix.shape[0]) @ matrix


def test_a_rank_r_matrix_reads_as_r():
    """The entropy form tracks exact rank: sigma has r nonzero values, so the
    normalised spectrum is uniform over r and exp(H) = r."""
    for r in (1, 3, 16):
        assert float(effective_rank(_rank(torch.randn(r, HIDDEN)))) == pytest.approx(
            r, rel=0.05
        )


def test_a_gaussian_matrix_reads_near_full_rank():
    """The upper anchor. A [128, 128] Gaussian is not exactly full rank but is
    spectrally flat enough to land close, so a rank that has collapsed by orders
    of magnitude cannot hide inside this."""
    assert float(effective_rank(torch.randn(N, HIDDEN))) > 0.7 * HIDDEN


def test_the_result_can_never_fall_below_one():
    """exp of an entropy, so 1 is the floor: an exactly rank-1 input reads 1.0,
    not 0. The all-zero case must not divide by a zero norm."""
    assert float(effective_rank(torch.zeros(N, HIDDEN))) == 1.0


def test_tokens_are_the_sample_axis():
    """The caller passes [batch, tokens, hidden]; flattening is the caller's
    contract, so [8, 16, hidden] must equal the [128, hidden] matrix."""
    stacked = torch.randn(8, 16, HIDDEN)
    assert float(effective_rank(stacked)) == pytest.approx(
        float(effective_rank(stacked.reshape(-1, HIDDEN)))
    )


def test_bf16_input_does_not_lose_the_low_rank_collapse():
    """The stats forward runs under autocast, so the features arrive bf16.

    Exact rank is not recoverable there: 8 mantissa bits leave ~1e-3 relative
    noise in the singular values that are nominally zero, and the entropy form
    is sensitive to exactly that. Measured 1.11 against fp32's 1.00. What
    matters is the order of magnitude, so the bound is loose and paired with the
    full-rank anchor below -- a broken normalisation reads in the tens.
    """
    bf16 = _rank(torch.randn(1, HIDDEN)).to(torch.bfloat16)
    collapsed = float(effective_rank(bf16))
    assert collapsed == pytest.approx(1.0, abs=0.2)
    assert collapsed < 0.05 * float(effective_rank(torch.randn(N, HIDDEN).bfloat16()))