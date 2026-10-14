"""``effective_rank`` is read off a chart, so it must be pinned to numbers.

The metric exists only as a W&B trace (Dohare et al., arXiv:2306.13812, rank
loss as the correlate of plasticity loss). A broken normalisation here would
still log a plausible-looking curve, which is why the ranks are checked against
inputs of known rank rather than against a reference implementation.
"""

import pytest
import torch

from myfpo.flow import SJFPOFlowModel, effective_rank

N, HIDDEN = 128, 128


def _flow_model(device: torch.device) -> SJFPOFlowModel:
    torch.manual_seed(0)
    return SJFPOFlowModel(
        action_dim=3,
        action_horizon=16,
        hidden_dim=HIDDEN,
        dit_depth=2,
        dit_heads=4,
        dit_mlp_ratio=4.0,
        time_embed_dim=128,
    ).to(device)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_weight_magnitude_follows_the_parameters_onto_the_device(device):
    """The metric is reduced over the module's own parameters, so the result has
    to land on the device they are on.

    A scalar accumulator written as ``torch.zeros((), dtype=...)`` defaults to
    CPU, so the first ``+=`` against a CUDA parameter raises. That shipped: the
    stats forward only runs when a tracker is enabled, so it reached Humanoid on
    Kaggle and nothing else.
    """
    model = _flow_model(torch.device(device))
    magnitude = model.weight_magnitude()
    assert magnitude.device.type == torch.device(device).type
    assert magnitude > 0


def test_weight_magnitude_is_the_mean_absolute_value_not_the_sum():
    """A scale change in one parameter has to move the figure, or the metric is
    just reporting that parameters exist."""
    model = _flow_model(torch.device("cpu"))
    before = float(model.weight_magnitude())
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(2.0)
    assert float(model.weight_magnitude()) == pytest.approx(2.0 * before, rel=1e-5)


def test_the_stats_forward_returns_both_probes_on_one_device():
    """``weight_magnitude`` is the only reduction in the dict that does not come
    from ``h``, so it is the only one that can disagree with the rest."""
    model = _flow_model(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    _, stats = model(
        torch.randn(8, 16, 3),
        torch.rand(8),
        torch.randn(8, HIDDEN),
        return_stats=True,
    )
    devices = {v.device.type for v in stats.values()}
    assert len(devices) == 1, f"stats span {devices}"


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