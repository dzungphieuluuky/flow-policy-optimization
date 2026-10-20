"""``effective_rank`` is read off a chart, so it must be pinned to numbers."""

import pytest
import torch

from myfpo.flow import MyfpoFlowModel, effective_rank

N, HIDDEN = 128, 128


def _flow_model(device: torch.device) -> MyfpoFlowModel:
    torch.manual_seed(0)
    return MyfpoFlowModel(
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
    to land on the device they are on."""
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
    """weight_magnitude is the only reduction not coming from h, so it is the only one that can disagree."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _flow_model(device)
    _, stats = model.forward_with_stats(
        torch.randn(8, 16, 3, device=device),
        torch.rand(8, device=device),
        torch.randn(8, HIDDEN, device=device),
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
    """The upper anchor."""
    assert float(effective_rank(torch.randn(N, HIDDEN))) > 0.7 * HIDDEN


def test_the_result_can_never_fall_below_one():
    """exp of an entropy, so 1 is the floor: an exactly rank-1 input reads 1.0,
    not 0."""
    assert float(effective_rank(torch.zeros(N, HIDDEN))) == 1.0


def test_tokens_are_the_sample_axis():
    """The caller passes [batch, tokens, hidden]. Flattening is the caller's
    contract, so [8, 16, hidden] must equal the [128, hidden] matrix."""
    stacked = torch.randn(8, 16, HIDDEN)
    assert float(effective_rank(stacked)) == pytest.approx(
        float(effective_rank(stacked.reshape(-1, HIDDEN)))
    )


def test_bf16_input_does_not_lose_the_low_rank_collapse():
    """The stats forward runs under autocast, so the features arrive bf16."""
    bf16 = _rank(torch.randn(1, HIDDEN)).to(torch.bfloat16)
    collapsed = float(effective_rank(bf16))
    assert collapsed == pytest.approx(1.0, abs=0.2)
    assert collapsed < 0.05 * float(effective_rank(torch.randn(N, HIDDEN).bfloat16()))