"""The timestep encoding must be the log-spaced DDPM/DiT table, scaled to the flow.

The old table was ``2**i``. Over ``t in [0, 1]`` that runs to 32768 rad, so the
code was high-frequency noise: neighbouring times were near-orthogonal. These
tests pin the frequency table and the scale that sets its resolution.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from myfpo.nn import TIME_SCALE, SinusoidalEncoding


def test_the_frequencies_are_log_spaced_and_decaying():
    enc = SinusoidalEncoding(32)
    expected = torch.exp(
        -math.log(10000.0) * torch.arange(16, dtype=torch.float32) / 16
    )
    assert torch.allclose(enc.freqs, expected, atol=0.0)


def test_the_table_starts_at_one_and_ends_far_below_the_old_peak():
    enc = SinusoidalEncoding(32)
    assert enc.freqs[0].item() == pytest.approx(1.0)
    # exp(-log(10000) * 15/16). The old 2**i table peaked at 32768.
    assert enc.freqs[-1].item() == pytest.approx(10000.0 ** (-15 / 16))
    assert enc.freqs.max().item() == 1.0


def test_max_period_changes_the_table():
    a = SinusoidalEncoding(16)
    b = SinusoidalEncoding(16, max_period=100.0)
    assert not torch.allclose(a.freqs, b.freqs)
    assert b.freqs.max().item() == pytest.approx(1.0)


def test_max_period_must_be_positive():
    with pytest.raises(ValueError, match="max_period must be > 0"):
        SinusoidalEncoding(16, max_period=0.0)


def test_the_scale_covers_the_trained_band():
    """The code must resolve times inside [0.1, 0.9], not just separate 0 from 1.

    The draw bounds of ``MyfpoConfig`` are 0.1 and 0.9. A span under one cycle
    at the top channel leaves the whole band on one arc.
    """
    enc = SinusoidalEncoding(32)
    span = (0.9 - 0.1) * TIME_SCALE * enc.freqs[0]
    assert span > 2 * math.pi, float(span)


def test_a_hundredth_step_is_resolvable():
    """Two training draws that differ by 0.01 must not look the same."""
    enc = SinusoidalEncoding(32)
    unit = F.normalize(enc.encode_times(torch.tensor([0.1, 0.11])), dim=-1)
    assert (unit[0] @ unit[1]).item() < 0.8


def test_a_thousandth_step_is_still_close():
    """The code stays continuous: it does not turn a tiny step into noise."""
    enc = SinusoidalEncoding(32)
    unit = F.normalize(enc.encode_times(torch.tensor([0.1, 0.101])), dim=-1)
    assert (unit[0] @ unit[1]).item() > 0.9


def test_the_table_still_separates_the_two_ends():
    enc = SinusoidalEncoding(32)
    ts = torch.tensor([0.0, 1.0])
    unit = F.normalize(enc.encode_times(ts), dim=-1)
    assert (unit[0] @ unit[1]).item() < 0.5


def test_the_buffer_is_not_persistent():
    enc = SinusoidalEncoding(32)
    assert "freqs" not in enc.state_dict()


@pytest.mark.parametrize("bad", [1, 0, -4])
def test_dim_must_be_at_least_two(bad):
    with pytest.raises(ValueError, match="dim must be >= 2"):
        SinusoidalEncoding(bad)
