"""The timestep encoding must be the log-spaced DDPM/DiT table.

The old table was ``2**i``. Over ``t in [0, 1]`` that runs to 32768 rad, so the
code was high-frequency noise: neighbouring times were near-orthogonal. These
tests pin the frequency table, not just the shape.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from myfpo.layers import SinusoidalEncoding


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


def test_neighbouring_times_stay_close():
    """The property the old table lost: adjacent t embed to nearby vectors."""
    enc = SinusoidalEncoding(32)
    ts = torch.tensor([0.0, 0.05, 0.1, 0.5, 0.9, 1.0])
    unit = F.normalize(enc.encode_times(ts), dim=-1)
    sim = unit @ unit.T
    assert sim.min().item() > 0.9, sim


def test_the_table_still_separates_the_two_ends():
    enc = SinusoidalEncoding(32)
    ts = torch.tensor([0.0, 1.0])
    unit = F.normalize(enc.encode_times(ts), dim=-1)
    assert (unit[0] @ unit[1]).item() < 0.999


def test_the_buffer_is_not_persistent():
    enc = SinusoidalEncoding(32)
    assert "freqs" not in enc.state_dict()


@pytest.mark.parametrize("bad", [1, 0, -4])
def test_dim_must_be_at_least_two(bad):
    with pytest.raises(ValueError, match="dim must be >= 2"):
        SinusoidalEncoding(bad)