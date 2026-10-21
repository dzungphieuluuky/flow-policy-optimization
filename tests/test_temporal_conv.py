"""The temporal filter. It keeps the width and it keeps the offset."""

import pytest
import torch
from myfpo.layers import TemporalConv


def test_the_width_and_the_length_do_not_change():
    for causal in (False, True):
        conv = TemporalConv(3, 3, causal=causal)
        out = conv(torch.randn(2, 3, 8))
        assert out.shape == (2, 3, 8)


def test_an_even_or_zero_kernel_is_rejected():
    for kernel in (0, 2, 4):
        with pytest.raises(ValueError, match="kernel_size"):
            TemporalConv(3, kernel)


def test_the_filter_keeps_the_offset():
    """A constant shift must move the output. A norm would erase it."""
    torch.manual_seed(0)
    conv = TemporalConv(3, 3)
    x = torch.zeros(1, 3, 8)
    assert not torch.allclose(conv(x + 5.0), conv(x))


def test_the_causal_filter_ignores_the_future():
    torch.manual_seed(0)
    conv = TemporalConv(3, 3, causal=True)
    x = torch.randn(1, 3, 8)
    moved = x.clone()
    moved[:, :, -1] = torch.randn(1, 3)
    assert torch.equal(conv(x)[:, :, :-1], conv(moved)[:, :, :-1])


def test_the_symmetric_filter_reads_the_future():
    torch.manual_seed(0)
    conv = TemporalConv(3, 3)
    x = torch.randn(1, 3, 8)
    moved = x.clone()
    moved[:, :, -1] = torch.randn(1, 3)
    assert not torch.allclose(conv(x)[:, :, :-1], conv(moved)[:, :, :-1])
