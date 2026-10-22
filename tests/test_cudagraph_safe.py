"""The cudagraph copy is off when nothing is compiled.

The clone exists to leave the cudagraph memory pool. With ``compile=False``
the encoder is not compiled, so the clone is pure overhead.
"""

import torch

from myfpo.layers import cudagraph_safe


def test_disabled_returns_the_same_tensor():
    x = torch.zeros(2, 3)
    assert cudagraph_safe(x, enabled=False) is x


def test_enabled_returns_an_equal_copy():
    x = torch.zeros(2, 3)
    out = cudagraph_safe(x)
    assert out is not x
    assert torch.equal(out, x)
