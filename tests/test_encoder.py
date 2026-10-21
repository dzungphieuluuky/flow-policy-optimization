"""The state encoder normalises its output.

It was the only unnormalised path in the stack: every observation reaches the
actor through here and then through an unnormalised adaLN projection.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from myfpo.processor import StateEncoder

OBS, HIDDEN, STEPS = 39, 128, 16


def _encoder(**kw) -> StateEncoder:
    torch.manual_seed(0)
    return StateEncoder(OBS, HIDDEN, STEPS, **kw)


def _window(seed: int = 0, batch: int = 4) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, STEPS, OBS)


def test_the_output_is_normalised():
    out = _encoder()(_window())
    rms = out.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_the_output_stays_normalised_whatever_the_input_scale():
    """The property the norm buys.

    Exact scale invariance only holds in the eps->0 limit, since the norm is
    ``z / sqrt(mean(z**2) + eps)``. Assert the unit-RMS invariant instead.
    """
    enc = _encoder()
    x = _window()
    for k in (0.01, 0.1, 1.0, 3.0, 100.0):
        rms = enc(k * x).float().pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3), k


def test_the_shape_is_unchanged():
    assert _encoder()(_window(batch=7)).shape == (7, HIDDEN)


def test_the_norm_carries_an_affine_scale():
    assert isinstance(_encoder().out_norm, nn.RMSNorm)
    assert _encoder().out_norm.normalized_shape == (HIDDEN,)


def test_the_norm_adds_exactly_one_hidden_width_of_parameters():
    plain = sum(p.numel() for p in _encoder().net.parameters())
    assert sum(p.numel() for p in _encoder().parameters()) == plain + HIDDEN


def test_the_norm_is_applied_after_the_mlp():
    """Order matters: normalising the flattened window would discard the MLP."""
    enc = _encoder()
    x = _window()
    assert torch.allclose(enc(x), enc.out_norm(enc.net(x.flatten(1))), atol=0.0)


def test_the_filter_is_off_by_default():
    assert _encoder().frame_conv is None
    assert _encoder(temporal_kernel=3).frame_conv is not None


def test_the_filter_keeps_the_shape_and_the_width():
    enc = _encoder(temporal_kernel=3)
    assert enc(_window(batch=7)).shape == (7, HIDDEN)
    assert enc.net.fc1.in_features == STEPS * OBS


def test_the_filter_sits_before_the_flatten():
    enc = _encoder(temporal_kernel=3)
    x = _window()
    frames = enc.frame_conv(x.transpose(1, 2))
    want = enc.out_norm(enc.net(frames.transpose(1, 2).flatten(1)))
    assert torch.allclose(enc(x), want, atol=0.0)


def test_the_filter_changes_the_conditioning():
    enc = _encoder(temporal_kernel=3)
    x = _window()
    plain = enc.out_norm(enc.net(x.flatten(1)))
    assert not torch.allclose(enc(x), plain, atol=1e-6)


def test_the_bad_shapes_still_raise():
    enc = _encoder()
    with pytest.raises(ValueError, match="states must have shape"):
        enc(torch.randn(4, STEPS, OBS + 1))
    with pytest.raises(ValueError, match="built for max_obs_steps"):
        enc(torch.randn(4, STEPS + 1, OBS))


@pytest.mark.parametrize("bad", [0, -3])
def test_the_step_count_must_be_positive(bad):
    with pytest.raises(ValueError, match="max_obs_steps must be >= 1"):
        StateEncoder(OBS, HIDDEN, bad)


def test_the_hidden_width_must_be_at_least_two():
    with pytest.raises(ValueError, match="hidden_dim must be at least 2"):
        StateEncoder(OBS, 1, STEPS)