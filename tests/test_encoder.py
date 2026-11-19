"""The state encoder projects each frame, mixes the window, and normalizes.

It was the only unnormalized path in the stack: every observation reached the
actor through here and then through an unnormalized adaLN projection. It was
also the capacity sink: one dense over ``T * obs_dim`` held 713k of the
encoder's 729k on Humanoid.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from flow_policy_optimization.flow_policy_optimization import StateEncoder

OBS, HIDDEN, STEPS = 39, 128, 16


def _encoder(**kw) -> StateEncoder:
    torch.manual_seed(0)
    return StateEncoder(OBS, HIDDEN, STEPS, **kw)


def _window(seed: int = 0, batch: int = 4) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, STEPS, OBS)


def _mixed(enc: StateEncoder, x: torch.Tensor) -> torch.Tensor:
    """The per-frame vector after the attention pass, before the readout."""
    h = enc.frame(x)
    return h + enc.attn(enc.attn_norm(h))


def test_the_output_is_normalized():
    out = _encoder()(_window())
    rms = out.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_the_output_stays_normalized_whatever_the_input_scale():
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


def test_the_last_mixed_frame_is_the_output():
    """The readout is one frame, so the width does not grow with the window."""
    enc = _encoder()
    x = _window()
    want = enc.out_norm(_mixed(enc, x)[:, -1])
    assert torch.allclose(enc(x), want, atol=0.0)


def test_the_projection_width_follows_obs_dim_not_the_window():
    """The old flat dense read ``T * obs_dim``. This is the regression."""
    enc = _encoder()
    assert enc.frame.fc1.in_features == OBS
    assert enc.frame.fc1.in_features != STEPS * OBS


def test_the_projection_is_shared_over_the_frames():
    """One module sees every frame, so the cost does not grow with T."""
    enc = _encoder()
    torch.manual_seed(3)
    v = torch.randn(2, OBS)
    same = v[:, None, :].expand(-1, STEPS, -1)
    h = enc.frame(same)
    torch.testing.assert_close(h, enc.frame(v[:, None, :]).expand(-1, STEPS, -1))


def test_every_frame_changes_the_conditioning():
    enc = _encoder()
    x = _window()
    base = enc(x)
    for frame in (0, STEPS - 1):
        moved = x.clone()
        moved[:, frame] += 1.0
        assert not torch.allclose(base, enc(moved), atol=1e-6), f"frame {frame}"


def test_the_frame_order_changes_the_conditioning():
    """A pool would pass this window and its roll as one input."""
    enc = _encoder()
    x = _window()
    assert not torch.allclose(enc(x), enc(torch.roll(x, 1, dims=1)), atol=1e-6)


def test_attention_is_causal():
    """Frame 0 must not read the later frames. The last frame reads them all."""
    enc = _encoder()
    x = _window()
    moved = x.clone()
    moved[:, 5] += 1.0
    a, b = _mixed(enc, x), _mixed(enc, moved)
    torch.testing.assert_close(a[:, 0], b[:, 0])
    assert not torch.allclose(a[:, -1], b[:, -1], atol=1e-6)


def test_the_capacity_follows_obs_dim():
    """Exact count for Humanoid (348 obs, 16 frames, 128 wide).

    The old encoder was 729_472 here. The per-frame projection is 61_184 and
    the attention layer is 66_048, which is the whole cost at any window.
    """
    enc = StateEncoder(348, 128, 16)
    assert sum(p.numel() for p in enc.parameters()) == 127_488


@pytest.mark.parametrize("bad", [0, -3])
def test_the_step_count_must_be_positive(bad):
    with pytest.raises(ValueError, match="max_obs_steps must be >= 1"):
        StateEncoder(OBS, HIDDEN, bad)


def test_the_hidden_width_must_be_at_least_two():
    with pytest.raises(ValueError, match="hidden_dim must be at least 2"):
        StateEncoder(OBS, 1, STEPS)
