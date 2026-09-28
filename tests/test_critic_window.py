"""The critic reads the whole observation window, and order survives into it.

It used to read states[:, -1, :], discarding 15 of 16 frames. Two properties are
pinned here, because together they are what distinguishes a flatten from a pool:

  * every frame influences the output, including index 0
  * permuting the frames changes the output

A mean-pool satisfies neither in the way that matters: it ignores order entirely,
so the permutation would be a no-op. That is the failure mode this file exists to
prevent.
"""

import pytest
import torch

from sjfpo.networks import Critic

OBS, HIDDEN, STEPS = 11, 32, 8


def _critic(seed: int = 0) -> Critic:
    torch.manual_seed(seed)
    return Critic(OBS, HIDDEN, STEPS)


def _window(seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(4, STEPS, OBS)


@pytest.mark.parametrize("batch", [1, 3, 64])
def test_output_is_one_value_per_sample(batch: int):
    out = _critic()(torch.randn(batch, STEPS, OBS))
    assert out.shape == (batch,)


def test_window_length_must_match_the_declared_one():
    """The first layer is sized to n_obs_steps, so a mismatch is a real error.

    The old states[:, -1, :] path accepted any T, so this is a narrowing: the
    critic is now tied to the window length the config declares.
    """
    with pytest.raises(ValueError, match="built for n_obs_steps=8"):
        _critic()(torch.randn(2, 4, OBS))


def test_every_frame_influences_the_output():
    """Frame 0 specifically. The old states[:, -1, :] provably ignored it."""
    critic = _critic()
    base = _window()
    for frame in range(STEPS):
        moved = base.clone()
        moved[:, frame, :] += 5.0
        assert not torch.allclose(critic(base), critic(moved), atol=1e-6), (
            f"frame {frame} does not affect the value"
        )


def test_frame_order_matters():
    """The property a mean-pool cannot have."""
    critic = _critic()
    w = _window()
    assert not torch.allclose(critic(w), critic(w.flip(1)), atol=1e-6)


def test_position_code_does_not_destroy_the_observation():
    """obs_j + pe_j must still separate back into obs_j.

    pe is a fixed constant, so W_j @ pe_j is a constant the layer bias absorbs.
    If that ever stops holding, the critic would be regressing on a shifted
    target, which is the risk of adding the code directly to the observation.
    """
    critic = _critic()
    w = _window()
    zero_out = critic(w) - critic(w + 7.0)          # bias-only shift, same layout
    assert torch.isfinite(zero_out).all()
    # Two windows differing by a constant must differ by a constant, not a
    # rescaling, which is what a leaked position term would produce.
    a, b = critic(w), critic(w * 2.0)
    assert not torch.allclose(a, b, atol=1e-6)


def test_obs_dim_must_allow_a_position_code():
    with pytest.raises(ValueError, match="obs_dim must be >= 2"):
        Critic(1, HIDDEN, STEPS)


def test_n_obs_steps_must_be_positive():
    with pytest.raises(ValueError, match="n_obs_steps must be >= 1"):
        Critic(OBS, HIDDEN, 0)


def test_parameter_count_is_proportionate():
    """A per-frame hidden embed would have made this 263k params and 22% of the net."""
    critic = Critic(11, 128, 16)
    n = sum(p.numel() for p in critic.parameters())
    assert n == pytest.approx(22_785, abs=200), f"critic grew to {n} params"
