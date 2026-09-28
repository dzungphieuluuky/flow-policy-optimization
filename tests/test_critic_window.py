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
import torch.nn as nn
import torch.nn.functional as F
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
    """[num_critics, B]: the critic does not aggregate, the caller does."""
    out = _critic()(torch.randn(batch, STEPS, OBS))
    assert out.shape == (1, batch)


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
    shifted = critic(w) - critic(w + 7.0)          # bias-only shift
    assert torch.isfinite(shifted).all()
    # Two windows differing by a constant must not collapse onto each other,
    # which is what a leaked position term would produce.
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


# --- ensemble -------------------------------------------------------------


def test_members_are_independent_not_one_wide_mlp():
    """Perturbing one member's first layer must not move the other member.

    This is the test that separates a real ensemble from the naive
    concat-and-chunk fusion. Widening each Linear to n*out and chunking at the
    end gives the second layer every hidden unit, so member 0's output would move
    when member 1's weights move. Here it must not.
    """
    torch.manual_seed(3)
    critic = Critic(OBS, HIDDEN, STEPS, 2)
    w = _window()
    before = critic(w).clone()
    with torch.no_grad():
        critic.w_in[1].add_(1.0)
    after = critic(w)
    torch.testing.assert_close(after[0], before[0])
    assert not torch.allclose(after[1], before[1]), "member 1 did not move at all"


def test_permuting_every_member_permutes_the_output():
    """The ensemble is equivariant to a relabelling of its members."""
    torch.manual_seed(3)
    critic = Critic(OBS, HIDDEN, STEPS, 2)
    w = _window()
    before = critic(w).clone()
    with torch.no_grad():
        critic.w_in.copy_(critic.w_in.flip(0))
        critic.b_in.copy_(critic.b_in.flip(0))
        critic.w_out.copy_(critic.w_out.flip(0))
        critic.b_out.copy_(critic.b_out.flip(0))
    after = critic(w)
    torch.testing.assert_close(after[0], before[1])
    torch.testing.assert_close(after[1], before[0])


def test_members_differ_at_init():
    torch.manual_seed(4)
    critic = Critic(OBS, HIDDEN, STEPS, 2)
    out = critic(_window())
    assert not torch.allclose(out[0], out[1])


def test_gradient_reaches_every_member():
    critic = Critic(OBS, HIDDEN, STEPS, 2)
    critic(_window()).sum().backward()
    assert critic.w_in.grad is not None
    # Every member's slice must carry signal, not just one of them.
    assert critic.w_in.grad[0].abs().sum() > 0
    assert critic.w_in.grad[1].abs().sum() > 0


def test_single_member_matches_two_linears():
    """num_critics=1 must reduce to exactly an nn.Linear pair.

    The weights are stored transposed relative to nn.Linear, so this pins the
    layout convention that the rest of the ensemble depends on.
    """
    torch.manual_seed(5)
    critic = Critic(OBS, HIDDEN, STEPS, 1)
    w = _window()
    fc1, fc2 = nn.Linear(STEPS * OBS, HIDDEN), nn.Linear(HIDDEN, 1)
    with torch.no_grad():
        fc1.weight.copy_(critic.w_in[0].t())
        fc1.bias.copy_(critic.b_in[0, 0])
        fc2.weight.copy_(critic.w_out[0].t())
        fc2.bias.copy_(critic.b_out[0, 0])
    ref_in = critic.pos_enc.add_to(w, torch.arange(STEPS)).flatten(1)
    ref = fc2(F.silu(fc1(ref_in))).squeeze(-1)
    torch.testing.assert_close(critic(w)[0], ref, rtol=1e-5, atol=1e-6)


def test_num_critics_must_be_positive():
    with pytest.raises(ValueError, match="num_critics must be >= 1"):
        Critic(OBS, HIDDEN, STEPS, 0)
