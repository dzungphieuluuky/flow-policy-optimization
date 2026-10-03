"""The critic reads the whole window, in order, and its ensemble stays independent.

It used to read states[:, -1, :], discarding 15 of 16 frames, and later a
flatten plus a position code. The window properties still have to hold:

  * every frame influences the output, including index 0
  * permuting the frames changes the output

A mean-pool satisfies neither in the way that matters: it ignores order entirely,
so the permutation would be a no-op. That is the failure mode this file exists to
prevent.

The critic is now a per-member SiLU MLP over the flattened window, so the
window properties are structural rather than incidental: a permuted window is a
different input vector to the first Linear. The ensemble property is the
load-bearing one here: members must not share a trunk, or min-over-members is
meaningless.
"""

import pytest
import torch

from spectral_joint_policy.networks import Critic

OBS, HIDDEN, STEPS = 11, 32, 8


def _critic(seed: int = 0, num_critics: int = 1) -> Critic:
    torch.manual_seed(seed)
    return Critic(OBS, HIDDEN, STEPS, num_critics)


def _window(seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(4, STEPS, OBS)


@pytest.mark.parametrize("batch", [1, 3, 64])
def test_output_is_one_value_per_sample(batch: int):
    """[num_critics, B]: the critic does not aggregate, the caller does."""
    out = _critic()(torch.randn(batch, STEPS, OBS))
    assert out.shape == (1, batch)


def test_window_length_must_match_the_declared_one():
    """The collector builds windows to n_obs_steps, so a mismatch is a real error.

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


def test_temporal_frequency_content_is_visible():
    """Two windows with identical per-frame value multisets must separate.

    The flatten makes this structural: a permutation is a different input to
    the first Linear, and an MLP is not permutation-invariant.
    """
    critic = _critic()
    w = _window()
    shuffled = w[:, torch.randperm(STEPS, generator=torch.Generator().manual_seed(7))]
    torch.testing.assert_close(w.sum(1), shuffled.sum(1))
    assert not torch.allclose(critic(w), critic(shuffled), atol=1e-6)


def test_n_obs_steps_must_be_positive():
    with pytest.raises(ValueError, match="n_obs_steps must be >= 1"):
        Critic(OBS, HIDDEN, 0)


def test_parameter_count_is_proportionate():
    """The per-member conditioner is the dominant cost; guard against widening it."""
    n = sum(p.numel() for p in Critic(11, 128, 16, 2).parameters())
    assert n == pytest.approx(45_570, abs=2_000), f"critic is {n} params"


# --- ensemble -------------------------------------------------------------


def test_members_are_independent_not_one_wide_mlp():
    """Perturbing one member's conditioner must not move the other member.

    This is the test that separates a real ensemble from a shared trunk with n
    heads, where member 0's output would move when member 1's weights move.
    """
    critic = _critic(seed=3, num_critics=2)
    w = _window()
    before = critic(w).clone()
    with torch.no_grad():
        critic.networks[1].fc2.weight.add_(1.0)
    after = critic(w)
    torch.testing.assert_close(after[0], before[0])
    assert not torch.allclose(after[1], before[1]), "member 1 did not move at all"


def test_per_member_network_has_no_shared_parameters():
    """The independence above is structural, not just numerically lucky."""
    critic = _critic(seed=3, num_critics=2)
    assert critic.networks[0] is not critic.networks[1]
    ids = {id(p) for net in critic.networks for p in net.parameters()}
    assert len(ids) == sum(1 for net in critic.networks for p in net.parameters())


def test_permuting_every_member_permutes_the_output():
    """The ensemble is equivariant to a relabelling of its members."""
    critic = _critic(seed=3, num_critics=2)
    w = _window()
    before = critic(w).clone()
    with torch.no_grad():
        critic.networks[0], critic.networks[1] = (
            critic.networks[1],
            critic.networks[0],
        )
    after = critic(w)
    torch.testing.assert_close(after[0], before[1])
    torch.testing.assert_close(after[1], before[0])


def test_members_differ_at_init():
    critic = _critic(seed=4, num_critics=2)
    out = critic(_window())
    assert not torch.allclose(out[0], out[1])


def test_gradient_reaches_every_member():
    critic = _critic(seed=4, num_critics=2)
    critic(_window()).sum().backward()
    # Every member must carry signal, not just one of them.
    for net in critic.networks:
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert grads, "a member received no gradient at all"
        assert sum(g.abs().sum() for g in grads) > 0


def test_critic_member_reads_the_whole_window_at_the_actor_encoder_width():
    """The critic must encode the window the actor conditions on, at the same width.

    If these drift, the critic silently becomes a weaker approximator of the
    policy it is critiquing, which is the failure this design exists to prevent.
    """
    member = _critic().networks[0]
    assert member.fc1.in_features == STEPS * OBS
    assert member.fc1.out_features == HIDDEN


def test_num_critics_must_be_positive():
    with pytest.raises(ValueError, match="num_critics must be >= 1"):
        Critic(OBS, HIDDEN, STEPS, 0)
