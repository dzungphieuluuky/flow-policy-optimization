"""The critic is Q(s, a_chunk), it reads the whole window in order, and the
ensemble stays independent.

It used to read states[:, -1, :], discarding 15 of 16 frames, and later a
flatten plus a position code. The window properties still have to hold:

  * every frame influences the output, including index 0
  * permuting the frames changes the output

A mean-pool satisfies neither in the way that matters: it ignores order entirely,
so the permutation would be a no-op. That is the failure mode this file exists to
prevent.

The critic is a per-member SiLU MLP over the flattened window, so the window
properties are structural rather than incidental: a permuted window is a
different input vector to the first Linear. The ensemble property is the
load-bearing one here: members must not share a trunk, or min-over-members is
meaningless.

Since the chunked critic, each member is also *action* conditioned -- the chunk
modulates the state feature through FiLM -- so the action properties get the
same treatment: Q must separate two chunks that differ only in ordering, and a
member perturbed in its chunk branch must move its own output only.
"""

import pytest
import torch

from spectral_joint_policy.networks import Critic

OBS, HIDDEN, STEPS, ACT, H = 11, 32, 8, 3, 6


def _critic(seed: int = 0, num_critics: int = 1) -> Critic:
    torch.manual_seed(seed)
    return Critic(OBS, HIDDEN, ACT, STEPS, H, num_critics)


def _window(seed: int = 1, batch: int = 4) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, STEPS, OBS)


def _chunk(seed: int = 2, batch: int = 4) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, H, ACT)


@pytest.mark.parametrize("batch", [1, 3, 64])
def test_output_is_one_value_per_sample(batch: int):
    """[num_critics, B]: the critic does not aggregate, the caller does."""
    out = _critic()(_window(batch=batch), _chunk(batch=batch))
    assert out.shape == (1, batch)


def test_window_length_must_match_the_declared_one():
    """The collector builds windows to n_obs_steps, so a mismatch is a real error.

    The old states[:, -1, :] path accepted any T, so this is a narrowing: the
    critic is now tied to the window length the config declares.
    """
    with pytest.raises(ValueError, match="built for n_obs_steps=8"):
        _critic()(torch.randn(2, 4, OBS), _chunk()[:, :H])


def test_chunk_length_must_match_the_horizon():
    with pytest.raises(ValueError, match="action_horizon=6"):
        _critic()(_window(), torch.randn(4, H - 1, ACT))


def test_action_dim_must_match():
    with pytest.raises(ValueError, match="actions feature dimension must be 3"):
        _critic()(_window(), torch.randn(4, H, ACT + 1))


def test_batch_mismatch_is_rejected():
    with pytest.raises(ValueError, match="actions batch 3 != states batch 4"):
        _critic()(_window(), torch.randn(3, H, ACT))


def test_every_frame_influences_the_output():
    """Frame 0 specifically. The old states[:, -1, :] provably ignored it."""
    critic = _critic()
    base, chunk = _window(), _chunk()
    for frame in range(STEPS):
        moved = base.clone()
        moved[:, frame, :] += 5.0
        assert not torch.allclose(critic(base, chunk), critic(moved, chunk), atol=1e-6), (
            f"frame {frame} does not affect the value"
        )


def test_frame_order_matters():
    """The property a mean-pool cannot have."""
    critic = _critic()
    w, c = _window(), _chunk()
    assert not torch.allclose(critic(w, c), critic(w.flip(1), c), atol=1e-6)


def test_temporal_frequency_content_is_visible():
    """Two windows with identical per-frame value multisets must separate.

    The flatten makes this structural: a permutation is a different input to
    the first Linear, and an MLP is not permutation-invariant.
    """
    critic = _critic()
    w, c = _window(), _chunk()
    shuffled = w[:, torch.randperm(STEPS, generator=torch.Generator().manual_seed(7))]
    torch.testing.assert_close(w.sum(1), shuffled.sum(1))
    assert not torch.allclose(critic(w, c), critic(shuffled, c), atol=1e-6)


# --- the chunk is a first-class input ---------------------------------------


def test_film_starts_as_identity():
    """Zero-init FiLM head: at step 0 the chunk must not modulate the state
    feature, so Q is chunk-independent until a gradient step un-zeroes it."""
    critic = _critic()
    s = _window()
    assert torch.allclose(critic(s, _chunk()), critic(s, _chunk(seed=9)), atol=1e-6)


def _unzero_film(critic: Critic) -> Critic:
    """Stand-in for 'after one training step': FiLM head is zero-init, so
    un-perturb it to exercise the chunk dependence these probes check for."""
    with torch.no_grad():
        for net in critic.networks:
            net.act_enc.fc2.weight.normal_(0, 0.05)
    return critic


def test_every_action_step_influences_the_output():
    critic = _unzero_film(_critic())
    base, chunk = _window(), _chunk()
    for step in range(H):
        moved = chunk.clone()
        moved[:, step, :] += 5.0
        assert not torch.allclose(critic(base, chunk), critic(base, moved), atol=1e-6), (
            f"action step {step} does not affect Q"
        )


def test_chunk_order_matters():
    """An order-insensitive chunk encoder could not tell a reversed chunk."""
    critic = _unzero_film(_critic())
    w, c = _window(), _chunk()
    assert not torch.allclose(critic(w, c), critic(w, c.flip(1)), atol=1e-6)


def test_q_is_not_additive_in_state_and_chunk():
    """The FiLM branch must make Q non-additive, which is why it is not a concat.

    If Q(s1,a1) + Q(s2,a2) == Q(s1,a2) + Q(s2,a1) held everywhere, the
    interaction collapsed to a sum and the extra parameters bought nothing.
    """
    critic = _unzero_film(_critic(seed=5))
    s1, s2 = _window(seed=11), _window(seed=12)
    a1, a2 = _chunk(seed=13), _chunk(seed=14)
    cross = (critic(s1, a1) + critic(s2, a2)) - (critic(s1, a2) + critic(s2, a1))
    assert not torch.allclose(cross, torch.zeros_like(cross), atol=1e-4), (
        "Q is additive in (state, chunk); FiLM is doing nothing"
    )


# --- ensemble --------------------------------------------------------------


def test_num_critics_two_min_le_mean_contract():
    """The caller's aggregator contract holds for the 2-member ensemble:
    min (backup) is never above mean (value loss), members stay distinct."""
    out = _critic(seed=3, num_critics=2)(_window(), _chunk())
    assert out.shape == (2, 4)
    assert not torch.allclose(out[0], out[1])
    assert (out.amin(dim=0) <= out.mean(dim=0)).all()



def test_members_are_independent_not_one_wide_mlp():
    """Perturbing one member must not move the other member.

    This is the test that separates a real ensemble from a shared trunk with n
    heads, where member 0's output would move when member 1's weights move.
    """
    critic = _critic(seed=3, num_critics=2)
    w, c = _window(), _chunk()
    before = critic(w, c).clone()
    with torch.no_grad():
        critic.networks[1].head.fc2.weight.add_(1.0)
    after = critic(w, c)
    torch.testing.assert_close(after[0], before[0])
    assert not torch.allclose(after[1], before[1]), "member 1 did not move at all"


def test_perturbing_the_chunk_branch_moves_only_that_member():
    """The action path is per-member too, not shared behind a single encoder."""
    critic = _critic(seed=3, num_critics=2)
    w, c = _window(), _chunk()
    before = critic(w, c).clone()
    with torch.no_grad():
        critic.networks[1].act_enc.fc2.weight.add_(1.0)
    after = critic(w, c)
    torch.testing.assert_close(after[0], before[0])
    assert not torch.allclose(after[1], before[1])


def test_per_member_network_has_no_shared_parameters():
    """The independence above is structural, not just numerically lucky."""
    critic = _critic(seed=3, num_critics=2)
    assert critic.networks[0] is not critic.networks[1]
    ids = {id(p) for net in critic.networks for p in net.parameters()}
    assert len(ids) == sum(1 for net in critic.networks for p in net.parameters())


def test_permuting_every_member_permutes_the_output():
    """The ensemble is equivariant to a relabelling of its members."""
    critic = _critic(seed=3, num_critics=2)
    w, c = _window(), _chunk()
    before = critic(w, c).clone()
    with torch.no_grad():
        critic.networks[0], critic.networks[1] = (
            critic.networks[1],
            critic.networks[0],
        )
    after = critic(w, c)
    torch.testing.assert_close(after[0], before[1])
    torch.testing.assert_close(after[1], before[0])


def test_members_differ_at_init():
    critic = _critic(seed=4, num_critics=2)
    out = critic(_window(), _chunk())
    assert not torch.allclose(out[0], out[1])


def test_gradient_reaches_every_member_and_both_branches():
    critic = _critic(seed=4, num_critics=2)
    critic(_window(), _chunk()).sum().backward()
    for net in critic.networks:
        for branch in (net.state_enc, net.act_enc, net.head):
            grads = [p.grad for p in branch.parameters() if p.grad is not None]
            assert grads, f"{type(branch).__name__} received no gradient"
            assert sum(g.abs().sum() for g in grads) > 0


def test_critic_member_reads_the_whole_window_and_the_whole_chunk():
    """Both inputs must be consumed in full, at the actor encoder's width.

    If either narrows, the critic silently becomes a weaker approximator of the
    policy it is critiquing, which is the failure this design exists to prevent.
    """
    member = _critic().networks[0]
    assert member.state_enc.fc1.in_features == STEPS * OBS
    assert member.state_enc.fc1.out_features == HIDDEN
    assert member.act_enc.fc1.in_features == H * ACT
    assert member.act_enc.fc2.out_features == 2 * HIDDEN


def test_parameter_count_is_proportionate():
    """Three CondMLPs per member; guard against silently widening any of them."""
    n = sum(p.numel() for p in Critic(11, 128, 17, 16, 16, 2).parameters())
    assert n == pytest.approx(247_554, abs=5_000), f"critic is {n} params"


def test_n_obs_steps_must_be_positive():
    with pytest.raises(ValueError, match="n_obs_steps must be >= 1"):
        Critic(OBS, HIDDEN, ACT, 0, H, 2)


def test_num_critics_must_be_positive():
    with pytest.raises(ValueError, match="num_critics must be >= 1"):
        Critic(OBS, HIDDEN, ACT, STEPS, H, 0)