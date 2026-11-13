"""Critic Q(s, a_chunk, f): reads the whole window in order, chunk is an input, ensemble stays independent."""

import pytest
import torch

from myfpo.nn import Critic

OBS, HIDDEN, STEPS, ACT, H = 11, 32, 8, 3, 6


def _critic(
    seed: int = 0,
    num_critics: int = 1,
    *,
    condition_on_exec_length: bool = True,
    num_layers: int = 2,
) -> Critic:
    torch.manual_seed(seed)
    return Critic(
        OBS,
        HIDDEN,
        ACT,
        STEPS,
        H,
        num_critics,
        condition_on_exec_length=condition_on_exec_length,
        num_layers=num_layers,
    )


def _window(seed: int = 1, batch: int = 4) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, STEPS, OBS)


def _chunk(seed: int = 2, batch: int = 4) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, H, ACT)


def _frac(batch: int = 4, value: float = 0.5) -> torch.Tensor:
    return torch.full((batch,), value)


def _q(critic: Critic, w: torch.Tensor, c: torch.Tensor, f=None) -> torch.Tensor:
    """Forward with the length feature only when the critic wants it."""
    if critic.condition_on_exec_length and f is None:
        f = _frac(w.shape[0])
    return critic(w, c, f)


@pytest.mark.parametrize("batch", [1, 3, 64])
def test_output_is_one_value_per_sample(batch: int):
    """[num_critics, B]: the critic does not aggregate, the caller does."""
    assert _q(_critic(), _window(batch=batch), _chunk(batch=batch)).shape == (1, batch)


def test_window_length_must_match_the_declared_one():
    """The collector builds windows to num_obs_steps, so a mismatch is a real error."""
    with pytest.raises(ValueError, match=f"built for num_obs_steps={STEPS}"):
        _q(_critic(), torch.randn(2, 4, OBS), _chunk())


def test_chunk_length_must_match_the_horizon():
    with pytest.raises(ValueError, match=f"action_horizon={H}"):
        _q(_critic(), _window(), torch.randn(4, H - 1, ACT))


def test_action_dim_must_match():
    with pytest.raises(ValueError, match=f"actions feature dimension must be {ACT}"):
        _q(_critic(), _window(), torch.randn(4, H, ACT + 1))


def test_batch_mismatch_is_rejected():
    with pytest.raises(ValueError, match=r"actions batch 3 != states batch 4"):
        _q(_critic(), _window(), torch.randn(3, H, ACT))


def test_every_frame_influences_the_output():
    """Frame 0 specifically."""
    critic = _critic()
    base, chunk = _window(), _chunk()
    for frame in range(STEPS):
        moved = base.clone()
        moved[:, frame, :] += 5.0
        assert not torch.allclose(
            _q(critic, base, chunk), _q(critic, moved, chunk), atol=1e-6
        ), f"frame {frame} does not affect the value"


def test_frame_order_matters():
    """The property a mean-pool cannot have."""
    critic = _critic()
    w, c = _window(), _chunk()
    assert not torch.allclose(_q(critic, w, c), _q(critic, w.flip(1), c), atol=1e-6)


def test_a_state_offset_is_not_normalized_away():
    """A LayerNorm subtracts the window's mean, so ``w`` and ``w + c`` reach the
    first layer as one input. The critic must see how high the state sits."""
    critic = _critic()
    w, c = _window(), _chunk()
    assert not torch.allclose(
        _q(critic, w, c), _q(critic, w + 3.0, c), atol=1e-6
    ), "the state norm removed the window's offset"


def test_temporal_frequency_content_is_visible():
    """Two windows with identical per-frame value multisets must separate."""
    critic = _critic()
    w, c = _window(), _chunk()
    shuffled = w[:, torch.randperm(STEPS, generator=torch.Generator().manual_seed(7))]
    torch.testing.assert_close(w.sum(1), shuffled.sum(1))
    assert not torch.allclose(_q(critic, w, c), _q(critic, shuffled, c), atol=1e-6)


# --- the chunk is a first-class input ---------------------------------------.


def test_every_action_step_influences_the_output():
    critic = _critic()
    base, chunk = _window(), _chunk()
    for step in range(H):
        moved = chunk.clone()
        moved[:, step, :] += 5.0
        assert not torch.allclose(
            _q(critic, base, chunk), _q(critic, base, moved), atol=1e-6
        ), f"action step {step} does not affect Q"


def test_chunk_order_matters():
    """An order-insensitive chunk encoder could not tell a reversed chunk."""
    critic = _critic()
    w, c = _window(), _chunk()
    assert not torch.allclose(_q(critic, w, c), _q(critic, w, c.flip(1)), atol=1e-6)


def test_q_is_not_additive_in_state_and_chunk():
    """Q must not factor as f(s) + g(a), or the concat buys nothing."""
    critic = _critic(seed=5)
    s1, s2 = _window(seed=11), _window(seed=12)
    a1, a2 = _chunk(seed=13), _chunk(seed=14)
    cross = (_q(critic, s1, a1) + _q(critic, s2, a2)) - (
        _q(critic, s1, a2) + _q(critic, s2, a1)
    )
    assert not torch.allclose(cross, torch.zeros_like(cross), atol=1e-4), (
        "Q is additive in (state, chunk)"
    )


# --- the executed-length input ---------------------------------------------.


def test_input_dim_grows_by_exactly_one_with_the_length_feature():
    """The +1 is what makes pre-existing checkpoints unloadable, so pin it."""
    on = _critic(condition_on_exec_length=True)
    off = _critic(condition_on_exec_length=False)
    assert on.input_dim - off.input_dim == 1
    assert off.input_dim == 2 * HIDDEN


def test_flag_off_ignores_exec_frac_entirely():
    """The ablation has to reproduce the old critic exactly, so a stray argument
    must not leak in."""
    critic = _critic(condition_on_exec_length=False)
    w, c = _window(), _chunk()
    torch.testing.assert_close(critic(w, c, None), critic(w, c, _frac(w.shape[0], 0.9)))


def test_missing_exec_frac_is_rejected_rather_than_defaulted():
    """Silently defaulting f would reintroduce exactly the bug this input fixes."""
    critic = _critic(condition_on_exec_length=True)
    with pytest.raises(ValueError, match="exec_length_frac is required"):
        critic(_window(), _chunk(), None)


def test_exec_frac_shape_must_match_the_batch():
    critic = _critic(condition_on_exec_length=True)
    with pytest.raises(ValueError, match=r"exec_length_frac must be \[B\]"):
        critic(_window(), _chunk(), _frac(3))


def test_q_separates_chunks_differing_only_in_the_unexecuted_tail():
    """The regression this whole input exists for."""
    critic = _critic()
    w = _window(batch=1)
    a = torch.zeros(1, H, ACT)
    executed = 3
    a[:, :executed, :] = 0.4
    tail_a, tail_b = a.clone(), a.clone()
    tail_a[:, executed:, :] = 0.1
    tail_b[:, executed:, :] = -0.9
    q_a = critic(w, tail_a, _frac(1, executed / H))
    q_b = critic(w, tail_b, _frac(1, executed / H))
    assert not torch.allclose(q_a, q_b, atol=1e-6)


def test_exec_frac_itself_moves_q():
    """Same state and chunk, different executed fraction."""
    critic = _critic()
    w, c = _window(batch=1), _chunk(batch=1)
    assert not torch.allclose(
        critic(w, c, _frac(1, 0.25)), critic(w, c, _frac(1, 1.0)), atol=1e-6
    )


def test_gradient_reaches_the_length_column():
    """Guards a detached or accidentally non-differentiable feature."""
    critic = _critic()
    critic(_window(), _chunk(), _frac(4)).sum().backward()
    first = critic.networks[0].mlp[0].weight
    assert first.grad is not None
    assert first.grad[:, -1].abs().sum() > 0, "exec_length_frac column got no gradient"


# --- ensemble --------------------------------------------------------------.


def test_num_critics_two_min_le_mean_contract():
    """The caller's aggregator contract holds for the 2-member ensemble:
    min (backup) is never above mean (value loss), members stay distinct."""
    out = _q(_critic(seed=3, num_critics=2), _window(), _chunk())
    assert out.shape == (2, 4)
    assert not torch.allclose(out[0], out[1])
    assert (out.amin(dim=0) <= out.mean(dim=0)).all()


def test_members_are_independent_not_one_wide_mlp():
    """Perturbing one member must not move the other member."""
    critic = _critic(seed=3, num_critics=2)
    w, c = _window(), _chunk()
    before = _q(critic, w, c).clone()
    with torch.no_grad():
        critic.networks[1].mlp[-1].weight.add_(1.0)
    after = _q(critic, w, c)
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
    w, c = _window(), _chunk()
    before = _q(critic, w, c).clone()
    with torch.no_grad():
        critic.networks[0], critic.networks[1] = critic.networks[1], critic.networks[0]
    after = _q(critic, w, c)
    torch.testing.assert_close(after[0], before[1])
    torch.testing.assert_close(after[1], before[0])


def test_members_differ_at_init():
    critic = _critic(seed=4, num_critics=2)
    out = _q(critic, _window(), _chunk())
    assert not torch.allclose(out[0], out[1])


def test_gradient_reaches_every_member():
    critic = _critic(seed=4, num_critics=2)
    _q(critic, _window(), _chunk()).sum().backward()
    for net in critic.networks:
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert grads
        assert sum(g.abs().sum() for g in grads) > 0


def test_the_window_trunk_reads_every_frame_and_the_chunk_is_a_full_input():
    """Both inputs must be consumed in full. If either narrows, the critic
    silently becomes a weaker approximator of the policy it is critiquing."""
    member = _critic().networks[0]
    first = member.mlp[0]
    assert member.state_encoder.frame.fc1.in_features == OBS
    assert first.in_features == 2 * HIDDEN + 1
    assert first.out_features == HIDDEN


def test_the_trunk_width_is_separate_from_the_critic_width():
    """A narrow window trunk must not force a narrow Q MLP, or the reverse."""
    member = Critic(OBS, HIDDEN, ACT, STEPS, H, 1, state_hidden=16).networks[0]
    assert member.mlp[0].in_features == 2 * 16 + 1


def test_num_layers_controls_hidden_width():
    """One shared hidden_dim serves Pendulum (obs 3) and Humanoid (obs 348). A
    separate critic_num_layers is how that gets tuned without touching the actor."""
    deep = _critic(num_layers=4).networks[0]
    linears = [m for m in deep.mlp if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 5, [type(m).__name__ for m in deep.mlp]
    assert all(m.out_features == HIDDEN for m in linears[:-1])
    assert linears[-1].out_features == 1


def test_parameter_count_is_proportionate():
    """Guard against silently widening or deepening a member."""
    n = sum(
        p.numel()
        for p in Critic(11, 128, 17, 16, 16, 2).parameters()
    )
    # state_hidden defaults to hidden_dim=128. Per member:
    # norm_state 352, window trunk 84352, norm_action 544, act_in 2304,
    # act_attn 66048 + its norm 128, obs_to_act 66048, act_to_obs 66048,
    # two cross norms 256, pool 2*128+1=257 -> MLP 50177
    # -> 336257, times 2 members.
    assert n == pytest.approx(672_514, abs=50), f"critic is {n} params"


def test_num_obs_steps_must_be_positive():
    with pytest.raises(ValueError, match="max_obs_steps must be >= 1"):
        Critic(OBS, HIDDEN, ACT, 0, H, 2)


