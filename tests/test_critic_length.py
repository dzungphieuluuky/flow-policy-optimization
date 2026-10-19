"""The critic is told how much of each planned chunk actually ran."""

import pytest
import torch

from myfpo.configuration import SJFPOConfig
from myfpo.critic import Critic


def _cfg(**kw) -> SJFPOConfig:
    base = dict(
        action_horizon=8,
        n_action_steps=4,
        n_obs_steps=4,
    )
    base.update(kw)
    cfg = SJFPOConfig(**base)
    cfg.validate()
    return cfg


def _policy_with_critic(cfg: SJFPOConfig):
    from myfpo.modeling import SJFPOPolicy

    policy = SJFPOPolicy.__new__(SJFPOPolicy)
    # nn.Module.__setattr__ refuses submodule assignment before __init__ ran.
    torch.nn.Module.__init__(policy)
    policy.config = cfg
    policy.critic = Critic(
        5,
        cfg.critic_hidden_dim if cfg.critic_hidden_dim is not None else cfg.hidden_dim,
        2,
        cfg.n_obs_steps,
        cfg.action_horizon,
        cfg.num_critics,
        condition_on_exec_length=cfg.critic_condition_on_exec_length,
        num_layers=cfg.critic_num_layers,
    )
    return policy


def test_exec_frac_normalises_by_the_horizon():
    policy = _policy_with_critic(_cfg())
    got = policy.exec_length_frac(torch.tensor([0, 4, 8]))
    torch.testing.assert_close(got, torch.tensor([0.0, 0.5, 1.0]))


def test_exec_frac_is_fp32_even_from_a_long_tensor():
    """The critic runs under autocast. An int64 column would not concatenate."""
    policy = _policy_with_critic(_cfg())
    got = policy.exec_length_frac(torch.tensor([2, 4], dtype=torch.long))
    assert got.dtype == torch.float32


def test_bootstrap_uses_n_action_steps_not_the_whole_horizon():
    """The regression guard for the row that is easiest to get wrong."""
    policy = _policy_with_critic(_cfg(action_horizon=8, n_action_steps=4))
    boot = policy.exec_length_frac(torch.full((3,), 4))
    torch.testing.assert_close(boot, torch.full((3,), 0.5))
    assert not torch.allclose(boot, torch.ones(3))


def test_flag_off_leaves_the_critic_width_unchanged():
    on = _policy_with_critic(_cfg(critic_condition_on_exec_length=True))
    off = _policy_with_critic(_cfg(critic_condition_on_exec_length=False))
    assert on.critic.input_dim - off.critic.input_dim == 1


def test_critic_hidden_dim_defaults_to_the_actor_width():
    """None must mean 'reuse hidden_dim' so old constructions are unchanged."""
    policy = _policy_with_critic(_cfg(hidden_dim=64, critic_hidden_dim=None))
    assert policy.critic.networks[0].mlp[0].out_features == 64


def test_critic_hidden_dim_overrides_the_actor_width():
    """The point of the flag: size the critic for a 348-dim Humanoid state
    without doubling the actor."""
    policy = _policy_with_critic(_cfg(hidden_dim=64, critic_hidden_dim=256))
    assert policy.critic.networks[0].mlp[0].out_features == 256
    assert policy.critic.networks[0].mlp[0].in_features == 5 * 4 + 2 * 8 + 1


@pytest.mark.parametrize("bad", [0, -3])
def test_critic_num_layers_must_be_positive(bad: int):
    with pytest.raises(ValueError, match="critic_num_layers"):
        _cfg(critic_num_layers=bad).validate()


@pytest.mark.parametrize("bad", [0, -1])
def test_critic_hidden_dim_must_be_positive(bad: int):
    with pytest.raises(ValueError, match="critic_hidden_dim"):
        _cfg(critic_hidden_dim=bad).validate()


def test_a_full_length_chunk_and_a_truncated_one_are_distinguishable():
    """End-to-end: same state and same prefix, different executed count."""
    torch.manual_seed(0)
    policy = _policy_with_critic(_cfg(critic_num_layers=2))
    states = torch.randn(1, 4, 5)
    actions = torch.zeros(1, 8, 2)
    full = policy.critic(states, actions, torch.tensor([1.0]))
    half = policy.critic(states, actions, torch.tensor([0.25]))
    assert not torch.allclose(full, half, atol=1e-6)