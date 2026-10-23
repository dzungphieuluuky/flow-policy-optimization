"""Check that every reachable ValueError still names its condition.

The message text is not pinned. A guard is identified by the field or the
condition it names, and a reworded message does not break the suite.
"""

from __future__ import annotations

import pytest
import torch

# : Guard count the table below is expected to cover.
EXPECTED_CASES = 79

_CASES: list[tuple[str, object]] = []


def case(cid):
    def wrap(thunk):
        _CASES.append((cid, thunk))
        return thunk

    return wrap


def _raises(thunk) -> str:
    try:
        thunk()
    except ValueError as exc:
        return str(exc)
    raise AssertionError("expected ValueError, nothing was raised")


# --- configuration.MyfpoConfig -----------------------------------------.
from myfpo.configuration import MyfpoConfig  # noqa: E402

_BAD_CONFIG = {
    "action_perturb_std": -1.0,
    "advantage_clamp_pos": 0.0,
    "batch_size": 0,
    "cfm_clamp": 0.0,
    "cfm_loss_clamp_negative_advantages_max": 0.0,
    "cfm_loss_reduction": "median",
    "chunks_per_update": 0,
    "clip_param": 1.0,
    "cfm_loss_huber_delta": 0.0,
    "critic_hidden_dim": 0,
    "critic_lr": 0.0,
    "critic_num_layers": 0,
    "eval_max_episodes": 0,
    "gae_lambda": 1.5,
    "gamma": 0.0,
    "grad_accum_steps": 0,
    "log_ratio_cap": 0.0,
    "max_grad_norm": 0.0,
    "min_sampled_timestep": -1.0,
    "noise_sampling_mode": "laplace",
    "num_critics": 0,
    "num_steps_noise_start": 0,
    "num_envs": 0,
    "num_steps_zero_start": 0,
    "num_learning_epochs": 0,
    "num_mc_samples": 0,
    "optimizer_weight_decay": -1.0,
    "policy_lr": 0.0,
    "ratio_mode": "softmax",
    "report_to": "mlflow",
    "cfm_average_group_size": 0,

    "temporal_probe_every": -1,
    "timestep_inverse_cdf_beta": 0.0,
    "timestep_sampling_mode": "cosine",
    "transported_clip_value": 0.0,
    "trust_region_mode": "grpo",
    "value_loss_coef": -1.0,
}
for _field, _bad in sorted(_BAD_CONFIG.items()):
    case(f"config:{_field}")(lambda f=_field, b=_bad: MyfpoConfig(**{f: b}))

# An even count is required: sample_eps draws half and mirrors it.
case("config:num_mc_samples-odd")(lambda: MyfpoConfig(num_mc_samples=3))


@case("config:missing-input-features")
def _missing_input_features() -> None:
    MyfpoConfig().validate_features()


# --- network constructors ----------------------------------------------------.
from myfpo.layers import CrossAttention, DiTBlock, SelfAttention  # noqa: E402
from myfpo.critic import Critic, CriticMember  # noqa: E402
from myfpo.processor import RunningMeanStd, StateEncoder  # noqa: E402
from myfpo.flow import MyfpoFlowModel  # noqa: E402
from myfpo.layers import CondMLP, SinusoidalEncoding, MishFFN  # noqa: E402

case("layers:MishFFN:zero-dims")(lambda: MishFFN(0, 4, 4))
case("layers:CondMLP:zero-dims")(lambda: CondMLP(0, 4, 4))
case("layers:SinusoidalEncoding:dim")(lambda: SinusoidalEncoding(1))
case("encoder:RunningMeanStd:dim")(lambda: RunningMeanStd(0))
case("encoder:StateEncoder:hidden")(lambda: StateEncoder(3, 1, 4))
case("encoder:StateEncoder:steps")(lambda: StateEncoder(3, 8, 0))
case("attention:SelfAttention:hidden")(lambda: SelfAttention(0, 2))
case("attention:SelfAttention:heads")(lambda: SelfAttention(8, 0))
case("attention:SelfAttention:split")(lambda: SelfAttention(8, 3))
case("attention:CrossAttention:hidden")(lambda: CrossAttention(0, 2))
case("attention:CrossAttention:heads")(lambda: CrossAttention(8, 0))
case("attention:CrossAttention:split")(lambda: CrossAttention(8, 3))
case("attention:DiTBlock:hidden")(lambda: DiTBlock(1, 2, 2.0))
case("attention:DiTBlock:ratio")(lambda: DiTBlock(8, 2, 0.0))
case("flow:actor:action_dim")(lambda: MyfpoFlowModel(0, 8, 128, 3, 2, 2.0, 64))
case("flow:actor:horizon")(lambda: MyfpoFlowModel(3, 0, 128, 3, 2, 2.0, 64))
case("flow:actor:depth")(lambda: MyfpoFlowModel(3, 8, 128, 0, 2, 2.0, 64))
case("flow:actor:mlp_ratio")(lambda: MyfpoFlowModel(3, 8, 128, 3, 2, 0.0, 64))
case("critic:member:action_dim")(lambda: CriticMember(3, 0, 8, 4, 8, num_layers=2))
case("critic:member:horizon")(lambda: CriticMember(3, 2, 8, 4, 0, num_layers=2))
case("critic:member:hidden")(lambda: CriticMember(3, 2, 0, 4, 8, num_layers=2))
case("critic:member:layers")(lambda: CriticMember(3, 2, 8, 4, 8, num_layers=0))
case("critic:ensemble:obs_dim")(lambda: Critic(0, 8, 2, 4, 8, num_critics=2))
case("critic:ensemble:critics")(lambda: Critic(3, 8, 2, 4, 8, num_critics=0))
case("critic:ensemble:obs_steps")(lambda: Critic(3, 8, 2, 0, 8, num_critics=2))

# DiTBlock.forward, not the constructor: a batch disagreement between x and c.
case("attention:DiTBlock:forward:token-batch")(
    lambda: DiTBlock(8, 2, 2.0)(torch.randn(2, 4, 8), torch.randn(3, 4, 8))
)
case("attention:DiTBlock:forward:cond-batch")(
    lambda: DiTBlock(8, 2, 2.0)(torch.randn(2, 4, 8), torch.randn(3, 8))
)

# --- pure functions ----------------------------------------------------------.
from myfpo.fpo_core import (  # noqa: E402
    compute_cfm_loss,
    compute_policy_ratio,
    broadcast_over_mc_axis,
    sample_action,
    sample_timesteps,
)
from myfpo.rollout import compute_chunk_advantages  # noqa: E402


def _gae(**kw):
    base = dict(
        step_rewards=torch.zeros(4),
        q_current_chunk=torch.zeros(2),
        q_next_state=torch.zeros(2),
        chunk_starts=torch.zeros(2, dtype=torch.long),
        chunk_lengths=torch.ones(2, dtype=torch.long),
        chunk_terminated=torch.zeros(2, dtype=torch.bool),
        chunk_size=2,
    )
    base.update(kw)
    return compute_chunk_advantages(**base)


case("gae:gamma")(lambda: _gae(gamma=1.5))
case("gae:gae_lambda")(lambda: _gae(gae_lambda=-0.1))
case("gae:lengths")(lambda: _gae(chunk_lengths=torch.ones(5, dtype=torch.long)))
case("gae:requires-truncated")(lambda: _gae(gae_lambda=0.95))


def _mc_t(**kw):
    base = dict(
        action=torch.zeros(2, 4, 3),
        eps=torch.zeros(2, 2, 4, 3),
        t=torch.zeros(2, 2, 4, 1),
    )
    base.update(kw)
    return broadcast_over_mc_axis(**base)


case("fpo_core:mc:eps-shape")(lambda: _mc_t(eps=torch.zeros(2, 2, 4, 2)))
case("fpo_core:mc:t-shape")(lambda: _mc_t(t=torch.zeros(2, 2)))
case("fpo_core:cfm:vhat-mismatch")(
    lambda: compute_cfm_loss(
        torch.zeros(2, 4, 3), torch.zeros(2, 5, 3), MyfpoConfig(), "sample"
    )
)
case("fpo_core:cfm:reduce-mode")(
    lambda: compute_cfm_loss(
        torch.zeros(2, 4, 3), torch.zeros(2, 4, 3), MyfpoConfig(), "mean"
    )
)
case("fpo_core:sample_action:device")(
    lambda: sample_action(
        lambda x, _t, _c: x,
        (2, 4, 3),
        None,
        2,
        torch.zeros(2, 4, 3),
        None,
        t_min=0.0,
        t_max=1.0,
    )
)
case("fpo_core:sample_action:shape")(
    lambda: sample_action(
        lambda x, _t, _c: x,
        (2, 4, 3),
        None,
        2,
        torch.zeros(2, 1, 3),
        "cpu",
        t_min=0.0,
        t_max=1.0,
    )
)
case("fpo_core:sample_action:t_min")(
    lambda: sample_action(
        lambda x, _t, _c: x,
        (2, 4, 3),
        None,
        2,
        torch.zeros(2, 4, 3),
        "cpu",
        t_min=0.5,
        t_max=0.4,
    )
)
case("fpo_core:sample_timesteps:mode")(
    lambda: sample_timesteps(
        4, MyfpoConfig(timestep_sampling_mode="cosine"), "cpu", None
    )
)
case("fpo_core:ratio:velocity-inputs")(
    lambda: compute_policy_ratio(
        torch.ones(1, 1, 1),
        torch.ones(1, 1, 1),
        MyfpoConfig(ratio_mode="linear"),
    )
)

_IDS = [cid for cid, _ in _CASES]
_THUNKS = dict(_CASES)

# The word each message has to carry. The field or condition it names is the
# contract; the wording around it is not.
KEYWORDS = {
    'attention:DiTBlock:forward:cond-batch': 'batch',
    'attention:DiTBlock:forward:token-batch': 'batch',
    'attention:DiTBlock:hidden': 'hidden_dim',
    'attention:DiTBlock:ratio': 'mlp_ratio',
    'attention:SelfAttention:heads': 'n_heads',
    'attention:SelfAttention:hidden': 'hidden_dim',
    'attention:SelfAttention:split': 'split',
    'attention:CrossAttention:heads': 'n_heads',
    'attention:CrossAttention:hidden': 'hidden_dim',
    'attention:CrossAttention:split': 'split',
    'config:action_perturb_std': 'action_perturb_std',
    'config:advantage_clamp_pos': 'advantage_clamp_pos',
    'config:batch_size': 'batch_size',
    'config:cfm_average_group_size': 'cfm_average_group_size',
    'config:cfm_clamp': 'cfm_clamp',
    'config:cfm_loss_clamp_negative_advantages_max': 'cfm_loss_clamp_negative_advantages_max',
    'config:cfm_loss_huber_delta': 'cfm_loss_huber_delta',
    'config:cfm_loss_reduction': 'cfm_loss_reduction',
    'config:chunks_per_update': 'chunks_per_update',
    'config:clip_param': 'clip_param',
    'config:critic_hidden_dim': 'critic_hidden_dim',
    'config:critic_lr': 'critic_lr',
    'config:critic_num_layers': 'critic_num_layers',
    'config:eval_max_episodes': 'eval_max_episodes',
    'config:gae_lambda': 'gae_lambda',
    'config:gamma': 'gamma',
    'config:grad_accum_steps': 'grad_accum_steps',
    'config:log_ratio_cap': 'log_ratio_cap',
    'config:max_grad_norm': 'max_grad_norm',
    'config:min_sampled_timestep': 'min_sampled_timestep',
    'config:missing-input-features': 'input_features',
    'config:noise_sampling_mode': 'noise_sampling_mode',
    'config:num_critics': 'num_critics',
    'config:num_envs': 'num_envs',
    'config:num_learning_epochs': 'num_learning_epochs',
    'config:num_mc_samples': 'num_mc_samples',
    'config:num_mc_samples-odd': 'num_mc_samples',
    'config:num_steps_noise_start': 'num_steps_noise_start',
    'config:num_steps_zero_start': 'num_steps_zero_start',
    'config:optimizer_weight_decay': 'optimizer_weight_decay',
    'config:policy_lr': 'policy_lr',
    'config:ratio_mode': 'ratio_mode',
    'config:report_to': 'report_to',
    'config:temporal_probe_every': 'temporal_probe_every',
    'config:timestep_inverse_cdf_beta': 'timestep_inverse_cdf_beta',
    'config:timestep_sampling_mode': 'timestep_sampling_mode',
    'config:transported_clip_value': 'transported_clip_value',
    'config:trust_region_mode': 'trust_region_mode',
    'config:value_loss_coef': 'value_loss_coef',
    'critic:ensemble:critics': 'num_critics',
    'critic:ensemble:obs_dim': 'obs_dim',
    'critic:ensemble:obs_steps': 'n_obs_steps',
    'critic:member:action_dim': 'action_dim',
    'critic:member:hidden': 'hidden_dim',
    'critic:member:horizon': 'action_horizon',
    'critic:member:layers': 'num_layers',

    'encoder:RunningMeanStd:dim': 'dim',
    'encoder:StateEncoder:hidden': 'hidden_dim',
    'encoder:StateEncoder:steps': 'max_obs_steps',
    'flow:actor:action_dim': 'action_dim',
    'flow:actor:depth': 'dit_depth',
    'flow:actor:horizon': 'action_horizon',
    'flow:actor:mlp_ratio': 'dit_mlp_ratio',
    'fpo_core:cfm:reduce-mode': 'reduce_mode',
    'fpo_core:cfm:vhat-mismatch': 'v_hat shape',
    'fpo_core:mc:eps-shape': 'eps shape',
    'fpo_core:mc:t-shape': 't shape',
    'fpo_core:sample_action:device': 'device',
    'fpo_core:sample_action:shape': 'eps shape',
    'fpo_core:sample_action:t_min': 't_min',
    'fpo_core:sample_timesteps:mode': 'timestep_sampling_mode',
    'fpo_core:ratio:velocity-inputs': 'velocity_old',
    'gae:gae_lambda': 'gae_lambda',
    'gae:gamma': 'gamma',
    'gae:lengths': 'chunk_lengths',
    'gae:requires-truncated': 'chunk_truncated',
    'layers:CondMLP:zero-dims': 'dimensions must be positive',
    'layers:SinusoidalEncoding:dim': 'SinusoidalEncoding',
    'layers:MishFFN:zero-dims': 'dimensions must be positive',
}


def test_every_case_has_a_keyword():
    """A guard dropped from the table would stop being checked silently."""
    assert len(_CASES) == EXPECTED_CASES, (
        f"expected {EXPECTED_CASES} guarded cases, found {len(_CASES)}. "
        "Update EXPECTED_CASES deliberately when a guard is added or removed."
    )
    missing = sorted(set(_IDS) - set(KEYWORDS))
    assert not missing, f"no keyword for: {missing}"
    extra = sorted(set(KEYWORDS) - set(_IDS))
    assert not extra, f"keyword for a case that does not exist: {extra}"


@pytest.mark.parametrize("cid", _IDS)
def test_guard_message_names_its_condition(cid):
    """The message still names the field or condition it guards."""
    actual = _raises(_THUNKS[cid])
    keyword = KEYWORDS[cid]
    assert keyword in actual, (
        f"{cid}: the message no longer mentions {keyword!r}: {actual!r}"
    )
