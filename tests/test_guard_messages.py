"""Snapshot the exact text of every reachable ValueError message."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

SNAPSHOT = Path(__file__).with_name("guard_messages.json")

# : Guard count the table below is expected to cover.
EXPECTED_CASES = 76

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


# --- configuration_sjfpo.SJFPOConfig -----------------------------------------.
from myfpo.configuration_sjfpo import SJFPOConfig

_BAD_CONFIG = {
    "action_perturb_std": -1.0,
    "actor_lr_warmup_steps": -1,
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
    "critic_lr_warmup_steps": -1,
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
    "report_to": "mlflow",
    "cfm_average_group_size": 0,
    "sigreg_mode": "barlow",
    "sigreg_coef": -1.0,
    "sigreg_sketch_dim": 1,
    "timestep_inverse_cdf_beta": 0.0,
    "timestep_sampling_mode": "cosine",
    "transported_clip_value": 0.0,
    "trust_region_mode": "grpo",
    "value_loss_coef": -1.0,
}
for _field, _bad in sorted(_BAD_CONFIG.items()):
    case(f"config:{_field}")(lambda f=_field, b=_bad: SJFPOConfig(**{f: b}))


@case("config:missing-input-features")
def _missing_input_features() -> None:
    SJFPOConfig().validate_features()


# --- network constructors ----------------------------------------------------.
from myfpo.attention import DiTBlock, SelfAttention
from myfpo.critic import Critic, CriticMember
from myfpo.encoder import RunningMeanStd, StateEncoder
from myfpo.flow import SJFPOFlowModel
from myfpo.layers import CondMLP, SinusoidalEncoding, SwiGLU

case("layers:SwiGLU:zero-dims")(lambda: SwiGLU(0, 4, 4))
case("layers:CondMLP:zero-dims")(lambda: CondMLP(0, 4, 4))
case("layers:SinusoidalEncoding:dim")(lambda: SinusoidalEncoding(1))
case("encoder:RunningMeanStd:dim")(lambda: RunningMeanStd(0))
case("encoder:StateEncoder:hidden")(lambda: StateEncoder(3, 1, 4))
case("encoder:StateEncoder:steps")(lambda: StateEncoder(3, 8, 0))
case("attention:SelfAttention:hidden")(lambda: SelfAttention(0, 2))
case("attention:SelfAttention:heads")(lambda: SelfAttention(8, 0))
case("attention:SelfAttention:split")(lambda: SelfAttention(8, 3))
case("attention:DiTBlock:hidden")(lambda: DiTBlock(1, 2, 2.0))
case("attention:DiTBlock:ratio")(lambda: DiTBlock(8, 2, 0.0))
case("flow:actor:action_dim")(lambda: SJFPOFlowModel(0, 8, 128, 3, 2, 2.0, 64))
case("flow:actor:horizon")(lambda: SJFPOFlowModel(3, 0, 128, 3, 2, 2.0, 64))
case("flow:actor:depth")(lambda: SJFPOFlowModel(3, 8, 128, 0, 2, 2.0, 64))
case("flow:actor:mlp_ratio")(lambda: SJFPOFlowModel(3, 8, 128, 3, 2, 0.0, 64))
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
from myfpo.fpo_core import (
    compute_cfm_loss,
    broadcast_over_mc_axis,
    sample_action,
    sample_timesteps,
)
from myfpo.gae import compute_chunk_advantages


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
        torch.zeros(2, 4, 3), torch.zeros(2, 5, 3), SJFPOConfig(), "sample"
    )
)
case("fpo_core:cfm:reduce-mode")(
    lambda: compute_cfm_loss(
        torch.zeros(2, 4, 3), torch.zeros(2, 4, 3), SJFPOConfig(), "mean"
    )
)
case("fpo_core:sample_action:device")(
    lambda: sample_action(
        lambda x, t, c: x, (2, 4, 3), None, 2, torch.zeros(2, 4, 3), None
    )
)
case("fpo_core:sample_action:shape")(
    lambda: sample_action(
        lambda x, t, c: x, (2, 4, 3), None, 2, torch.zeros(2, 1, 3), "cpu"
    )
)
case("fpo_core:sample_timesteps:mode")(
    lambda: sample_timesteps(
        4, SJFPOConfig(timestep_sampling_mode="cosine"), "cpu", None
    )
)

_IDS = [cid for cid, _ in _CASES]
_THUNKS = dict(_CASES)


def _collect() -> dict[str, str]:
    return {cid: _raises(_THUNKS[cid]) for cid in _IDS}


def test_snapshot_is_complete(request):
    """A guard dropped from the table would stop being checked silently."""
    assert len(_CASES) == EXPECTED_CASES, (
        f"expected {EXPECTED_CASES} guarded cases, found {len(_CASES)}. "
        "Update EXPECTED_CASES deliberately when a guard is added or removed."
    )
    if request.config.getoption("--snapshot-regen"):
        return  # the file is being rewritten at session end.
    snap = json.loads(SNAPSHOT.read_text())
    missing = sorted(set(_IDS) - set(snap))
    assert not missing, f"snapshot is missing: {missing}"


@pytest.mark.parametrize("cid", _IDS)
def test_guard_message_matches_snapshot(cid, request):
    actual = _raises(_THUNKS[cid])
    if request.config.getoption("--snapshot-regen"):
        return
    snap = json.loads(SNAPSHOT.read_text())
    assert actual == snap[cid], (
        f"message changed for {cid}\n"
        f"  was: {snap[cid]!r}\n  now: {actual!r}\n"
        "If deliberate, rerun with --snapshot-regen and read the diff."
    )
