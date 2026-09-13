"""Tests for src/sjfpo/modeling_sjfpo.py (LeRobot-native SJFPO policy)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402

from sjfpo.configuration_sjfpo import SJFPOConfig  # noqa: E402
from sjfpo.modeling_sjfpo import SJFPOPolicy  # noqa: E402
from tests.fixtures.fake_env import FakeGymEnv  # noqa: E402

OBS_DIM, ACT_DIM = 4, 2


def make_config(**overrides) -> SJFPOConfig:
    base = dict(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (OBS_DIM,))
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT_DIM,))},
        num_mc_samples=2,
        dit_depth=2,
    )
    base.update(overrides)
    return SJFPOConfig(**base)


def make_batch(n: int = 2, h: int = 16, pad_tail: int = 0) -> dict:
    is_pad = torch.zeros(n, h, dtype=torch.bool)
    if pad_tail:
        is_pad[:, -pad_tail:] = True
    return {
        "observation.state": torch.randn(n, 16, OBS_DIM),
        "action": torch.randn(n, h, ACT_DIM).clamp(-1.0, 1.0),
        "action_is_pad": is_pad,
        "advantages": torch.randn(n) * 0.1,
        "returns": torch.randn(n) * 10.0,
    }


def test_name_and_config_class_and_modules():
    policy = SJFPOPolicy(make_config())
    assert policy.name == "sjfpo"
    assert policy.config_class is SJFPOConfig
    for m in (
        policy.actor,
        policy.spectral,
        policy.critic,
        policy.obs_normalizer,
        policy.old_actor,
    ):
        assert isinstance(m, torch.nn.Module)
    # Snapshot matches live weights at init (unit ratios).
    for p_old, p_live in zip(
        policy.old_actor.parameters(), policy.actor.parameters()
    ):
        assert torch.equal(p_old, p_live)


def test_predict_action_chunk_shape_and_bounds():
    policy = SJFPOPolicy(make_config())
    chunk = policy.predict_action_chunk(make_batch(n=3))
    assert chunk.shape == (3, 16, ACT_DIM)
    assert chunk.dtype == torch.float32
    assert torch.isfinite(chunk).all()
    assert (chunk <= 1.0).all() and (chunk >= -1.0).all()


def test_select_action_buffers_and_reset():
    policy = SJFPOPolicy(make_config())
    obs = torch.randn(OBS_DIM)
    a1 = policy.select_action({"observation.state": obs})
    assert a1.shape == (ACT_DIM,)
    assert len(policy._action_buffer) == 7  # 8-step prefix, one consumed
    policy.reset()
    assert len(policy._action_buffer) == 0 and len(policy._frame_buffer) == 0
    a1_again = policy.select_action({"observation.state": obs})
    assert torch.equal(a1, a1_again)  # deterministic replans match


def test_forward_finite_loss_and_native_dict():
    policy = SJFPOPolicy(make_config())
    policy.train()
    loss, info = policy(make_batch(pad_tail=8))
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert set(info) == {"policy_loss", "value_loss", "mean_log_ratio"}
    assert all(isinstance(v, float) for v in info.values())


def test_optim_params_exclude_snapshot():
    policy = SJFPOPolicy(make_config())
    optim_ids = {id(p) for p in policy.get_optim_params()["params"]}
    live_ids = {id(p) for p in policy.actor.parameters()} | {
        id(p) for p in policy.critic.parameters()
    }
    assert optim_ids == live_ids
    snap_ids = {id(p) for p in policy.old_actor.parameters()}
    assert optim_ids.isdisjoint(snap_ids)


def test_compute_target_velocity_broadcast():
    a = torch.randn(2, 5, 2)
    eps = torch.randn(2, 3, 5, 2)
    v_star = SJFPOPolicy.compute_target_velocity(a, eps)
    assert v_star.shape == (2, 3, 5, 2)
    assert torch.equal(v_star[0, 1], eps[0, 1] - a[0])


def test_compute_predicted_velocity_old_vs_live():
    policy = SJFPOPolicy(make_config())
    x_t = torch.randn(4, 5, ACT_DIM)
    tau = torch.rand(4)
    condition = torch.randn(4, 5, OBS_DIM)
    states = torch.randn(4, 16, OBS_DIM)
    v_live = policy.compute_predicted_velocity(x_t, tau, condition, states)
    v_old = policy.compute_predicted_velocity(x_t, tau, condition, states, old=True)
    assert torch.equal(v_live, v_old)  # snapshot matches at init
    with torch.no_grad():
        # Perturb the readout bias: the zero-init final layer would swallow
        # any upstream perturbation, so touch the output directly.
        policy.actor.out.bias.add_(1.0)
    v_live2 = policy.compute_predicted_velocity(x_t, tau, condition, states)
    v_old2 = policy.compute_predicted_velocity(x_t, tau, condition, states, old=True)
    assert not torch.equal(v_live2, v_old2)


def test_snapshot_refresh_tracks_live_weights():
    policy = SJFPOPolicy(make_config())
    with torch.no_grad():
        next(policy.actor.parameters()).add_(1.0)
    policy.snapshot_behavior_policy()
    for p_old, p_live in zip(
        policy.old_actor.parameters(), policy.actor.parameters()
    ):
        assert torch.equal(p_old, p_live)


def test_forward_reduction_none_returns_per_chunk_vector():
    policy = SJFPOPolicy(make_config())
    batch = make_batch()
    loss_vec, info = policy(batch, reduction="none")
    assert loss_vec.shape == (2,)
    assert torch.isfinite(loss_vec).all()
    loss_scalar, _ = policy(batch)
    assert loss_vec.mean().item() == pytest.approx(loss_scalar.item(), rel=1e-5)
    assert set(info) == {"policy_loss", "value_loss", "mean_log_ratio"}
    with pytest.raises(ValueError, match="reduction"):
        policy(batch, reduction="sum")


def test_predict_noisy_kwarg_and_select_extra_kwargs():
    torch.manual_seed(0)
    policy = SJFPOPolicy(make_config())
    batch = {"observation.state": torch.randn(2, 16, OBS_DIM)}
    det = policy.predict_action_chunk(batch)
    noisy = policy.predict_action_chunk(batch, noise=True)
    assert noisy.shape == det.shape
    assert torch.isfinite(noisy).all()
    assert not torch.equal(noisy, det)
    obs = torch.randn(OBS_DIM)
    a = policy.select_action({"observation.state": obs}, foo=1)
    assert a.shape == (ACT_DIM,)


def test_forward_deterministic_same_input():
    torch.manual_seed(0)
    p1 = SJFPOPolicy(make_config())
    torch.manual_seed(0)
    loss1, info1 = p1(make_batch())
    torch.manual_seed(0)
    p2 = SJFPOPolicy(make_config())
    torch.manual_seed(0)
    loss2, info2 = p2(make_batch())
    assert loss1.item() == loss2.item()
    assert info1 == info2


def test_forward_nan_raises_aborted():
    from sjfpo.modeling_sjfpo import SJFPOAborted

    policy = SJFPOPolicy(make_config())
    batch = make_batch()
    batch["returns"] = torch.full((2,), float("nan"))
    with pytest.raises(SJFPOAborted):
        policy(batch)


def test_forward_clipped_value_loss_engages():
    torch.manual_seed(3)
    p_plain = SJFPOPolicy(make_config())
    _, info_plain = p_plain(make_batch())
    torch.manual_seed(3)
    p_clip = SJFPOPolicy(make_config())
    batch_far = dict(make_batch(), values=torch.full((2,), 1e6))
    _, info_clip = p_clip(batch_far)
    assert math.isfinite(info_clip["value_loss"])
    assert info_clip["value_loss"] != info_plain["value_loss"]


def test_critic_encoder_receives_grads():
    policy = SJFPOPolicy(make_config())
    loss, _ = policy(make_batch())
    policy.zero_grad()
    loss.backward()
    grads = [p.grad for p in policy.critic.state_encoder.parameters()]
    assert all(
        g is not None and g.abs().sum() > 0 for g in grads
    ), "critic state encoder missing grads"


def test_generate_rollout_returns_forward_ready_batch():
    torch.manual_seed(0)
    policy = SJFPOPolicy(make_config())
    env = FakeGymEnv(obs_dim=OBS_DIM, act_dim=ACT_DIM, max_steps=20)
    batch = policy.generate_rollout(env, total_steps=32)
    n = batch["observation.state"].shape[0]
    assert n >= 1
    assert batch["observation.state"].shape == (n, 16, OBS_DIM)
    assert batch["action"].shape == (n, 16, ACT_DIM)
    assert batch["action_is_pad"].shape == (n, 16)
    assert batch["action_is_pad"].sum() > 0  # truncated tail present
    for k in ("advantages", "returns", "values"):
        assert batch[k].shape == (n,)
        assert torch.isfinite(batch[k]).all()
    policy.train()
    loss, info = policy(batch)
    assert torch.isfinite(loss)
    assert set(info) == {"policy_loss", "value_loss", "mean_log_ratio"}


def test_save_pretrained_roundtrip(tmp_path):
    policy = SJFPOPolicy(make_config())
    policy.save_pretrained(tmp_path)
    loaded = SJFPOPolicy.from_pretrained(tmp_path)
    assert loaded.config.type == "sjfpo"
    for (n1, p1), (n2, p2) in zip(policy.named_parameters(), loaded.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), n1
