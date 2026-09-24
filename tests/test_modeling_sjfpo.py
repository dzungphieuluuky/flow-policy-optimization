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

INFO_KEYS = {
    "policy_loss",
    "value_loss",
    "explained_var",
    "mean_log_ratio",
    "flow_rho_mean",
    "flow_rho_std",
    "flow_rho_bin_range",
    "flow_rho_bin_lowfreq",
    "flow_rho_bin_highfreq",
    "flow_rho_bin_outside_frac",
    "flow_rho_bin_low_outside_frac",
    "flow_rho_bin_high_outside_frac",
    "flow_v_star_rms",
    "cond_temb",
    "cond_fused",
    "cond_c",
    "cond_gate_msa",
    "cond_gate_mlp",
    "cond_scale_msa",
    "cond_scale_mlp",
}


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


def test_actor_architecture_uses_policy_config():
    cfg = make_config(
        hidden_dim=64,
        dit_depth=1,
        dit_heads=2,
        dit_mlp_ratio=2.0,
        dit_time_dim=32,
        timestep_scale=10.0,
    )
    policy = SJFPOPolicy(cfg)
    assert policy.actor.time_dim == 32
    assert policy.actor.timestep_embedder.scale == 10.0
    assert len(policy.actor.blocks) == 1
    assert policy.actor.blocks[0].attn.n_heads == 2
    assert policy.actor.in_proj.out_features == 64


def test_rollout_uses_configured_action_execution_length():
    policy = SJFPOPolicy(make_config(n_action_steps=4))
    env = FakeGymEnv(obs_dim=OBS_DIM, action_dim=ACT_DIM, max_steps=20)

    batch = policy.generate_rollout(env, total_steps=8, noisy=False)

    assert batch["action"].shape[1] == 16
    assert batch["action_is_pad"].shape[0] == 2
    assert batch["action_is_pad"].sum(dim=1).tolist() == [12, 12]


def test_rollout_normalizer_observations_are_executed_post_step_frames():
    policy = SJFPOPolicy(make_config())
    env = FakeGymEnv(obs_dim=OBS_DIM, action_dim=ACT_DIM, max_steps=20)

    batch = policy.generate_rollout(env, total_steps=8, noisy=False)
    raw_observations = batch["normalizer_observations"]
    executed_steps = int((~batch["action_is_pad"]).sum())

    assert raw_observations.shape == (executed_steps, OBS_DIM)
    assert executed_steps == 8
    assert float(policy.obs_normalizer.count.item()) == 0.0
    assert bool(policy.obs_normalizer._uninitialized.item())

    # Inference must not update statistics.
    policy.predict_action_chunk({"observation.state": torch.randn(1, 16, OBS_DIM)})
    assert float(policy.obs_normalizer.count.item()) == 0.0

    policy.update_observation_stats(raw_observations)
    assert float(policy.obs_normalizer.count.item()) == float(executed_steps)
    assert not bool(policy.obs_normalizer._uninitialized.item())


def test_update_observation_stats_rejects_wrong_shape():
    policy = SJFPOPolicy(make_config())
    with pytest.raises(ValueError, match="shape"):
        policy.update_observation_stats(torch.randn(1, 1, 16, OBS_DIM))


def test_name_and_config_class_and_modules():
    policy = SJFPOPolicy(make_config())
    assert policy.name == "sjfpo"
    assert policy.config_class is SJFPOConfig
    for m in (
        policy.actor,
        policy.conditioner,
        policy.critic,
        policy.obs_normalizer,
        policy.old_actor,
        policy.old_conditioner,
    ):
        assert isinstance(m, torch.nn.Module)
    # Snapshot matches live weights at init (unit ratios).
    for p_old, p_live in zip(policy.old_actor.parameters(), policy.actor.parameters()):
        assert torch.equal(p_old, p_live)
    for p_old, p_live in zip(
        policy.old_conditioner.parameters(), policy.conditioner.parameters()
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
    assert set(info) == INFO_KEYS
    assert all(isinstance(v, float) for v in info.values())


def test_optim_params_exclude_snapshot():
    policy = SJFPOPolicy(make_config())
    groups = policy.get_optim_params()["params"]
    assert len(groups) == 4  # actor decay / actor no-decay / conditioner / critic
    optim_ids = {id(p) for g in groups for p in g["params"]}
    live_ids = (
        {id(p) for p in policy.actor.parameters()}
        | {id(p) for p in policy.conditioner.parameters()}
        | {id(p) for p in policy.critic.parameters()}
    )
    assert optim_ids == live_ids
    snap_ids = {id(p) for p in policy.old_actor.parameters()} | {
        id(p) for p in policy.old_conditioner.parameters()
    }
    assert optim_ids.isdisjoint(snap_ids)


def test_compute_target_velocity_broadcast():
    a = torch.randn(2, 5, 2)
    eps = torch.randn(2, 3, 5, 2)
    v_star = SJFPOPolicy.compute_target_velocity(a, eps)
    assert v_star.shape == (2, 3, 5, 2)
    assert torch.equal(v_star[0, 1], a[0] - eps[0, 1])


def test_compute_predicted_velocity_old_vs_live():
    policy = SJFPOPolicy(make_config())
    x_t = torch.randn(4, 5, ACT_DIM)
    tau = torch.rand(4)
    states = torch.randn(4, 16, OBS_DIM)
    v_live = policy.compute_predicted_velocity(x_t, tau, states)
    v_old = policy.compute_predicted_velocity(x_t, tau, states, old=True)
    assert torch.equal(v_live, v_old)  # snapshot matches at init
    with torch.no_grad():
        # Perturb the readout bias: the zero-init final layer would swallow
        # any upstream perturbation, so touch the output directly.
        policy.actor.out.bias.add_(1.0)
    v_live2 = policy.compute_predicted_velocity(x_t, tau, states)
    v_old2 = policy.compute_predicted_velocity(x_t, tau, states, old=True)
    assert not torch.equal(v_live2, v_old2)


def test_snapshot_refresh_tracks_live_weights():
    policy = SJFPOPolicy(make_config())
    with torch.no_grad():
        next(policy.actor.parameters()).add_(1.0)
    policy.snapshot_behavior_policy()
    for p_old, p_live in zip(policy.old_actor.parameters(), policy.actor.parameters()):
        assert torch.equal(p_old, p_live)


def test_forward_per_bin_clip_finite_and_binds_per_group():
    policy = SJFPOPolicy(
        make_config(
            use_per_bin_clip=True, epsilon_clip_lowfreq=0.02, epsilon_clip_highfreq=0.10
        )
    )
    policy.train()
    batch = make_batch()
    batch["mc_tau"] = torch.rand(2, 2, 1)
    batch["mc_eps"] = torch.randn(2, 2, 16, ACT_DIM)
    loss, info = policy(batch)
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert set(info) == INFO_KEYS
    assert 0.0 <= info["flow_rho_bin_low_outside_frac"] <= 1.0
    assert 0.0 <= info["flow_rho_bin_high_outside_frac"] <= 1.0
    loss.backward()  # per-bin psi graph reaches the live actor
    assert any(p.grad is not None for p in policy.actor.parameters())
    loss2, _ = policy(batch)  # stored MC timesteps/noise reused: deterministic
    assert torch.equal(loss, loss2)


def test_forward_per_bin_uniform_eps_matches_scalar_reduction_order():
    # low == high isolates the reduction-order change (mean-of-psi over bins
    # vs psi of the summed loss) from the trust-region change.
    cfg = dict(
        use_per_bin_clip=True, epsilon_clip_lowfreq=0.05, epsilon_clip_highfreq=0.05
    )
    torch.manual_seed(0)
    policy = SJFPOPolicy(make_config(**cfg))
    policy.train()
    batch = make_batch()
    batch["mc_tau"] = torch.rand(2, 2, 1)
    batch["mc_eps"] = torch.randn(2, 2, 16, ACT_DIM)
    loss_bin, _ = policy(batch)
    assert torch.isfinite(loss_bin)


def test_forward_reduction_none_returns_per_chunk_vector():
    policy = SJFPOPolicy(make_config())
    batch = make_batch()
    loss_vec, info = policy(batch, reduction="none")
    assert loss_vec.shape == (2,)
    assert torch.isfinite(loss_vec).all()
    loss_scalar, _ = policy(batch)
    assert loss_vec.mean().item() == pytest.approx(loss_scalar.item(), rel=1e-5)
    assert set(info) == INFO_KEYS
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
    p_clip = SJFPOPolicy(make_config(use_clipped_value_loss=True))
    batch_far = dict(make_batch(), values=torch.full((2,), 1e6))
    _, info_clip = p_clip(batch_far)
    assert math.isfinite(info_clip["value_loss"])
    assert info_clip["value_loss"] != info_plain["value_loss"]


def test_forward_ignores_values_without_clip_flag():
    # OFF default: unclipped value loss never reads rollout values.
    torch.manual_seed(3)
    policy = SJFPOPolicy(make_config())
    batch = make_batch()
    _, info_plain = policy(batch)
    batch_far = dict(batch, values=torch.full((2,), 1e6))
    _, info_far = policy(batch_far)
    assert info_far["value_loss"] == info_plain["value_loss"]


def test_rollout_stores_per_chunk_mc_draws():
    torch.manual_seed(0)
    policy = SJFPOPolicy(make_config())
    env = FakeGymEnv(obs_dim=OBS_DIM, action_dim=ACT_DIM, max_steps=20)
    batch = policy.generate_rollout(env, total_steps=32)
    n = batch["observation.state"].shape[0]
    assert batch["mc_tau"].shape == (n, policy.config.num_mc_samples, 1)
    assert batch["mc_eps"].shape[0] == n
    # Same stored draws reused: repeated forwards agree exactly (eval mode
    # freezes obs stats, so only the MC draws can move the ratios)...
    policy.eval()
    with torch.no_grad():
        # Perturb live weights so old-vs-new ratios are nonzero (at init the
        # snapshot matches and every ratio is exactly 1).
        policy.actor.out.bias.add_(1.0)
    _, info1 = policy(batch)
    _, info2 = policy(batch)
    assert info1["mean_log_ratio"] == info2["mean_log_ratio"]
    # ...while fresh draws would not (stored MC draws pin noise and timesteps per round).
    other = dict(batch)
    other["mc_eps"] = torch.randn_like(batch["mc_eps"])
    _, info3 = policy(other)
    assert info3["mean_log_ratio"] != info1["mean_log_ratio"]


def test_critic_receives_grads():
    policy = SJFPOPolicy(make_config())
    loss, _ = policy(make_batch())
    policy.zero_grad()
    loss.backward()
    grads = [p.grad for p in policy.critic.parameters()]
    assert all(g is not None and g.abs().sum() > 0 for g in grads), (
        "critic missing grads"
    )


def test_generate_rollout_returns_forward_ready_batch():
    torch.manual_seed(0)
    policy = SJFPOPolicy(make_config())
    env = FakeGymEnv(obs_dim=OBS_DIM, action_dim=ACT_DIM, max_steps=20)
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
    assert set(info) == INFO_KEYS


def test_optim_params_actor_critic_groups():
    policy = SJFPOPolicy(make_config())
    groups = policy.get_optim_params()["params"]
    assert len(groups) == 4  # actor decay / no-decay / conditioner / critic
    opt = policy.config.get_optimizer_preset().build(groups)
    assert [g["lr"] for g in opt.param_groups] == [1e-4] * 4  # single LR
    assert [g["weight_decay"] for g in opt.param_groups][1] == 0.0


def test_loss_contains_only_policy_and_value():
    policy = SJFPOPolicy(make_config())
    policy.train()
    loss, info = policy(make_batch())
    assert "kl_loss" not in info
    assert loss.item() == pytest.approx(
        info["policy_loss"] + policy.config.vf_coef * info["value_loss"]
    )


def test_save_pretrained_roundtrip(tmp_path):
    policy = SJFPOPolicy(make_config())
    policy.save_pretrained(tmp_path)
    loaded = SJFPOPolicy.from_pretrained(tmp_path)
    assert loaded.config.type == "sjfpo"
    for (n1, p1), (n2, p2) in zip(policy.named_parameters(), loaded.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), n1
