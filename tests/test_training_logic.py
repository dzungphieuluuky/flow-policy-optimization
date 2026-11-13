"""Unit test: training logic produces expected losses and updates."""
import inspect
import warnings

import torch
from myfpo.train import (
    make_optimizer,
    make_policy,
    policy_update_step,
    run_training_loop,
    split_grad_params,
    update_metrics,
)
from myfpo.config import MyfpoConfig
from fixtures import make_batch, make_test_policy
from myfpo.nn import cudagraph_safe


def test_observation_stats_land_before_the_update():
    """The batch normalizes at load time (``modeling.to_device_batch``), so a stats
    update after ``train_update`` leaves the first rollout unnormalized."""
    source = inspect.getsource(run_training_loop)
    # Frozen-stats invariant: stats updated after policy_update_step, before target_critic update.
    assert source.index("update_observation_stats") > source.index("policy_update_step(")
    assert source.index("update_observation_stats") < source.index("update_target_critic()")

def test_training_logic():
    cfg = MyfpoConfig(
        action_horizon=4,
        num_action_steps=2,
        hidden_dim=16,
    )
    policy = make_policy(5, 3, cfg, device="cpu")
    batch = {
        "observation.state": torch.zeros(1, cfg.num_obs_steps, 5),
        "action": torch.zeros(1, cfg.action_horizon, 3),
        "action_is_pad": torch.zeros(1, cfg.action_horizon, dtype=torch.bool),
        "chunk_truncated": torch.zeros(1, dtype=torch.bool),
        "raw_advantages": torch.ones(1),
        "normalized_advantages": torch.ones(1),
        "raw_returns": torch.ones(1),
        "mc_t": torch.ones(1, cfg.num_mc_samples, cfg.action_horizon, 1),
        "mc_eps": torch.randn(1, cfg.num_mc_samples, cfg.action_horizon, 3),
        "behavior_cfm_loss": torch.ones(1, cfg.num_mc_samples),
        "behavior_cfm_loss_token": torch.ones(1, cfg.num_mc_samples, cfg.action_horizon),
        "normalizer_observations": torch.zeros(10, 5),
    }
    loss = policy(batch)
    assert loss.dim() == 0, f"loss must be scalar, got {loss.shape}"
    assert torch.isfinite(loss).all(), "loss must be finite"
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters() if p.requires_grad)
    assert has_grad, "actor gradients must be non-zero"


def test_one_policy_update_runs():
    """The update path must execute end to end.

    ``autocast_context`` was called but never imported, so the first update of
    a run raised NameError. Nothing in the suite called this path: the loss
    tests call ``policy(batch)`` directly.
    """
    cfg = MyfpoConfig(
        action_horizon=2,
        num_action_steps=2,
        num_obs_steps=2,
        hidden_dim=16,
        num_mc_samples=2,
        num_learning_epochs=1,
        chunks_per_update=2,
        chunks_per_minibatch=1,
    )
    policy = make_policy(5, 3, cfg, device="cpu")
    n = 2
    batch = {
        "observation.state": torch.zeros(n, cfg.num_obs_steps, 5),
        "action": torch.zeros(n, cfg.action_horizon, 3),
        "action_is_pad": torch.zeros(n, cfg.action_horizon, dtype=torch.bool),
        "chunk_truncated": torch.zeros(n, dtype=torch.bool),
        "raw_advantages": torch.ones(n),
        "normalized_advantages": torch.ones(n),
        "raw_returns": torch.ones(n),
        "mc_t": torch.ones(n, cfg.num_mc_samples, cfg.action_horizon, 1),
        "mc_eps": torch.randn(n, cfg.num_mc_samples, cfg.action_horizon, 3),
        "behavior_cfm_loss": torch.ones(n, cfg.num_mc_samples),
        "behavior_cfm_loss_token": torch.ones(n, cfg.num_mc_samples, cfg.action_horizon),
    }
    optimizer = make_optimizer(policy)
    _, grad_norms_pre, _, _ = policy_update_step(
        policy, optimizer, batch, None, cfg, split_grad_params(policy)
    )
    assert len(grad_norms_pre) == cfg.num_minibatches
    assert set(grad_norms_pre[0]) == {"actor", "critic", "total"}
    assert policy.step_count == cfg.num_minibatches

# ---- CUDA-graph clone guard (from test_cudagraph_safe) ----
def test_cudagraph_safe_disabled_returns_the_same_tensor():
    x = torch.zeros(2, 3)
    assert cudagraph_safe(x, enabled=False) is x


def test_cudagraph_safe_enabled_returns_an_equal_copy():
    x = torch.zeros(2, 3)
    out = cudagraph_safe(x)
    assert out is not x
    assert torch.equal(out, x)

# ---- Temporal-sensitivity probe (from test_temporal_sensitivity) ----
def _info(probe_every: int) -> dict[str, float]:
    policy = make_test_policy(temporal_probe_every=probe_every)
    cfg = policy.config
    batch = make_batch(
        policy,
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        horizon=cfg.action_horizon,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, info = policy.loss_with_info(batch)
    dead = [w for w in caught if "temporal_sensitivity" in str(w.message)]
    assert not dead, dead[0].message
    return info


def test_the_probe_reports_both_axes():
    info = _info(1)
    assert any(k.startswith("critic_action_grad_norm_pos") for k in info), sorted(info)
    assert any(k.startswith("obs_frame_grad_norm_pos") for k in info), sorted(info)


def test_the_probe_stays_off_until_its_interval():
    info = _info(0)
    assert not any("grad_norm_pos" in k for k in info), sorted(info)

# ---- Per-update metric average (from test_update_metrics) ----
def _metrics(infos):
    return update_metrics(
        infos,
        [torch.zeros(1)],
        [torch.zeros(1)],
        None,
        [{"total": torch.zeros(1), "actor": torch.zeros(1), "critic": torch.zeros(1)}],
    )


def test_a_key_is_averaged_over_the_minibatches_that_carry_it():
    infos = [
        {"loss": 1.0, "probe": 10.0},
        {"loss": 3.0},
        {"loss": 5.0, "probe": 20.0},
    ]
    out = _metrics(infos).info
    assert out["loss"] == 3.0
    assert out["probe"] == 15.0


def test_a_missing_key_does_not_stop_the_update():
    infos = [{"loss": 1.0, "probe": 10.0}, {"loss": 3.0}]
    out = _metrics(infos).info
    assert out["loss"] == 2.0
    assert out["probe"] == 10.0


def test_no_minibatch_means_no_info():
    assert _metrics([]).info is None


def test_the_probe_key_appears_only_on_its_own_minibatches():
    """The shape of the defect: the first minibatch carries the key."""
    infos = [{"loss": 1.0, "probe": 1.0}, {"loss": 2.0}, {"loss": 3.0}]
    out = _metrics(infos).info
    assert "probe" in out
    assert out["probe"] == 1.0
