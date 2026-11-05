"""The normalized flow target must match the behavior losses bit for bit, and
round-trip through the checkpoint.

The z-score action normalizer sits in the CFM target frame: ``x_t``, the
regression target, and the stored behavior losses all see it. The critic
never does. The frozen-per-cycle statistics mean the rollout and the update
are always in the same frame.
"""

from __future__ import annotations

import torch

from fixtures import cfg

from myfpo.checkpointing import (
    CHECKPOINT_FILE,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from myfpo.layers import resolve_storage_dtype
from myfpo.training import make_policy

OBS, ACT, HORIZON = 4, 2, 4


def _pcfg(**overrides):
    base = dict(
        obs_dim=OBS, act_dim=ACT,
        action_horizon=HORIZON, n_action_steps=HORIZON,
        n_obs_steps=2, hidden_dim=32, dit_depth=1, dit_heads=2,
        time_embed_dim=16, num_mc_samples=4,
        compile=False, report_to="none",
    )
    base.update(overrides)
    return cfg(**base)


def _policy(**overrides):
    pcfg = _pcfg(**overrides)
    return make_policy(
        OBS, ACT, pcfg, device="cpu",
        action_low=[-1.0] * ACT, action_high=[1.0] * ACT,
    )


def _prime_stats(policy, rows=64, seed=3):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    fake = torch.randn(rows, HORIZON, ACT, generator=gen)
    pad = torch.zeros(rows, HORIZON, dtype=torch.bool)
    policy.update_action_stats(fake, pad)
    return policy


def test_the_flag_defaults_to_on_and_the_policy_carries_the_normalizer():
    policy = _policy()
    assert policy.config.normalize_actions is True
    assert hasattr(policy, "action_normalizer")


def test_update_action_stats_is_a_no_op_when_off():
    policy = _policy(normalize_actions=False)
    assert not hasattr(policy, "action_normalizer")
    policy.update_action_stats(
        torch.ones(4, HORIZON, ACT), torch.zeros(4, HORIZON, dtype=torch.bool)
    )
    assert not hasattr(policy, "action_normalizer")


def test_a_checkpoint_predating_the_flag_resumes_in_raw_space(tmp_path):
    """A config without the key must resume exactly as the run trained."""
    policy = _policy(normalize_actions=False)
    save_policy_checkpoint(tmp_path, policy=policy, optimizer=None, global_step=2)
    payload = torch.load(tmp_path / CHECKPOINT_FILE, weights_only=True)
    assert "normalize_actions" in payload["config"]
    del payload["config"]["normalize_actions"]
    torch.save(payload, tmp_path / CHECKPOINT_FILE)
    loaded, _, _ = load_policy_checkpoint(tmp_path)
    assert loaded.config.normalize_actions is False
    assert not hasattr(loaded, "action_normalizer")


def test_update_action_stats_uses_only_executed_tokens():
    policy = _policy(normalize_actions=True)
    torch.manual_seed(0)
    actions = torch.randn(2, HORIZON, ACT)
    pad = torch.tensor([[False, True, True, True],
                        [False, False, True, True]])
    executed = actions[~pad]
    policy.update_action_stats(actions, pad)
    mean = executed.mean(dim=0)
    var = executed.var(dim=0, unbiased=False)
    assert torch.allclose(policy.action_normalizer.mean, mean, atol=1e-6)
    assert torch.allclose(policy.action_normalizer.var, var, atol=1e-6)


def test_the_behavior_and_training_losses_agree_in_the_normalized_frame():
    from myfpo.layers import autocast_context

    policy = _prime_stats(_policy(normalize_actions=True))
    policy.update_observation_stats(torch.randn(64, OBS))

    n = 2
    torch.manual_seed(0)
    batch = {
        "observation.state": torch.randn(n, policy.config.n_obs_steps, OBS),
        "action": torch.randn(n, HORIZON, ACT),
        "action_is_pad": torch.zeros(n, HORIZON, dtype=torch.bool),
        "chunk_truncated": torch.zeros(n, dtype=torch.bool),
        "normalized_advantages": torch.ones(n),
        "raw_returns": torch.ones(n),
        "mc_t": torch.ones(n, policy.config.num_mc_samples, HORIZON, 1),
        "mc_eps": torch.randn(n, policy.config.num_mc_samples, HORIZON, ACT),
    }
    pre = policy.obs_normalizer(batch["observation.state"])
    old, old_token = policy.compute_behavior_cfm_losses(
        pre, batch["action"], batch["mc_t"], batch["mc_eps"],
        batch["action_is_pad"],
    )
    batch["behavior_cfm_loss"] = old
    batch["behavior_cfm_loss_token"] = old_token

    b = policy.to_device_batch(
        batch, torch.device("cpu"), resolve_storage_dtype(torch.device("cpu"))
    )
    with autocast_context(torch.device("cpu")):
        v_new = policy.compute_predicted_velocity(
            b.x_t, b.actor_t, b.condition_flat
        ).float()
    terms = policy.policy_loss(b, v_new, torch.device("cpu"))

    loss_b, _ = policy.compute_behavior_cfm_losses(
        b.states, batch["action"], b.t, b.eps, b.action_is_pad
    )
    assert torch.equal(loss_b.detach(), terms.cfm_loss_new.detach())


def test_the_train_and_rollout_views_use_the_same_stats_frame():
    """``x_t`` and the target must be the normalized action, not the raw one."""
    policy = _prime_stats(_policy(normalize_actions=True))
    torch.manual_seed(0)
    raw = torch.randn(3, HORIZON, ACT)
    out = policy.action_normalizer(raw)
    mean, var = policy.action_normalizer.mean, policy.action_normalizer.var
    want = (raw - mean) * torch.rsqrt(var + policy.action_normalizer.eps)
    torch.testing.assert_close(out, want, rtol=0, atol=0)
    # The normalized action cannot be a constant for a nonzero raw spread.
    assert out.std(unbiased=False) > 0.0


def test_checkpoint_round_trip_preserves_the_action_stats(tmp_path):
    policy = _prime_stats(_policy(normalize_actions=True), rows=128)
    save_policy_checkpoint(
        tmp_path, policy=policy, optimizer=None, global_step=3
    )
    loaded, _, counters = load_policy_checkpoint(tmp_path)
    assert counters["global_step"] == 3
    for key in ("mean", "var", "count"):
        assert torch.equal(getattr(loaded.action_normalizer, key),
                           getattr(policy.action_normalizer, key))



