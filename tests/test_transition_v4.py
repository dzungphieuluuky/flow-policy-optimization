"""Transition dict (LeRobot keys) + horizon split + next-state loss.

- Transition: dict with LeRobot keys observation.state / action /
  next.action / next.observation.state / next.reward / next.done / next.truncated
  (+ repo extensions n_action_steps / clip_count).
- H (chunk_size) = 16 predict, n (n_action_steps) = 8 execute, T = 16.
- next-state loss target = trainable critic state encoder encoding of next.observation.state
  (grad flows, no target net); critic V(s,a) bootstraps V(s',a').
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.collector import Transition, collect_chunks  # noqa: E402

OBS_DIM = 4
ACT_DIM = 2

EXPECTED_KEYS = {
    "observation.state",
    "action",
    "next.action",
    "next.observation.state",
    "next.reward",
    "next.done",
    "next.truncated",
    "n_action_steps",
    "clip_count",
}

# Config v4 defaults + guard

class TestConfigV4:
    def test_horizon_split_defaults(self):
        from sjfpo.configuration_sjfpo import SJFPOConfig

        cfg = SJFPOConfig()
        assert cfg.horizon == 16, "H must be 16 (predict)"
        assert cfg.n_obs_steps == 16, "T must be 16"
        assert not hasattr(cfg, "context_len"), "context_len must be gone"

    def test_n_action_steps_guard(self):
        env, policy = ScriptedEnv(), lambda obs: np.zeros((16, ACT_DIM))
        low, high = np.full(ACT_DIM, -1.0), np.full(ACT_DIM, 1.0)
        with pytest.raises(ValueError, match="n_action_steps"):
            collect_chunks(env, policy, 40, low, high, n_action_steps=17)
        with pytest.raises(ValueError, match="n_action_steps"):
            collect_chunks(env, policy, 40, low, high, n_action_steps=0)

            # Collector: Transition dicts with LeRobot keys

class ScriptedEnv(gym.Env):
    """Deterministic env: obs[t] = full(marker=t), optional terminate_at."""

    metadata: dict = {}

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, terminate_at=None):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )
        self.t = 0
        self.terminate_at = terminate_at

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return np.full(OBS_DIM, 0.0, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        obs = np.full(OBS_DIM, float(self.t), dtype=np.float32)
        terminated = self.terminate_at is not None and self.t == self.terminate_at
        return obs, 0.1 * self.t, terminated, False, {}


def collect(**kw):
    env = ScriptedEnv(**kw)
    policy = lambda obs: np.zeros((16, ACT_DIM), dtype=np.float32)  # full-H proposal
    return collect_chunks(
        env,
        policy,
        40,
        np.full(ACT_DIM, -1.0),
        np.full(ACT_DIM, 1.0),
        n_action_steps=8,  # local: executed steps are caller-passed
    )


class TestCollectorV4:
    def test_emits_transition_dicts_with_keys(self):
        chunks = collect()
        assert len(chunks) > 0
        for t in chunks:
            assert isinstance(t, dict), type(t)
            assert set(t.keys()) == EXPECTED_KEYS, set(t.keys()) ^ EXPECTED_KEYS

    def test_shapes_and_dtypes(self):
        chunks = collect()
        for t in chunks:
            assert t["observation.state"].shape == (16, OBS_DIM)
            assert t["action"].shape == (16, ACT_DIM)  # full-H anchor, always
            assert t["next.action"].shape == (16, ACT_DIM)  # chained/next proposal
            assert t["next.observation.state"].shape == (16, OBS_DIM)
            assert t["observation.state"].dtype == torch.float32
            assert t["action"].dtype == torch.float32
            assert t["next.done"].dtype == torch.bool
            assert t["next.truncated"].dtype == torch.bool
            assert int(t["n_action_steps"]) == len(t["next.reward"])

    def test_next_action_chains_to_next_proposal(self):
        """continue chunk[k].next.action == chunk[k+1].action."""
        chunks = collect()
        checked = 0
        for prev, cur in zip(chunks, chunks[1:]):
            if bool(prev["next.done"]) or bool(prev["next.truncated"]):
                continue  # boundary: own behavior proposal, not chained
            assert torch.equal(prev["next.action"], cur["action"]), (
                "continue chunks must chain next.action from the next proposal"
            )
            checked += 1
        assert checked > 0, "no continue pairs found"

    def test_terminated_next_action_is_zeros(self):
        chunks = collect(terminate_at=10)
        term = [t for t in chunks if bool(t["next.done"])]
        assert len(term) >= 1
        for t in term:
            assert t["next.action"].shape == (16, ACT_DIM)
            assert bool((t["next.action"] == 0).all()), "terminated: no bootstrap"

    def test_continue_chunk_executes_n_steps(self):
        chunks = collect()
        first = chunks[0]
        assert first["next.done"] == torch.tensor(False)
        assert int(first["n_action_steps"]) == 8
        assert len(first["next.reward"]) == 8

    def test_next_state_continuity(self):
        """chunk[k+1].state == chunk[k].next_state for continue pairs."""
        chunks = collect()
        checked = 0
        for prev, cur in zip(chunks, chunks[1:]):
            if prev["next.done"]:
                continue  # fresh episode after termination: no continuity
            assert torch.equal(
                cur["observation.state"], prev["next.observation.state"]
            ), "window must advance exactly by executed steps"
            checked += 1
        assert checked > 0, "no continue pairs found"

    def test_terminated_chunk_keeps_full_h_action(self):
        chunks = collect(terminate_at=10)
        term = [t for t in chunks if bool(t["next.done"])]
        assert len(term) >= 1, "at least one terminated chunk expected"
        for t in term:
            assert t["action"].shape == (16, ACT_DIM)
            assert int(t["n_action_steps"]) < 8

            # Trainer: DiT actor (actions only) + transformer critic, no aux head

def make_policy_v4(**overrides):
    from lerobot.configs.types import FeatureType, PolicyFeature

    from sjfpo.configuration_sjfpo import SJFPOConfig
    from sjfpo.modeling_sjfpo import SJFPOPolicy

    cfg = SJFPOConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (OBS_DIM,))
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT_DIM,))},
        **overrides,
    )
    return SJFPOPolicy(cfg)


def make_transitions(n: int = 32):
    out: list[Transition] = []
    for _ in range(n):
        out.append(
            {
                "observation.state": torch.randn(16, OBS_DIM),
                "action": torch.randn(16, ACT_DIM) * 0.5,
                "next.action": torch.randn(16, ACT_DIM) * 0.5,
                "next.observation.state": torch.randn(16, OBS_DIM),
                "next.reward": torch.randn(8) * 0.1,
                "next.done": torch.tensor(False),
                "next.truncated": torch.tensor(False),
                "n_action_steps": torch.tensor(8),
                "clip_count": torch.tensor(0),
            }
        )
    return out


def make_forward_batch(chunks, n: int):
    lengths = [int(c["n_action_steps"]) for c in chunks]
    h = chunks[0]["action"].shape[0]
    return {
        "observation.state": torch.stack([c["observation.state"] for c in chunks]),
        "action": torch.stack([c["action"] for c in chunks]),
        "action_is_pad": torch.arange(h).unsqueeze(0)
        >= torch.tensor(lengths).unsqueeze(1),
        "advantages": torch.randn(n) * 0.1,
        "returns": torch.randn(n) * 10.0,
    }


class TestNoAuxHeadV4:
    def test_no_aux_head_or_next_state_loss(self):
        """The next-state path is deleted: no head, no loss key."""
        policy = make_policy_v4()
        assert not hasattr(policy, "aux_head"), "aux_head must be gone"
        _, info = policy(make_forward_batch(make_transitions(8), 8))
        assert "next_state_loss" not in info
        assert set(info) == {"policy_loss", "value_loss", "mean_log_ratio"}

    def test_policy_head_in_optimizer_and_moves(self):
        policy = make_policy_v4()
        opt = torch.optim.AdamW(policy.get_optim_params()["params"])
        opt_ids = {id(p) for group in opt.param_groups for p in group["params"]}
        for name, p in policy.actor.named_parameters():
            assert id(p) in opt_ids, f"actor.{name} not in optimizer"
        before = {n: p.clone() for n, p in policy.actor.named_parameters()}
        opt.zero_grad()
        loss, _ = policy(make_forward_batch(make_transitions(32), 32))
        loss.backward()
        opt.step()
        moved = any(
            not torch.equal(p, before[n]) for n, p in policy.actor.named_parameters()
        )
        assert moved, "actor parameters did not move after update"
