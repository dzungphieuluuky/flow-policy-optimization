"""``select_action`` must hand out a chunk in the order it planned it."""

import dataclasses
import math

import numpy as np
import pytest
import torch

from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType

from myfpo.configuration import MyfpoConfig
from myfpo.training import build_policy
from test_collector_batch_equivalence import FakeVecEnv

OBS, ACT, HORIZON, EXECUTE = 5, 3, 8, 4
LOW = [-1.0] * ACT
HIGH = [1.0] * ACT


def _cfg(**overrides):
    return dataclasses.replace(
        MyfpoConfig(
            action_horizon=HORIZON,
            n_action_steps=EXECUTE,
            hidden_dim=16,
            dit_depth=1,
            dit_heads=2,
            n_obs_steps=EXECUTE,
            num_mc_samples=2,
            input_features={"observation.state": PolicyFeature(FeatureType.STATE, (OBS,))},
            output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT,))},
        ),
        **overrides,
    )


def _policy(cfg=None):
    cfg = cfg or _cfg()
    return build_policy(OBS, ACT, cfg, device="cpu", action_low=LOW, action_high=HIGH), cfg


def _ramp(horizon=HORIZON, scale=0.1):
    """A chunk whose steps are all distinguishable, so a swap cannot hide."""
    return (
        torch.arange(1, horizon + 1, dtype=torch.float32).reshape(1, horizon, 1) * scale
    ).expand(1, horizon, ACT).contiguous()


def _window(cfg):
    return {"observation.state": torch.zeros(cfg.n_obs_steps, OBS)}


def _squashed(chunk: torch.Tensor) -> torch.Tensor:
    """The env-facing action of a raw chunk: the raw chunk squashed."""
    from myfpo.processor import squash_to_env_bounds

    low = torch.tensor(LOW)
    high = torch.tensor(HIGH)
    return squash_to_env_bounds(chunk, low, high)


def test_the_chunk_comes_back_in_plan_order():
    """The bug: a deque fills right and pops left, so reverse-filling inverts it."""
    policy, cfg = _policy()
    chunk = _ramp()
    policy.predict_action_chunk = lambda *a, **k: chunk

    got = [policy.select_action(_window(cfg)).tolist() for _ in range(EXECUTE)]
    want = [_squashed(chunk)[0, i].tolist() for i in range(EXECUTE)]
    assert got == want, f"executed {got}, planned {want}"


def test_the_order_is_exactly_ascending_across_a_full_chunk():
    """Pins the whole thing at once: EXECUTE calls, strictly increasing values."""
    policy, cfg = _policy()
    policy.predict_action_chunk = lambda *a, **k: _ramp()
    values = [policy.select_action(_window(cfg))[0].item() for _ in range(EXECUTE)]
    assert values == sorted(values), values
    assert len(set(values)) == len(values), "a step repeated, so order is not a permutation"


def test_replans_once_the_cached_prefix_is_spent():
    """One plan per ``n_action_steps``: not per call, not per episode."""
    policy, cfg = _policy()
    calls = []

    def fake_predict(*args, **kwargs):
        calls.append(1)
        return torch.full((1, HORIZON, ACT), float(len(calls)))

    policy.predict_action_chunk = fake_predict

    first = [policy.select_action(_window(cfg))[0].item() for _ in range(EXECUTE)]
    assert first == pytest.approx([math.tanh(1.0)] * EXECUTE), first
    assert len(calls) == 1, "replanned before the cached prefix ran out"

    second = [policy.select_action(_window(cfg))[0].item() for _ in range(EXECUTE)]
    assert second == pytest.approx([math.tanh(2.0)] * EXECUTE), second
    assert len(calls) == 2, "did not replan after the prefix ran out"


def test_reset_drops_an_unfinished_chunk():
    """A new episode must not inherit the previous episode's tail."""
    policy, cfg = _policy()
    calls = []

    def fake_predict(*args, **kwargs):
        calls.append(1)
        return torch.full((1, HORIZON, ACT), float(len(calls)))

    policy.predict_action_chunk = fake_predict

    policy.select_action(_window(cfg))
    assert len(policy.action_buffer) == EXECUTE - 1
    policy.reset()
    assert not policy.action_buffer
    assert policy.select_action(_window(cfg))[0].item() == pytest.approx(
        math.tanh(2.0)
    ), "drained the old chunk"


def test_executes_no_more_than_the_chunk_holds():
    """A short plan must not over-run: ``n_execute`` is clamped to the horizon."""
    cfg = _cfg(action_horizon=2, n_action_steps=2)
    policy, cfg = _policy(cfg)
    chunk = _ramp(horizon=2)
    policy.predict_action_chunk = lambda *a, **k: chunk

    got = [policy.select_action(_window(cfg)).tolist() for _ in range(2)]
    want = [_squashed(chunk)[0, i].tolist() for i in range(2)]
    assert got == want, got


def test_the_collector_executes_the_same_order():
    """Both paths agree, which is what made the divergence invisible."""
    cfg = _cfg(action_horizon=4, n_action_steps=4)
    policy, cfg = _policy(cfg)
    chunk = _ramp(horizon=4)
    policy.predict_action_chunk = lambda *a, **k: chunk.expand(2, -1, -1).contiguous()

    # 8 steps over 2 envs is 4 vec-steps each: one full chunk, no cut.
    env = FakeVecEnv(n=2, obs_dim=OBS, act_dim=ACT, episode=1000)
    policy.generate_rollout(env, 8, action_low=LOW, action_high=HIGH)

    seen = np.stack(env.actions_seen[:4])          # [4 steps, 2 envs, ACT].
    want = _squashed(chunk)[0, :, 0].numpy()       # [4 steps].
    for step in range(4):
        assert np.allclose(seen[step, :, 0], want[step]), (
            f"step {step}: env saw {seen[step, :, 0]}, plan said {want[step]}"
        )


def test_eval_and_collection_agree_on_the_first_chunk():
    """Same policy, same obs: eval's first action equals the collector's first."""
    cfg = _cfg(action_horizon=4, n_action_steps=4)
    policy, cfg = _policy(cfg)
    chunk = _ramp(horizon=4)
    policy.predict_action_chunk = lambda *a, **k: chunk.expand(2, -1, -1).contiguous()

    env = FakeVecEnv(n=2, obs_dim=OBS, act_dim=ACT, episode=1000)
    policy.generate_rollout(env, 4, action_low=LOW, action_high=HIGH)
    collected_first = np.array(env.actions_seen[0][0])

    policy.reset()
    # eval path returns env-bounds actions, collector stored scaled-to-[-1,1].
    eval_first = policy.select_action(_window(cfg))[0].numpy()
    assert np.allclose(eval_first, collected_first), (
        f"eval {eval_first} vs collector {collected_first}"
    )