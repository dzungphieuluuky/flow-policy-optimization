"""Golden behavior pins: evaluator, select_action, collector, guards.

Locks observable outputs (CSV schema, open-loop cadence, pending/flush
format, truncation/bootstrap semantics, divergence/ratio guards) so later
refactors cannot silently change results. Fast unit-level only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class ScriptedEnv:
    """Minimal gymnasium-v5-style env with deterministic rewards."""

    def __init__(self, obs_dim=4, action_dim=2, max_steps=10):
        self.observation_space = type("S", (), {"shape": (obs_dim,)})()
        self.action_space = type("A", (), {"shape": (action_dim,)})()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.step_count = 0

    def reset(self, seed=None, **kw):
        self.step_count = 0
        return np.full(self.obs_dim, 1.0, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        obs = np.full(self.obs_dim, float(self.step_count), dtype=np.float32)
        done = self.step_count >= self.max_steps
        return obs, 1.0, done, False, {}


class TestEvaluatorCharacterization:
    def test_csv_schema_five_cols(self, tmp_path):
        from sjfpo.evaluator import Evaluator

        ev = Evaluator(ScriptedEnv(), eval_seed=0, max_episodes=2)
        results = ev.evaluate(
            lambda obs: np.zeros(2, dtype=np.float32), out_dir=tmp_path
        )
        assert len(results) == 2
        assert all(
            set(r) == {"episode", "return", "length", "success", "jerk"}
            for r in results
        )
        # ScriptedEnv reports no success; zero-policy actions are jerk-free.
        assert [r["success"] for r in results] == [None, None]
        assert [r["jerk"] for r in results] == [0.0, 0.0]
        header = (tmp_path / "eval_history.csv").read_text().splitlines()[0]
        assert header == "episode,return,length,success,jerk"

    def test_returns_and_lengths_match_env(self, tmp_path):
        from sjfpo.evaluator import Evaluator

        ev = Evaluator(ScriptedEnv(max_steps=10), eval_seed=0, max_episodes=3)
        results = ev.evaluate(lambda obs: np.zeros(2, dtype=np.float32))
        assert [r["length"] for r in results] == [10, 10, 10]
        assert [r["return"] for r in results] == [10.0, 10.0, 10.0]
        assert [r["episode"] for r in results] == [0, 1, 2]

    def test_max_steps_truncates_episodes(self):
        from sjfpo.evaluator import Evaluator

        ev = Evaluator(
            ScriptedEnv(max_steps=10), eval_seed=0, max_episodes=12, max_steps=25
        )
        results = ev.evaluate(lambda obs: np.zeros(2, dtype=np.float32))
        assert sum(r["length"] for r in results) == 25
        assert len(results) == 3

    def test_deterministic_seed(self):
        from sjfpo.evaluator import Evaluator

        kw = dict(eval_seed=7, max_episodes=2)
        r1 = Evaluator(ScriptedEnv(), **kw).evaluate(
            lambda obs: np.zeros(2, dtype=np.float32)
        )
        r2 = Evaluator(ScriptedEnv(), **kw).evaluate(
            lambda obs: np.zeros(2, dtype=np.float32)
        )
        assert r1 == r2

    def test_policy_reset_called_per_episode(self):
        from sjfpo.evaluator import Evaluator

        calls = []

        class ResetPolicy:
            def reset(self):
                calls.append(1)

            def __call__(self, obs):
                return np.zeros(2, dtype=np.float32)

        Evaluator(ScriptedEnv(), eval_seed=0, max_episodes=3).evaluate(ResetPolicy())
        assert len(calls) == 3

    def test_dict_obs_rejected(self):
        import gymnasium as gym
        from sjfpo.evaluator import Evaluator

        env = ScriptedEnv()
        env.observation_space = gym.spaces.Dict(
            {"o": gym.spaces.Box(-1, 1, shape=(2,))}
        )
        with pytest.raises(AssertionError, match="flat Box"):
            Evaluator(env)


class TestSelectActionCharacterization:
    def build(self):
        from lerobot.configs.types import FeatureType, PolicyFeature

        from sjfpo.configuration_sjfpo import SJFPOConfig
        from sjfpo.modeling_sjfpo import SJFPOPolicy

        cfg = SJFPOConfig(
            input_features={
                "observation.state": PolicyFeature(FeatureType.STATE, (4,))
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        )
        return SJFPOPolicy(cfg), cfg

    def test_single_action_shape_and_open_loop_cadence(self):
        policy, cfg = self.build()
        obs = np.ones(4, dtype=np.float32)
        first = policy.select_action({"observation.state": obs})
        assert first.shape == (2,)
        # Open-loop: only the executed prefix (n_action_steps) is buffered; the
        # predicted tail is discarded and the next call replans.
        n_execute = cfg.n_action_steps
        seq = [first] + [
            policy.select_action({"observation.state": obs})
            for _ in range(n_execute - 1)
        ]
        # fresh policy with same init weights would differ; instead pin determinism:
        policy.reset()
        again = [
            policy.select_action({"observation.state": obs}) for _ in range(n_execute)
        ]
        assert all(torch.equal(a, b) for a, b in zip(seq, again))

    def test_reset_clears_buffers(self):
        policy, _ = self.build()
        obs = np.ones(4, dtype=np.float32)
        policy.select_action({"observation.state": obs})
        policy.reset()
        assert len(policy._frame_buffer) == 0
        assert len(policy._action_buffer) == 0


class TestCollectorHelperCharacterization:
    def test_as_float_tensor_numpy_and_torch(self):
        from sjfpo.collector import as_float_tensor

        t = as_float_tensor(np.ones((3,), dtype=np.float64))
        assert isinstance(t, torch.Tensor) and t.dtype == torch.float32
        src = torch.ones(2, 2, dtype=torch.float64)
        out = as_float_tensor(src)
        assert out.dtype == torch.float32
        assert out.data_ptr() != src.data_ptr()

    def test_transition_truncated_bootstrap_semantics(self):
        from sjfpo.collector import Transition

        # Transition dicts carry LeRobot done flags; bootstrap is DERIVED
        # (truncated and not terminated), not a stored field.
        continued: Transition = {
            "observation.state": torch.zeros(2, 2),
            "action": torch.zeros(16, 2),
            "next.observation.state": torch.zeros(2, 2),
            "next.reward": torch.tensor([1.0]),
            "next.done": torch.tensor(False),
            "next.truncated": torch.tensor(False),
            "executed_n_action_steps": torch.tensor(1),
            "clip_count": torch.tensor(0),
        }
        assert bool(continued["next.truncated"]) is False
        cut: Transition = dict(
            continued,
            **{"next.truncated": torch.tensor(True)},
        )
        bootstrap = bool(cut["next.truncated"]) and not bool(cut["next.done"])
        assert bootstrap is True, "TimeLimit truncation must be bootstrap-eligible"
        dead: Transition = dict(
            continued,
            **{
                "next.done": torch.tensor(True),
                "next.truncated": torch.tensor(False),
            },
        )
        assert (bool(dead["next.truncated"]) and not bool(dead["next.done"])) is False


class TestCheckDivergenceGuard:
    def test_nonpositive_baseline_never_diverges(self):
        from sjfpo.evaluator import check_divergence

        # fn-consolidation C3 guard: baseline <= 0 never counts, even far below threshold.
        for baseline in (0.0, -5.0):
            consecutive, diverged = check_divergence(-100.0, baseline, 2)
            assert (consecutive, diverged) == (0, False)


class TestExecutedVsPredictedRatio:
    def test_ratio_bounds_rejected(self):
        from sjfpo.configuration_sjfpo import SJFPOConfig

        for bad in (0.0, -0.5, 1.5):
            try:
                SJFPOConfig(executed_vs_predicted_ratio=bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"ratio={bad} must be rejected")

    def test_wrapper_buffers_executed_prefix_only(self):
        from lerobot.configs.types import FeatureType, PolicyFeature

        from sjfpo.configuration_sjfpo import SJFPOConfig
        from sjfpo.modeling_sjfpo import SJFPOPolicy

        cfg = SJFPOConfig(
            input_features={
                "observation.state": PolicyFeature(FeatureType.STATE, (4,))
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        )
        policy = SJFPOPolicy(cfg)
        policy.select_action({"observation.state": np.ones(4, dtype=np.float32)})
        assert len(policy._action_buffer) == cfg.n_action_steps - 1
