"""Characterization tests for fn-consolidation Todo 3 (written BEFORE splits).

Pins current observable behavior of the runner+sweep+collector+evaluator
chain so behavior-identical pure-extraction splits cannot change results.
Fast unit-level only (no train_run — covered by test_runner_mfpo.py).
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

    def __init__(self, obs_dim=4, act_dim=2, max_steps=10):
        self.observation_space = type("S", (), {"shape": (obs_dim,)})()
        self.action_space = type(
            "A", (), {"shape": (act_dim,)}
        )()
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._max_steps = max_steps
        self._step = 0

    def reset(self, seed=None, **kw):
        self._step = 0
        return np.full(self._obs_dim, 1.0, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        obs = np.full(self._obs_dim, float(self._step), dtype=np.float32)
        done = self._step >= self._max_steps
        return obs, 1.0, done, False, {}


class TestEvaluatorCharacterization:
    def test_csv_schema_three_cols(self, tmp_path):
        from mfpo.evaluator import Evaluator

        ev = Evaluator(ScriptedEnv(), eval_seed=0, max_episodes=2)
        results = ev.evaluate(lambda obs: np.zeros(2, dtype=np.float32), out_dir=tmp_path)
        assert len(results) == 2
        assert all(set(r) == {"episode", "return", "length"} for r in results)
        header = (tmp_path / "eval_history.csv").read_text().splitlines()[0]
        assert header == "episode,return,length"

    def test_returns_and_lengths_match_env(self, tmp_path):
        from mfpo.evaluator import Evaluator

        ev = Evaluator(ScriptedEnv(max_steps=10), eval_seed=0, max_episodes=3)
        results = ev.evaluate(lambda obs: np.zeros(2, dtype=np.float32))
        assert [r["length"] for r in results] == [10, 10, 10]
        assert [r["return"] for r in results] == [10.0, 10.0, 10.0]
        assert [r["episode"] for r in results] == [0, 1, 2]

    def test_max_steps_truncates_episodes(self):
        from mfpo.evaluator import Evaluator

        ev = Evaluator(ScriptedEnv(max_steps=10), eval_seed=0, max_episodes=12, max_steps=25)
        results = ev.evaluate(lambda obs: np.zeros(2, dtype=np.float32))
        assert sum(r["length"] for r in results) == 25
        assert len(results) == 3

    def test_deterministic_seed(self):
        from mfpo.evaluator import Evaluator

        kw = dict(eval_seed=7, max_episodes=2)
        r1 = Evaluator(ScriptedEnv(), **kw).evaluate(lambda obs: np.zeros(2, dtype=np.float32))
        r2 = Evaluator(ScriptedEnv(), **kw).evaluate(lambda obs: np.zeros(2, dtype=np.float32))
        assert r1 == r2

    def test_policy_reset_called_per_episode(self):
        from mfpo.evaluator import Evaluator

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
        from mfpo.evaluator import Evaluator

        env = ScriptedEnv()
        env.observation_space = gym.spaces.Dict({"o": gym.spaces.Box(-1, 1, shape=(2,))})
        with pytest.raises(AssertionError, match="flat Box"):
            Evaluator(env)


class TestMpoEvalWrapperCharacterization:
    def _build(self):
        from mfpo.config import MFPOConfig
        from mfpo.networks import ContextTrunk, PolicyFlowHead, SpectralTokens
        from mfpo.runner import MpoEvalWrapper

        cfg = MFPOConfig()
        trunk = ContextTrunk(4, context_len=cfg.context_len, z_dim=cfg.Z_dim)
        head = PolicyFlowHead(2, z_dim=cfg.Z_dim, cond_dims=(64,))
        spectral = SpectralTokens(z_dim=cfg.Z_dim, k_spectral=cfg.K_spectral,
                                  context_len=cfg.context_len)
        with torch.no_grad():
            spectral.capture_stats(torch.randn(4, cfg.context_len, cfg.Z_dim))
        return MpoEvalWrapper(trunk, head, spectral, cfg), cfg

    def test_single_action_shape_and_open_loop_cadence(self):
        from mfpo.runner import MpoEvalWrapper  # noqa: F401

        wrapper, cfg = self._build()
        obs = np.ones(4, dtype=np.float32)
        first = wrapper(obs)
        assert first.shape == (2,)
        # Open-loop: next chunk_len-1 calls reuse buffered chunk deterministically
        seq = [first] + [wrapper(obs) for _ in range(cfg.chunk_len - 1)]
        wrapper2, _ = self._build()
        # fresh wrapper with same init weights would differ; instead pin determinism:
        wrapper.reset()
        again = [wrapper(obs) for _ in range(cfg.chunk_len)]
        assert all(np.array_equal(a, b) for a, b in zip(seq, again))

    def test_reset_clears_buffers(self):
        wrapper, _ = self._build()
        obs = np.ones(4, dtype=np.float32)
        wrapper(obs)
        wrapper.reset()
        assert len(wrapper._frame_buffer) == 0
        assert len(wrapper._action_buffer) == 0


class TestSweepCharacterization:
    def test_dir_without_manifest_is_pending(self, tmp_path):
        from mfpo.sweep import scan_runs
        from mfpo.config import RunSpec

        spec = RunSpec(method="mfpo", env="Hopper-v5", seed=0,
                       total_steps=100, eval_interval=10, ckpt_interval=10)
        (tmp_path / "mfpo_Hopper-v5_seed0").mkdir(parents=True)
        result = scan_runs((spec,), tmp_path)
        assert len(result.pending) == 1
        assert len(result.skipped) == 0

    def test_flush_error_message_format(self, tmp_path, capsys):
        from mfpo.sweep import flush_pending_runs
        from mfpo.config import RunSpec
        import time

        spec = RunSpec(method="mfpo", env="Hopper-v5", seed=0,
                       total_steps=100, eval_interval=10, ckpt_interval=10)

        def boom(spec, base):
            raise RuntimeError("kablam")

        results = flush_pending_runs([spec], tmp_path, boom, time.time())
        assert results == []
        assert "ERROR in run mfpo_Hopper-v5_seed0: kablam" in capsys.readouterr().out


class TestCollectorHelperCharacterization:
    def test_as_float_tensor_numpy_and_torch(self):
        from mfpo.collector import as_float_tensor

        t = as_float_tensor(np.ones((3,), dtype=np.float64))
        assert isinstance(t, torch.Tensor) and t.dtype == torch.float32
        src = torch.ones(2, 2, dtype=torch.float64)
        out = as_float_tensor(src)
        assert out.dtype == torch.float32
        assert out.data_ptr() != src.data_ptr()

    def test_chunk_record_bootstrap_semantics(self):
        from mfpo.collector import ChunkRecord

        rec = ChunkRecord(context=torch.zeros(2, 2), action_dct=torch.zeros(1, 2),
                          rewards=[1.0], done="truncated", bootstrap=True,
                          clip_count=0, exec_len=1)
        assert rec.bootstrap is True
        assert rec.done == "truncated"


class TestCheckDivergenceGuard:
    def test_nonpositive_baseline_never_diverges(self):
        from mfpo.evaluator import check_divergence

        # fn-consolidation C3 guard: baseline <= 0 never counts, even far below threshold.
        for baseline in (0.0, -5.0):
            consecutive, diverged = check_divergence(-100.0, baseline, 2)
            assert (consecutive, diverged) == (0, False)
