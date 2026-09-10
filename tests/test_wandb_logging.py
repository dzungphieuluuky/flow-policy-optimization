"""Tests for the wandb SDK-direct surface (wandb-simplify, red phase).

Locked contract — see ``.omo/evidence/wandb-simplify/t1-sdk-probe.md``:
- Core presence assertion: ``wandb_disabled``, ``init_wandb_run_for_method``,
  ``log_metrics``, ``log_eval_details``, ``finish_run`` (plus the later additive
  shared helper ``log_eval_summary`` from fn-consolidation C4).
- Deleted: ``wandb_init``, ``wandb_table``, ``initWandbRun`` (+ renames
  ``wandb_log`` -> ``log_metrics``, ``wandb_finish`` -> ``finish_run``).
- ``init_wandb_run_for_method(*, method, env_name, seed, total_steps,
  cfg_dict) -> bool`` with SDK-direct ``wandb.init`` kwargs
  (``project=method``, ``name=None``, ``group={method}_{env}``,
  ``job_type`` smoke<=10000/train, locked ``mode`` passthrough,
  ``reinit="finish_previous"``, ``force=True``) + locked global
  ``wandb.define_metric("*", step_metric="global_step")``.
- ``log_eval_details`` builds ``wandb.Table(columns=cols,
  data=[list(r) for r in rows])`` (list-of-lists) and routes improvement
  ``(curr - prev) / max(abs(prev), 1e-8)`` via ``log_metrics``.
- Bool return only (no ``ctx``); init+log+finish errors propagate.

These tests FAIL on the pre-simplification 8-fn ``(bool, ctx)`` src (red
proof) and pass after Todo 3. The autouse fixture in tests/conftest.py
mocks ``wl.wandb`` — no test touches the network. All SDK access via the
``wl.wandb`` prefix; no ``from wandb import`` anywhere.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mfpo.config import MFPOConfig, build_legacy_run_matrix
from tests.fixtures.fake_env import FakeGymEnv


def _init_kwargs(recorder):
    """Return the kwargs of the single wandb.init call on the mock."""
    recorder.init.assert_called_once()
    return recorder.init.call_args.kwargs


class TestWandbDisabled:
    """Disabled-path tests — MFPO_DISABLE_WANDB=1, helpers no-op."""

    def test_wandb_disabled_flag(self, monkeypatch):
        """wandb_disabled() is True only for exactly '1'."""
        monkeypatch.delenv("MFPO_DISABLE_WANDB", raising=False)
        from mfpo.wandb_logging import wandb_disabled

        assert wandb_disabled() is False
        monkeypatch.setenv("MFPO_DISABLE_WANDB", "1")
        assert wandb_disabled() is True
        monkeypatch.setenv("MFPO_DISABLE_WANDB", "0")
        assert wandb_disabled() is False

    def test_log_metrics_noop_when_disabled(self, monkeypatch):
        """MFPO_DISABLE_WANDB=1 → wandb.log is NOT called even if mocked."""
        monkeypatch.setenv("MFPO_DISABLE_WANDB", "1")
        import mfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.log_metrics({"a": 1}, step=5)
        recorder.log.assert_not_called()

    def test_finish_run_noop_when_disabled(self, monkeypatch):
        """MFPO_DISABLE_WANDB=1 → wandb.finish is NOT called even if mocked."""
        monkeypatch.setenv("MFPO_DISABLE_WANDB", "1")
        import mfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.finish_run(exit_code=0)
        recorder.finish.assert_not_called()

    def test_init_wandb_run_for_method_returns_false_without_sdk_call(self, monkeypatch):
        """MFPO_DISABLE_WANDB=1 → init returns False, zero SDK calls."""
        monkeypatch.setenv("MFPO_DISABLE_WANDB", "1")
        import mfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        result = wl.init_wandb_run_for_method(
            method="mfpo",
            env_name="Hopper-v5",
            seed=3047,
            total_steps=5000,
            cfg_dict={},
        )
        assert result is False
        recorder.init.assert_not_called()
        recorder.define_metric.assert_not_called()

    def test_wandb_train_run_completes_when_disabled(self, tmp_path, monkeypatch):
        """MFPO_DISABLE_WANDB=1 → train_run completes without crash."""
        monkeypatch.setenv("MFPO_DISABLE_WANDB", "1")
        from mfpo.runner import train_run

        env = FakeGymEnv()
        out = tmp_path / "wandb_disabled_run"
        cfg = MFPOConfig(
            N_mc=2,
            euler_train=4,
            euler_eval=4,
            chunks_per_update=16,
            epochs=1,
            minibatch=16,
            context_len=16,
        )
        result = train_run(
            env=env,
            out_dir=out,
            cfg=cfg,
            seed=0,
            total_steps=200,
            eval_interval=100,
            ckpt_interval=100,
            obs_dim=4,
            act_dim=2,
        )
        # Return dict has expected keys
        assert "reward_mean_last10" in result
        assert "fps" in result
        assert "artifacts" in result

    def test_csv_contract_unchanged(self, tmp_path, monkeypatch):
        """MFPO_DISABLE_WANDB=1 → 3-col CSV [episode,return,length] still written."""
        monkeypatch.setenv("MFPO_DISABLE_WANDB", "1")
        from mfpo.runner import train_run

        env = FakeGymEnv()
        out = tmp_path / "wandb_csv_check"
        cfg = MFPOConfig(
            N_mc=2,
            euler_train=4,
            euler_eval=4,
            chunks_per_update=16,
            epochs=1,
            minibatch=16,
            context_len=16,
        )
        train_run(
            env=env,
            out_dir=out,
            cfg=cfg,
            seed=0,
            total_steps=200,
            eval_interval=100,
            ckpt_interval=100,
            obs_dim=4,
            act_dim=2,
        )
        csv_path = out / "eval_history.csv"
        assert csv_path.exists()
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["episode", "return", "length"], (
                f"header mismatch: {header}"
            )
            rows = list(reader)
            assert len(rows) >= 1
            for r in rows:
                assert len(r) == 3


class TestInitWandbRunForMethod:
    """Enabled-path init tests — switch unset, wandb mocked."""

    def test_returns_bool_true(self, monkeypatch):
        """Enabled init returns exactly True (bool only, no ctx tuple)."""
        monkeypatch.delenv("MFPO_DISABLE_WANDB", raising=False)
        import mfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        result = wl.init_wandb_run_for_method(
            method="mfpo",
            env_name="Hopper-v5",
            seed=3047,
            total_steps=5000,
            cfg_dict={},
        )
        assert result is True
        assert isinstance(result, bool)

    def test_project_group_name_kwargs(self, monkeypatch):
        """project=method, name=None, group={method}_{env}."""
        monkeypatch.delenv("MFPO_DISABLE_WANDB", raising=False)
        import mfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.init_wandb_run_for_method(
            method="mfpo",
            env_name="Hopper-v5",
            seed=3047,
            total_steps=5000,
            cfg_dict={},
        )
        kw = _init_kwargs(recorder)
        assert kw["project"] == "mfpo"
        assert kw["name"] is None
        assert kw["group"] == "mfpo_Hopper-v5"

    def test_job_type_smoke_and_train(self, monkeypatch):
        """job_type smoke<=10000 (boundary), else train."""
        monkeypatch.delenv("MFPO_DISABLE_WANDB", raising=False)
        import mfpo.wandb_logging as wl

        for total_steps, expected in (
            (500, "smoke"),
            (10000, "smoke"),
            (10001, "train"),
            (50000, "train"),
        ):
            recorder = MagicMock()
            monkeypatch.setattr(wl, "wandb", recorder)
            wl.init_wandb_run_for_method(
                method="ppo",
                env_name="Walker2d-v5",
                seed=0,
                total_steps=total_steps,
                cfg_dict={},
            )
            kw = _init_kwargs(recorder)
            assert kw["job_type"] == expected, (
                f"total_steps={total_steps}: {kw['job_type']!r} != {expected!r}"
            )

    def test_tags_and_config_merge(self, monkeypatch):
        """tags=[method,mujoco,cpu-first]; config merges cfg_dict+method/seed/env."""
        monkeypatch.delenv("MFPO_DISABLE_WANDB", raising=False)
        import mfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.init_wandb_run_for_method(
            method="sac",
            env_name="HalfCheetah-v5",
            seed=7,
            total_steps=50000,
            cfg_dict={"lr": 3e-4},
        )
        kw = _init_kwargs(recorder)
        assert kw["tags"] == ["sac", "mujoco", "cpu-first"]
        assert kw["config"]["lr"] == 3e-4
        assert kw["config"]["method"] == "sac"
        assert kw["config"]["seed"] == 7
        assert kw["config"]["env"] == "HalfCheetah-v5"

    def test_reinit_force_locked_forms(self, monkeypatch):
        """Probe-locked SDK literals: reinit='finish_previous', force=True."""
        monkeypatch.delenv("MFPO_DISABLE_WANDB", raising=False)
        import mfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.init_wandb_run_for_method(
            method="mfpo",
            env_name="Hopper-v5",
            seed=3047,
            total_steps=5000,
            cfg_dict={},
        )
        kw = _init_kwargs(recorder)
        assert kw["reinit"] == "finish_previous"
        assert kw["force"] is True

    def test_define_metric_locked_form(self, monkeypatch):
        """Probe-locked global proxy: wandb.define_metric('*', step_metric='global_step')."""
        monkeypatch.delenv("MFPO_DISABLE_WANDB", raising=False)
        import mfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.init_wandb_run_for_method(
            method="mfpo",
            env_name="Hopper-v5",
            seed=3047,
            total_steps=5000,
            cfg_dict={},
        )
        recorder.define_metric.assert_called_once_with(
            "*", step_metric="global_step"
        )

    def test_mode_default_online(self, monkeypatch):
        """WANDB_MODE unset → mode='online' passthrough."""
        monkeypatch.delenv("MFPO_DISABLE_WANDB", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        import mfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.init_wandb_run_for_method(
            method="mfpo",
            env_name="Hopper-v5",
            seed=3047,
            total_steps=5000,
            cfg_dict={},
        )
        kw = _init_kwargs(recorder)
        assert kw["mode"] == "online"

    def test_mode_offline_passthrough(self, monkeypatch):
        """WANDB_MODE=offline → mode='offline' straight through to the SDK."""
        monkeypatch.delenv("MFPO_DISABLE_WANDB", raising=False)
        monkeypatch.setenv("WANDB_MODE", "offline")
        import mfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.init_wandb_run_for_method(
            method="mfpo",
            env_name="Hopper-v5",
            seed=3047,
            total_steps=5000,
            cfg_dict={},
        )
        kw = _init_kwargs(recorder)
        assert kw["mode"] == "offline"

    def test_mode_other_values_pass_through(self, monkeypatch):
        """Non-allowlist WANDB_MODE passes through to the SDK (no clamp)."""
        monkeypatch.delenv("MFPO_DISABLE_WANDB", raising=False)
        monkeypatch.setenv("WANDB_MODE", "shared")
        import mfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.init_wandb_run_for_method(
            method="mfpo",
            env_name="Hopper-v5",
            seed=3047,
            total_steps=5000,
            cfg_dict={},
        )
        kw = _init_kwargs(recorder)
        assert kw["mode"] == "shared"

    def test_raising_init_propagates(self, monkeypatch):
        """A raising wandb.init is NEVER swallowed — the run fails fast."""
        monkeypatch.delenv("MFPO_DISABLE_WANDB", raising=False)
        import mfpo.wandb_logging as wl

        recorder = MagicMock()
        recorder.init.side_effect = RuntimeError("no key")
        monkeypatch.setattr(wl, "wandb", recorder)

        with pytest.raises(RuntimeError, match="no key"):
            wl.init_wandb_run_for_method(
                method="mfpo",
                env_name="Hopper-v5",
                seed=3047,
                total_steps=5000,
                cfg_dict={},
            )


class TestLogMetrics:
    """Thin train-metrics guard around wandb.log."""

    def test_log_metrics_calls_wandb_log_by_default(self, monkeypatch):
        """Switch unset → wandb.log IS called (default-on)."""
        monkeypatch.delenv("MFPO_DISABLE_WANDB", raising=False)
        import mfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.log_metrics({"a": 1}, step=5)
        recorder.log.assert_called_once_with({"a": 1}, step=5)

    def test_log_metrics_raising_propagates(self, monkeypatch):
        """A raising wandb.log propagates — never swallowed."""
        monkeypatch.delenv("MFPO_DISABLE_WANDB", raising=False)
        import mfpo.wandb_logging as wl

        recorder = MagicMock()
        recorder.log.side_effect = RuntimeError("send failed")
        monkeypatch.setattr(wl, "wandb", recorder)

        with pytest.raises(RuntimeError, match="send failed"):
            wl.log_metrics({"a": 1}, step=5)


class TestLogEvalDetails:
    """Eval Table + improvement logging via the locked forms."""

    def test_eval_table_direct_constructor_list_of_lists(self, monkeypatch):
        """Table built via wandb.Table(columns=cols, data=[[..]]) list-of-lists."""
        monkeypatch.delenv("MFPO_DISABLE_WANDB", raising=False)
        import mfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.log_eval_details(
            [{"episode": 0, "return": 1.5, "length": 10}],
            [1.5],
            global_step=7,
        )
        recorder.Table.assert_called_once()
        table_kw = recorder.Table.call_args.kwargs
        assert table_kw["columns"] == ["episode", "return", "length"]
        assert table_kw["data"] == [[0, 1.5, 10]]
        assert all(type(row) is list for row in table_kw["data"])

    def test_improvement_math_routes_via_log_metrics(self, monkeypatch):
        """improvement=(curr-prev)/max(abs(prev),1e-8) routed via log_metrics."""
        monkeypatch.delenv("MFPO_DISABLE_WANDB", raising=False)
        import mfpo.wandb_logging as wl

        logged = []
        monkeypatch.setattr(
            wl, "log_metrics", lambda data, step: logged.append((data, step))
        )

        wl.log_eval_details([], [2.0, 3.0], global_step=9)
        assert len(logged) == 1
        data, step = logged[0]
        assert step == 9
        assert data["eval/reward_improvement"] == pytest.approx(1.0)
        assert data["eval/reward_improvement_pct"] == pytest.approx(0.5)

    def test_improvement_zero_prev_uses_1e8_floor(self, monkeypatch):
        """prev=0.0 → divisor max(abs(0.0),1e-8)=1e-8, no ZeroDivisionError."""
        monkeypatch.delenv("MFPO_DISABLE_WANDB", raising=False)
        import mfpo.wandb_logging as wl

        logged = []
        monkeypatch.setattr(
            wl, "log_metrics", lambda data, step: logged.append((data, step))
        )

        wl.log_eval_details([], [0.0, 2.0], global_step=3)
        assert len(logged) == 1
        data, _step = logged[0]
        assert data["eval/reward_improvement"] == pytest.approx(2.0)
        assert data["eval/reward_improvement_pct"] == pytest.approx(2.0 / 1e-8)

    def test_empty_results_no_table(self, monkeypatch):
        """No episodes → no Table construction, improvement still logged."""
        monkeypatch.delenv("MFPO_DISABLE_WANDB", raising=False)
        import mfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)
        logged = []
        monkeypatch.setattr(
            wl, "log_metrics", lambda data, step: logged.append((data, step))
        )

        wl.log_eval_details([], [1.0, 2.0], global_step=4)
        recorder.Table.assert_not_called()
        assert len(logged) == 1

    def test_log_eval_details_noop_when_disabled(self, monkeypatch):
        """MFPO_DISABLE_WANDB=1 → zero SDK calls."""
        monkeypatch.setenv("MFPO_DISABLE_WANDB", "1")
        import mfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.log_eval_details(
            [{"episode": 0, "return": 1.5, "length": 10}],
            [1.0, 1.5],
            global_step=7,
        )
        recorder.Table.assert_not_called()
        recorder.log.assert_not_called()


class TestFinishRun:
    """Thin finish guard around wandb.finish."""

    def test_finish_run_calls_wandb_finish_by_default(self, monkeypatch):
        """Switch unset → wandb.finish IS called with the exit code."""
        monkeypatch.delenv("MFPO_DISABLE_WANDB", raising=False)
        import mfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.finish_run(exit_code=1)
        recorder.finish.assert_called_once_with(exit_code=1)

    def test_finish_run_raising_propagates(self, monkeypatch):
        """A raising wandb.finish propagates — never swallowed."""
        monkeypatch.delenv("MFPO_DISABLE_WANDB", raising=False)
        import mfpo.wandb_logging as wl

        recorder = MagicMock()
        recorder.finish.side_effect = RuntimeError("finish failed")
        monkeypatch.setattr(wl, "wandb", recorder)

        with pytest.raises(RuntimeError, match="finish failed"):
            wl.finish_run(exit_code=0)


class TestSurfaceAndImportGuards:
    """Regression tests — core-fn presence, wandb required, RUN_MATRIX 45."""

    def test_five_fn_surface(self):
        """Core kept fns present; all old (pre-simplification) names gone."""
        import mfpo.wandb_logging as wl

        for name in (
            "wandb_disabled",
            "init_wandb_run_for_method",
            "log_metrics",
            "log_eval_details",
            "finish_run",
        ):
            assert callable(getattr(wl, name, None)), f"missing: {name}"
        for name in (
            "wandb_init",
            "wandb_log",
            "wandb_finish",
            "initWandbRun",
            "wandb_table",
        ):
            assert not hasattr(wl, name), f"deleted wrapper still present: {name}"

    def test_wandb_attribute_mockable(self, monkeypatch):
        """wl.wandb is settable — log_metrics routes through the mock."""
        monkeypatch.delenv("MFPO_DISABLE_WANDB", raising=False)
        import mfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)
        wl.log_metrics({"x": 1}, step=0)
        recorder.log.assert_called_once()

    def test_run_matrix_unchanged(self):
        """len(build_legacy_run_matrix()) == 45 — cardinality locked."""
        assert len(build_legacy_run_matrix()) == 45

    def test_wandb_absent_is_hard_import_error(self):
        """Subprocess guard: wandb hidden → import raises ImportError (hard)."""
        code = (
            "import sys\n"
            "sys.modules['wandb'] = None\n"
            "sys.path.insert(0, 'src')\n"
            "try:\n    import mfpo.wandb_logging as wl\n"
            "except ImportError:\n    print('wandb required guard ok')\n"
            "else:\n    raise SystemExit('expected ImportError')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, (
            f"required guard failed: {proc.stderr}\n{proc.stdout}"
        )
        assert "wandb required guard ok" in proc.stdout

    def test_wandb_default_on_guard(self):
        """Subprocess guard: default-on → import ok, wl.wandb mockable."""
        code = (
            "import os, sys; "
            "os.environ.pop('MFPO_DISABLE_WANDB', None); "
            "sys.path.insert(0, 'src'); "
            "import mfpo.wandb_logging as wl; "
            "assert wl.wandb_disabled() is False; "
            "from unittest.mock import MagicMock; wl.wandb = MagicMock(); "
            "wl.log_metrics({'x': 1}, step=0); "
            "wl.wandb.log.assert_called_once(); "
            "print('wandb default-on guard ok')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, (
            f"default-on guard failed: {proc.stderr}\n{proc.stdout}"
        )
        assert "wandb default-on guard ok" in proc.stdout
