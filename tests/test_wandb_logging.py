"""Tests for the wandb SDK-direct surface (wandb-simplify, red phase).

Locked contract — see ``.omo/evidence/wandb-simplify/t1-sdk-probe.md``:
- Core presence assertion: ``wandb_disabled``, ``init_wandb_run_for_method``,
  ``log_metrics``, ``log_eval_summary``, ``finish_run``.
- Deleted: ``wandb_init``, ``wandb_table``, ``initWandbRun`` (+ renames
  ``wandb_log`` -> ``log_metrics``, ``wandb_finish`` -> ``finish_run``).
- ``init_wandb_run_for_method(*, method, env_name, seed, total_steps,
  cfg_dict) -> bool`` with SDK-direct ``wandb.init`` kwargs
  (``project=method``, ``name=None``, ``group={method}_{env}``,
  ``job_type`` smoke<=10000/train, locked ``mode`` passthrough,
  ``reinit="finish_previous"``, ``force=True``) with no custom step metric
  (default ``step=global_step`` is the x-axis, matching system auto-logging).
- ``log_eval_summary`` builds ``wandb.Table(columns=cols,
  data=[list(r) for r in rows])`` (list-of-lists) and routes improvement
  ``(curr - prev) / max(abs(prev), 1e-8)`` via ``log_metrics``.
- Bool return only (no ``ctx``); init+log+finish errors propagate.

These tests FAIL on the pre-simplification 8-fn ``(bool, ctx)`` src (red
proof) and pass after Todo 3. The autouse fixture in tests/conftest.py
mocks ``wl.wandb`` — no test touches the network. All SDK access via the
``wl.wandb`` prefix; no ``from wandb import`` anywhere.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.config import build_run_matrix


def init_kwargs(recorder):
    """Return the kwargs of the single wandb.init call on the mock."""
    recorder.init.assert_called_once()
    return recorder.init.call_args.kwargs


class TestWandbDisabled:
    """Disabled-path tests — SJFPO_DISABLE_WANDB=1, helpers no-op."""

    def test_wandb_disabled_flag(self, monkeypatch):
        """wandb_disabled() is True only for exactly '1'."""
        monkeypatch.delenv("SJFPO_DISABLE_WANDB", raising=False)
        from sjfpo.wandb_logging import wandb_disabled

        assert wandb_disabled() is False
        monkeypatch.setenv("SJFPO_DISABLE_WANDB", "1")
        assert wandb_disabled() is True
        monkeypatch.setenv("SJFPO_DISABLE_WANDB", "0")
        assert wandb_disabled() is False

    def test_log_metrics_noop_when_disabled(self, monkeypatch):
        """SJFPO_DISABLE_WANDB=1 → wandb.log is NOT called even if mocked."""
        monkeypatch.setenv("SJFPO_DISABLE_WANDB", "1")
        import sjfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.log_metrics({"a": 1}, step=5)
        recorder.log.assert_not_called()

    def test_finish_run_noop_when_disabled(self, monkeypatch):
        """SJFPO_DISABLE_WANDB=1 → wandb.finish is NOT called even if mocked."""
        monkeypatch.setenv("SJFPO_DISABLE_WANDB", "1")
        import sjfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.finish_run(exit_code=0)
        recorder.finish.assert_not_called()

    def test_init_wandb_run_for_method_returns_false_without_sdk_call(
        self, monkeypatch
    ):
        """SJFPO_DISABLE_WANDB=1 → init returns False, zero SDK calls."""
        monkeypatch.setenv("SJFPO_DISABLE_WANDB", "1")
        import sjfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        result = wl.init_wandb_run_for_method(
            method="sjfpo",
            env_name="Hopper-v5",
            seed=3047,
            total_steps=5000,
            cfg_dict={},
        )
        assert result is False
        recorder.init.assert_not_called()
        recorder.define_metric.assert_not_called()


class TestInitWandbRunForMethod:
    """Enabled-path init tests — switch unset, wandb mocked."""

    def test_returns_bool_true(self, monkeypatch):
        """Enabled init returns exactly True (bool only, no ctx tuple)."""
        monkeypatch.delenv("SJFPO_DISABLE_WANDB", raising=False)
        import sjfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        result = wl.init_wandb_run_for_method(
            method="sjfpo",
            env_name="Hopper-v5",
            seed=3047,
            total_steps=5000,
            cfg_dict={},
        )
        assert result is True
        assert isinstance(result, bool)

    def test_project_group_name_kwargs(self, monkeypatch):
        """project=method, name=None, group={method}_{env}."""
        monkeypatch.delenv("SJFPO_DISABLE_WANDB", raising=False)
        import sjfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.init_wandb_run_for_method(
            method="sjfpo",
            env_name="Hopper-v5",
            seed=3047,
            total_steps=5000,
            cfg_dict={},
        )
        kw = init_kwargs(recorder)
        assert kw["project"] == "sjfpo"
        assert kw["name"] is None
        assert kw["group"] == "sjfpo_Hopper-v5"

    def test_job_type_smoke_and_train(self, monkeypatch):
        """job_type smoke<=10000 (boundary), else train."""
        monkeypatch.delenv("SJFPO_DISABLE_WANDB", raising=False)
        import sjfpo.wandb_logging as wl

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
            kw = init_kwargs(recorder)
            assert kw["job_type"] == expected, (
                f"total_steps={total_steps}: {kw['job_type']!r} != {expected!r}"
            )

    def test_tags_and_config_merge(self, monkeypatch):
        """tags=[method,mujoco,cpu-first]; config merges cfg_dict+method/seed/env."""
        monkeypatch.delenv("SJFPO_DISABLE_WANDB", raising=False)
        import sjfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.init_wandb_run_for_method(
            method="sac",
            env_name="HalfCheetah-v5",
            seed=7,
            total_steps=50000,
            cfg_dict={"lr": 3e-4},
        )
        kw = init_kwargs(recorder)
        assert kw["tags"] == ["sac", "mujoco", "cpu-first"]
        assert kw["config"]["lr"] == 3e-4
        assert kw["config"]["method"] == "sac"
        assert kw["config"]["seed"] == 7
        assert kw["config"]["env"] == "HalfCheetah-v5"

    def test_reinit_force_locked_forms(self, monkeypatch):
        """Probe-locked SDK literals: reinit='finish_previous', force=True."""
        monkeypatch.delenv("SJFPO_DISABLE_WANDB", raising=False)
        import sjfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.init_wandb_run_for_method(
            method="sjfpo",
            env_name="Hopper-v5",
            seed=3047,
            total_steps=5000,
            cfg_dict={},
        )
        kw = init_kwargs(recorder)
        assert kw["reinit"] == "finish_previous"
        assert kw["force"] is True

    def test_no_custom_step_metric(self, monkeypatch):
        """No custom x-axis: default step=global_step renders line graphs."""
        monkeypatch.delenv("SJFPO_DISABLE_WANDB", raising=False)
        import sjfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.init_wandb_run_for_method(
            method="sjfpo",
            env_name="Hopper-v5",
            seed=3047,
            total_steps=5000,
            cfg_dict={},
        )
        recorder.define_metric.assert_not_called()

    def test_mode_default_online(self, monkeypatch):
        """WANDB_MODE unset → mode='online' passthrough."""
        monkeypatch.delenv("SJFPO_DISABLE_WANDB", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        import sjfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.init_wandb_run_for_method(
            method="sjfpo",
            env_name="Hopper-v5",
            seed=3047,
            total_steps=5000,
            cfg_dict={},
        )
        kw = init_kwargs(recorder)
        assert kw["mode"] == "online"

    def test_mode_offline_passthrough(self, monkeypatch):
        """WANDB_MODE=offline → mode='offline' straight through to the SDK."""
        monkeypatch.delenv("SJFPO_DISABLE_WANDB", raising=False)
        monkeypatch.setenv("WANDB_MODE", "offline")
        import sjfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.init_wandb_run_for_method(
            method="sjfpo",
            env_name="Hopper-v5",
            seed=3047,
            total_steps=5000,
            cfg_dict={},
        )
        kw = init_kwargs(recorder)
        assert kw["mode"] == "offline"

    def test_mode_other_values_pass_through(self, monkeypatch):
        """Non-allowlist WANDB_MODE passes through to the SDK (no clamp)."""
        monkeypatch.delenv("SJFPO_DISABLE_WANDB", raising=False)
        monkeypatch.setenv("WANDB_MODE", "shared")
        import sjfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.init_wandb_run_for_method(
            method="sjfpo",
            env_name="Hopper-v5",
            seed=3047,
            total_steps=5000,
            cfg_dict={},
        )
        kw = init_kwargs(recorder)
        assert kw["mode"] == "shared"

    def test_raising_init_propagates(self, monkeypatch):
        """A raising wandb.init is NEVER swallowed — the run fails fast."""
        monkeypatch.delenv("SJFPO_DISABLE_WANDB", raising=False)
        import sjfpo.wandb_logging as wl

        recorder = MagicMock()
        recorder.init.side_effect = RuntimeError("no key")
        monkeypatch.setattr(wl, "wandb", recorder)

        with pytest.raises(RuntimeError, match="no key"):
            wl.init_wandb_run_for_method(
                method="sjfpo",
                env_name="Hopper-v5",
                seed=3047,
                total_steps=5000,
                cfg_dict={},
            )


class TestLogMetrics:
    """Thin train-metrics guard around wandb.log."""

    def test_log_metrics_calls_wandb_log_by_default(self, monkeypatch):
        """Switch unset → wandb.log IS called (default-on)."""
        monkeypatch.delenv("SJFPO_DISABLE_WANDB", raising=False)
        import sjfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.log_metrics({"a": 1}, step=5)
        recorder.log.assert_called_once_with({"a": 1}, step=5)

    def test_log_metrics_raising_propagates(self, monkeypatch):
        """A raising wandb.log propagates — never swallowed."""
        monkeypatch.delenv("SJFPO_DISABLE_WANDB", raising=False)
        import sjfpo.wandb_logging as wl

        recorder = MagicMock()
        recorder.log.side_effect = RuntimeError("send failed")
        monkeypatch.setattr(wl, "wandb", recorder)

        with pytest.raises(RuntimeError, match="send failed"):
            wl.log_metrics({"a": 1}, step=5)


class TestLogEvalSummary:
    """Full eval payload lands in one wandb.log commit."""

    def test_all_eval_metrics_logged(self, monkeypatch):
        monkeypatch.delenv("SJFPO_DISABLE_WANDB", raising=False)
        import sjfpo.wandb_logging as wl

        logged = []
        monkeypatch.setattr(
            wl, "log_metrics", lambda data, step: logged.append((data, step))
        )

        results = [
            {"episode": 0, "return": 1.0, "length": 10},
            {"episode": 1, "return": 3.0, "length": 20},
        ]
        wl.log_eval_summary(
            2.0,
            results,
            [1.0, 2.0],
            4,
            100,
            r_rand=0.5,
            consecutive_below=1,
        )
        assert len(logged) == 1
        data, step = logged[0]
        assert step == 100
        assert data["eval/mean_return"] == pytest.approx(2.0)
        assert data["eval/min_return"] == pytest.approx(1.0)
        assert data["eval/max_return"] == pytest.approx(3.0)
        assert data["eval/std_return"] == pytest.approx(1.0)
        assert data["eval/mean_length"] == pytest.approx(15.0)
        assert data["eval/max_length"] == pytest.approx(20.0)
        assert data["eval/std_length"] == pytest.approx(5.0)
        assert data["eval/episode_count"] == 4
        assert data["eval/random_baseline"] == pytest.approx(0.5)
        assert data["eval/consecutive_below"] == 1
        assert data["eval/reward_improvement"] == pytest.approx(1.0)

    def test_optional_baseline_keys_absent_when_none(self, monkeypatch):
        monkeypatch.delenv("SJFPO_DISABLE_WANDB", raising=False)
        import sjfpo.wandb_logging as wl

        logged = []
        monkeypatch.setattr(
            wl, "log_metrics", lambda data, step: logged.append((data, step))
        )

        wl.log_eval_summary(1.0, [], [1.0], 1, 7)
        data, _ = logged[0]
        assert "eval/random_baseline" not in data
        assert "eval/consecutive_below" not in data
        assert data["eval/mean_length"] == 0.0

    def test_eval_table_list_of_lists(self, monkeypatch):
        """Table built via wandb.Table(columns=cols, data=[[..]]) list-of-lists."""
        monkeypatch.delenv("SJFPO_DISABLE_WANDB", raising=False)
        import sjfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)
        logged = []
        monkeypatch.setattr(
            wl, "log_metrics", lambda data, step: logged.append((data, step))
        )

        wl.log_eval_summary(
            1.5, [{"episode": 0, "return": 1.5, "length": 10}], [1.5], 1, 7
        )
        table_kw = recorder.Table.call_args.kwargs
        assert table_kw["columns"] == ["episode", "return", "length"]
        assert table_kw["data"] == [[0, 1.5, 10]]
        assert all(type(row) is list for row in table_kw["data"])

    def test_log_eval_summary_noop_when_disabled(self, monkeypatch):
        """SJFPO_DISABLE_WANDB=1 → zero SDK calls."""
        monkeypatch.setenv("SJFPO_DISABLE_WANDB", "1")
        import sjfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.log_eval_summary(
            1.5, [{"episode": 0, "return": 1.5, "length": 10}], [1.0, 1.5], 1, 7
        )
        recorder.Table.assert_not_called()
        recorder.log.assert_not_called()


class TestFinishRun:
    """Thin finish guard around wandb.finish."""

    def test_finish_run_calls_wandb_finish_by_default(self, monkeypatch):
        """Switch unset → wandb.finish IS called with the exit code."""
        monkeypatch.delenv("SJFPO_DISABLE_WANDB", raising=False)
        import sjfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)

        wl.finish_run(exit_code=1)
        recorder.finish.assert_called_once_with(exit_code=1)

    def test_finish_run_raising_propagates(self, monkeypatch):
        """A raising wandb.finish propagates — never swallowed."""
        monkeypatch.delenv("SJFPO_DISABLE_WANDB", raising=False)
        import sjfpo.wandb_logging as wl

        recorder = MagicMock()
        recorder.finish.side_effect = RuntimeError("finish failed")
        monkeypatch.setattr(wl, "wandb", recorder)

        with pytest.raises(RuntimeError, match="finish failed"):
            wl.finish_run(exit_code=0)


class TestSurfaceAndImportGuards:
    """Regression tests — core-fn presence, wandb required, RUN_MATRIX 45."""

    def test_five_fn_surface(self):
        """Core kept fns present; all old (pre-simplification) names gone."""
        import sjfpo.wandb_logging as wl

        for name in (
            "wandb_disabled",
            "init_wandb_run_for_method",
            "log_metrics",
            "log_eval_summary",
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
        monkeypatch.delenv("SJFPO_DISABLE_WANDB", raising=False)
        import sjfpo.wandb_logging as wl

        recorder = MagicMock()
        monkeypatch.setattr(wl, "wandb", recorder)
        wl.log_metrics({"x": 1}, step=0)
        recorder.log.assert_called_once()

    def test_run_matrix_unchanged(self):
        """len(build_run_matrix()) == 45 — cardinality locked."""
        assert len(build_run_matrix()) == 45

    def test_wandb_absent_is_hard_import_error(self):
        """Subprocess guard: wandb hidden → import raises ImportError (hard)."""
        code = (
            "import sys\n"
            "sys.modules['wandb'] = None\n"
            "sys.path.insert(0, 'src')\n"
            "try:\n    import sjfpo.wandb_logging as wl\n"
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
            "os.environ.pop('SJFPO_DISABLE_WANDB', None); "
            "sys.path.insert(0, 'src'); "
            "import sjfpo.wandb_logging as wl; "
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
