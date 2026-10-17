import dataclasses
from pathlib import Path

import pytest
from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType

from myfpo.configuration_sjfpo import SJFPOConfig
from myfpo.tracking import build_tracker
from myfpo.tracking.base import (
    ExperimentTracker,
    build_run_name,
    config_hash,
)


def _cfg(**kw):
    return dataclasses.replace(
        SJFPOConfig(**kw),
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (5,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
    )


def test_report_to_is_validated():
    with pytest.raises(ValueError, match="report_to"):
        _cfg(report_to="mlflow").__post_init__()


def test_none_gives_a_disabled_tracker():
    t = build_tracker(
        _cfg(report_to="none"),
        out_dir="/tmp/tt",
        method="sjfpo",
        env_name="Hopper-v5",
        seed=1,
        total_steps=100,
    )
    assert isinstance(t, ExperimentTracker)
    assert t.enabled is False
    t.log({"x": 1.0}, 0)
    t.finish(0)


def test_all_fans_out_to_every_backend():
    t = build_tracker(
        _cfg(report_to="all"),
        out_dir="/tmp/tt",
        method="sjfpo",
        env_name="Hopper-v5",
        seed=1,
        total_steps=100,
    )
    assert t.enabled is True
    assert len(t.backends) == 2
    seen = []

    class Spy:
        def log(self, data, step):
            seen.append((data, step))

        def finish(self, code):
            pass

    t.backends = [Spy(), Spy()]
    t.log({"a": 1.0}, 7)
    assert seen == [({"a": 1.0}, 7), ({"a": 1.0}, 7)]


def test_unknown_report_to_raises_from_the_factory():
    cfg = _cfg()
    object.__setattr__(cfg, "report_to", "mlflow")
    with pytest.raises(ValueError, match="report_to"):
        build_tracker(
            cfg,
            out_dir="/tmp/tt",
            method="sjfpo",
            env_name="Hopper-v5",
            seed=1,
            total_steps=100,
        )


def _name(**kw) -> str:
    return build_run_name("sjfpo", "Hopper-v5", 1, dataclasses.asdict(_cfg(**kw)))


def test_run_name_keeps_the_readable_tags():
    """The tags are the scan-friendly part. The hash only breaks ties."""
    c = _cfg()
    name = _name()
    prefix = (
        f"sjfpo_Hopper-v5_s1_h{c.action_horizon}_d{c.hidden_dim}x{c.dit_depth}"
        f"_ep{c.num_learning_epochs}_lr{c.policy_lr:g}_eps{c.clip_param:g}"
        f"_mc{c.num_mc_samples}"
    )
    assert name.startswith(prefix), (name, prefix)


def test_run_name_is_stable_for_one_config():
    """Same config, same name: the tensorboard log dir must not churn."""
    assert _name() == _name()


def test_untagged_config_still_gets_a_hash():
    """No tags must not mean no name: two such runs still need to differ."""
    name = _name(noise_sampling_mode="uniform")
    assert name.endswith(
        config_hash(dataclasses.asdict(_cfg(noise_sampling_mode="uniform")))
    )


def test_one_changed_value_changes_the_name():
    """The collision this exists to fix: gamma is not in any tag."""
    assert _name() != _name(gamma=0.95)
    assert _name(num_envs=8) != _name(num_envs=64)


def test_feature_dicts_do_not_change_the_name():
    """They are fixed by the env, and they render badly in a log path."""
    d = dataclasses.asdict(_cfg())
    with_features = dict(d, input_features={"observation.state": "other"})
    assert build_run_name("sjfpo", "Hopper-v5", 1, d) == build_run_name(
        "sjfpo", "Hopper-v5", 1, with_features
    )




def test_two_out_dirs_do_not_share_one_tensorboard_directory(tmp_path):
    """The defect: out_dir was accepted and ignored.

    The run name carries no out_dir, so with the log directory pinned to the
    working directory two runs with the same config wrote to one place and their
    series interleaved. The path is now under out_dir.
    """
    dirs = []
    for name in ("run_a", "run_b"):
        out = tmp_path / name
        out.mkdir()
        t = build_tracker(
            _cfg(report_to="tensorboard"),
            out_dir=str(out),
            method="sjfpo",
            env_name="Hopper-v5",
            seed=1,
            total_steps=100,
        )
        t.log({"x": 1.0}, 0)
        backend = t.backends[0]
        dirs.append(Path(backend.writer.log_dir))
        t.finish(0)

    assert dirs[0] != dirs[1], "two out_dirs resolved to one log directory"
    for name, d in zip(("run_a", "run_b"), dirs):
        assert d.parent.name == "tensorboard", d
        assert d.parent.parent == tmp_path / name, f"{d} is not under its out_dir"


def test_the_log_directory_is_not_in_the_working_directory(tmp_path, monkeypatch):
    """A tracker must not write into whatever directory pytest was started in."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    out = tmp_path / "elsewhere"
    out.mkdir()
    t = build_tracker(
        _cfg(report_to="tensorboard"),
        out_dir=str(out),
        method="sjfpo",
        env_name="Hopper-v5",
        seed=1,
        total_steps=100,
    )
    t.log({"x": 1.0}, 0)
    assert not (work / "tensorboard").exists(), (
        "the tracker wrote into the working directory instead of out_dir"
    )
    t.finish(0)
