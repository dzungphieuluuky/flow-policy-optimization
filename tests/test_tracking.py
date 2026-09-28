import dataclasses

import pytest
from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType

from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.tracking import build_tracker
from sjfpo.tracking.base import ExperimentTracker


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
    t.log_video([], 0, fps=30)
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

        def log_video(self, *a, **k):
            pass

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
