import dataclasses

import pytest
from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType

from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
from spectral_joint_policy.evaluator import check_regression
from spectral_joint_policy.tracking import build_tracker
from spectral_joint_policy.tracking.base import (
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


def _name(**kw) -> str:
    return build_run_name("sjfpo", "Hopper-v5", 1, dataclasses.asdict(_cfg(**kw)))


def test_run_name_keeps_the_readable_tags():
    """The tags are the scan-friendly part; the hash only breaks ties.

    Built from the config rather than written out, so a hyperparameter change
    does not have to be mirrored here or the test fails for the wrong reason.
    """
    c = _cfg()
    name = _name()
    prefix = (
        f"sjfpo_Hopper-v5_s1_k{c.num_kept_tokens}_d{c.hidden_dim}x{c.dit_depth}"
        f"_ep{c.num_learning_epochs}_lr{c.policy_lr:g}_eps{c.clip_param:g}"
        f"_mc{c.num_mc_samples}"
    )
    assert name.startswith(prefix), (name, prefix)


def test_run_name_is_stable_for_one_config():
    """Same config, same name: the tensorboard log dir must not churn."""
    assert _name() == _name()


def test_untagged_config_still_gets_a_hash():
    """No tags must not mean no name: two such runs still need to differ."""
    name = _name(num_kept_tokens=1)
    assert name.endswith(config_hash(dataclasses.asdict(_cfg(num_kept_tokens=1))))


def test_one_changed_value_changes_the_name():
    """The collision this exists to fix: gamma is not in any tag."""
    assert _name() != _name(gamma=0.95)
    assert _name(gae_lambda=0.5) != _name(gae_lambda=0.95)


def test_feature_dicts_do_not_change_the_name():
    """They are fixed by the env, and they render badly in a log path."""
    d = dataclasses.asdict(_cfg())
    with_features = dict(d, input_features={"observation.state": "other"})
    assert build_run_name("sjfpo", "Hopper-v5", 1, d) == build_run_name(
        "sjfpo", "Hopper-v5", 1, with_features
    )


# --- forgetting: the peak-regression stop ----------------------------------


def _regression_streak(returns: list[float], best: float = 500.0) -> list[tuple[int, bool]]:
    streak, out = 0, []
    for r in returns:
        streak, window = check_regression(
            r, best, streak, return_frac=0.7, patience=3
        )
        out.append((streak, window))
    return out


def test_a_new_peak_resets_the_streak():
    """One good eval has to clear the memory of the bad ones."""
    # 0.7 * 500 = 350, so 300 is a regression and 600 is a new peak.
    streak = _regression_streak([300, 300, 600])
    assert [s for s, _ in streak] == [1, 2, 0]


def test_a_window_completes_once_per_patience_bad_evals():
    """Each window costs one lr decay, so the plateau gets a smaller step first."""
    streak = _regression_streak([300, 300, 300, 300, 300, 300])
    assert [w for _, w in streak] == [False, False, True, False, False, True]


def test_being_just_above_the_threshold_is_not_a_regression():
    """0.7 * 500 = 350, and 349 is below it. The boundary is the whole point."""
    assert check_regression(350, 500, 0, return_frac=0.7, patience=1) == (0, False)
    assert check_regression(349, 500, 0, return_frac=0.7, patience=1) == (0, True)


def test_no_peak_yet_is_not_a_regression():
    assert check_regression(0.0, None, 2, return_frac=0.7, patience=1) == (0, False)
