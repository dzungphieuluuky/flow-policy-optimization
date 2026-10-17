"""Which flow start the reported score and the stopper read.

Training collects with the stochastic start, so the critic bootstraps from the
stochastic start. The reported score used the zero start anyway, so the run
measured a policy that training never optimized.

The two names now cover the config, the logs, the CSV, and the return dict.
This file pins the naming, the plumbing, and the separation of the two
questions: what the policy returns, and when to stop.
"""

import dataclasses

import numpy as np
import pytest

from myfpo.configuration_sjfpo import SJFPOConfig

SAMPLINGS = ("zero_start", "noise_start")


# --- the config -----------------------------------------------------------


def test_both_fields_default_to_the_trained_distribution():
    cfg = SJFPOConfig()
    assert cfg.eval_sampling == "noise_start", (
        "the reported score must read the distribution training optimizes"
    )
    assert cfg.stop_eval_sampling == "noise_start", (
        "the stopper must read the distribution training optimizes"
    )


def test_the_two_fields_are_independent():
    """Reporting one start and stopping on the other is a valid request."""
    cfg = dataclasses.replace(
        SJFPOConfig(), eval_sampling="zero_start", stop_eval_sampling="noise_start"
    )
    cfg.validate()
    assert cfg.eval_sampling == "zero_start"
    assert cfg.stop_eval_sampling == "noise_start"


@pytest.mark.parametrize("field", ["eval_sampling", "stop_eval_sampling"])
def test_an_unknown_sampling_is_rejected(field):
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(SJFPOConfig(), **{field: "noise"}).validate()


@pytest.mark.parametrize("field", ["eval_sampling", "stop_eval_sampling"])
@pytest.mark.parametrize("value", SAMPLINGS)
def test_both_named_values_are_accepted(field, value):
    dataclasses.replace(SJFPOConfig(), **{field: value}).validate()


def test_the_released_defaults_table_is_untouched():
    """Item 3 adds fields. It must not edit an existing default."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "released_defaults", root / "tests" / "test_released_defaults.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = SJFPOConfig()
    for name, (want, _where) in mod.RELEASED.items():
        if name == "spo_clip_param_is_separate":
            continue
        assert getattr(cfg, name) == want, f"{name} drifted"


# --- the adapter ----------------------------------------------------------


def test_the_adapter_maps_the_config_value_to_the_start_point():
    from myfpo.train_sjfpo import _EvalPolicy

    class _P:
        pass

    assert _EvalPolicy.for_sampling(_P(), "zero_start").start_from_zero is True
    assert _EvalPolicy.for_sampling(_P(), "noise_start").start_from_zero is False


def test_the_adapter_rejects_an_unknown_start():
    from myfpo.train_sjfpo import _EvalPolicy

    class _P:
        pass

    with pytest.raises(ValueError, match="sampling"):
        _EvalPolicy.for_sampling(_P(), "noise")


def test_the_two_adapters_give_different_returns():
    """The whole point: the two starts must not be interchangeable."""
    import gymnasium as gym

    from myfpo.configuration_sjfpo import SJFPOConfig
    from myfpo.evaluator import Evaluator
    from myfpo.train_sjfpo import _EvalPolicy, build_policy

    env = gym.make("Hopper-v5")
    try:
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        low = np.atleast_1d(np.asarray(env.action_space.low, dtype=np.float32))
        high = np.atleast_1d(np.asarray(env.action_space.high, dtype=np.float32))
        policy = build_policy(
            obs_dim,
            act_dim,
            dataclasses.replace(SJFPOConfig(), compile=False),
            device="cpu",
            action_low=low,
            action_high=high,
        )
        ev = Evaluator(env, eval_seed=4, max_episodes=6)
        zero_start = ev.summarize(
            ev.evaluate(_EvalPolicy.for_sampling(policy, "zero_start")), seed=4
        )
        noise_start = ev.summarize(
            ev.evaluate(_EvalPolicy.for_sampling(policy, "noise_start")), seed=4
        )
        assert (
            zero_start["return_mean"] != noise_start["return_mean"]
        ), "a fresh policy gave the same score for both starts; the two paths are not wired"
    finally:
        env.close()


# --- the plumbing ---------------------------------------------------------


def test_the_trainer_builds_both_adapters_once():
    import inspect

    from myfpo.train_sjfpo import train_sjfpo_run

    src = inspect.getsource(train_sjfpo_run)
    assert '"zero_start": _EvalPolicy(policy, start_from_zero=True)' in src
    assert '"noise_start": _EvalPolicy(policy, start_from_zero=False)' in src


def test_the_trainer_reads_both_names_from_the_config():
    import inspect

    from myfpo.train_sjfpo import train_sjfpo_run

    src = inspect.getsource(train_sjfpo_run)
    assert "reported_name = cfg.eval_sampling" in src
    assert "stop_name = cfg.stop_eval_sampling" in src


def test_each_start_is_evaluated_once():
    """The reported start writes the history. A second pass would double the cost."""
    import inspect

    from myfpo.train_sjfpo import train_sjfpo_run

    src = inspect.getsource(train_sjfpo_run)
    assert src.count("evaluator.evaluate(") == 1, (
        "the eval dict comprehension must run each start once; an extra call "
        "doubles the eval cost"
    )
    assert 'out_dir=out_dir if name == reported_name else None' in src


def test_the_stopper_reads_the_stop_distribution():
    import inspect

    from myfpo.train_sjfpo import train_sjfpo_run

    src = inspect.getsource(train_sjfpo_run)
    assert "check_divergence(\n                    stop_return" in src, (
        "check_divergence must read stop_return, not the reported mean"
    )
    assert "stop_return,\n                    best_return," in src


def test_both_returns_are_logged_under_their_own_names():
    import inspect

    from myfpo.train_sjfpo import train_sjfpo_run

    src = inspect.getsource(train_sjfpo_run)
    assert 'f"eval/{name}_return_mean"' in src
    # The tracker takes floats only, so the two names must not be scalars.
    assert '"eval/reported_sampling"' not in src
    assert '"eval/stop_sampling"' not in src


def test_the_names_go_to_the_run_metadata_instead():
    """A string constant cannot be a per-step scalar. It belongs in the metadata."""
    import inspect

    from myfpo.train_sjfpo import train_sjfpo_run

    src = inspect.getsource(train_sjfpo_run)
    assert 'runtime["eval_sampling"] = cfg.eval_sampling' in src
    assert 'runtime["stop_eval_sampling"] = cfg.stop_eval_sampling' in src


def test_the_tracker_rejects_no_key_after_this_change():
    """Every value the trainer logs must be a float. The tensorboard backend
    calls ``float(value)``, so one string aborts the run."""
    import inspect

    from myfpo.train_sjfpo import train_sjfpo_run

    src = inspect.getsource(train_sjfpo_run)
    # The one remaining string-valued value is the EMA flag, which is already
    # wrapped in float(). Every literal string in a logged dict is a key, not a
    # value, so this asserts there is no bare string assigned as a value.
    assert '"train/stochastic_mean_return"' not in src
    for line in src.splitlines():
        stripped = line.strip()
        if not stripped.startswith('"eval/') or ":" not in stripped:
            continue
        value = stripped.split(":", 1)[1].strip().rstrip(",")
        if not value or value.startswith(('"', "f\"", "{")):
            # A key literal, an f-string key, or a spread of another mapping.
            continue
        literal = value.split("}")[0].strip() if value.startswith("(") else value
        if (
            literal.startswith("float(")
            or literal.endswith(")")
            or "[" in literal
            or "(" in literal
        ):
            # A float() call, a parenthesised expression, a dict subscript, or a
            # helper call. All of these resolve to a float at run time.
            continue
        try:
            float(literal)
        except ValueError:
            pytest.fail(f"tracker value is not numeric: {stripped}")


def test_the_stochastic_mean_return_key_is_gone():
    """The old key named one distribution and hid the other."""
    import inspect

    from myfpo.train_sjfpo import train_sjfpo_run

    src = inspect.getsource(train_sjfpo_run)
    assert "stochastic_mean_return" not in src, (
        "the single-adapter key is replaced by one key per named start"
    )


def test_the_messages_name_the_start_they_read():
    """A stop message must say which distribution triggered it."""
    import inspect

    from myfpo.train_sjfpo import train_sjfpo_run

    src = inspect.getsource(train_sjfpo_run)
    # The messages span two f-strings each, so the shared fragment is what the
    # assertion reads.
    assert src.count("on the {stop_name}") == 3, (
        "the DIVERGED, plateau, and REGRESSED messages must each name the start"
    )
    for token in ("DIVERGED at step {global_step}", "REGRESSED at step {global_step}"):
        assert token in src, token
    assert "plateau at step {global_step} on the {stop_name}" in src


# --- the generated artifact ----------------------------------------------


def test_both_fields_reach_the_generated_yaml():
    """A new field must sit in the SECTIONS table, or parity fails."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "configs" / "sjfpo.yaml").read_text())
    cfg = SJFPOConfig()
    assert data["eval_sampling"] == cfg.eval_sampling
    assert data["stop_eval_sampling"] == cfg.stop_eval_sampling


def test_the_generator_emits_both_fields():
    import importlib.util

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "gen_cfg", root / "scripts" / "gen_sjfpo_config.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    emitted = {name for _title, names in mod.SECTIONS for name in names}
    assert "eval_sampling" in emitted
    assert "stop_eval_sampling" in emitted