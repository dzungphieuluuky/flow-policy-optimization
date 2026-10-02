"""``scripts/train.py``: the ``--method sjfpo_jax`` branch.

Each test here is a way the branch can look fine while being wrong:

* **the config prefix follows the method.** ``--sjfpo.policy_lr`` must land on
  the torch config and ``--sjfpo_jax.policy_lr`` on the JAX one. A shared prefix
  would let a torch flag silently retune the JAX run.
* **the two backends' imports do not cross.** ``run_single`` imports the torch
  trainer inside the ``sjfpo`` branch on purpose; hoisting it puts torch in the
  JAX process, which is the one thing ``sjfpo_jax.train_sjfpo`` exists to avoid.
* **the config is parseable without the dims.** draccus constructs it before the
  env is known, which is why the dims check moved to ``build_state``.

The W&B key set is pinned as an explicit list rather than compared against the
torch loop: that comparison needs both loops to run, and the keys are an
interface either way.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def parse(argv):
    import draccus
    from train import TrainConfig

    return draccus.parse(config_class=TrainConfig, args=argv)


# ------------------------------------------------------------- the CLI surface


def test_both_methods_parse():
    assert parse(["--method", "sjfpo", "--env", "Reacher-v5"]).method == "sjfpo"
    assert parse(["--method", "sjfpo_jax", "--env", "Reacher-v5"]).method == "sjfpo_jax"


def test_the_config_prefix_follows_the_method(tmp_path):
    """The prefix decides whether a YAML overlay clobbers a CLI flag.

    draccus routes ``--sjfpo_jax.policy_lr`` correctly no matter what
    ``_method_cli_fields`` says -- that helper exists for the *YAML* overlay,
    which would otherwise win over a value the user set explicitly on the command
    line. So the test needs a config file: without one the prefix is unobservable.
    """
    from train import _method_cli_fields, select_method_cfg

    config = tmp_path / "overlay.yaml"
    config.write_text("policy_lr: 9.0e-4\n")

    argv = [
        "--method", "sjfpo_jax",
        "--sjfpo_jax.policy_lr", "3e-4",
        "--config", str(config),
    ]
    args = parse(argv)
    assert _method_cli_fields(argv, "sjfpo_jax") == {"policy_lr"}

    # The CLI value wins over the overlay, on the jax config.
    cfg = select_method_cfg(args, {"policy_lr": 9.0e-4}, cli_fields={"policy_lr"})
    assert cfg.policy_lr == pytest.approx(3e-4), cfg.policy_lr
    # And with no CLI touch, the overlay applies.
    assert select_method_cfg(
        parse(["--method", "sjfpo_jax", "--env", "Reacher-v5"]),
        {"policy_lr": 9.0e-4},
        cli_fields=set(),
    ).policy_lr == pytest.approx(9.0e-4)


def test_a_torch_flag_does_not_reach_the_jax_config():
    """The two configs are independent fields, not one shared object."""
    args = parse(
        ["--method", "sjfpo_jax", "--env", "Reacher-v5", "--sjfpo.policy_lr", "9e-4"]
    )
    assert args.sjfpo.policy_lr == pytest.approx(9e-4)
    assert args.sjfpo_jax.policy_lr == pytest.approx(1e-5), args.sjfpo_jax.policy_lr


def test_select_method_cfg_returns_the_backend_s_own_dataclass():
    from train import select_method_cfg

    from sjfpo_jax.configuration_sjfpo import SJFPOConfig as JaxCfg
    from spectral_joint_policy.configuration_sjfpo import SJFPOConfig as TorchCfg

    torch_args = parse(["--method", "sjfpo", "--env", "Reacher-v5"])
    jax_args = parse(["--method", "sjfpo_jax", "--env", "Reacher-v5"])
    assert isinstance(select_method_cfg(torch_args, {}, set()), TorchCfg)
    assert isinstance(select_method_cfg(jax_args, {}, set()), JaxCfg)


def test_the_jax_config_is_parseable_without_the_dims():
    """draccus constructs it during ``parse()``, before the env is known.

    This is why the dims check moved to ``build_state``; see
    ``test_config_parity.test_unset_dims_are_allowed_here_and_rejected_by_build_state``.
    """
    args = parse(["--method", "sjfpo_jax", "--env", "Reacher-v5"])
    assert args.sjfpo_jax.obs_dim == 0
    assert args.sjfpo_jax.action_dim == 0


def test_an_unknown_method_is_rejected():
    with pytest.raises(SystemExit):
        parse(["--method", "nonsense", "--env", "Reacher-v5"])


# ------------------------------------------------------------- the dispatch


def _patch_dims(monkeypatch):
    import train as T

    monkeypatch.setattr(T, "get_observation_action_dim", lambda env_id: (10, 2))


def test_run_single_dispatches_to_the_jax_trainer(monkeypatch):
    import train as T

    import sjfpo_jax.train_sjfpo as jax_train

    seen = {}

    monkeypatch.setattr(
        jax_train, "train_sjfpo_run", lambda **kw: seen.update(kw) or {"method": "sjfpo_jax"}
    )
    _patch_dims(monkeypatch)
    result = T.run_single(
        method="sjfpo_jax",
        env_id="Reacher-v5",
        seed=1,
        out_dir=Path("/tmp/does-not-matter"),
        cfg=T.TrainConfig().sjfpo_jax,
        total_steps=8,
        eval_steps=4,
        save_steps=4,
        obs_dim=None,
        action_dim=None,
    )
    assert result["method"] == "sjfpo_jax"
    assert seen["total_steps"] == 8
    assert seen["resume_from_checkpoint"] is None


def test_run_single_dispatches_to_the_torch_trainer(monkeypatch):
    import train as T

    import spectral_joint_policy.train_sjfpo as torch_train

    seen = {}

    monkeypatch.setattr(
        torch_train, "train_sjfpo_run", lambda **kw: seen.update(kw) or {"method": "sjfpo"}
    )
    _patch_dims(monkeypatch)
    T.run_single(
        method="sjfpo",
        env_id="Reacher-v5",
        seed=1,
        out_dir=Path("/tmp/does-not-matter"),
        cfg=T.TrainConfig().sjfpo,
        total_steps=8,
        eval_steps=4,
        save_steps=4,
        obs_dim=None,
        action_dim=None,
    )
    assert seen["total_steps"] == 8
    assert seen["resume_from_checkpoint"] is None


def test_a_jax_dispatch_does_not_import_the_torch_trainer():
    """``run_single`` must not hoist the torch trainer's import.

    Note the boundary: ``scripts/train.py`` itself imports torch, because
    ``spectral_joint_policy.config`` and the torch ``SJFPOConfig`` are parsed at
    module level for the ``--method sjfpo`` branch. So the torch-free property
    belongs to the **library** -- ``sjfpo_jax.train_sjfpo`` and its imports --
    which ``test_train_run_parity`` gates in a subprocess. What is checked here is
    the part this branch controls: a ``sjfpo_jax`` dispatch must not reach for
    ``spectral_joint_policy.train_sjfpo``.
    """
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT / 'scripts')!r})\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        "import train\n"
        "import sjfpo_jax.train_sjfpo as J\n"
        # Stub the loop: the question is which module got imported, not
        # whether training works.\n"
        "J.train_sjfpo_run = lambda **kw: {'method': 'sjfpo_jax'}\n"
        "train.get_observation_action_dim = lambda env_id: (10, 2)\n"
        "train.run_single(\n"
        "    method='sjfpo_jax', env_id='Reacher-v5', seed=1,\n"
        "    out_dir='.', cfg=train.TrainConfig().sjfpo_jax,\n"
        "    total_steps=1, eval_steps=1, save_steps=1,\n"
        "    obs_dim=None, action_dim=None)\n"
        "print('spectral_joint_policy.train_sjfpo' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=False
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "False", "a jax dispatch imported the torch trainer"


def test_resume_is_allowed_for_both_sjfpo_methods(monkeypatch, capsys, tmp_path):
    """``main`` used to reject --resume for anything but --method sjfpo."""
    import train as T

    _patch_dims(monkeypatch)
    monkeypatch.setattr(T, "load_yaml_dict", lambda path: {})
    monkeypatch.setattr(T, "resolve_run_dir", lambda *a, **k: tmp_path / "run")
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    monkeypatch.setattr(T, "run_single", lambda *a, **kw: {"method": a[0]})

    for method in ("sjfpo", "sjfpo_jax"):
        argv = [
            "--method", method, "--env", "Reacher-v5",
            "--out_dir", str(tmp_path),
            "--resume", str(checkpoint),
        ]
        assert T.main(argv) == 0, (method, capsys.readouterr().err)
