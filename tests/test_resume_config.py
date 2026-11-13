"""A resumed run must read the checkpoint's config, not the CLI's.

``make_or_resume`` returns a policy built from the checkpoint config, while
``configure_run`` went on building the scheduler and the evaluators from the
CLI config. A resumed run then wrapped one policy in another run's settings.
"""

from __future__ import annotations

import numpy as np
import pytest

from myfpo import train
from myfpo.checkpointing import ResumeError, save_policy_checkpoint
from myfpo.algo import make_optimizer
from myfpo.train import configure_run, make_or_resume

from fixtures import ACT_DIM, OBS_DIM, cfg as fixture_cfg, make_test_policy


class FakeEnv:
    """Flat-obs env stub: enough for ``configure_run``, and no stepping."""

    def __init__(self, action_dim: int = ACT_DIM):
        self.action_space = type(
            "Space",
            (),
            {
                "low": -np.ones(action_dim, dtype=np.float32),
                "high": np.ones(action_dim, dtype=np.float32),
            },
        )()


def test_a_resume_takes_the_checkpoint_config(tmp_path, monkeypatch, capsys):
    trained = make_test_policy(num_envs=2, eval_max_episodes=3, report_to="none")
    ckpt = tmp_path / "ckpt"
    save_policy_checkpoint(
        ckpt, policy=trained, optimizer=make_optimizer(trained), global_step=0
    )

    cli = fixture_cfg(num_envs=1, eval_max_episodes=99, report_to="none")
    monkeypatch.setattr(train, "sample_random_baseline", lambda _evaluator: 0.0)
    setup = configure_run(
        FakeEnv(),
        tmp_path / "run",
        cli,
        seed=0,
        total_steps=64,
        obs_dim=OBS_DIM,
        action_dim=ACT_DIM,
        resume_from_checkpoint=ckpt,
        runtime=None,
    )

    # The checkpoint says 3 episodes; the CLI said 99.
    assert setup.evaluator.max_episodes == 3
    props = capsys.readouterr().err
    assert "num_envs=2" in props and "asked for 1" in props


def test_a_policy_only_checkpoint_cannot_resume_training(tmp_path):
    trained = make_test_policy()
    ckpt = tmp_path / "ckpt"
    save_policy_checkpoint(ckpt, policy=trained, optimizer=None, global_step=0)

    with pytest.raises(ResumeError, match="carries no optimizer state"):
        make_or_resume(
            fixture_cfg(),
            "cpu",
            OBS_DIM,
            ACT_DIM,
            -np.ones(ACT_DIM, dtype=np.float32),
            np.ones(ACT_DIM, dtype=np.float32),
            ckpt,
        )


def test_a_fresh_run_keeps_the_cli_config(tmp_path, monkeypatch):
    cli = fixture_cfg(num_envs=1, eval_max_episodes=7, report_to="none")
    monkeypatch.setattr(train, "sample_random_baseline", lambda _evaluator: 0.0)
    setup = configure_run(
        FakeEnv(),
        tmp_path / "run",
        cli,
        seed=0,
        total_steps=64,
        obs_dim=OBS_DIM,
        action_dim=ACT_DIM,
        resume_from_checkpoint=None,
        runtime=None,
    )
    assert setup.evaluator.max_episodes == 7
    assert setup.policy.config.eval_max_episodes == 7
