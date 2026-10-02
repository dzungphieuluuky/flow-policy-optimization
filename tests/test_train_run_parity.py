"""The trainer, end to end: ``train_sjfpo_run``.

Three of these are gates that only the trainer can provide, and all three failed
at least once while this was being written:

* **the import stays torch-free.** ``lerobot`` is where ``CHECKPOINTS_DIR`` and
  ``set_seed`` live in the torch tree, and importing either pulls torch in. A
  subprocess, because ``sys.modules`` cannot be un-popped.
* **x64 does not promote the network to fp64.** ``runtime.configure_device``
  enables ``jax_enable_x64`` (the running-mean counts need fp64), which turns
  *every* unpinned ``jnp`` and ``jax.random`` call fp64. flax's
  ``variance_scaling`` default follows it, so the conditioner returned fp64 and
  the Euler sampler's ``lax.scan`` carry changed dtype mid-scan and refused to
  compile. No test enabled x64 before this one, so nothing caught it.
* **a resume round-trips.** The checkpointing work in milestone 6 had no caller
  until this.

The smoke run is deliberately tiny: 64 env steps, two envs, 2 MC samples and 4
kept tokens. It exists to prove the loop is *wired*, not to measure learning.
"""
import json
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from sjfpo_jax import checkpointing as JCK
from sjfpo_jax import train_sjfpo as X
from sjfpo_jax.configuration_sjfpo import SJFPOConfig
from sjfpo_jax.state import build_state

ENV = "Reacher-v5"
# Reacher-v5: 4 cos/sin + 6 joint offsets, 2 joint torques.
OBS_DIM, ACT_DIM = 10, 2


def tiny_cfg(**overrides):
    cfg = SJFPOConfig(
        report_to="none",
        num_envs=2,
        num_mc_samples=2,
        num_kept_tokens=4,
        chunks_per_update=8,
        batch_size=4,
        eval_max_episodes=1,
        # 8, not the default 500: a 64-step run would never reach it, so the EMA
        # would stay cold and swap_in_ema would never be called. The warm path is
        # where the milestone-6 shadow bugs lived, so it has to run.
        ema_warmup_steps=8,
    )
    for key, value in overrides.items():
        cfg = type(cfg)(**{**cfg.__dict__, key: value})
    return cfg


def run(tmp_path, **kwargs):
    env = gym.make(ENV)
    try:
        return X.train_sjfpo_run(
            env, tmp_path / "run", tiny_cfg(), seed=1,
            total_steps=64, eval_steps=32, save_steps=32, **kwargs
        )
    finally:
        env.close()


# ------------------------------------------------------------- the import gate


def test_importing_the_trainer_does_not_pull_in_torch_or_lerobot():
    """A subprocess, because ``sys.modules`` cannot be un-popped.

    ``CHECKPOINTS_DIR`` and ``set_seed`` both live under ``lerobot`` in the torch
    tree, and either import drags torch into the JAX process. So the local
    constants in ``train_sjfpo`` are load-bearing, not duplication.
    """
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        "import sjfpo_jax.train_sjfpo\n"
        "leaked = [m for m in ('torch', 'lerobot') if m in sys.modules]\n"
        "print(','.join(leaked))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=False
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "", f"leaked into the process: {out.stdout.strip()}"


def test_grad_accum_steps_greater_than_one_is_refused():
    """There is no group boundary to accumulate towards.

    ``update_step`` is one clip-and-AdamW, so a second accumulation boundary
    would have to live inside it. Refusing beats silently running a different
    experiment than the config asks for.
    """
    env = gym.make(ENV)
    try:
        with pytest.raises(ValueError, match="grad_accum_steps"):
            X.train_sjfpo_run(
                env, "/tmp/x", tiny_cfg(grad_accum_steps=2), seed=1, total_steps=8,
            )
    finally:
        env.close()


# ------------------------------------------------------------ the x64 regression


def test_the_network_stays_fp32_with_x64_enabled():
    """The bug this pins: every unpinned call follows ``jax_enable_x64``.

    ``RunningMeanStd`` counts are fp64, so the trainer genuinely needs x64 on.
    Every fp32 array in the model has to *say* fp32, or the conditioner's output
    becomes fp64 and ``sample_action``'s ``lax.scan`` fails to compile because
    the carry's dtype changes.
    """
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        "import jax\n"
        "jax.config.update('jax_enable_x64', True)\n"
        "import jax.numpy as jnp, numpy as np\n"
        "from sjfpo_jax.configuration_sjfpo import SJFPOConfig\n"
        "from sjfpo_jax.state import build_state\n"
        "from sjfpo_jax.policy import predict_action_chunk\n"
        "cfg = SJFPOConfig(obs_dim=11, action_dim=2)\n"
        "st = build_state(cfg, 11, 2, jax.random.key(0))\n"
        "obs = jnp.asarray(np.zeros((1, 16, 11), np.float32))\n"
        # Compiling is the assertion: an fp64 carry would raise here.
        "out = jax.jit(lambda s, o: predict_action_chunk(s, o, cfg, zero_sampling=True))(st, obs)\n"
        "print(out.dtype)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=False
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.strip() == "float32", out.stdout.strip()


# ------------------------------------------------------------------- the smoke


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    return run(tmp_path_factory.mktemp("smoke"))


def test_the_run_completes_and_reports_the_torch_shaped_dict(smoke):
    assert smoke["method"] == "sjfpo_jax"
    assert smoke["env_name"] == ENV
    assert smoke["global_step"] >= 64
    assert smoke["diverged"] is False and smoke["regressed"] is False
    assert smoke["fps"] > 0
    assert set(smoke) == {
        "method", "env_name", "seed", "reward_mean_last10", "fps", "artifacts",
        "best_artifact", "best_return", "diverged", "regressed", "lr_decays",
        "global_step",
    }, sorted(smoke)


def test_the_artifacts_are_on_disk(smoke, tmp_path_factory):
    root = Path(smoke["artifacts"][0]).parent.parent
    assert (root / "checkpoints" / "final").is_dir(), sorted(p.name for p in root.rglob("*"))
    assert (root / "checkpoints" / "best").is_dir()
    assert (root / "eval_history.csv").is_file(), "the evaluator never ran"
    assert (root / "checkpoints" / "step_64").is_dir()


def test_the_unnormalised_action_guard_did_not_trip(tmp_path_factory):
    """``action_abs_max > 1.05`` would have raised. Reaching the end is the assert.

    It trips when the chunk is rescaled before it is clipped, so it is the one
    cheap detector for an unscale-order regression in the rollout path.
    """
    assert run(tmp_path_factory.mktemp("guard"))["global_step"] > 0


# ------------------------------------------------------------------- the resume


def test_a_run_resumes_from_its_own_final_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("resume")
    first = run(root)
    final = next(a for a in first["artifacts"] if a.endswith("final"))

    resumed = _resume(final, root / "resumed", total_steps=96)
    assert resumed["global_step"] > first["global_step"], (
        first["global_step"], resumed["global_step"]
    )
    info = json.loads((root / "resumed" / "resume_info.json").read_text())
    assert info["restored_global_step"] == first["global_step"], info


def _resume(checkpoint, out_dir, total_steps):
    env = gym.make(ENV)
    try:
        return X.train_sjfpo_run(
            env, out_dir, tiny_cfg(), seed=1, total_steps=total_steps,
            eval_steps=32, save_steps=32, resume_from_checkpoint=checkpoint,
        )
    finally:
        env.close()


def test_resume_restores_the_optimizer_moments_and_both_latches(tmp_path_factory):
    """The three things a resume exists for.

    ``checkpointing.py`` grew three silent bugs -- a mangled optax tree, a
    mis-keyed EMA shadow, a wrong-keyed name table -- all of which produce a run
    that *continues* with silently wrong state. So this asserts the restored
    moments are non-zero and the latches are set, not merely that it runs.
    """

    root = tmp_path_factory.mktemp("latches")
    first = run(root)
    final = next(a for a in first["artifacts"] if a.endswith("final"))
    state, opt_state, cfg, counters = JCK.load_checkpoint(final)

    assert counters["global_step"] == first["global_step"]
    assert state.ema_initialised is True, "the EMA latch was lost, so the average restarts"
    assert state.ema_shadow is not None
    assert state.target_critic_initialised is True
    mu = JCK.flatten(opt_state[1][0].mu)
    assert mu, "no moments came back"
    assert float(opt_state[1][0].count) > 0, "the step counter reset"
    # Not "every moment is non-zero". A *fresh* actor has 40 of 80 leaves with a
    # structurally zero gradient -- the zero-init adaLN holds the gates at 0, so
    # attention and the modulation heads are off the loss path entirely -- and
    # those leaves' Adam state legitimately stays zero. The critic is never gated,
    # so all 14 of its leaves must have trained.
    critic = {k: v for k, v in mu.items() if k.startswith("['critic']")}
    assert len(critic) == 14, len(critic)
    assert all(float(jnp.abs(v).sum()) > 0 for v in critic.values()), (
        "the critic's Adam state is zero, so the resume restarts it"
    )
    assert sum(1 for v in mu.values() if float(jnp.abs(v).sum()) > 0) >= 20
    assert cfg.obs_dim == OBS_DIM and cfg.action_dim == ACT_DIM


# ----------------------------------------------------------------- the eval path


def test_EvalPolicy_resets_its_buffer_between_episodes():
    """``Evaluator`` calls ``reset()`` and discards the return value."""
    cfg = tiny_cfg()
    state = _small_state(cfg)
    policy = X.EvalPolicy(state, cfg, zero_sampling=True)

    first = policy(np.zeros(OBS_DIM, dtype=np.float32))
    n_consumed = int(policy.buffer.length)
    assert n_consumed < cfg.action_horizon, "the whole chunk went in one step"
    policy.reset()
    assert int(policy.buffer.length) == 0
    assert int(policy.buffer.frame_len) == 0
    second = policy(np.zeros(OBS_DIM, dtype=np.float32))
    assert np.allclose(first, second), "the reset did not take"


def test_the_two_eval_noise_modes_produce_different_actions():
    """Same seed, different start point: eps=0 versus sampled noise."""
    cfg = tiny_cfg()
    state = _small_state(cfg)
    obs = np.random.default_rng(0).standard_normal(OBS_DIM).astype(np.float32)
    key = jax.random.key(0)
    deterministic = X.EvalPolicy(state, cfg, zero_sampling=True)(obs)
    stochastic = X.EvalPolicy(state, cfg, zero_sampling=False, key=key)(obs)
    assert not np.allclose(deterministic, stochastic), (
        "the stochastic path is not drawing noise"
    )


def test_the_eval_action_is_a_flat_float_vector():
    cfg = tiny_cfg()
    policy = X.EvalPolicy(_small_state(cfg), cfg)
    action = policy(np.zeros(OBS_DIM, dtype=np.float32))
    assert action.shape == (ACT_DIM,), action.shape
    assert action.dtype == np.float32, action.dtype


def _small_state(cfg):
    state = build_state(cfg, OBS_DIM, ACT_DIM, jax.random.key(0))
    env = gym.make(ENV)
    try:
        return state.replace(
            action_low=jnp.asarray(env.action_space.low, dtype=jnp.float32),
            action_high=jnp.asarray(env.action_space.high, dtype=jnp.float32),
        )
    finally:
        env.close()
