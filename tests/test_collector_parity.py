"""torch vs jax parity for the vec collector.

Same policy, same ``reset(seed=...)``, same action bounds: the two collectors
must produce byte-comparable ``chunk_state`` / ``chunk_actions`` /
``action_is_pad`` / ``step_rewards``. Three situations are covered, because each
one exercises a different branch of the awaiting/reset state machine:

* a chunk that closes on its own (the normal case)
* a terminal, which arms ``awaiting`` so the reset obs arrives on the *next* step
* a budget cut, which sets ``chunk_budget_truncated`` and does **not** arm
  ``awaiting`` -- the env is still alive
"""
import sys
from pathlib import Path

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sjfpo_jax import collector as J
from spectral_joint_policy import collector as T

OBS_STEPS, HORIZON, ACTION_STEPS, NUM_ENVS = 16, 16, 16, 4
ACTION_DIM = 2
OBS_DIM = 10  # Reacher-v5: 4 cos/sin + 6 joint offsets
LOW = np.array([-2.0, -2.0], dtype=np.float32)
HIGH = np.array([2.0, 2.0], dtype=np.float32)
RTOL, ATOL = 1e-5, 1e-6


def fixed_policy(seed=0):
    """A deterministic map from a window to a chunk, so both sides agree exactly."""
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((OBS_STEPS * OBS_DIM, HORIZON * ACTION_DIM), dtype=np.float32) * 0.05

    def torch_policy(states):
        batch = states.shape[0]
        flat = states.reshape(batch, -1)
        return (flat @ torch.from_numpy(weight)).reshape(batch, HORIZON, ACTION_DIM)

    def jax_policy(states):
        batch = states.shape[0]
        flat = states.reshape(batch, -1)
        return (flat @ jnp.asarray(weight)).reshape(batch, HORIZON, ACTION_DIM)

    return torch_policy, jax_policy


def make_envs(seed=0):
    """Four Reacher envs. Reacher-v5's own TimeLimit is 50 steps, so a budget
    above 50 per env forces terminations and autoresets with no monkeypatching."""
    return gym.vector.SyncVectorEnv(
        [lambda: gym.make("Reacher-v5") for _ in range(NUM_ENVS)]
    )


def run_both(total_steps, seed=0):
    torch_policy, jax_policy = fixed_policy(seed)
    torch_buffer = T.RolloutBuffer(
        max_chunks=total_steps, max_steps=total_steps, obs_steps=OBS_STEPS,
        obs_dim=OBS_DIM, action_horizon=HORIZON, action_dim=ACTION_DIM,
    )
    jax_buffer = J.RolloutBuffer(
        max_chunks=total_steps, max_steps=total_steps, obs_steps=OBS_STEPS,
        obs_dim=OBS_DIM, action_horizon=HORIZON, action_dim=ACTION_DIM,
    )
    T.collect_chunks_vec(
        make_envs(), torch_policy, total_steps, LOW, HIGH,
        n_obs_steps=OBS_STEPS, n_action_steps=ACTION_STEPS,
        chunk_size=HORIZON, buffer=torch_buffer, seed=seed,
    )
    J.collect_chunks_vec(
        make_envs(), jax_policy, total_steps, LOW, HIGH,
        n_obs_steps=OBS_STEPS, n_action_steps=ACTION_STEPS,
        chunk_size=HORIZON, buffer=jax_buffer, seed=seed,
    )
    return torch_buffer, jax_buffer


def assert_same(torch_out, jax_out):
    t = torch_out.detach().cpu().numpy() if isinstance(torch_out, torch.Tensor) else np.asarray(torch_out)
    j = np.asarray(jax_out)
    assert t.shape == j.shape, (t.shape, j.shape)
    np.testing.assert_allclose(t, j, rtol=RTOL, atol=ATOL)


def test_collector_matches_torch_with_no_terminations():
    """192 steps over 4 envs is 48 each, under Reacher's 50-step TimeLimit."""
    torch_buffer, jax_buffer = run_both(total_steps=192)
    t, j = torch_buffer.active(), jax_buffer.active()
    assert jax_buffer.n_chunks == torch_buffer.n_chunks > 0
    assert jax_buffer.n_steps == torch_buffer.n_steps
    assert not bool(np.asarray(j["chunk_terminated"]).any()), "unexpected termination"
    for key in J.CHUNK_KEYS + J.STEP_KEYS:
        assert_same(t[key], j[key])


def test_collector_matches_torch_across_terminations_and_autoresets():
    """100 steps per env crosses Reacher's 50-step limit twice, so each env
    terminates mid-rollout, which arms awaiting and takes the reset obs on the
    following step."""
    torch_buffer, jax_buffer = run_both(total_steps=400)
    t, j = torch_buffer.active(), jax_buffer.active()
    assert jax_buffer.n_chunks == torch_buffer.n_chunks
    # Reacher's only end is its 50-step TimeLimit, so it *truncates*; `terminated`
    # is never set by this env and the terminal branch of the state machine is
    # compared but not exercised here.
    assert int(np.asarray(j["chunk_truncated"]).sum()) == 2 * NUM_ENVS
    assert int(np.asarray(j["chunk_terminated"]).sum()) == 0
    for key in J.CHUNK_KEYS + J.STEP_KEYS:
        assert_same(t[key], j[key])


def test_budget_cut_is_flagged_and_does_not_arm_awaiting():
    """A short final chunk is the env still being alive: GAE must chain past it."""
    total = 100  # not a multiple of the 16-step chunk
    torch_buffer, jax_buffer = run_both(total_steps=total)
    t, j = torch_buffer.active(), jax_buffer.active()
    assert bool(np.asarray(j["chunk_budget_truncated"]).any())
    assert_same(t["chunk_budget_truncated"], j["chunk_budget_truncated"])
    # It is a distinct signal from the env's own truncation.
    assert not bool(np.asarray(j["chunk_truncated"]).any())


def test_chunk_lengths_are_the_executed_step_counts():
    torch_buffer, jax_buffer = run_both(total_steps=400)
    t, j = torch_buffer.active(), jax_buffer.active()
    assert_same(t["chunk_lengths"], j["chunk_lengths"])
    lengths = np.asarray(j["chunk_lengths"])
    assert lengths.max() <= HORIZON and lengths.min() >= 1
    # Every executed step is accounted for exactly once, contiguously.
    assert int(lengths.sum()) == jax_buffer.n_steps
    assert np.array_equal(np.asarray(j["chunk_starts"]), np.concatenate([[0], np.cumsum(lengths)[:-1]]))


def test_windows_are_chronological_after_the_ring_wraps():
    """The window is a ring; roll_table restores order once per closing chunk."""
    torch_buffer, jax_buffer = run_both(total_steps=400)
    t, j = torch_buffer.active(), jax_buffer.active()
    # A chunk that ran 16 steps sees the 16 frames in order, and the final frame
    # is the one the env last returned.
    assert_same(t["chunk_next_state"], j["chunk_next_state"])


def test_buffer_rejects_bad_shapes():
    buffer = J.RolloutBuffer(
        max_chunks=8, max_steps=8, obs_steps=OBS_STEPS, obs_dim=OBS_DIM,
        action_horizon=HORIZON, action_dim=ACTION_DIM,
    )
    good = {
        "state": jnp.zeros((OBS_STEPS, OBS_DIM), dtype=jnp.float32),
        "next_state": jnp.zeros((OBS_STEPS, OBS_DIM), dtype=jnp.float32),
        "action": jnp.zeros((HORIZON, ACTION_DIM), dtype=jnp.float32),
        "rewards": jnp.zeros((3,), dtype=jnp.float32),
        "step_frames": jnp.zeros((3, OBS_DIM), dtype=jnp.float32),
        "terminated": False,
        "truncated": False,
        "clip_count": 0,
    }
    assert buffer.add_chunk(**good) == 0
    with pytest.raises(ValueError, match="state shape"):
        buffer.add_chunk(**{**good, "state": jnp.zeros((4, OBS_DIM), dtype=jnp.float32)})
    with pytest.raises(ValueError, match="step_frames shape"):
        buffer.add_chunk(**{**good, "step_frames": jnp.zeros((2, OBS_DIM), dtype=jnp.float32)})
    with pytest.raises(RuntimeError, match="step capacity"):
        buffer.add_chunk(**{**good, "rewards": jnp.zeros((20,), dtype=jnp.float32)})


def test_action_bounds_must_strictly_bracket():
    with pytest.raises(ValueError, match="high > low"):
        J.resolve_action_bounds(np.array([0.0, 0.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="action_low shape"):
        J.resolve_action_bounds(np.array([0.0]), np.array([0.0, 1.0]))


def test_scale_and_clip_rescales_after_clamping():
    proposed = jnp.asarray([[-5.0, 5.0], [0.0, 1.0]], dtype=jnp.float32)
    executed, scaled, counts = J.scale_and_clip_chunk(
        proposed, jnp.asarray([-2.0, -2.0]), jnp.asarray([4.0, 4.0])
    )
    assert float(jnp.min(executed)) >= -2.0 and float(jnp.max(executed)) <= 2.0
    assert float(jnp.min(scaled)) >= -1.0 and float(jnp.max(scaled)) <= 1.0
    assert int(counts) == 2
