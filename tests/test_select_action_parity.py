"""torch vs jax parity for ``select_action``: the receding eval chunk.

The evaluator drives the policy one frame at a time and calls ``reset()`` when an
episode ends, so the gate does the same: a fixed obs sequence fed one frame per
call, with a terminal part-way through and both buffers cleared at that boundary.
Comparing only a whole episode would pass even if the refill-on-terminate were
wrong, because the two would only disagree on the first action of the next
episode.

``predict_action_chunk`` is gated separately, so this file is really about the two
buffers: the obs ring (chronological order, and the *first frame repeated* to
fill a short window rather than zero padding) and the action queue (torch fills
its deque back-to-front and pops from the front, so the first action out is
``chunk[0, 0]`` -- the chunk plays in **reverse**.
"""
import sys
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_policy_parity import CFG, HIGH, LOW
from torch_jax_weights import to_jax_tree

from sjfpo_jax import policy as J
from sjfpo_jax.state import build_state, load_pair
from spectral_joint_policy.train_sjfpo import build_policy

# Reacher-v5: 4 cos/sin + 6 joint offsets, 2 joint torques.
ENV = "Reacher-v5"
OBS_DIM = 10
ACTION_DIM = 2
EPISODE_STEPS = 12
TERMINAL_AT = 7
RTOL, ATOL = 1e-4, 1e-5


def pair_at(cfg, seed=0, obs_dim=OBS_DIM, action_dim=ACTION_DIM):
    """``test_policy_parity.pair`` with the dims as parameters.

    That helper hardcodes OBS=11/ACT=2, and Reacher-v5 is 10/2, so the
    Evaluator end-to-end gate needs a policy built at the env's real widths.
    Otherwise this is the same thing: same seed, same warmed normaliser, weights
    copied across.
    """
    torch.manual_seed(seed)
    t_policy = build_policy(
        obs_dim, action_dim, cfg, device="cpu",
        action_low=LOW, action_high=HIGH,
    )
    state = build_state(cfg, obs_dim, action_dim, jax.random.key(seed))
    state = state.replace(
        actor=load_pair(state.actor, to_jax_tree(t_policy.actor.state_dict())),
        conditioner=load_pair(state.conditioner, to_jax_tree(t_policy.conditioner.state_dict())),
        critic=load_pair(
            state.critic,
            to_jax_tree(t_policy.critic.state_dict(), stack=["conditioners", "heads"]),
        ),
        old_actor=load_pair(state.old_actor, to_jax_tree(t_policy.old_actor.state_dict())),
        old_conditioner=load_pair(
            state.old_conditioner, to_jax_tree(t_policy.old_conditioner.state_dict())
        ),
        action_low=jnp.asarray(LOW, dtype=jnp.float32),
        action_high=jnp.asarray(HIGH, dtype=jnp.float32),
    )
    frames = np.random.default_rng(99).standard_normal((64, obs_dim), dtype=np.float32)
    t_policy.update_observation_stats(torch.from_numpy(frames.copy()))
    state.obs_normalizer.update(jnp.asarray(frames))
    return t_policy, state


def gentle_wake(state, t_policy, seed=5, std=0.01):
    """A *gentle* wake: enough to move the actor, not enough to saturate it.

    ``wake_actor_and_adaln`` uses std=0.1, which drives the normalised chunk past
    the ``[-1, 1]`` clip and the unscale onto the +/-2 rail. Every action then
    comes back exactly [-2, -2], so a buffer bug cannot change a single number and
    the gate compares two constants. std=0.01 keeps the chunk interior, which is
    what makes "step 3 differs from step 0" a real assertion.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        for module in [t_policy.actor.out] + [b.adaLN.fc2 for b in t_policy.actor.blocks]:
            module.weight.normal_(0.0, std)
            module.bias.normal_(0.0, std)
    return state.replace(actor=load_pair(state.actor, to_jax_tree(t_policy.actor.state_dict())))


def fixture(seed=0, n_action_steps=16):
    """Both policies awake and in eval mode, at Reacher-v5's dims."""
    import dataclasses

    cfg = dataclasses.replace(CFG, n_action_steps=n_action_steps)
    t_policy, state = pair_at(cfg, seed=seed)
    state = gentle_wake(state, t_policy)
    t_policy.eval()
    return cfg, t_policy, state


def obs_sequence(n=EPISODE_STEPS):
    """A fixed, reproducible obs stream. Not a real env: the gate is the buffer,
    and the same numbers on both sides is what makes the two comparable."""
    rng = np.random.default_rng(11)
    return rng.standard_normal((n, OBS_DIM)).astype(np.float32) * 0.3


def torch_actions(t_policy, obs_seq, terminal_at=None):
    """``[obs]`` per call, reset at ``terminal_at`` -- what Evaluator does."""
    t_policy.reset()
    out = []
    for i, obs in enumerate(obs_seq):
        if terminal_at is not None and i == terminal_at:
            t_policy.reset()
        action = t_policy.select_action(
            {"observation.state": torch.from_numpy(obs)}, zero_sampling=True
        )
        out.append(np.asarray(action.detach().cpu().numpy(), dtype=np.float32))
    return out


def jax_actions(state, cfg, obs_seq, terminal_at=None):
    buffer = J.make_action_buffer(OBS_DIM, cfg.n_obs_steps, ACTION_DIM)
    out = []
    for i, obs in enumerate(obs_seq):
        if terminal_at is not None and i == terminal_at:
            buffer = J.make_action_buffer(OBS_DIM, cfg.n_obs_steps, ACTION_DIM)
        action, buffer = J.select_action(
            state, buffer, jnp.asarray(obs), cfg, zero_sampling=True
        )
        out.append(np.asarray(action, dtype=np.float32))
    return out


def assert_actions_match(got, want, label=""):
    assert len(got) == len(want), (len(got), len(want))
    for step, (a, b) in enumerate(zip(got, want)):
        np.testing.assert_allclose(
            a, b, rtol=RTOL, atol=ATOL, err_msg=f"{label} step {step}"
        )


def test_the_policy_does_not_saturate_the_chunk():
    """A guard on every other test in this file.

    A chunk that clips to +/-1 in normalised space unscales onto the +/-2 rail,
    so every action becomes exactly [-2, -2] and no buffer bug can move a number.
    """
    cfg, _, state = fixture()
    stacked = np.stack(jax_actions(state, cfg, obs_sequence(n=8)))
    rail = np.abs(stacked).max()
    assert rail < 1.9, f"the chunk is saturated at {rail}, so the gate is vacuous"
    assert rail > 0.01, f"every action is ~0 ({rail}), so nothing is exercised"
    # Successive steps must differ, or the obs window is not reaching the actor.
    assert not np.allclose(stacked[0], stacked[4]), "the action never changes"


def test_one_frame_at_a_time_matches_torch():
    cfg, t_policy, state = fixture()
    obs_seq = obs_sequence()
    assert_actions_match(
        jax_actions(state, cfg, obs_seq), torch_actions(t_policy, obs_seq)
    )


def test_a_terminal_mid_episode_matches_torch():
    """The refill-on-terminate: both buffers cleared, then the same stream."""
    cfg, t_policy, state = fixture()
    obs_seq = obs_sequence()
    got = jax_actions(state, cfg, obs_seq, terminal_at=TERMINAL_AT)
    want = torch_actions(t_policy, obs_seq, terminal_at=TERMINAL_AT)
    assert_actions_match(got, want, "after reset")
    # And the reset changed something, or the test cannot see a wrong refill: the
    # frame ring restarts, so step TERMINAL_AT's window is built from one frame
    # where the previous step's was built from sixteen.
    assert not np.allclose(got[TERMINAL_AT], got[TERMINAL_AT - 1])


def test_a_full_window_argument_replaces_the_ring():
    """[T, obs] is used as-is and leaves the ring untouched, as torch's 2-D branch."""
    cfg, t_policy, state = fixture()
    window_np = obs_sequence(cfg.n_obs_steps)
    window = torch.from_numpy(window_np)

    t_policy.reset()
    first = t_policy.select_action({"observation.state": window}, zero_sampling=True)
    second = t_policy.select_action({"observation.state": window}, zero_sampling=True)

    buffer = J.make_action_buffer(OBS_DIM, cfg.n_obs_steps, ACTION_DIM)
    j_first, buffer = J.select_action(
        state, buffer, jnp.asarray(window_np), cfg, zero_sampling=True
    )
    j_second, buffer = J.select_action(
        state, buffer, jnp.asarray(window_np), cfg, zero_sampling=True
    )
    np.testing.assert_allclose(
        np.asarray(j_first), first.detach().numpy(), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        np.asarray(j_second), second.detach().numpy(), rtol=RTOL, atol=ATOL
    )
    # The second call served the cached chunk, so it is a different row of it.
    assert not np.allclose(np.asarray(j_second), np.asarray(j_first))
    # Neither path wrote a frame.
    assert int(buffer.frame_len) == 0, int(buffer.frame_len)


def test_the_chunk_is_cached_for_n_action_steps_before_refilling():
    """The queue must hold exactly ``n_action_steps`` actions from one chunk.

    Asserted by counting the proposals rather than by comparing actions: inside
    one chunk consecutive actions are *different rows* of it, so ``got[0]`` and
    ``got[1]`` differ whether or not the chunk was cached. The number of proposals
    is the thing that distinguishes ``n_action_steps=1`` (one per step) from
    ``n_action_steps=16`` (one for the whole sequence).
    """
    for n_steps in (1, 4, 16):
        cfg, t_policy, state = fixture(n_action_steps=n_steps)
        obs_seq = obs_sequence(n=4)
        got = jax_actions(state, cfg, obs_seq)
        want = torch_actions(t_policy, obs_seq)
        assert_actions_match(got, want, f"n_action_steps={n_steps}")

        calls = {"n": 0}
        real = J.predict_action_chunk

        # Bound as defaults: `calls` and `real` are rebound on the next
        # iteration of this loop, and a closure would follow them.
        def counting(*args, _calls=calls, _real=real, **kwargs):
            _calls["n"] += 1
            return _real(*args, **kwargs)

        J.predict_action_chunk = counting
        try:
            jax_actions(state, cfg, obs_seq)
        finally:
            J.predict_action_chunk = real
        expected = -(-len(obs_seq) // n_steps)
        assert calls["n"] == expected, (
            f"n_action_steps={n_steps}: {calls['n']} proposals for {len(obs_seq)} "
            f"steps, expected {expected}"
        )


def test_the_actions_within_one_chunk_are_distinct_rows():
    """The queue plays the chunk in reverse, so step 0 is the chunk's *last* row.

    Pins the order itself rather than inferring it: the first action out is
    ``chunk[0, n_action_steps - 1]`` and the last is ``chunk[0, 0]``.
    """
    cfg, _, state = fixture(n_action_steps=16)
    obs_seq = obs_sequence(n=cfg.n_obs_steps)
    buffer = J.make_action_buffer(OBS_DIM, cfg.n_obs_steps, ACTION_DIM)
    window = np.broadcast_to(obs_seq[0], (cfg.n_obs_steps, OBS_DIM)).astype(np.float32)
    # One call to fill the ring, and it also consumes chunk[0, 15] -- so it is
    # out[0], not a throwaway.
    first, buffer = J.select_action(
        state, buffer, jnp.asarray(window[0]), cfg, zero_sampling=True
    )
    chunk = np.asarray(
        J.predict_action_chunk(
            state, jnp.asarray(window)[None], cfg, zero_sampling=True
        )
    )[0]

    out = [np.asarray(first)]
    for i in range(1, 3):
        action, buffer = J.select_action(
            state, buffer, jnp.asarray(obs_seq[i]), cfg, zero_sampling=True
        )
        out.append(np.asarray(action))
    for step, row in enumerate(out):
        np.testing.assert_allclose(
            row, chunk[15 - step], rtol=RTOL, atol=ATOL, err_msg=f"step {step}"
        )


def test_the_obs_window_repeats_its_first_frame_until_it_is_full():
    """torch pads with the oldest frame, not with zeros.

    Zero padding feeds the conditioner a window that does not exist yet, so the
    first ``n_obs_steps - 1`` actions of every episode come from a different
    observation history than they should.
    """
    cfg, _, state = fixture()
    obs_seq = obs_sequence(cfg.n_obs_steps)
    buffer = J.make_action_buffer(OBS_DIM, cfg.n_obs_steps, ACTION_DIM)

    for seen in range(1, cfg.n_obs_steps + 1):
        _, buffer = J.select_action(
            state, buffer, jnp.asarray(obs_seq[seen - 1]), cfg, zero_sampling=True
        )
        window = np.asarray(J._chronological_frames(buffer, cfg.n_obs_steps))
        assert window.shape == (cfg.n_obs_steps, OBS_DIM), window.shape
        pad = cfg.n_obs_steps - seen
        # The tail is the frames seen so far; the head is a repeat of the first.
        np.testing.assert_allclose(window[pad:], obs_seq[:seen], rtol=1e-6, atol=1e-6)
        if pad:
            np.testing.assert_allclose(
                window[:pad], np.broadcast_to(obs_seq[0], (pad, OBS_DIM))
            )


def test_the_window_keeps_sliding_after_the_ring_wraps():
    """The wrap case, which the fill test above stops just short of.

    Once the ring is full the write cursor is mid-array, so the oldest frame is
    no longer at index 0 and the window has to be **rotated**. Reading
    ``frames[:n]`` unconditionally would silently replay the oldest n_obs_steps
    observations for the rest of the episode.
    """
    cfg, _, state = fixture()
    n = cfg.n_obs_steps
    obs_seq = obs_sequence(n + 4)
    buffer = J.make_action_buffer(OBS_DIM, n, ACTION_DIM)

    for seen in range(1, n + 5):
        _, buffer = J.select_action(
            state, buffer, jnp.asarray(obs_seq[seen - 1]), cfg, zero_sampling=True
        )
        window = np.asarray(J._chronological_frames(buffer, n))
        expected = obs_seq[max(0, seen - n):seen]
        pad = n - len(expected)
        np.testing.assert_allclose(
            window[n - len(expected):], expected, rtol=1e-6, atol=1e-6,
            err_msg=f"seen={seen}",
        )
        if pad:
            np.testing.assert_allclose(
                window[:pad], np.broadcast_to(expected[0], (pad, OBS_DIM))
            )
    # The oldest frame is genuinely gone from the front, so the test above is
    # not passing because the rotation is a no-op.
    assert not np.allclose(window[0], obs_seq[0])


def test_the_traced_window_matches_the_eager_one():
    """Two branches, one answer.

    ``_chronological_frames`` has a Python branch on ``frame_len`` when it is a
    concrete value and a ``jnp.where`` when it is a tracer. Only the eager one is
    reached by the other tests here, so without this the traced branch could
    disagree and nothing would notice.
    """
    cfg, _, state = fixture()
    n = cfg.n_obs_steps
    obs_seq = obs_sequence(n + 3)
    buffer = J.make_action_buffer(OBS_DIM, n, ACTION_DIM)

    def traced(buffer):
        # Force the tracer branch by returning frame_len through a traced value.
        return J._chronological_frames(
            buffer.replace(frame_len=buffer.frame_len * 1), n
        )

    for seen in range(1, n + 4):
        buffer = J.select_action(
            state, buffer, jnp.asarray(obs_seq[seen - 1]), cfg, zero_sampling=True
        )[1]
        want = np.asarray(J._chronological_frames(buffer, n))
        # A jitted call traces with a tracer, so this is the other branch.
        got = np.asarray(jax.jit(traced)(buffer))
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6, err_msg=f"seen={seen}")


def test_a_window_of_the_wrong_length_is_rejected():
    cfg, _, state = fixture()
    buffer = J.make_action_buffer(OBS_DIM, cfg.n_obs_steps, ACTION_DIM)
    with pytest.raises(ValueError, match="n_obs_steps"):
        J.select_action(
            state, buffer, jnp.zeros((cfg.n_obs_steps + 1, OBS_DIM), jnp.float32), cfg
        )
    with pytest.raises(ValueError, match=r"\[obs\] or \[T, obs\]"):
        J.select_action(state, buffer, jnp.zeros((2, 3, OBS_DIM), jnp.float32), cfg)


def test_the_buffer_is_jittable():
    """The reason it is two arrays and not two deques.

    torch's version cannot be jitted at all. This does not prove a vmapped eval
    is correct -- it proves the step is one traced program, which is what makes
    that possible later.
    """
    cfg, _, state = fixture()
    obs = jnp.asarray(obs_sequence(1)[0])

    def run(state, buffer, obs):
        return J.select_action(state, buffer, obs, cfg, zero_sampling=True)

    buffer = J.make_action_buffer(OBS_DIM, cfg.n_obs_steps, ACTION_DIM)
    want_action, want_buffer = run(state, buffer, obs)
    got_action, got_buffer = jax.jit(run)(state, buffer, obs)
    np.testing.assert_allclose(
        np.asarray(got_action), np.asarray(want_action), rtol=RTOL, atol=ATOL
    )
    for a, b in zip(
        jax.tree_util.tree_leaves(want_buffer), jax.tree_util.tree_leaves(got_buffer)
    ):
        # allclose, not array_equal: jit reassociates, so it differs in the last
        # fp32 bits. Same reasoning as update_step's jit-vs-eager gate.
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=RTOL, atol=ATOL)


def test_the_actions_land_inside_the_env_bounds():
    """select_action unscales into the env range, and the trainer's
    ``action_abs_max > 1.05`` guard reads the number this produces."""
    cfg, _, state = fixture()
    state = state.replace(
        action_low=jnp.asarray(LOW, dtype=jnp.float32),
        action_high=jnp.asarray(HIGH, dtype=jnp.float32),
    )
    stacked = np.stack(jax_actions(state, cfg, obs_sequence(n=8)))
    assert np.abs(stacked).max() > 0.0, "every action was zero, so nothing was exercised"
    # LOW/HIGH are the test suite's +/-2 bounds, and the chunk is clipped to
    # [-1, 1] before the unscale, so nothing can leave that range.
    assert stacked.min() >= LOW.min() - 1e-5, stacked.min()
    assert stacked.max() <= HIGH.max() + 1e-5, stacked.max()


def test_it_runs_on_a_real_env_the_way_the_evaluator_drives_it():
    """End to end through ``Evaluator`` itself, which is the real caller."""
    cfg, t_policy, state = fixture()
    from spectral_joint_policy.evaluator import Evaluator

    evaluator = Evaluator(gym.make(ENV), eval_seed=1234, max_episodes=2, max_steps=60)

    def torch_policy(obs):
        return t_policy.select_action(
            {"observation.state": torch.from_numpy(np.asarray(obs, dtype=np.float32))},
            zero_sampling=True,
        ).detach().cpu().numpy()

    # Evaluator takes a plain callable, so the buffer rides in a closure.
    box: dict = {}

    def jax_policy(obs):
        if "buffer" not in box:
            box["buffer"] = J.make_action_buffer(OBS_DIM, cfg.n_obs_steps, ACTION_DIM)
        action, box["buffer"] = J.select_action(
            state, box["buffer"], jnp.asarray(obs, dtype=jnp.float32), cfg,
            zero_sampling=True,
        )
        return np.asarray(action, dtype=np.float32)

    torch_results = evaluator.evaluate(torch_policy, out_dir=None)
    jax_results = evaluator.evaluate(jax_policy, out_dir=None)

    assert len(torch_results) == len(jax_results) == 2
    for t, j in zip(torch_results, jax_results):
        assert t["length"] == j["length"], (t["length"], j["length"])
        np.testing.assert_allclose(t["return"], j["return"], rtol=1e-3, atol=1e-4)
        assert np.isfinite(j["return"]), j["return"]