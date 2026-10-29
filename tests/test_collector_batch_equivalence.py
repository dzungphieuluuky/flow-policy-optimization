"""The batched collector must fill, window, chain, and store every chunk."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from myfpo.rollout import (
    RolloutBuffer,
    collect_chunks_vec,
)

N_ENVS, OBS_DIM, ACT_DIM = 8, 11, 3
N_OBS, HORIZON = 16, 16
TOTAL_STEPS = 512


class FakeVecEnv:
    """Deterministic stand-in for a gym NEXT_STEP vec env."""

    def __init__(self, n=N_ENVS, obs_dim=OBS_DIM, act_dim=ACT_DIM, episode=1000):
        self.num_envs = n
        self.observation_space = type("S", (), {"shape": (obs_dim,)})()
        self.single_observation_space = self.observation_space
        self.single_action_space = type("S", (), {"shape": (act_dim,)})()
        self.autoreset_mode = type("M", (), {"name": "NEXT_STEP"})()
        self.episode = episode
        self.t = np.zeros(n, dtype=np.int64)
        self.rng = np.random.default_rng(1234)
        self.actions_seen: list[np.ndarray] = []

    def _obs(self):
        return self.rng.standard_normal(
            (self.num_envs, self.observation_space.shape[0])
        ).astype(np.float32)

    def reset(self, seed=None):  # noqa: ARG002 - gym calls reset(seed=...)
        self.t[:] = 0
        return self._obs(), {}

    def step(self, actions):
        self.actions_seen.append(np.array(actions, copy=True))
        done = self.t >= self.episode
        self.t = np.where(done, 0, self.t + 1)
        rew = self.rng.standard_normal(self.num_envs).astype(np.float32)
        return self._obs(), rew, done, np.zeros_like(done), {}


def collect(
    episode: int,
    n_envs: int = N_ENVS,
    total_steps: int = TOTAL_STEPS,
    n_obs_steps: int = N_OBS,
):
    """Run the real collector on a deterministic env. Return the buffer + env."""
    env = FakeVecEnv(n=n_envs, episode=episode)
    # Fixed projection so the proposal depends on every frame of the window.
    w = torch.randn(N_OBS * OBS_DIM, ACT_DIM, generator=torch.Generator().manual_seed(3))
    ramp = torch.arange(1, HORIZON + 1, dtype=torch.float32).reshape(1, HORIZON, 1)

    def policy(states):
        n = states.shape[0]
        flat = states.reshape(n, -1)
        return torch.addmm(flat.new_zeros(n, ACT_DIM), flat, w).unsqueeze(1) * ramp

    buf = RolloutBuffer(
        max_chunks=total_steps, max_steps=total_steps, obs_steps=n_obs_steps,
        obs_dim=OBS_DIM, action_horizon=HORIZON, action_dim=ACT_DIM,
    )
    collect_chunks_vec(
        env, policy, total_steps,
        -torch.ones(ACT_DIM), torch.ones(ACT_DIM),
        n_obs_steps=n_obs_steps, n_action_steps=HORIZON,
        buffer=buf, seed=0,
    )
    return buf, env


def _per_env_chunks(buf):
    """chunk indices grouped by env, in time order within each env."""
    a = buf.active_prefix()
    env_id = a["chunk_env_id"].to(torch.long)
    order = torch.argsort(env_id, stable=True)
    groups: dict[int, list[int]] = {}
    for idx in order.tolist():
        groups.setdefault(int(env_id[idx]), []).append(idx)
    return a, groups


def test_collector_fills_the_budget_exactly():
    """Every env must run its whole share of the step budget."""
    for episode in (1000, 50, 16, 1):
        buf, _ = collect(episode)
        assert buf.n_steps == TOTAL_STEPS, (episode, buf.n_steps)
        assert buf.n_chunks >= 1


def test_obs_windows_slide_by_exactly_one_frame_per_step():
    """chunk_next_state[i] is chunk_state[i] slid by n_exec frames."""
    checked = 0
    for episode in (1000, 50, 16, 1):
        buf, _ = collect(episode)
        a = buf.active_prefix()
        for i in range(buf.n_chunks):
            n = int(a["chunk_lengths"][i])
            keep = N_OBS - n
            if keep <= 0:
                continue  # the chunk replaced the whole window.
            assert torch.equal(a["chunk_next_state"][i, :keep],
                               a["chunk_state"][i, n:]), (episode, i)
            # and the frames that entered are the ones the chunk recorded.
            s = int(a["chunk_starts"][i])
            assert torch.equal(a["chunk_next_state"][i, -1],
                               a["step_frames"][s + n - 1]), (episode, i)
            checked += 1
    assert checked > 40, f"only {checked} windows checked"


def test_ring_wraps_and_still_yields_the_right_window():
    """episode=1 forces a terminal on (almost) every step."""
    buf, _ = collect(1)
    a, groups = _per_env_chunks(buf)
    for e, idxs in groups.items():
        for _k, i in enumerate(idxs):
            w = a["chunk_state"][i]
            assert w.shape == (N_OBS, OBS_DIM) and torch.isfinite(w).all()
            n_exec = int(a["chunk_lengths"][i])
            assert 1 <= n_exec <= 2, (e, i, n_exec)
            s = int(a["chunk_starts"][i])
            assert torch.equal(a["chunk_next_state"][i][-1],
                               a["step_frames"][s + n_exec - 1]), (e, i)


def test_next_state_chains_only_across_continue_chunks():
    """chunk_next_state[k] equals chunk_state[k+1] only if no terminal intervened."""
    buf, _ = collect(50)
    a, groups = _per_env_chunks(buf)
    chained = severed = 0
    for e, idxs in groups.items():
        for k in range(len(idxs) - 1):
            i, j = idxs[k], idxs[k + 1]
            ended = bool(a["chunk_terminated"][i]) or bool(a["chunk_truncated"][i])
            if ended:
                severed += 1
                assert not torch.equal(a["chunk_next_state"][i],
                                       a["chunk_state"][j]), (e, i)
            else:
                chained += 1
                assert torch.equal(a["chunk_next_state"][i],
                                   a["chunk_state"][j]), (e, i)
    assert chained > 0 and severed > 0, (chained, severed)


def test_autoreset_refills_the_window_with_the_reset_observation():
    """After a terminal the window is the reset obs repeated, not stale frames."""
    buf, _ = collect(episode=4)  # falls every 4 steps, horizon 16.
    a, groups = _per_env_chunks(buf)
    checked = 0
    for e, idxs in groups.items():
        for k in range(len(idxs) - 1):
            i, j = idxs[k], idxs[k + 1]
            if not bool(a["chunk_terminated"][i]):
                continue
            w = a["chunk_state"][j]
            assert torch.equal(w, w[:1].expand_as(w)), (e, j)
            checked += 1
    assert checked > 0, "expected terminal-cut chunks at episode=4"


def test_actions_sent_to_the_env_match_the_stored_chunks():
    """Guards the batched actions_t gather.

    The env receives the squash of the stored chunk: the stored chunk is the
    flow's raw output, and the squash is applied on the way to the env.
    """
    buf, env = collect(50)
    a, groups = _per_env_chunks(buf)
    sent = torch.from_numpy(np.stack(env.actions_seen))  # [vec_steps, n_envs, A].
    for e, idxs in groups.items():
        # only the executed prefix was sent. The tail of the chunk never ran.
        expected = torch.cat(
            [a["chunk_actions"][i, : int(a["chunk_lengths"][i])] for i in idxs],
            dim=0,
        )
        # The bounds are [-1, 1], so the affine map is the identity and the
        # squash is the tanh.
        expected = torch.tanh(expected)
        got = sent[:, e, :]
        nonzero = got.abs().sum(dim=1) > 0
        assert torch.allclose(got[nonzero], expected, atol=1e-6), e
        n_terminals = sum(bool(a["chunk_terminated"][i]) for i in idxs)
        assert int((~nonzero).sum()) == n_terminals, (e, int((~nonzero).sum()))


def test_a_buffer_is_required_and_returns_itself():
    """There is one write path, so ``buffer`` is not optional."""
    env = FakeVecEnv(n=N_ENVS, episode=50)

    def policy(states):
        return torch.zeros(states.shape[0], HORIZON, ACT_DIM)

    kwargs = {
        "n_obs_steps": N_OBS, "n_action_steps": HORIZON,
        "seed": 0,
    }
    with pytest.raises(TypeError):
        collect_chunks_vec(
            env, policy, 32,
            -torch.ones(ACT_DIM), torch.ones(ACT_DIM), **kwargs
        )
    buf = RolloutBuffer(
        max_chunks=32, max_steps=32, obs_steps=N_OBS, obs_dim=OBS_DIM,
        action_horizon=HORIZON, action_dim=ACT_DIM,
    )
    got = collect_chunks_vec(
        env, policy, 32,
        -torch.ones(ACT_DIM), torch.ones(ACT_DIM), buffer=buf, **kwargs
    )
    assert got is buf and buf.n_steps == 32


def test_a_chunk_as_deep_as_the_window_still_lands_chronologically():
    """n_action_steps == n_obs_steps: the cursor reaches n_obs_steps."""
    assert HORIZON == N_OBS, "this test is the boundary case"
    buf, _ = collect(1000)  # no terminal: every chunk runs the full horizon.
    a = buf.active_prefix()
    full = [i for i in range(buf.n_chunks) if int(a["chunk_lengths"][i]) == N_OBS]
    assert len(full) >= N_ENVS, f"no full-width chunks: {buf.n_chunks}"
    for i in full:
        n = int(a["chunk_lengths"][i])
        s = int(a["chunk_starts"][i])
        assert torch.equal(
            a["chunk_next_state"][i], a["step_frames"][s : s + n]
        ), i


def test_a_chunk_longer_than_the_obs_window_is_rejected():
    """The ring buffer cannot hold a chunk deeper than itself."""
    with pytest.raises(ValueError, match="exceeds n_obs_steps"):
        collect(50, n_obs_steps=N_OBS // 4)
