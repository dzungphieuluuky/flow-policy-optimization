"""Batched obs-window writes must be bitwise identical to row-at-a-time.

The collector used to slide the obs window with a per-env, per-step
``windows[i, :-1].copy_(windows[i, 1:].clone())``. It now keeps a ring cursor
and writes the whole ``[n, obs]`` batch that ``envs.step`` already returns, so
one vec step costs a handful of dispatches instead of twelve per env.

These tests pin the properties the refactor could break. They assert
invariants rather than a recorded golden buffer, so they keep protecting the
collector after the shape of the implementation moves again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spectral_joint_policy.collector import (
    RolloutBuffer,
    collect_chunks_vec,
)

N_ENVS, OBS_DIM, ACT_DIM = 8, 11, 3
N_OBS, HORIZON = 16, 16
TOTAL_STEPS = 512


class FakeVecEnv:
    """Deterministic stand-in for a gym NEXT_STEP vec env.

    The counter resets on the step after a terminal, which is what makes
    ``done`` a pulse rather than a level -- otherwise the collector's
    ``awaiting`` handshake never clears.
    """

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

    def reset(self, seed=None):
        self.t[:] = 0
        return self._obs(), {}

    def step(self, actions):
        self.actions_seen.append(np.array(actions, copy=True))
        done = self.t >= self.episode
        self.t = np.where(done, 0, self.t + 1)
        rew = self.rng.standard_normal(self.num_envs).astype(np.float32)
        return self._obs(), rew, done, np.zeros_like(done), {}


def collect(episode: int, n_envs: int = N_ENVS, total_steps: int = TOTAL_STEPS):
    """Run the real collector on a deterministic env; return the buffer + env."""
    env = FakeVecEnv(n=n_envs, episode=episode)
    # Fixed projection so the proposal depends on every frame of the window in
    # order: a rotated or torn window changes the actions the env receives.
    w = torch.randn(N_OBS * OBS_DIM, ACT_DIM, generator=torch.Generator().manual_seed(3))
    ramp = torch.arange(1, HORIZON + 1, dtype=torch.float32).reshape(1, HORIZON, 1)

    def policy(states):
        n = states.shape[0]
        flat = states.reshape(n, -1)
        return torch.addmm(flat.new_zeros(n, ACT_DIM), flat, w).unsqueeze(1) * ramp

    buf = RolloutBuffer(
        max_chunks=total_steps, max_steps=total_steps, obs_steps=N_OBS,
        obs_dim=OBS_DIM, action_horizon=HORIZON, action_dim=ACT_DIM,
    )
    collect_chunks_vec(
        env, policy, total_steps,
        -torch.ones(ACT_DIM), torch.ones(ACT_DIM),
        n_obs_steps=N_OBS, n_action_steps=HORIZON, chunk_size=HORIZON,
        buffer=buf, seed=0,
    )
    return buf, env


def _per_env_chunks(buf):
    """chunk indices grouped by env, in time order within each env.

    Chunks are appended in wavefront order, so an env's chunks are num_envs
    apart in the buffer; the env_id sort is what puts them back in sequence.
    """
    a = buf.active()
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
    """chunk_next_state[i] is chunk_state[i] slid by n_exec frames.

    This is the property a ring cursor has to preserve, and unlike a slice of
    ``step_frames`` it does not depend on the buffer's layout: the step
    timeline is wavefront-major, so env 0's second chunk starts at row 128,
    not row 16, and any 16-row window of it spans eight different envs.
    """
    checked = 0
    for episode in (1000, 50, 16, 1):
        buf, _ = collect(episode)
        a = buf.active()
        for i in range(buf.n_chunks):
            n = int(a["chunk_lengths"][i])
            keep = N_OBS - n
            if keep <= 0:
                continue  # the chunk replaced the whole window
            assert torch.equal(a["chunk_next_state"][i, :keep],
                               a["chunk_state"][i, n:]), (episode, i)
            # and the frames that entered are the ones the chunk recorded
            s = int(a["chunk_starts"][i])
            assert torch.equal(a["chunk_next_state"][i, -1],
                               a["step_frames"][s + n - 1]), (episode, i)
            checked += 1
    assert checked > 40, f"only {checked} windows checked"


def test_ring_wraps_and_still_yields_the_right_window():
    """episode=1 forces a terminal on (almost) every step.

    Each chunk is then 1-2 steps, so the window wraps many times per chunk.
    A shift-vs-ring mistake still returns 16 valid frames, just rotated, so
    the assertion is against the step timeline rather than a shape check.
    """
    buf, _ = collect(1)
    a, groups = _per_env_chunks(buf)
    for e, idxs in groups.items():
        for k, i in enumerate(idxs):
            w = a["chunk_state"][i]
            assert w.shape == (N_OBS, OBS_DIM) and torch.isfinite(w).all()
            n_exec = int(a["chunk_lengths"][i])
            assert 1 <= n_exec <= 2, (e, i, n_exec)
            s = int(a["chunk_starts"][i])
            assert torch.equal(a["chunk_next_state"][i][-1],
                               a["step_frames"][s + n_exec - 1]), (e, i)


def test_next_state_chains_only_across_continue_chunks():
    """chunk_next_state[k] equals chunk_state[k+1] only if no terminal intervened.

    After a terminal the collector refills the window from the reset
    observation, so the two differ. This is the condition item C depends on,
    and it is why C's saving is bounded by the terminal rate rather than
    being the full 75%.
    """
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
    buf, _ = collect(episode=4)  # falls every 4 steps, horizon 16
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

    Per env, the non-zero actions handed to the env must be exactly its stored
    chunk actions in order. The zero rows are the ``awaiting`` handshake steps,
    one per terminal, which the collector spends with a zero action.
    """
    buf, env = collect(50)
    a, groups = _per_env_chunks(buf)
    sent = torch.from_numpy(np.stack(env.actions_seen))  # [vec_steps, n_envs, A]
    for e, idxs in groups.items():
        # only the executed prefix was sent; the tail of the chunk never ran
        expected = torch.cat(
            [a["chunk_actions"][i, : int(a["chunk_lengths"][i])] for i in idxs],
            dim=0,
        )
        got = sent[:, e, :]
        nonzero = got.abs().sum(dim=1) > 0
        assert torch.allclose(got[nonzero], expected, atol=1e-6), e
        n_terminals = sum(bool(a["chunk_terminated"][i]) for i in idxs)
        assert int((~nonzero).sum()) == n_terminals, (e, int((~nonzero).sum()))
