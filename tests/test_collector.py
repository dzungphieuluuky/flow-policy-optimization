"""Unit tests for src/sjfpo/collector.py.

Boundary semantics under test (plan Scope §Chunk-level semi-MDP GAE):
    terminated (anywhere)  -> stop executing, discard remaining actions,
                              next.done=True, bootstrap FORBIDDEN
    truncated (TimeLimit)  -> stop executing, next.done=False,
                              next.truncated=True (bootstrap-eligible)
    neither                -> chunk continues to completion, both flags False

Transitions are plain dicts with LeRobot keys (observation.state / action /
next.*). The policy proposes ACTION_HORIZON actions but only N_STEPS are executed
(receding horizon); the stored action anchor is always the real-space SCALED
FULL proposed chunk, shape [ACTION_HORIZON, ACT_DIM] (one row per predicted step;
DCT happens downstream in forward).

The fake env is scripted gymnasium-v5 style: step() returns the 5-tuple
(obs, reward, terminated, truncated, info) and termination/truncation can
be planted at exact global step offsets. STATE_T / ACTION_HORIZON / N_STEPS are
imported from sjfpo.config — never hardcoded here.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.collector import Transition, collect_chunks, scale_and_clip_chunk  # noqa: E402, F401
from sjfpo.configuration_sjfpo import SJFPOConfig  # noqa: E402

_DEFAULTS = SJFPOConfig()

STATE_T = _DEFAULTS.n_obs_steps  # imported constant — NEVER hardcode 16
ACTION_HORIZON = _DEFAULTS.action_horizon  # imported constant — NEVER hardcode 16
N_STEPS = 8  # local: executed steps are caller-passed, no config default

OBS_DIM = 4
ACT_DIM = 3


class FakeEnv:
    """Scripted gymnasium-v5-style env with controllable done flags.

    Observations are constant-filled vectors whose scalar marker uniquely
    identifies (episode, position-within-episode): marker = 1000*resets + t.
    Rewards equal the 1-based global step count. terminate_at/truncate_at
    fire when the 1-based global step counter equals that value; they are
    mutually exclusive by construction in every fixture below.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        terminate_at: int | None = None,
        truncate_at: int | None = None,
        raise_at_step: int | None = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.terminate_at = terminate_at
        self.truncate_at = truncate_at
        self.raise_at_step = raise_at_step
        self.applied_actions: list[np.ndarray] = []
        self.obs_markers: list[int] = []  # marker after reset and after each step
        self.resets = 0
        self.global_step = 0
        self.episode_step = 0

    def marker(self) -> int:
        return 1000 * self.resets + self.episode_step

    def reset(self, *, seed=None, options=None):
        self.resets += 1
        self.episode_step = 0
        obs = np.full(self.obs_dim, float(self.marker()), dtype=np.float32)
        self.obs_markers.append(int(obs[0]))
        return obs, {}

    def step(self, action):
        if (
            self.raise_at_step is not None
            and self.global_step + 1 == self.raise_at_step
        ):
            raise RuntimeError(f"fake env boom at step {self.global_step + 1}")
        self.applied_actions.append(np.array(action, dtype=np.float32, copy=True))
        self.global_step += 1
        self.episode_step += 1
        terminated = (
            self.terminate_at is not None and self.global_step == self.terminate_at
        )
        truncated = (
            self.truncate_at is not None and self.global_step == self.truncate_at
        )
        assert not (terminated and truncated), "fixture bug: flags must be exclusive"
        obs = np.full(self.obs_dim, float(self.marker()), dtype=np.float32)
        self.obs_markers.append(int(obs[0]))
        return obs, float(self.global_step), terminated, truncated, {}


class FakePolicy:
    """Records conditioning states; returns a fixed full-horizon proposal."""

    def __init__(self, chunk: torch.Tensor) -> None:
        assert chunk.shape[0] == ACTION_HORIZON, "policy must propose exactly H actions"
        self.chunk = chunk
        self.states: list[torch.Tensor] = []

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        assert isinstance(state, torch.Tensor)
        self.states.append(state.detach().clone())
        return self.chunk


def expected_window(markers: list[int], upto: int) -> torch.Tensor:
    """Stacked sliding window of the last STATE_T markers before `upto`."""
    recent = markers[max(0, upto - STATE_T) : upto]
    while len(recent) < STATE_T:
        recent.insert(0, markers[0])  # warmup replication of the first obs
    rows = [np.full(OBS_DIM, float(m), dtype=np.float32) for m in recent]
    return torch.from_numpy(np.stack(rows))


def is_continue(t: Transition) -> bool:
    return not bool(t["next.done"]) and not bool(t["next.truncated"])

    # (a) chunk counts + open-loop replan cadence (every N_STEPS)


def test_a_exact_chunk_count_and_replan_every_n_steps():
    env = FakeEnv()
    policy = FakePolicy(torch.zeros(ACTION_HORIZON, ACT_DIM))

    records = collect_chunks(
        env,
        policy,
        total_steps=20,
        action_low=-1.0,
        action_high=1.0,
        n_action_steps=N_STEPS,
    )

    assert len(records) == math.ceil(20 / N_STEPS) == 3
    # 3 replans + 1 stream-end tail proposal (bootstrap next.action, no execution).
    assert len(policy.states) == 4, "3 replans + 1 tail next-action proposal"
    assert len(env.applied_actions) == 20
    assert [int(r["executed_n_action_steps"]) for r in records] == [8, 8, 4]  # ceil budget split
    assert all(is_continue(r) for r in records)
    assert all(r["action"].shape == (ACTION_HORIZON, ACT_DIM) for r in records)
    assert all(torch.isfinite(r["next.action"]).all() for r in records)

    # Exact multiple: 24 steps -> 3 full chunks.
    env24 = FakeEnv()
    pol24 = FakePolicy(torch.zeros(ACTION_HORIZON, ACT_DIM))
    recs24 = collect_chunks(
        env24,
        pol24,
        total_steps=24,
        action_low=-1.0,
        action_high=1.0,
        n_action_steps=N_STEPS,
    )
    assert len(recs24) == math.ceil(24 / N_STEPS) == 3
    assert [int(r["executed_n_action_steps"]) for r in recs24] == [8, 8, 8]
    assert len(pol24.states) == 4  # 3 replans + 1 tail proposal

    # (b) mid-chunk termination: discard rest, no bootstrap, full-H anchor kept


def test_b_mid_chunk_termination_discards_remaining_and_forbids_bootstrap():
    # Global steps 9,10,11 are actions 1..3 of chunk index 1; termination
    # fires ON step 11 => n=3, remaining proposed actions discarded.
    env = FakeEnv(terminate_at=11)
    policy = FakePolicy(torch.zeros(ACTION_HORIZON, ACT_DIM))

    records = collect_chunks(
        env,
        policy,
        total_steps=24,
        action_low=-1.0,
        action_high=1.0,
        n_action_steps=N_STEPS,
    )

    assert [int(r["executed_n_action_steps"]) for r in records] == [8, 3, 8, 5]
    assert sum(int(r["executed_n_action_steps"]) for r in records) == 24
    assert len(env.applied_actions) == 24, "collection must continue after reset"

    dead = records[1]
    assert bool(dead["next.done"]) is True
    assert bool(dead["next.truncated"]) is False, "V-bootstrap FORBIDDEN on termination"
    assert int(dead["executed_n_action_steps"]) == 3
    assert dead["next.reward"].tolist() == [9.0, 10.0, 11.0]
    assert dead["action"].shape == (ACTION_HORIZON, ACT_DIM)  # full-H anchor kept

    # Exactly 3 actions were applied inside the dying chunk — the remaining
    # proposed actions were NOT executed.
    dying_chunk_actions = env.applied_actions[8:11]
    assert len(dying_chunk_actions) == 3
    assert int(dead["clip_count"]) == 0

    # Post-reset chunks are clean 'continue' chunks.
    assert is_continue(records[2]) and int(records[2]["executed_n_action_steps"]) == 8
    assert is_continue(records[3]) and int(records[3]["executed_n_action_steps"]) == 5

    # (c) truncation (TimeLimit semantics): boundary + mid-chunk, bootstrap-eligible


def test_c_truncation_at_chunk_boundary_is_bootstrap_eligible():
    env = FakeEnv(truncate_at=8)  # TimeLimit fires exactly on the last executed action
    policy = FakePolicy(torch.zeros(ACTION_HORIZON, ACT_DIM))

    records = collect_chunks(
        env,
        policy,
        total_steps=16,
        action_low=-1.0,
        action_high=1.0,
        n_action_steps=N_STEPS,
    )

    assert len(records) == 2
    cut = records[0]
    assert bool(cut["next.done"]) is False
    assert bool(cut["next.truncated"]) is True, "truncation must be bootstrap-eligible"
    assert int(cut["executed_n_action_steps"]) == N_STEPS
    assert cut["action"].shape == (ACTION_HORIZON, ACT_DIM)
    assert is_continue(records[1])


def test_c_truncation_mid_chunk_also_bootstraps():
    # Steps 9,10 are actions 1..2 of chunk 1; truncation fires ON step 10.
    env = FakeEnv(truncate_at=10)
    policy = FakePolicy(torch.zeros(ACTION_HORIZON, ACT_DIM))

    records = collect_chunks(
        env,
        policy,
        total_steps=16,
        action_low=-1.0,
        action_high=1.0,
        n_action_steps=N_STEPS,
    )

    assert [int(r["executed_n_action_steps"]) for r in records] == [8, 2, 6]
    cut = records[1]
    assert bool(cut["next.done"]) is False
    assert int(cut["executed_n_action_steps"]) == 2
    assert bool(cut["next.truncated"]) is True
    assert cut["action"].shape == (ACTION_HORIZON, ACT_DIM)
    assert len(env.applied_actions) == 16

    # (d) frame buffer: warmup replication + sliding windows


def test_d_warmup_replicates_first_obs_n_obs_steps_times_then_slides():
    env = FakeEnv()
    policy = FakePolicy(torch.zeros(ACTION_HORIZON, ACT_DIM))
    total_steps = 20  # 3 replans

    records = collect_chunks(
        env,
        policy,
        total_steps=total_steps,
        action_low=-1.0,
        action_high=1.0,
        n_action_steps=N_STEPS,
    )

    first_obs_marker = env.obs_markers[0]
    warm = policy.states[0]
    assert warm.shape == (STATE_T, OBS_DIM)
    # Warmup replicates the first obs EXACTLY STATE_T times.
    assert torch.equal(warm, torch.full((STATE_T, OBS_DIM), float(first_obs_marker)))

    # Sliding windows: replay the marker stream independently and compare
    # every captured state against the expected stacked window.
    ptr = 1  # obs_markers[0] consumed by warmup
    window_markers = [first_obs_marker] * STATE_T
    executed = 0
    for j, record in enumerate(records):
        expected = expected_window(window_markers, len(window_markers))
        assert torch.allclose(policy.states[j], expected), (
            f"state mismatch at chunk {j}"
        )
        for _ in range(int(record["executed_n_action_steps"])):
            window_markers.append(env.obs_markers[ptr])
            ptr += 1
            executed += 1
    assert executed == total_steps
    assert ptr == len(env.obs_markers)

    # (e) stored per-chunk fields: state, real SCALED FULL proposal, rewards,
    #     done flags, clip count


def test_e_stores_state_rewards_done_and_real_scaled_chunk():
    # Bounds [-2, 2]; constant proposal 1.0 is IN bounds and scales to 0.5.
    env = FakeEnv()
    chunk = torch.full((ACTION_HORIZON, ACT_DIM), 1.0)
    policy = FakePolicy(chunk)

    records = collect_chunks(
        env,
        policy,
        total_steps=ACTION_HORIZON,
        action_low=-2.0,
        action_high=2.0,
        n_action_steps=N_STEPS,
    )

    rec = records[0]
    assert rec["observation.state"].shape == (STATE_T, OBS_DIM)
    assert rec["next.observation.state"].shape == (STATE_T, OBS_DIM)
    assert bool(rec["next.done"]) is False
    assert bool(rec["next.truncated"]) is False
    assert int(rec["clip_count"]) == 0
    assert rec["next.reward"].tolist() == [float(i) for i in range(1, N_STEPS + 1)]

    # Stored action is the real SCALED FULL proposal: scaled(1.0; [-2,2]) = 0.5.
    expected = torch.full((ACTION_HORIZON, ACT_DIM), 0.5)
    assert torch.equal(rec["action"], expected)


def test_e_clip_count_counts_out_of_bound_elements_and_executes_clipped():
    env = FakeEnv()
    # Col 0 exceeds high, col 1 exceeds low, col 2 in bounds -> 2*ACTION_HORIZON clipped
    # elements over the proposal; clipping must be COUNTED, not silent.
    chunk = torch.tensor([[3.0, -5.0, 0.5]] * ACTION_HORIZON)
    policy = FakePolicy(chunk)

    records = collect_chunks(
        env,
        policy,
        total_steps=ACTION_HORIZON,
        action_low=-2.0,
        action_high=2.0,
        n_action_steps=N_STEPS,
    )

    rec = records[0]
    assert int(rec["clip_count"]) == 2 * ACTION_HORIZON
    # Executed actions were clipped into bounds before reaching the env.
    applied = np.stack(env.applied_actions)
    assert np.all(applied[:, 0] <= 2.0) and np.allclose(applied[:, 0], 2.0)
    assert np.all(applied[:, 1] >= -2.0) and np.allclose(applied[:, 1], -2.0)
    assert np.allclose(applied[:, 2], 0.5)
    # Stored action is the real SCALED CLIPPED proposal: col0 -> +1.0, col1 -> -1.0,
    # col2 raw 0.5 scales to 0.25 on [-2, 2].
    assert torch.equal(rec["action"][:, 0], torch.full((ACTION_HORIZON,), 1.0))
    assert torch.equal(rec["action"][:, 1], torch.full((ACTION_HORIZON,), -1.0))
    assert torch.equal(rec["action"][:, 2], torch.full((ACTION_HORIZON,), 0.25))

    # QA failure scenario: env raising mid-chunk propagates AFTER flush attempt


def test_env_exception_propagates_after_checkpoint_flush_attempt():
    env = FakeEnv(raise_at_step=5)
    policy = FakePolicy(torch.zeros(ACTION_HORIZON, ACT_DIM))
    hook_calls: list[int] = []

    def failing_flush() -> None:
        hook_calls.append(1)
        raise ValueError("flush hook itself exploded")

    with pytest.raises(RuntimeError, match="boom at step 5") as excinfo:
        collect_chunks(
            env,
            policy,
            total_steps=24,
            action_low=-1.0,
            action_high=1.0,
            n_action_steps=N_STEPS,
            on_error_flush=failing_flush,
        )

        # Original exception type preserved (NOT the hook's ValueError).
    assert type(excinfo.value) is RuntimeError
    assert hook_calls == [1], "flush hook must be attempted exactly once"


def test_zero_steps_collects_nothing():
    env = FakeEnv()
    policy = FakePolicy(torch.zeros(ACTION_HORIZON, ACT_DIM))
    records = collect_chunks(
        env,
        policy,
        total_steps=0,
        action_low=-1.0,
        action_high=1.0,
        n_action_steps=N_STEPS,
    )
    assert records == []
    assert len(env.applied_actions) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_bootstrap_action_lengths_rule():
    """Interior-continue masks to the NEXT chunk's length; rest stay full."""
    from sjfpo.collector import bootstrap_action_lengths

    def chunk(done, truncated, n):
        return {
            "next.done": torch.tensor(done),
            "next.truncated": torch.tensor(truncated),
            "executed_n_action_steps": torch.tensor(n),
        }

    chunks = [
        chunk(False, False, 16),  # interior continue → L_1 = 8
        chunk(False, False, 8),  # interior continue → L_2 = 5
        chunk(False, True, 5),  # truncated proposal stays full
        chunk(True, False, 3),  # terminated unused → full
        chunk(False, False, 16),  # budget-cut tail → full
    ]
    assert bootstrap_action_lengths(chunks, 16) == [8, 5, 16, 16, 16]
