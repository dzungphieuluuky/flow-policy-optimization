"""Unit tests for src/mfpo/collector.py (plan task 6).

TDD: written BEFORE the implementation exists — the first pytest run must
fail RED on the missing module, then go GREEN once collector.py lands.

Boundary semantics under test (plan Scope §Chunk-level semi-MDP GAE):
    terminated (anywhere)  -> stop executing, discard remaining actions,
                              done='terminated', bootstrap FORBIDDEN
    truncated (TimeLimit)  -> stop executing, done='truncated',
                              bootstrap-eligible (V(s_last))
    neither                -> chunk continues to completion, done='continue'

The fake env is scripted gymnasium-v5 style: step() returns the 5-tuple
(obs, reward, terminated, truncated, info) and termination/truncation can
be planted at exact global step offsets. CONTEXT_LEN / CHUNK_LEN are
imported from mfpo.config — never hardcoded here.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mfpo.collector import ChunkRecord, collect_chunks  # noqa: E402, F401
from mfpo.config import MFPO_CFG  # noqa: E402
from mfpo.dct_utils import dct as dct_fn  # noqa: E402

CONTEXT_LEN = MFPO_CFG.context_len  # imported constant — NEVER hardcode 16
CHUNK_LEN = MFPO_CFG.chunk_len  # imported constant — NEVER hardcode 8

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
        self._global_step = 0
        self._episode_step = 0

    def _marker(self) -> int:
        return 1000 * self.resets + self._episode_step

    def reset(self, *, seed=None, options=None):
        self.resets += 1
        self._episode_step = 0
        obs = np.full(self.obs_dim, float(self._marker()), dtype=np.float32)
        self.obs_markers.append(int(obs[0]))
        return obs, {}

    def step(self, action):
        if (
            self.raise_at_step is not None
            and self._global_step + 1 == self.raise_at_step
        ):
            raise RuntimeError(f"fake env boom at step {self._global_step + 1}")
        self.applied_actions.append(np.array(action, dtype=np.float32, copy=True))
        self._global_step += 1
        self._episode_step += 1
        terminated = (
            self.terminate_at is not None and self._global_step == self.terminate_at
        )
        truncated = (
            self.truncate_at is not None and self._global_step == self.truncate_at
        )
        assert not (terminated and truncated), "fixture bug: flags must be exclusive"
        obs = np.full(self.obs_dim, float(self._marker()), dtype=np.float32)
        self.obs_markers.append(int(obs[0]))
        return obs, float(self._global_step), terminated, truncated, {}


class FakePolicy:
    """Records conditioning contexts; returns a fixed proposed chunk."""

    def __init__(self, chunk: torch.Tensor) -> None:
        self.chunk = chunk
        self.contexts: list[torch.Tensor] = []

    def __call__(self, context: torch.Tensor) -> torch.Tensor:
        assert isinstance(context, torch.Tensor)
        self.contexts.append(context.detach().clone())
        return self.chunk


def _expected_window(markers: list[int], upto: int) -> torch.Tensor:
    """Stacked sliding window of the last CONTEXT_LEN markers before `upto`."""
    recent = markers[max(0, upto - CONTEXT_LEN) : upto]
    while len(recent) < CONTEXT_LEN:
        recent.insert(0, markers[0])  # warmup replication of the first obs
    rows = [np.full(OBS_DIM, float(m), dtype=np.float32) for m in recent]
    return torch.from_numpy(np.stack(rows))


# ---------------------------------------------------------------------------
# (a) chunk counts + open-loop replan cadence
# ---------------------------------------------------------------------------


def test_a_exact_chunk_count_and_replan_every_8_steps():
    env = FakeEnv()
    policy = FakePolicy(torch.zeros(CHUNK_LEN, ACT_DIM))

    records = collect_chunks(
        env, policy, total_steps=20, action_low=-1.0, action_high=1.0
    )

    assert len(records) == math.ceil(20 / CHUNK_LEN) == 3
    assert len(policy.contexts) == 3, "must replan exactly once per chunk"
    assert len(env.applied_actions) == 20
    assert [r.exec_len for r in records] == [8, 8, 4]  # ceil budget split
    assert all(r.done == "continue" for r in records)
    assert all(not r.bootstrap for r in records)

    # Exact multiple: 24 steps -> 3 full chunks.
    env24 = FakeEnv()
    pol24 = FakePolicy(torch.zeros(CHUNK_LEN, ACT_DIM))
    recs24 = collect_chunks(
        env24, pol24, total_steps=24, action_low=-1.0, action_high=1.0
    )
    assert len(recs24) == math.ceil(24 / CHUNK_LEN) == 3
    assert [r.exec_len for r in recs24] == [8, 8, 8]
    assert len(pol24.contexts) == 3


# ---------------------------------------------------------------------------
# (b) mid-chunk termination: discard rest, no bootstrap
# ---------------------------------------------------------------------------


def test_b_mid_chunk_termination_discards_remaining_and_forbids_bootstrap():
    # Global steps 9,10,11 are actions 1..3 of chunk index 1; termination
    # fires ON step 11 => exec_len=3, remaining 5 actions discarded.
    env = FakeEnv(terminate_at=11)
    policy = FakePolicy(torch.zeros(CHUNK_LEN, ACT_DIM))

    records = collect_chunks(
        env, policy, total_steps=24, action_low=-1.0, action_high=1.0
    )

    assert [r.exec_len for r in records] == [8, 3, 8, 5]
    assert sum(r.exec_len for r in records) == 24
    assert len(env.applied_actions) == 24, "collection must continue after reset"

    dead = records[1]
    assert dead.done == "terminated"
    assert dead.exec_len == 3
    assert len(dead.rewards) == 3
    assert dead.rewards == [9.0, 10.0, 11.0]
    assert dead.bootstrap is False, "V-bootstrap is FORBIDDEN on termination"
    assert dead.action_dct.shape == (3, ACT_DIM)

    # Exactly 3 actions were applied inside the dying chunk — the remaining
    # 5 proposed actions were NOT executed.
    dying_chunk_actions = env.applied_actions[8:11]
    assert len(dying_chunk_actions) == 3
    assert dead.clip_count == 0

    # Post-reset chunks are clean 'continue' chunks.
    assert records[2].done == "continue" and records[2].exec_len == 8
    assert records[3].done == "continue" and records[3].exec_len == 5


# ---------------------------------------------------------------------------
# (c) truncation (TimeLimit semantics): boundary + mid-chunk, bootstrap-eligible
# ---------------------------------------------------------------------------


def test_c_truncation_at_chunk_boundary_is_bootstrap_eligible():
    env = FakeEnv(truncate_at=8)  # TimeLimit fires exactly on the last chunk action
    policy = FakePolicy(torch.zeros(CHUNK_LEN, ACT_DIM))

    records = collect_chunks(
        env, policy, total_steps=16, action_low=-1.0, action_high=1.0
    )

    assert len(records) == 2
    cut = records[0]
    assert cut.done == "truncated"
    assert cut.exec_len == CHUNK_LEN
    assert cut.bootstrap is True, "TimeLimit truncation must be bootstrap-eligible"
    assert cut.action_dct.shape == (CHUNK_LEN, ACT_DIM)
    assert records[1].done == "continue" and records[1].bootstrap is False


def test_c_truncation_mid_chunk_also_bootstraps():
    # Steps 9,10 are actions 1..2 of chunk 1; truncation fires ON step 10.
    env = FakeEnv(truncate_at=10)
    policy = FakePolicy(torch.zeros(CHUNK_LEN, ACT_DIM))

    records = collect_chunks(
        env, policy, total_steps=16, action_low=-1.0, action_high=1.0
    )

    assert [r.exec_len for r in records] == [8, 2, 6]
    cut = records[1]
    assert cut.done == "truncated"
    assert cut.exec_len == 2
    assert cut.bootstrap is True
    assert cut.action_dct.shape == (2, ACT_DIM)
    assert len(env.applied_actions) == 16


# ---------------------------------------------------------------------------
# (d) frame buffer: warmup replication + sliding windows
# ---------------------------------------------------------------------------


def test_d_warmup_replicates_first_obs_context_len_times_then_slides():
    env = FakeEnv()
    policy = FakePolicy(torch.zeros(CHUNK_LEN, ACT_DIM))
    total_steps = 20  # 3 replans

    records = collect_chunks(
        env, policy, total_steps=total_steps, action_low=-1.0, action_high=1.0
    )

    first_obs_marker = env.obs_markers[0]
    warm = policy.contexts[0]
    assert warm.shape == (CONTEXT_LEN, OBS_DIM)
    # Warmup replicates the first obs EXACTLY CONTEXT_LEN times.
    assert torch.equal(
        warm, torch.full((CONTEXT_LEN, OBS_DIM), float(first_obs_marker))
    )

    # Sliding windows: replay the marker stream independently and compare
    # every captured context against the expected stacked window.
    ptr = 1  # obs_markers[0] consumed by warmup
    window_markers = [first_obs_marker] * CONTEXT_LEN
    executed = 0
    for j, record in enumerate(records):
        expected = _expected_window(window_markers, len(window_markers))
        assert torch.allclose(policy.contexts[j], expected), (
            f"context mismatch at chunk {j}"
        )
        for _ in range(record.exec_len):
            window_markers.append(env.obs_markers[ptr])
            ptr += 1
            executed += 1
    assert executed == total_steps
    assert ptr == len(env.obs_markers)


# ---------------------------------------------------------------------------
# (e) stored per-chunk fields: context, DCT of SCALED prefix, rewards, done,
#     clip count
# ---------------------------------------------------------------------------


def test_e_stores_context_rewards_done_and_dct_of_scaled_chunk():
    # Bounds [-2, 2]; constant proposal 1.0 is IN bounds and scales to 0.5.
    env = FakeEnv()
    chunk = torch.full((CHUNK_LEN, ACT_DIM), 1.0)
    policy = FakePolicy(chunk)

    records = collect_chunks(
        env, policy, total_steps=CHUNK_LEN, action_low=-2.0, action_high=2.0
    )

    rec = records[0]
    assert rec.context.shape == (CONTEXT_LEN, OBS_DIM)
    assert rec.done == "continue"
    assert rec.bootstrap is False
    assert rec.clip_count == 0
    assert rec.rewards == [float(i) for i in range(1, CHUNK_LEN + 1)]

    # DCT must be taken on the SCALED chunk: scaled(1.0; [-2,2]) = 0.5.
    expected = dct_fn(torch.full((CHUNK_LEN, ACT_DIM), 0.5), dim=0)
    assert torch.allclose(rec.action_dct, expected, atol=1e-6)
    # Non-circular DC check: orthonormal DCT-II DC coeff = mean * sqrt(N).
    assert torch.allclose(
        rec.action_dct[0], torch.full((ACT_DIM,), 0.5 * math.sqrt(CHUNK_LEN)), atol=1e-6
    )


def test_e_clip_count_counts_out_of_bound_elements_and_executes_clipped():
    env = FakeEnv()
    # Col 0 exceeds high, col 1 exceeds low, col 2 in bounds -> 16 clipped
    # elements over the 8-step chunk; clipping must be COUNTED, not silent.
    chunk = torch.tensor([[3.0, -5.0, 0.5]] * CHUNK_LEN)
    policy = FakePolicy(chunk)

    records = collect_chunks(
        env, policy, total_steps=CHUNK_LEN, action_low=-2.0, action_high=2.0
    )

    rec = records[0]
    assert rec.clip_count == 2 * CHUNK_LEN
    # Executed actions were clipped into bounds before reaching the env.
    applied = np.stack(env.applied_actions)
    assert np.all(applied[:, 0] <= 2.0) and np.allclose(applied[:, 0], 2.0)
    assert np.all(applied[:, 1] >= -2.0) and np.allclose(applied[:, 1], -2.0)
    assert np.allclose(applied[:, 2], 0.5)
    # DCT of the SCALED CLIPPED prefix: col0 scales to +1.0, col1 to -1.0.
    assert torch.allclose(
        rec.action_dct[0, 0], torch.tensor(1.0 * math.sqrt(CHUNK_LEN)), atol=1e-6
    )
    assert torch.allclose(
        rec.action_dct[0, 1], torch.tensor(-1.0 * math.sqrt(CHUNK_LEN)), atol=1e-6
    )


# ---------------------------------------------------------------------------
# QA failure scenario: env raising mid-chunk propagates AFTER flush attempt
# ---------------------------------------------------------------------------


def test_env_exception_propagates_after_checkpoint_flush_attempt():
    env = FakeEnv(raise_at_step=5)
    policy = FakePolicy(torch.zeros(CHUNK_LEN, ACT_DIM))
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
            on_error_flush=failing_flush,
        )

    # Original exception type preserved (NOT the hook's ValueError).
    assert type(excinfo.value) is RuntimeError
    assert hook_calls == [1], "flush hook must be attempted exactly once"


def test_zero_steps_collects_nothing():
    env = FakeEnv()
    policy = FakePolicy(torch.zeros(CHUNK_LEN, ACT_DIM))
    records = collect_chunks(
        env, policy, total_steps=0, action_low=-1.0, action_high=1.0
    )
    assert records == []
    assert len(env.applied_actions) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
