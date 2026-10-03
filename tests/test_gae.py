"""Tests for the chunked Q backup (``sjfpo.gae``).

QC (Li, Zhou, Levine 2025) takes one backup per chunk:

    return_i = sum_{k < h_i} gamma^k r_{s_i+k} + gamma^{h_i} (1 - term_i) Q(s_{i+1}, a')

and nothing else. A lambda-GAE recursion over chunks used to sit on top, which
gave an effective horizon of ``1 / (1 - gamma^h * lam)`` chunks -- about five at
the defaults -- against QC's exactly one. The reference below is therefore
deliberately a one-line per-chunk expression: if the recursion ever comes back,
this file stops matching the paper and says so.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spectral_joint_policy.gae import compute_chunk_advantages

GAMMA, H = 0.99, 4


def expected_chunk_reward(lengths, reward=1.0):
    """R_i, the discounted reward sum over executed steps only."""
    return [sum(GAMMA**k * reward for k in range(h)) for h in lengths]


def run(lengths, terminated=None, chunk_q=0.0, chunk_next_q=2.0, reward=1.0,
        total_steps=None, chunk_size=H):
    """One rollout as the caller builds it: rewards per step, Q per chunk."""
    lengths_t = torch.tensor(lengths, dtype=torch.long)
    n = lengths_t.numel()
    starts = torch.cat([torch.zeros(1, dtype=torch.long), lengths_t.cumsum(0)[:-1]])
    total = int(total_steps if total_steps is not None else lengths_t.sum())
    return compute_chunk_advantages(
        torch.full((total,), float(reward)),
        torch.full((n,), float(chunk_q)),
        torch.full((n,), float(chunk_next_q)),
        starts, lengths_t,
        chunk_terminated=(
            torch.zeros(n, dtype=torch.bool)
            if terminated is None
            else torch.tensor(terminated, dtype=torch.bool)
        ),
        chunk_size=chunk_size, gamma=GAMMA,
    )


# --- the backup identity ---------------------------------------------------


def test_delta_is_one_chunked_backup():
    lengths = [H, H, 3, H]
    adv, ret = run(lengths)
    r = expected_chunk_reward(lengths)
    want = [r[i] + GAMMA ** lengths[i] * 2.0 for i in range(4)]
    assert adv.tolist() == pytest.approx(want, abs=1e-5)
    # returns - advantages is exactly the critic's own output, so returns reduces
    # to QC's target with no leftover term.
    torch.testing.assert_close(ret - adv, torch.full((4,), 0.0))


def test_returns_are_qcs_target_expression():
    """returns_i == R_i + gamma^{h_i} (1 - term_i) Q(s_{i+1}, a'), recomputed here.

    This is the whole point of dropping the GAE tail: the regression target is
    the paper's expression term for term.
    """
    lengths = [H, 2, H]
    q = torch.tensor([0.5, -3.0, 11.0])
    q_next = torch.tensor([2.0, 7.0, -1.0])
    rewards = torch.arange(1, int(sum(lengths)) + 1, dtype=torch.float32)
    starts = torch.cat(
        [torch.zeros(1, dtype=torch.long), torch.tensor(lengths).cumsum(0)[:-1]]
    )
    term = torch.tensor([False, True, False])
    _, returns = compute_chunk_advantages(
        rewards, q, q_next, starts, torch.tensor(lengths),
        chunk_terminated=term, chunk_size=H, gamma=GAMMA,
    )
    want = []
    for i, h in enumerate(lengths):
        r_i = sum(GAMMA**k * rewards[starts[i] + k].item() for k in range(h))
        want.append(r_i + (0.0 if term[i] else GAMMA**h * q_next[i].item()))
    assert returns.tolist() == pytest.approx(want, abs=1e-5)


def test_baseline_subtraction_scales_with_the_value():
    """A higher Q must lower the advantage, by exactly what it raised Q by.

    A unit-scale value next to multi-unit rewards cannot do this, which is what
    the old normalised critic produced: the baseline contributed almost nothing
    and the advantage was effectively the raw reward.
    """
    raw = run([H], terminated=[True], chunk_q=0.0)[0]
    raised = run([H], terminated=[True], chunk_q=9.0)[0]
    torch.testing.assert_close(raw - raised, torch.tensor([9.0]))
    assert float(raw) > 0 > float(raised)


# --- masks -----------------------------------------------------------------


def test_terminated_zeroes_the_bootstrap():
    lengths = [H] * 3
    r = expected_chunk_reward(lengths)
    adv, _ = run(lengths, terminated=[False, True, False])
    assert adv[1].item() == pytest.approx(r[1], abs=1e-5)  # reward sum only
    assert adv[2].item() == pytest.approx(r[2] + GAMMA**H * 2.0, abs=1e-5)


def test_a_neighbouring_termination_changes_nothing_upstream():
    """One backup per chunk, so no credit crosses a chunk boundary at all.

    This is the property the deleted recursion used to break on purpose, and it
    is now the whole isolation guarantee.
    """
    lengths = [H] * 3
    clean, _ = run(lengths, terminated=[False, False, False])
    cut, _ = run(lengths, terminated=[False, True, False])
    assert clean[0].item() == pytest.approx(cut[0].item(), abs=1e-6)


def test_truncation_flag_is_not_read_at_all():
    """Timeout and termination coincide in this backup: (1 - term) only.

    QC masks the last step of the trajectory, which is the terminal; a timeout
    that keeps the bootstrap was our own addition and the parameter is gone.
    """
    import inspect

    params = inspect.signature(compute_chunk_advantages).parameters
    assert "chunk_truncated" not in params
    assert "lam" not in params
    assert "chunk_env_id" not in params


def test_budget_cut_does_not_reach_the_backup():
    """A budget cut is not an episode end, so it must not appear as one.

    compute_chunk_advantages has no budget parameter at all, which is the fix: the
    rollout budget and the env's time limit shared chunk_truncated, so a live
    episode was read as finished.
    """
    import inspect

    assert "budget" not in inspect.signature(compute_chunk_advantages).parameters


# --- discounting and the reward gather --------------------------------------


def test_discount_uses_the_executed_length_not_the_horizon():
    """gamma^h over what ran: a chunk cut short must not be discounted as if whole."""
    lengths = [3, H]
    short = run(lengths)[0][0].item()
    whole = run([H, H])[0][0].item()
    assert short == pytest.approx(
        sum(GAMMA**k for k in range(3)) + GAMMA**3 * 2.0, abs=1e-6
    )
    assert whole > short


def test_uneven_chunk_lengths():
    lengths = [H, 2, 3, H, 1, H]
    adv, _ = run(lengths)
    r = expected_chunk_reward(lengths)
    want = [r[i] + GAMMA ** lengths[i] * 2.0 for i in range(len(lengths))]
    assert adv.tolist() == pytest.approx(want, abs=1e-5)


def test_row_order_does_not_matter():
    """With one backup per chunk there is no trajectory structure left to respect.

    Reordering the buffer permutes the result and nothing else, which is why the
    env_id sort is gone: it existed only to walk trajectories for the recursion.
    """
    lengths = [H, 2, 3, H]
    base, _ = run(lengths)
    perm = [2, 0, 3, 1]
    got, _ = run([lengths[i] for i in perm])
    assert got.tolist() == pytest.approx([base[i].item() for i in perm], abs=1e-5)


def test_no_fp32_precision_loss_at_large_offsets():
    """The gather must survive s_i ~ 4096, where gamma**-s would be ~1e18."""
    offset, n = 4096, 4
    starts = torch.arange(offset, offset + n * H, H, dtype=torch.long)
    lengths = torch.full((n,), H, dtype=torch.long)
    total = offset + n * H
    rewards = torch.zeros(total)
    rewards[starts[-1] : starts[-1] + H] = 1.0  # only the last chunk has reward
    adv, _ = compute_chunk_advantages(
        rewards, torch.zeros(n), torch.zeros(n), starts, lengths,
        chunk_terminated=torch.zeros(n, dtype=torch.bool),
        chunk_size=H, gamma=GAMMA,
    )
    assert adv[-1].item() == pytest.approx(sum(GAMMA**k for k in range(H)), rel=1e-5)
    # Every earlier chunk has no reward and no bootstrap, so it is exactly zero.
    assert adv[:-1].abs().max().item() == 0.0


def test_values_are_per_chunk_not_per_step():
    """The value arrays are gathered by the caller, so they are n_chunks long."""
    lengths = [H, 2, 3]
    starts = torch.cat(
        [torch.zeros(1, dtype=torch.long), torch.tensor(lengths).cumsum(0)[:-1]]
    )
    adv, _ = compute_chunk_advantages(
        torch.ones(sum(lengths)),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([-1.0, -2.0, -3.0]),
        starts, torch.tensor(lengths),
        chunk_terminated=torch.zeros(3, dtype=torch.bool),
        chunk_size=H, gamma=GAMMA,
    )
    want = [
        sum(GAMMA**k for k in range(h)) + GAMMA**h * nv - q
        for h, q, nv in zip(lengths, (1.0, 2.0, 3.0), (-1.0, -2.0, -3.0))
    ]
    assert adv.tolist() == pytest.approx(want, abs=1e-5)


# --- validation ------------------------------------------------------------


def test_empty_rollout():
    adv, ret = compute_chunk_advantages(
        torch.zeros(0), torch.zeros(0), torch.zeros(0),
        torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
        chunk_terminated=torch.zeros(0, dtype=torch.bool),
        chunk_size=H, gamma=GAMMA,
    )
    assert adv.numel() == 0 and ret.numel() == 0


def test_rejects_an_out_of_range_discount():
    with pytest.raises(ValueError, match="gamma"):
        compute_chunk_advantages(
            torch.ones(H), torch.zeros(1), torch.zeros(1),
            torch.zeros(1, dtype=torch.long), torch.tensor([H]),
            chunk_terminated=torch.zeros(1, dtype=torch.bool),
            chunk_size=H, gamma=1.5,
        )


def test_rejects_value_arrays_of_the_wrong_length():
    """A per-step-sized value array must raise, not silently misindex."""
    n = 3
    common = dict(
        step_rewards=torch.ones(16),
        chunk_starts=torch.arange(0, n * 4, 4, dtype=torch.long),
        chunk_lengths=torch.full((n,), 4, dtype=torch.long),
        chunk_terminated=torch.zeros(n, dtype=torch.bool),
        chunk_size=4,
    )
    with pytest.raises(ValueError, match="chunk_q"):
        compute_chunk_advantages(
            chunk_q=torch.zeros(16), chunk_next_q=torch.zeros(n), **common
        )
    with pytest.raises(ValueError, match="chunk_next_q"):
        compute_chunk_advantages(
            chunk_q=torch.zeros(n), chunk_next_q=torch.zeros(16), **common
        )


def test_rejects_mismatched_chunk_lengths():
    with pytest.raises(ValueError, match="chunk_lengths"):
        compute_chunk_advantages(
            torch.ones(H), torch.zeros(2), torch.zeros(2),
            torch.zeros(2, dtype=torch.long), torch.tensor([H]),
            chunk_terminated=torch.zeros(2, dtype=torch.bool),
            chunk_size=H, gamma=GAMMA,
        )