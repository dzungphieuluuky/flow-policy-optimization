"""Tests for the chunked Q backup (``sjfpo.gae``).

QC (Li, Zhou, Levine 2025) takes one backup per chunk:

    return_i = sum_{k < h_i} gamma^k r_{s_i+k} + gamma^{h_i} (1 - term_i) Q(s_{i+1}, a')

and nothing else. That is the default (``gae_lambda=0``) and the reference below
is deliberately a one-line per-chunk expression, so it keeps matching the paper.

A chunk-level lambda trace also exists, opt-in, with an effective horizon of
``1 / (1 - gamma^h * lam)`` chunks -- about five at gamma=0.99, h=16, lam=0.95.
It was removed in 7202519 to match QC and restored opt-in rather than by
default. Whether it belongs is a paper-level question: QC specifies one chunk,
FPO++ reports GAE lam 0.95 (0.99 on manipulation). The tests below pin both
behaviours so neither can drift silently.
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


def test_trace_inputs_are_absent_from_the_qc_backup():
    """The QC path must not consult the trace mask, only (1 - term).

    QC masks the last step of the trajectory, which is the terminal. The trace
    inputs are in the signature for the opt-in GAE, but at the default lambda=0
    they are unused -- and are permitted to be, so a caller reproducing QC does
    not have to supply them.
    """
    import inspect

    params = inspect.signature(compute_chunk_advantages).parameters
    assert params["gae_lambda"].default == 0.0
    assert params["chunk_truncated"].default is None
    assert params["chunk_env_id"].default is None

    # Supplying a truncated chunk must not change a QC backup, even mid-trace.
    lengths = [H] * 3
    without, _ = run(lengths, terminated=[False, False, False])
    truncated = torch.tensor([False, True, False], dtype=torch.bool)
    with_trunc, _ = compute_chunk_advantages(
        torch.full((sum(lengths),), 1.0),
        torch.zeros(3), torch.full((3,), 2.0),
        torch.tensor([0, H, 2 * H]), torch.tensor(lengths),
        chunk_terminated=torch.zeros(3, dtype=torch.bool),
        chunk_size=H, gamma=GAMMA,
        chunk_truncated=truncated,
        chunk_env_id=torch.zeros(3, dtype=torch.long),
    )
    assert without.tolist() == pytest.approx(with_trunc.tolist(), abs=1e-6)


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


# --- the opt-in chunk-level GAE trace ---------------------------------------


def _traced(lengths, env_id=None, terminated=None, truncated=None, lam=0.95,
            chunk_q=0.0, chunk_next_q=0.0, reward=1.0):
    """One rollout through the GAE path, with the masks the trace needs."""
    lengths_t = torch.tensor(lengths, dtype=torch.long)
    n = lengths_t.numel()
    starts = torch.cat([torch.zeros(1, dtype=torch.long), lengths_t.cumsum(0)[:-1]])
    n_env = len(set(env_id)) if env_id is not None else 1
    return compute_chunk_advantages(
        torch.full((int(lengths_t.sum()),), float(reward)),
        torch.full((n,), float(chunk_q)),
        torch.full((n,), float(chunk_next_q)),
        starts, lengths_t,
        chunk_terminated=(torch.zeros(n, dtype=torch.bool) if terminated is None
                          else torch.tensor(terminated, dtype=torch.bool)),
        chunk_size=H, gamma=GAMMA,
        chunk_truncated=(torch.zeros(n, dtype=torch.bool) if truncated is None
                         else torch.tensor(truncated, dtype=torch.bool)),
        chunk_env_id=(torch.arange(n) % n_env if env_id is None
                      else torch.tensor(env_id, dtype=torch.long)),
        gae_lambda=lam,
    )


def test_lambda_zero_is_bit_identical_to_the_qc_path():
    """The primary gate: the trace must not perturb the default at all.

    Same inputs, lambda explicitly 0.0, compared against the QC call that
    passes none of the trace arguments.
    """
    lengths = [H, 2, 3, H]
    qc_adv, qc_ret = run(lengths, chunk_q=1.0, chunk_next_q=2.5, reward=0.7)
    traced_adv, traced_ret = _traced(
        lengths, lam=0.0, chunk_q=1.0, chunk_next_q=2.5, reward=0.7
    )
    torch.testing.assert_close(traced_adv, qc_adv, rtol=0, atol=0)
    torch.testing.assert_close(traced_ret, qc_ret, rtol=0, atol=0)


def test_lambda_zero_returns_before_running_the_recursion():
    """Not merely equal -- not run. A broken successor map cannot be latent here."""
    lengths = [H] * 3
    # env ids deliberately wrong (interleaved, none matching buffer order). At
    # lambda=0 they are ignored, so this must still match QC exactly.
    adv, _ = _traced(lengths, env_id=[0, 1, 0], lam=0.0, chunk_next_q=2.0)
    qc_adv, _ = run(lengths, chunk_next_q=2.0)
    torch.testing.assert_close(adv, qc_adv, rtol=0, atol=0)


def test_trace_matches_the_reference_recursion_by_hand():
    """A_i = delta_i + gamma^{h_i} lam (1-term)(1-trunc) A_j, j = next same-env.

    Recomputed in Python from the definition rather than from the vectorised
    successor map, so a wrong map cannot agree with itself.
    """
    lengths = [H, H, H, H]
    terminated = [False, False, False, False]
    truncated = [False, False, True, False]
    lam = 0.9
    # Distinct per-chunk values so an index mix-up cannot hide.
    q = torch.tensor([0.0, 1.0, 2.0, 3.0])
    q_next = torch.tensor([0.5, 1.5, 2.5, 3.5])
    lengths_t = torch.tensor(lengths)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), lengths_t.cumsum(0)[:-1]])
    rewards = torch.ones(int(lengths_t.sum()))
    got, _ = compute_chunk_advantages(
        rewards, q, q_next, starts, lengths_t,
        chunk_terminated=torch.tensor(terminated),
        chunk_size=H, gamma=GAMMA,
        chunk_truncated=torch.tensor(truncated),
        chunk_env_id=torch.zeros(4, dtype=torch.long),
        gae_lambda=lam,
    )
    r = [sum(GAMMA**k for k in range(h)) for h in lengths]
    delta = [
        r[i] + GAMMA ** lengths[i] * q_next[i].item() - q[i].item()
        for i in range(4)
    ]
    want = [0.0] * 4
    for i in range(3, -1, -1):           # single env, so succ is i+1
        nxt = i + 1 if i + 1 < 4 else None
        trace = (
            GAMMA ** lengths[i] * lam
            * (1.0 - terminated[i]) * (1.0 - truncated[i])
        )
        want[i] = delta[i] + (trace * want[nxt] if nxt is not None else 0.0)
    assert got.tolist() == pytest.approx(want, abs=1e-5)


def test_trace_discount_is_gamma_h_times_lambda_not_gamma_lambda_to_the_h():
    """Pins the macro-MDP convention: lambda decays once per chunk.

    verl-vla and RLinf use (gamma*lambda)^h, which decays lambda per primitive
    env step. The two differ for h>1, so this is a real design choice and it is
    the one 7202519 described as gamma^h * lam.
    """
    # Two chunks, one env, bootstrap value zero so the only propagated term is
    # the trace itself.
    lengths = [H] * 2
    got, _ = _traced(lengths, lam=0.95, chunk_next_q=0.0)
    r = sum(GAMMA**k for k in range(H))
    trace = GAMMA**H * 0.95
    # delta_0 = r, delta_1 = r, so A_0 = r + trace * r.
    assert got[0].item() == pytest.approx(r + trace * r, abs=1e-5)
    # The (gamma*lambda)^h reading gives a strictly smaller trace here, so this
    # assertion would fail under the verl-vla convention.
    alt = (GAMMA * 0.95) ** H
    assert abs(alt - trace) > 1e-3, "conventions coincide; test cannot distinguish"


def test_truncation_stops_the_trace_but_keeps_the_bootstrap():
    """A timeout is not an episode end: bootstrap yes, trace no.

    This is the distinction a single (1 - term) mask cannot express, and the
    reason chunk_truncated came back with the trace.
    """
    lengths = [H] * 3
    truncated = [False, True, False]
    adv, ret = _traced(lengths, truncated=truncated, lam=0.9, chunk_next_q=2.0)
    r = sum(GAMMA**k for k in range(H))
    # Chunk 1 is truncated: its bootstrap survives, so its target still carries
    # gamma^h * Q'. Chunk 2's advantage is unchanged by chunk 1's trace.
    assert ret[1].item() == pytest.approx(r + GAMMA**H * 2.0, abs=1e-5)
    two_free = _traced([H] * 2, lam=0.9, chunk_next_q=2.0)[0]
    assert adv[2].item() == pytest.approx(two_free[1].item(), abs=1e-5)


def test_termination_still_zeroes_the_bootstrap_and_the_trace():
    lengths = [H] * 3
    adv, ret = _traced(lengths, terminated=[False, True, False],
                       lam=0.9, chunk_next_q=2.0)
    r = sum(GAMMA**k for k in range(H))
    # No bootstrap, and nothing to propagate: the advantage is the reward sum.
    assert ret[1].item() == pytest.approx(r, abs=1e-5)


def test_chunks_from_different_envs_do_not_interact():
    """The property the env sort exists for.

    Buffer order interleaves envs, so without grouping by env_id the trace
    would run from env 0's chunk straight into env 1's. Rescaling env 1's
    rewards must leave env 0 bit-identical.
    """
    lengths = [H] * 6
    env_id = [0, 1, 0, 1, 0, 1]
    lam = 0.95
    base, _ = _traced(lengths, env_id=env_id, lam=lam, chunk_next_q=1.0)
    # Same rollout, but env 1's rewards multiplied by 10 via a separate call.
    lengths_t = torch.tensor(lengths)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), lengths_t.cumsum(0)[:-1]])
    rewards = torch.full((int(lengths_t.sum()),), 1.0)
    rewards[starts[1] : starts[1] + H] = 10.0
    rewards[starts[3] : starts[3] + H] = 10.0
    rewards[starts[5] : starts[5] + H] = 10.0
    alt, _ = compute_chunk_advantages(
        rewards, torch.zeros(6), torch.ones(6), starts, lengths_t,
        chunk_terminated=torch.zeros(6, dtype=torch.bool),
        chunk_size=H, gamma=GAMMA,
        chunk_truncated=torch.zeros(6, dtype=torch.bool),
        chunk_env_id=torch.tensor(env_id),
        gae_lambda=lam,
    )
    for i in (0, 2, 4):
        assert base[i].item() == pytest.approx(alt[i].item(), abs=1e-6), (
            f"env 0 chunk {i} moved when env 1's rewards changed"
        )


def test_successor_is_the_next_chunk_of_the_same_env_not_the_next_buffer_row():
    """Buffer row 0's successor is row 2 here, not row 1."""
    lengths = [H] * 3
    env_id = [0, 1, 0]
    lam = 0.95
    got, _ = _traced(lengths, env_id=env_id, lam=lam, chunk_next_q=0.0)
    r = sum(GAMMA**k for k in range(H))
    # Chunk 0 -> chunk 2 (env 0); chunk 1 is env 1 and is not reachable.
    assert got[0].item() == pytest.approx(r + (GAMMA**H * lam) * r, abs=1e-5)
    # Chunk 2 is last in its env, so it has no trace successor.
    assert got[2].item() == pytest.approx(r, abs=1e-5)


def test_last_chunk_of_each_env_gets_a_one_chunk_advantage():
    lengths = [H] * 4
    env_id = [0, 1, 0, 1]
    got, _ = _traced(lengths, env_id=env_id, lam=0.95, chunk_next_q=0.0)
    qc, _ = run(lengths, chunk_next_q=0.0)
    # Every chunk here is last-in-its-env except none: 0->2, 1->3. Check the
    # final element of each env is exactly its own delta.
    for i in (2, 3):
        assert got[i].item() == pytest.approx(qc[i].item(), abs=1e-6)


def test_advantage_is_monotone_in_lambda_between_delta_and_the_full_return():
    """Higher lambda moves credit further back, never past the true return."""
    lengths = [H] * 4
    endpoint = _traced(lengths, lam=0.0, chunk_next_q=0.0)[0]
    lam = 0.95
    got, _ = _traced(lengths, lam=lam, chunk_next_q=0.0)
    assert (got[:-1] >= endpoint[:-1] - 1e-5).all(), "lambda shrank the advantage"
    # lambda=1 and a zero bootstrap value makes it the full discounted return.
    full, _ = _traced(lengths, lam=1.0, chunk_next_q=0.0)
    want = sum(
        sum(GAMMA ** (i * H + k) for k in range(H)) for i in range(4)
    )
    assert full[0].item() == pytest.approx(want, abs=1e-4)


def test_trace_requires_its_inputs():
    """Failing loudly beats silently running a cross-reset, cross-env trace."""
    lengths_t = torch.tensor([H])
    common = dict(
        step_rewards=torch.ones(H),
        chunk_q=torch.zeros(1),
        chunk_next_q=torch.zeros(1),
        chunk_starts=torch.zeros(1, dtype=torch.long),
        chunk_lengths=lengths_t,
        chunk_terminated=torch.zeros(1, dtype=torch.bool),
        chunk_size=H,
        gamma=GAMMA,
        gae_lambda=0.95,
    )
    with pytest.raises(ValueError, match="chunk_truncated is required"):
        compute_chunk_advantages(**common)
    with pytest.raises(ValueError, match="chunk_env_id is required"):
        compute_chunk_advantages(
            chunk_truncated=torch.zeros(1, dtype=torch.bool), **common
        )


def test_rejects_an_out_of_range_lambda():
    with pytest.raises(ValueError, match="gae_lambda"):
        compute_chunk_advantages(
            torch.ones(H), torch.zeros(1), torch.zeros(1),
            torch.zeros(1, dtype=torch.long), torch.tensor([H]),
            chunk_terminated=torch.zeros(1, dtype=torch.bool),
            chunk_size=H, gamma=GAMMA, gae_lambda=1.5,
        )


def test_trace_handles_an_empty_rollout():
    adv, ret = compute_chunk_advantages(
        torch.zeros(0), torch.zeros(0), torch.zeros(0),
        torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
        chunk_terminated=torch.zeros(0, dtype=torch.bool),
        chunk_size=H, gamma=GAMMA,
        chunk_truncated=torch.zeros(0, dtype=torch.bool),
        chunk_env_id=torch.zeros(0, dtype=torch.long),
        gae_lambda=0.95,
    )
    assert adv.numel() == 0 and ret.numel() == 0


def test_config_default_keeps_the_qc_path():
    """The opt-in must not have become the default."""
    from spectral_joint_policy.configuration_sjfpo import SJFPOConfig

    assert SJFPOConfig().gae_lambda == 0.0


def test_config_rejects_an_out_of_range_lambda():
    from spectral_joint_policy.configuration_sjfpo import SJFPOConfig

    with pytest.raises(ValueError, match="gae_lambda"):
        SJFPOConfig(gae_lambda=2.0).validate()


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