"""GAE backup."""
import inspect

import pytest
import torch

from myfpo.collect import compute_chunk_advantages, gae_trace

GAMMA, H = 0.99, 4


def prev_chain(env_id):
    """Previous chunk index per chunk, the buffer's ``chunk_prev`` layout."""
    last, prev = {}, []
    for i, e in enumerate(env_id):
        prev.append(last.get(e, -1))
        last[e] = i
    return torch.tensor(prev, dtype=torch.long)


def gae(lengths, terminated=None, truncated=None, env_id=None, lam=0.0,
        v_current=0.0, v_next=2.0, reward=1.0, chunk_size=H):
    """One rollout through the backup. ``lam=0`` gives the QC path."""
    lengths_t = torch.tensor(lengths, dtype=torch.long)
    n = lengths_t.numel()
    starts = torch.zeros(n, dtype=torch.long)
    starts[1:] = lengths_t.cumsum(0)[:-1]

    def flags(value):
        if value is None:
            return torch.zeros(n, dtype=torch.bool)
        return torch.tensor(value, dtype=torch.bool)

    return compute_chunk_advantages(
        torch.full((int(lengths_t.sum()),), float(reward)),
        torch.full((n,), float(v_current)),
        torch.full((n,), float(v_next)),
        starts, lengths_t,
        chunk_terminated=flags(terminated),
        chunk_size=chunk_size, gamma=GAMMA,
        chunk_truncated=flags(truncated),
        chunk_prev=prev_chain([0] * n if env_id is None else list(env_id)),
        gae_lambda=lam,
    )


def expected_chunk_reward(lengths, reward=1.0):
    """R_i: executed-step discount."""
    return [sum(GAMMA**k * reward for k in range(h)) for h in lengths]


# Backup identity.


def test_delta_is_one_chunked_backup():
    lengths = [H, H, 3, H]
    adv, ret = gae(lengths)
    r = expected_chunk_reward(lengths)
    want = [r[i] + GAMMA ** lengths[i] * 2.0 for i in range(4)]
    assert adv.tolist() == pytest.approx(want, abs=1e-5)
    # returns - advantages is the critic's own output, so it must vanish.
    torch.testing.assert_close(ret - adv, torch.full((4,), 0.0))


def test_returns_are_qcs_target_expression():
    """Regression target equals R_i + gamma^h (1 - term) Q(s')."""
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
        chunk_terminated=term, chunk_size=H, gamma=GAMMA, gae_lambda=0.0,
    )
    want = [
        sum(GAMMA**k * rewards[starts[i] + k].item() for k in range(h))
        + (0.0 if term[i] else GAMMA**h * q_next[i].item())
        for i, h in enumerate(lengths)
    ]
    assert returns.tolist() == pytest.approx(want, abs=1e-5)


def test_baseline_subtraction_scales_with_the_value():
    """A higher Q lowers the advantage by exactly what it raised Q by."""
    raw = gae([H], terminated=[True], v_current=0.0)[0]
    raised = gae([H], terminated=[True], v_current=9.0)[0]
    torch.testing.assert_close(raw - raised, torch.tensor([9.0]))
    assert float(raw) > 0 > float(raised)


# Masks.


def test_terminated_zeroes_the_bootstrap():
    lengths = [H] * 3
    r = expected_chunk_reward(lengths)
    adv, _ = gae(lengths, terminated=[False, True, False])
    assert adv[1].item() == pytest.approx(r[1], abs=1e-5)  # reward sum only.
    assert adv[2].item() == pytest.approx(r[2] + GAMMA**H * 2.0, abs=1e-5)


def test_a_neighbouring_termination_changes_nothing_upstream():
    """One backup per chunk: no credit crosses a chunk boundary."""
    lengths = [H] * 3
    clean, _ = gae(lengths, terminated=[False, False, False])
    cut, _ = gae(lengths, terminated=[False, True, False])
    assert clean[0].item() == pytest.approx(cut[0].item(), abs=1e-6)


def test_a_qc_backup_ignores_the_trace_mask():
    """The QC path must not read the trace mask, and must run without it."""
    lengths = [H] * 3
    without, _ = gae(lengths, terminated=[False, False, False])
    with_trunc, _ = gae(
        lengths, terminated=[False, False, False], truncated=[False, True, False]
    )
    assert without.tolist() == pytest.approx(with_trunc.tolist(), abs=1e-6)
    # And the QC path runs with no trace args at all.
    plain, _ = compute_chunk_advantages(
        torch.full((3 * H,), 1.0), torch.zeros(3), torch.full((3,), 2.0),
        torch.tensor([0, H, 2 * H]), torch.tensor([H] * 3),
        chunk_terminated=torch.zeros(3, dtype=torch.bool),
        chunk_size=H, gamma=GAMMA, gae_lambda=0.0,
    )
    assert plain.tolist() == pytest.approx(without.tolist(), abs=1e-6)


# Discount/reward.


def test_discount_uses_the_executed_length_not_the_horizon():
    """gamma^h over what actually ran, not the planned horizon."""
    lengths = [3, H]
    short = gae(lengths)[0][0].item()
    whole = gae([H, H])[0][0].item()
    assert short == pytest.approx(
        sum(GAMMA**k for k in range(3)) + GAMMA**3 * 2.0, abs=1e-6
    )
    assert whole > short


def test_uneven_chunk_lengths():
    lengths = [H, 2, 3, H, 1, H]
    adv, _ = gae(lengths)
    r = expected_chunk_reward(lengths)
    want = [r[i] + GAMMA ** lengths[i] * 2.0 for i in range(len(lengths))]
    assert adv.tolist() == pytest.approx(want, abs=1e-5)


def test_row_order_does_not_matter():
    """Reordering the buffer only permutes the result."""
    lengths = [H, 2, 3, H]
    base, _ = gae(lengths)
    perm = [2, 0, 3, 1]
    got, _ = gae([lengths[i] for i in perm])
    assert got.tolist() == pytest.approx([base[i].item() for i in perm], abs=1e-5)


def test_no_fp32_precision_loss_at_large_offsets():
    """The gather must survive s_i ~ 4096, where gamma**-s would be ~1e18."""
    offset, n = 4096, 4
    starts = torch.arange(offset, offset + n * H, H, dtype=torch.long)
    lengths = torch.full((n,), H, dtype=torch.long)
    rewards = torch.zeros(offset + n * H)
    rewards[starts[-1] : starts[-1] + H] = 1.0  # only the last chunk has reward.
    adv, _ = compute_chunk_advantages(
        rewards, torch.zeros(n), torch.zeros(n), starts, lengths,
        chunk_terminated=torch.zeros(n, dtype=torch.bool),
        chunk_size=H, gamma=GAMMA, gae_lambda=0.0,
    )
    assert adv[-1].item() == pytest.approx(sum(GAMMA**k for k in range(H)), rel=1e-5)
    # Every earlier chunk has no reward and no bootstrap, so it is exactly zero.
    assert adv[:-1].abs().max().item() == 0.0


def test_values_are_per_chunk_not_per_step():
    """Value arrays are n_chunks long, gathered by the caller."""
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
        chunk_size=H, gamma=GAMMA, gae_lambda=0.0,
    )
    want = [
        sum(GAMMA**k for k in range(h)) + GAMMA**h * nv - q
        for h, q, nv in zip(lengths, (1.0, 2.0, 3.0), (-1.0, -2.0, -3.0))
    ]
    assert adv.tolist() == pytest.approx(want, abs=1e-5)


# GAE trace.


def test_lambda_zero_is_bit_identical_to_the_qc_path():
    """lambda=0.0 is bit-identical to the QC call that omits the trace args."""
    lengths = [H, 2, 3, H]
    lengths_t = torch.tensor(lengths)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), lengths_t.cumsum(0)[:-1]])
    qc = compute_chunk_advantages(
        torch.full((int(lengths_t.sum()),), 0.7), torch.full((4,), 1.0),
        torch.full((4,), 2.5), starts, lengths_t,
        chunk_terminated=torch.zeros(4, dtype=torch.bool),
        chunk_size=H, gamma=GAMMA, gae_lambda=0.0,
    )
    traced = gae(lengths, lam=0.0, v_current=1.0, v_next=2.5, reward=0.7)
    for a, b in zip(qc, traced):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_lambda_zero_returns_before_running_the_recursion():
    """lambda=0.0 returns before the recursion runs, so a broken successor map cannot hide."""
    lengths = [H] * 3
    # env ids deliberately wrong (interleaved, none matching buffer order).
    adv, _ = gae(lengths, env_id=[0, 1, 0], lam=0.0, v_next=2.0)
    qc_adv, _ = gae(lengths, v_next=2.0)
    torch.testing.assert_close(adv, qc_adv, rtol=0, atol=0)


def test_trace_matches_the_reference_recursion_by_hand():
    """Recompute the trace in Python from its definition, not from the successor map."""
    lengths = [H, H, H, H]
    terminated = [False, False, False, False]
    truncated = [False, False, True, False]
    lam = 0.9
    # Distinct per-chunk values so an index mix-up cannot hide.
    q = torch.tensor([0.0, 1.0, 2.0, 3.0])
    q_next = torch.tensor([0.5, 1.5, 2.5, 3.5])
    lengths_t = torch.tensor(lengths)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), lengths_t.cumsum(0)[:-1]])
    got, _ = compute_chunk_advantages(
        torch.ones(int(lengths_t.sum())), q, q_next, starts, lengths_t,
        chunk_terminated=torch.tensor(terminated),
        chunk_size=H, gamma=GAMMA,
        chunk_truncated=torch.tensor(truncated),
        chunk_prev=prev_chain([0, 0, 0, 0]),
        gae_lambda=lam,
    )
    r = [sum(GAMMA**k for k in range(h)) for h in lengths]
    delta = [
        r[i] + GAMMA ** lengths[i] * q_next[i].item() - q[i].item()
        for i in range(4)
    ]
    want = [0.0] * 4
    for i in range(3, -1, -1):           # single env, so succ is i+1.
        nxt = i + 1 if i + 1 < 4 else None
        trace = (
            GAMMA ** lengths[i] * lam
            * (1.0 - terminated[i]) * (1.0 - truncated[i])
        )
        want[i] = delta[i] + (trace * want[nxt] if nxt is not None else 0.0)
    assert got.tolist() == pytest.approx(want, abs=1e-5)


def test_trace_discount_is_gamma_h_times_lambda_not_gamma_lambda_to_the_h():
    """Pins the macro-MDP convention: lambda decays once per chunk, not per env step."""
    # Two chunks, one env, bootstrap value zero, so only the trace propagates.
    lengths = [H] * 2
    got, _ = gae(lengths, lam=0.95, v_next=0.0)
    r = sum(GAMMA**k for k in range(H))
    trace = GAMMA**H * 0.95
    # delta_0 = r, delta_1 = r, so A_0 = r + trace * r.
    assert got[0].item() == pytest.approx(r + trace * r, abs=1e-5)
    # The (gamma*lambda)^h reading gives a strictly smaller trace here.
    alt = (GAMMA * 0.95) ** H
    assert abs(alt - trace) > 1e-3, "conventions coincide; test cannot distinguish"


def test_truncation_stops_the_trace_but_keeps_the_bootstrap():
    """A timeout is not an episode end: bootstrap yes, trace no."""
    lengths = [H] * 3
    adv, ret = gae(lengths, truncated=[False, True, False], lam=0.9, v_next=2.0)
    r = sum(GAMMA**k for k in range(H))
    # Chunk 1 is truncated: its bootstrap survives, so its target still carries gamma^h * Q'.
    assert ret[1].item() == pytest.approx(r + GAMMA**H * 2.0, abs=1e-5)
    two_free = gae([H] * 2, lam=0.9, v_next=2.0)[0]
    assert adv[2].item() == pytest.approx(two_free[1].item(), abs=1e-5)


def test_trace_returns_the_input_dtype_and_device():
    """The host loop must hand back a tensor the caller can add to its own."""
    delta = torch.tensor([1.0, 2.0])
    trace = torch.tensor([0.5, 0.5])
    successor = torch.tensor([1, -1])
    out = gae_trace(delta, trace, successor)
    assert out.dtype == delta.dtype
    assert out.device == delta.device
    assert out.tolist() == pytest.approx([2.0, 2.0])


def test_termination_still_zeroes_the_bootstrap_and_the_trace():
    lengths = [H] * 3
    _, ret = gae(lengths, terminated=[False, True, False], lam=0.9, v_next=2.0)
    r = sum(GAMMA**k for k in range(H))
    # No bootstrap, and nothing to propagate: the advantage is the reward sum.
    assert ret[1].item() == pytest.approx(r, abs=1e-5)


def test_chunks_from_different_envs_do_not_interact():
    """Rescaling env 1's rewards leaves env 0 bit-identical."""
    lengths = [H] * 6
    env_id = [0, 1, 0, 1, 0, 1]
    lam = 0.95
    base, _ = gae(lengths, env_id=env_id, lam=lam, v_next=1.0)
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
        chunk_prev=prev_chain(env_id),
        gae_lambda=lam,
    )
    for i in (0, 2, 4):
        assert base[i].item() == pytest.approx(alt[i].item(), abs=1e-6), (
            f"env 0 chunk {i} moved when env 1's rewards changed"
        )


def test_successor_is_the_next_chunk_of_the_same_env_not_the_next_buffer_row():
    """Buffer row 0's successor is row 2, not row 1."""
    lengths = [H] * 3
    env_id = [0, 1, 0]
    got, _ = gae(lengths, env_id=env_id, lam=0.95, v_next=0.0)
    r = sum(GAMMA**k for k in range(H))
    # Chunk 0 -> chunk 2 (env 0).
    assert got[0].item() == pytest.approx(r + (GAMMA**H * 0.95) * r, abs=1e-5)
    # Chunk 2 is last in its env, so it has no trace successor.
    assert got[2].item() == pytest.approx(r, abs=1e-5)


def test_last_chunk_of_each_env_gets_a_one_chunk_advantage():
    lengths = [H] * 4
    env_id = [0, 1, 0, 1]
    got, _ = gae(lengths, env_id=env_id, lam=0.95, v_next=0.0)
    qc, _ = gae(lengths, v_next=0.0)
    # Chunks 2 and 3 end their env, so their traces stop: 0->2, 1->3.
    for i in (2, 3):
        assert got[i].item() == pytest.approx(qc[i].item(), abs=1e-6)


def test_advantage_is_monotone_in_lambda_between_delta_and_the_full_return():
    """Higher lambda moves credit further back, never past the true return."""
    lengths = [H] * 4
    endpoint = gae(lengths, lam=0.0, v_next=0.0)[0]
    got, _ = gae(lengths, lam=0.95, v_next=0.0)
    assert (got[:-1] >= endpoint[:-1] - 1e-5).all(), "lambda shrank the advantage"
    # lambda=1 and a zero bootstrap value makes it the full discounted return.
    full, _ = gae(lengths, lam=1.0, v_next=0.0)
    want = sum(
        sum(GAMMA ** (i * H + k) for k in range(H)) for i in range(4)
    )
    assert full[0].item() == pytest.approx(want, abs=1e-4)


def test_trace_requires_its_inputs():
    """Ask for the trace mask rather than cross envs silently."""
    common = dict(
        step_rewards=torch.ones(H),
        v_current=torch.zeros(1),
        v_next=torch.zeros(1),
        chunk_starts=torch.zeros(1, dtype=torch.long),
        chunk_lengths=torch.tensor([H]),
        chunk_terminated=torch.zeros(1, dtype=torch.bool),
        chunk_size=H,
        gamma=GAMMA,
        gae_lambda=0.95,
    )
    with pytest.raises(ValueError, match="chunk_truncated is required"):
        compute_chunk_advantages(**common)
    with pytest.raises(ValueError, match="chunk_prev is required"):
        compute_chunk_advantages(
            chunk_truncated=torch.zeros(1, dtype=torch.bool), **common
        )


@pytest.mark.parametrize("name,value", [("gamma", 1.5), ("gae_lambda", 1.5)])
def test_rejects_an_out_of_range_discount_or_lambda(name, value):
    with pytest.raises(ValueError, match=name):
        compute_chunk_advantages(
            torch.ones(H), torch.zeros(1), torch.zeros(1),
            torch.zeros(1, dtype=torch.long), torch.tensor([H]),
            chunk_terminated=torch.zeros(1, dtype=torch.bool),
            chunk_size=H, **{"gamma": GAMMA, "gae_lambda": 0.0, name: value},
        )


@pytest.mark.parametrize("lam", [0.0, 0.95])
def test_an_empty_rollout_returns_empty_tensors(lam):
    adv, ret = gae([], lam=lam)
    assert adv.numel() == 0 and ret.numel() == 0


# Validation.


def test_rejects_value_arrays_of_the_wrong_length():
    """A per-step-sized value array must raise, not silently misindex."""
    n = 3
    common = dict(
        step_rewards=torch.ones(16),
        chunk_starts=torch.arange(0, n * 4, 4, dtype=torch.long),
        chunk_lengths=torch.full((n,), 4, dtype=torch.long),
        chunk_terminated=torch.zeros(n, dtype=torch.bool),
        chunk_size=4,
        gamma=GAMMA,
        gae_lambda=0.95,
    )
    with pytest.raises(ValueError, match="chunk tensors must have"):
        compute_chunk_advantages(
            v_current=torch.zeros(16), v_next=torch.zeros(n), **common
        )
    with pytest.raises(ValueError, match="chunk tensors must have"):
        compute_chunk_advantages(
            v_current=torch.zeros(n), v_next=torch.zeros(16), **common
        )


def test_rejects_mismatched_chunk_lengths():
    with pytest.raises(ValueError, match="chunk tensors must have"):
        compute_chunk_advantages(
            torch.ones(H), torch.zeros(2), torch.zeros(2),
            torch.zeros(2, dtype=torch.long), torch.tensor([H]),
            chunk_terminated=torch.zeros(2, dtype=torch.bool),
            chunk_size=H, gamma=GAMMA, gae_lambda=0.95,
        )


def test_discount_and_lambda_have_no_defaults():
    """The config owns the 0.99/0.95 defaults; the function must not restate them.

    It used to carry its own defaults, which could silently disagree with
    ``MyfpoConfig``. A call without them is a TypeError now.
    """
    params = inspect.signature(compute_chunk_advantages).parameters
    assert params["gamma"].default is inspect.Parameter.empty
    assert params["gae_lambda"].default is inspect.Parameter.empty
