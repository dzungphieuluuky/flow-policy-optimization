"""Tests for chunk-level GAE (``sjfpo.gae``)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.gae import compute_chunk_advantages

GAMMA, LAM, H = 0.99, 0.95, 4


def reference(env_id, lengths, gamma=GAMMA, lam=LAM, reward=1.0, value=0.0, next_value=2.0):
    """Explicit per-env recursion, emitted in buffer order."""
    out = [0.0] * len(lengths)
    for e in sorted(set(env_id)):
        idx = [i for i, x in enumerate(env_id) if x == e]
        acc = 0.0
        for j in range(len(idx) - 1, -1, -1):
            i = idx[j]
            h = lengths[i]
            delta = sum(gamma**k * reward for k in range(h)) + gamma**h * next_value - value
            acc = delta if j == len(idx) - 1 else delta + gamma**h * lam * acc
            out[i] = acc
    return out


def run(lengths, env_id, terminated=None, truncated=None, value=0.0, next_value=2.0,
        reward=1.0, total_steps=None, chunk_size=H):
    lengths = torch.tensor(lengths, dtype=torch.long)
    n = lengths.numel()
    starts = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]])
    total = int(total_steps if total_steps is not None else lengths.sum())
    zeros = lambda: torch.zeros(n, dtype=torch.bool)  # noqa: E731
    # Rewards are per executed step; the two value arrays are per chunk, already
    # gathered at the chunk's first and post-chunk state by the caller.
    return compute_chunk_advantages(
        torch.full((total,), float(reward)),
        torch.full((n,), float(value)),
        torch.full((n,), float(next_value)),
        starts, lengths,
        chunk_terminated=zeros() if terminated is None else torch.tensor(terminated),
        chunk_truncated=zeros() if truncated is None else torch.tensor(truncated),
        chunk_env_id=torch.tensor(env_id, dtype=torch.long),
        chunk_size=chunk_size, gamma=GAMMA, lam=LAM,
    )


def test_single_trajectory_matches_closed_form():
    adv, ret = run([H] * 3, [0, 0, 0])
    assert adv.tolist() == pytest.approx(reference([0, 0, 0], [H] * 3), abs=1e-5)
    assert torch.allclose(ret, adv)


def test_wavefront_interleave_isolates_envs():
    """The bug this guards: adjacency-masking finds no links in wavefront order."""
    lengths, env_id = [H] * 6, [0, 1, 2, 0, 1, 2]
    adv, _ = run(lengths, env_id)
    assert adv.tolist() == pytest.approx(reference(env_id, lengths), abs=1e-5)
    # An all-same-env buffer of the same total length must differ, proving the
    # env_id argument is actually load-bearing.
    flat, _ = run(lengths, [0] * 6)
    assert not torch.allclose(adv, flat)


def test_uneven_dropout():
    lengths, env_id = [H] * 6, [0, 1, 2, 1, 0, 2]
    adv, _ = run(lengths, env_id)
    assert adv.tolist() == pytest.approx(reference(env_id, lengths), abs=1e-5)


def test_permutation_invariance():
    """Regrouping the buffer by env must permute the advantages identically.

    The permutation has to preserve each env's temporal order, otherwise it
    describes a different trajectory rather than a reordering of the same one.
    """
    lengths = [H] * 6
    env_id = [0, 1, 2, 0, 1, 2]
    base, _ = run(lengths, env_id)
    perm = [0, 3, 1, 4, 2, 5]  # wavefront -> grouped by env
    got, _ = run([lengths[i] for i in perm], [env_id[i] for i in perm])
    assert got.tolist() == pytest.approx([base[i] for i in perm], abs=1e-5)


def test_terminated_zeroes_bootstrap_and_breaks_forward_trace():
    # reward sum only, and reward sum + discounted bootstrap
    r = sum(GAMMA**k for k in range(H))
    boot = r + GAMMA**H * 2.0
    c = GAMMA**H * LAM
    adv, _ = run([H] * 3, [0, 0, 0], terminated=[False, True, False])
    assert adv[1].item() == pytest.approx(r, abs=1e-5)  # no bootstrap
    assert adv[2].item() == pytest.approx(boot, abs=1e-5)
    # The 1->2 link is cut, but 0->1 is intact, so adv[0] keeps adv[1].
    assert adv[0].item() == pytest.approx(boot + c * r, abs=1e-5)


def test_truncated_keeps_bootstrap_but_breaks_forward_trace():
    r = sum(GAMMA**k for k in range(H))
    boot = r + GAMMA**H * 2.0
    c = GAMMA**H * LAM
    adv, _ = run([H] * 3, [0, 0, 0], truncated=[False, True, False])
    assert adv[1].item() == pytest.approx(boot, abs=1e-5)  # bootstrap kept
    assert adv[2].item() == pytest.approx(boot, abs=1e-5)
    assert adv[0].item() == pytest.approx(boot + c * boot, abs=1e-5)




def test_continuation_uses_own_chunk_length():
    """A_i+1 is anchored at chunk i+1's start, so the factor is gamma**h_i."""
    lengths = [H, H, 3, H]
    adv, _ = run(lengths, [0] * 4)
    assert adv.tolist() == pytest.approx(reference([0] * 4, lengths), abs=1e-5)
    # And it is genuinely not gamma**h_{i+1}: swapping the two differs.
    s, swap = 0.0, []
    for j in range(3, -1, -1):
        delta = sum(GAMMA**k for k in range(lengths[j])) + GAMMA ** lengths[j] * 2.0
        s = delta if j == 3 else delta + GAMMA ** lengths[j + 1] * LAM * s
        swap.append(s)
    assert adv.tolist() != pytest.approx(list(reversed(swap)), abs=1e-6)


def test_variable_lengths_per_env():
    lengths, env_id = [H, 2, 3, H, 1, H], [0, 1, 0, 1, 0, 1]
    adv, _ = run(lengths, env_id)
    assert adv.tolist() == pytest.approx(reference(env_id, lengths), abs=1e-5)


def test_empty_rollout():
    adv, ret = compute_chunk_advantages(
        torch.zeros(0), torch.zeros(0), torch.zeros(0),
        torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
        chunk_terminated=torch.zeros(0, dtype=torch.bool),
        chunk_truncated=torch.zeros(0, dtype=torch.bool),
        chunk_env_id=torch.zeros(0, dtype=torch.long),
        chunk_size=H, gamma=GAMMA, lam=LAM,
    )
    assert adv.numel() == 0 and ret.numel() == 0


def test_no_fp32_precision_loss_at_large_offsets():
    """The gather must survive s_i ~ 4096, where gamma**-s would be ~1e18."""
    offset, n = 4096, 4
    starts = torch.arange(offset, offset + n * H, H, dtype=torch.long)
    lengths = torch.full((n,), H, dtype=torch.long)
    total = offset + n * H
    rewards = torch.zeros(total)
    rewards[starts[-1]: starts[-1] + H] = 1.0  # only the last chunk has reward
    adv, _ = compute_chunk_advantages(
        rewards, torch.zeros(n), torch.zeros(n), starts, lengths,
        chunk_terminated=torch.zeros(n, dtype=torch.bool),
        chunk_truncated=torch.zeros(n, dtype=torch.bool),
        chunk_env_id=torch.zeros(n, dtype=torch.long),
        chunk_size=H, gamma=GAMMA, lam=LAM,
    )
    expected_r = sum(GAMMA**k for k in range(H))
    assert adv[-1].item() == pytest.approx(expected_r, rel=1e-5)
    # Earlier chunks are the same env, so they carry a decayed copy of it.
    c = GAMMA**H * LAM
    assert adv[n - 2].item() == pytest.approx(c * expected_r, rel=1e-5)


def test_rejects_out_of_range_discounts():
    args = dict(
        step_rewards=torch.ones(H), chunk_value=torch.zeros(1),
        chunk_next_value=torch.zeros(1),
        chunk_starts=torch.zeros(1, dtype=torch.long),
        chunk_lengths=torch.tensor([H]),
        chunk_terminated=torch.zeros(1, dtype=torch.bool),
        chunk_truncated=torch.zeros(1, dtype=torch.bool),
        chunk_env_id=torch.zeros(1, dtype=torch.long), chunk_size=H,
    )
    for bad in ({"gamma": 1.5}, {"lam": -0.1}):
        with pytest.raises(ValueError):
            compute_chunk_advantages(**args, **bad)


def test_rejects_mismatched_chunk_tensors():
    with pytest.raises(ValueError, match="chunk_env_id"):
        compute_chunk_advantages(
            torch.ones(H), torch.zeros(2), torch.zeros(2),
            torch.zeros(2, dtype=torch.long), torch.tensor([H, H]),
            chunk_terminated=torch.zeros(2, dtype=torch.bool),
            chunk_truncated=torch.zeros(2, dtype=torch.bool),
            chunk_env_id=torch.zeros(3, dtype=torch.long), chunk_size=H,
        )


def test_values_are_per_chunk_not_per_step():
    """The value arrays are gathered by the caller, so they are n_chunks long.

    n_chunks must differ from the step count for this to be a real check, and
    the per-chunk values must vary, or a per-step array of the right length
    would pass by coincidence.
    """
    lengths, env_id = [H, 2, 3], [0, 0, 0]
    total = sum(lengths)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), torch.tensor(lengths).cumsum(0)[:-1]])
    per_chunk_v = torch.tensor([1.0, 2.0, 3.0])
    per_chunk_nv = torch.tensor([-1.0, -2.0, -3.0])
    rewards = torch.ones(total)
    adv, _ = compute_chunk_advantages(
        rewards, per_chunk_v, per_chunk_nv, starts, torch.tensor(lengths),
        chunk_terminated=torch.zeros(3, dtype=torch.bool),
        chunk_truncated=torch.zeros(3, dtype=torch.bool),
        chunk_env_id=torch.zeros(3, dtype=torch.long),
        chunk_size=H, gamma=GAMMA, lam=LAM,
    )
    # Recompute the recursion by hand with the per-chunk values.
    exp = [0.0] * 3
    acc = 0.0
    for j in (2, 1, 0):
        h = lengths[j]
        delta = sum(GAMMA**k for k in range(h)) + GAMMA**h * per_chunk_nv[j].item() - per_chunk_v[j].item()
        acc = delta if j == 2 else delta + GAMMA**h * LAM * acc
        exp[j] = acc
    assert adv.tolist() == pytest.approx(exp, abs=1e-5)


def test_rejects_value_arrays_of_the_wrong_length():
    """A per-step-sized value array must raise, not silently misindex."""
    n = 3
    common = dict(
        step_rewards=torch.ones(16),
        chunk_starts=torch.arange(0, n * 4, 4, dtype=torch.long),
        chunk_lengths=torch.full((n,), 4, dtype=torch.long),
        chunk_terminated=torch.zeros(n, dtype=torch.bool),
        chunk_truncated=torch.zeros(n, dtype=torch.bool),
        chunk_env_id=torch.zeros(n, dtype=torch.long),
        chunk_size=4,
    )
    with pytest.raises(ValueError, match="chunk_value"):
        compute_chunk_advantages(
            chunk_value=torch.zeros(16), chunk_next_value=torch.zeros(n), **common
        )
    with pytest.raises(ValueError, match="chunk_next_value"):
        compute_chunk_advantages(
            chunk_value=torch.zeros(n), chunk_next_value=torch.zeros(16), **common
        )


def test_returns_equal_advantages_plus_values_in_raw_space():
    """returns = advantages + values must hold in the same units the critic emits.

    This held inside compute_chunk_advantages but not in generate_rollout, which
    used to normalize the critic output before comparing it against raw returns.
    That silently made the identity false and left the baseline an order of
    magnitude smaller than the reward term, so the advantage degenerated to "raw
    reward" with no per-state baseline.
    """
    import torch

    from sjfpo.gae import compute_chunk_advantages

    chunk_size, n_chunks = 4, 2
    rewards = torch.arange(1, 2 * chunk_size + 1, dtype=torch.float32)
    chunk_value = torch.tensor([0.0, 5.0])
    chunk_next_value = torch.tensor([0.0, 5.0])
    advantages, returns = compute_chunk_advantages(
        rewards,
        chunk_value,
        chunk_next_value,
        torch.tensor([0, chunk_size]),
        torch.tensor([chunk_size, chunk_size]),
        chunk_terminated=torch.tensor([False, True]),
        chunk_truncated=torch.tensor([False, False]),
        chunk_env_id=torch.tensor([0, 0]),
        chunk_size=chunk_size,
        gamma=0.99,
        lam=0.95,
    )
    torch.testing.assert_close(returns, advantages + chunk_value)


def test_baseline_subtraction_scales_with_the_value():
    """A higher V must lower the advantage, and by the same amount it raised V.

    A unit-scale value sitting next to multi-unit rewards cannot do this, which is
    exactly what the old normalized critic produced: the baseline contributed
    almost nothing and the advantage was effectively the raw reward.
    """
    import torch

    from sjfpo.gae import compute_chunk_advantages

    def advantage_for(value: float) -> torch.Tensor:
        advantages, _ = compute_chunk_advantages(
            torch.tensor([5.0, 6.0, 7.0, 8.0]),
            torch.tensor([value]),
            torch.tensor([value]),
            torch.tensor([0]),
            torch.tensor([4]),
            chunk_terminated=torch.tensor([True]),  # no bootstrap, so exact
            chunk_truncated=torch.tensor([False]),
            chunk_env_id=torch.tensor([0]),
            chunk_size=4,
            gamma=0.99,
            lam=0.95,
        )
        return advantages

    raw = advantage_for(0.0)
    raised = advantage_for(9.0)
    torch.testing.assert_close(raw - raised, torch.tensor([9.0]))
    # The reward term alone is 25.56, so a 9-unit baseline is a third of the
    # signal. Under the old wiring this term was O(1) against O(5) rewards.
    assert float(raw) > 0 and float(raised) < float(raw)
