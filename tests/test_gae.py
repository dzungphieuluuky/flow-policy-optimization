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
    return compute_chunk_advantages(
        torch.full((total,), float(reward)),
        torch.full((total,), float(value)),
        torch.full((total,), float(next_value)),
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
        rewards, torch.zeros(total), torch.zeros(total), starts, lengths,
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
        step_rewards=torch.ones(H), step_values=torch.zeros(H),
        step_next_values=torch.zeros(H),
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
            torch.ones(H), torch.zeros(H), torch.zeros(H),
            torch.zeros(2, dtype=torch.long), torch.tensor([H, H]),
            chunk_terminated=torch.zeros(2, dtype=torch.bool),
            chunk_truncated=torch.zeros(2, dtype=torch.bool),
            chunk_env_id=torch.zeros(3, dtype=torch.long), chunk_size=H,
        )
