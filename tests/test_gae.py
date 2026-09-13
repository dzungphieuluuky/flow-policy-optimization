"""Unit tests for src/sjfpo/gae.py (plan task 7).

TDD: written BEFORE the implementation exists — the first pytest run must
fail RED on the missing module, then go GREEN once gae.py lands.

BINDING semantics (plan Scope §Chunk-level semi-MDP GAE, Metis majors #8/#11):
each executed chunk is ONE semi-MDP decision epoch with
    R_k     = sum_i gamma^i * r_{chunk k, step i}      (i over EXECUTED steps only)
    delta_k = R_k + gamma^len_k * V_{k+1} * (1 - terminated_k) - V_k
    A_k     = delta_k + (gamma*lam)^len_k * (1 - terminated_k) * A_{k+1}
    returns = A + V ; advantages centered afterwards.

V_{k+1} is the value of the observation AFTER chunk k, i.e. the value at the
START of chunk k+1 for a continuing stream (the API takes `values[N]` plus a
single `final_value` bootstrap for the end of the stream — no redundant
parallel next-values array).

Mid-chunk termination: R_k sums ONLY executed steps, V-bootstrap = 0, and the
lambda-recursion is cut at that epoch. Truncated (TimeLimit) chunks DO
bootstrap from V(next obs). gamma compounds ONCE per chunk transition
(gamma^len_k), never per-step across the delta/bootstrap path.

The brute-force oracle below is deliberately written as an independent
FORWARD expansion over epochs (no shared code with the backward-recursion
implementation): T_k = sum_{j>=k} (prod_{m=k}^{j-1} c_m) *
[R_j + gamma^Lj (1-t_j)(1-lam^Lj) V_{j+1}],  c_m = (gamma*lam)^Lm (1-t_m),
with advantages = T - V. Equality with the implementation to <1e-6 pins the
semantics end-to-end.

Pure numpy / fp64; chunk-GAE consumes per-chunk V as black-box scalars
(history conditioning of the critic is out of scope here by design).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.gae import chunk_gae  # noqa: E402

GAMMA = 0.99
LAM = 0.95


# ---------------------------------------------------------------------------
# Brute-force oracle: forward-unrolled lambda-return over decision epochs.
# Shares ZERO code with the implementation (which recurses backwards).
# ---------------------------------------------------------------------------
def oracle_returns(
    per_step_rewards, values, final_value, n_action_steps, terminated, gamma, lam
):
    """Explicitly-unrolled return computation over semi-MDP epochs."""
    n = len(values)
    offsets = []
    start = 0
    for length in n_action_steps:
        offsets.append(start)
        start += length

    def v_next(j):  # V_{j+1}: start of chunk j+1, or the stream-end bootstrap
        return values[j + 1] if j + 1 < n else final_value

    returns = [0.0] * n
    for k in range(n):
        # Forward accumulation of every future epoch j >= k's contribution.
        total = 0.0
        prod_c = 1.0  # prod_{m=k}^{j-1} c_m
        for j in range(k, n):
            # R_j from RAW per-step rewards, discounted internally by gamma^i.
            r_j = 0.0
            for i in range(n_action_steps[j]):
                r_j += (gamma**i) * per_step_rewards[offsets[j] + i]
            lj = n_action_steps[j]
            if terminated[j]:
                b_j = r_j  # no bootstrap, nothing beyond a terminated epoch
            else:
                b_j = r_j + (gamma**lj) * (1.0 - lam**lj) * v_next(j)
            total += prod_c * b_j
            if terminated[j]:
                break
            prod_c *= (gamma * lam) ** lj
        returns[k] = total
    return np.array(returns, dtype=np.float64)


def random_stream(seed=7, n_chunks=40):
    """Random 40-chunk stream with known rewards/values/done-kinds.

    Includes a MID-CHUNK TERMINATION (chunk 17 dies after 3 of 8 steps),
    a mid-chunk TRUNCATION (chunk 23 fires TimeLimit after 5 steps), and a
    terminating final chunk. Variable executed lengths exercise gamma^len_k
    compounding.
    """
    rng = np.random.default_rng(seed)
    n_action_steps = [8] * n_chunks
    terminated = [False] * n_chunks
    n_action_steps[17] = 3
    terminated[17] = True  # mid-chunk DEATH: only 3 executed steps
    n_action_steps[23] = 5  # mid-chunk TimeLimit truncation, bootstrap-eligible
    terminated[39] = True  # final chunk terminates the stream
    rewards = rng.normal(0.0, 1.0, size=sum(n_action_steps))
    values = rng.uniform(-1.0, 2.0, size=n_chunks)
    final_value = float(rng.uniform(-1.0, 2.0))
    return rewards, values, final_value, n_action_steps, terminated


def test_bruteforce_mc_equality_random_stream():
    """Chunk-GAE output == explicitly-unrolled return computation (<1e-6)."""
    rewards, values, final_value, n_action_steps, terminated = random_stream(seed=7)

    adv, ret = chunk_gae(
        rewards,
        values,
        final_value,
        n_action_steps,
        terminated,
        gamma=GAMMA,
        lam=LAM,
        normalize=False,
    )

    oracle_ret = oracle_returns(
        rewards, values, final_value, n_action_steps, terminated, GAMMA, LAM
    )
    oracle_adv = oracle_ret - values

    adv_diff = float(np.max(np.abs(adv - oracle_adv)))
    ret_diff = float(np.max(np.abs(ret - oracle_ret)))
    print(
        f"\nbrute-force max-abs-diff advantages={adv_diff:.3e} returns={ret_diff:.3e}"
    )

    assert adv_diff < 1e-6, f"advantage mismatch {adv_diff}"
    assert ret_diff < 1e-6, f"return mismatch {ret_diff}"


def test_mid_chunk_death_semantics():
    """Death mid-chunk: R sums ONLY executed steps, V-bootstrap = 0 there,
    and the PREVIOUS epoch still recurses (its own flag is what matters)."""
    gamma, lam = GAMMA, LAM
    r = np.array(
        [0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.1, 1.2, 1.3]
    )  # chunk 0: 8 steps; chunk 1: dies after 3
    values = np.array([0.5, -0.25])
    final_value = 12345.0  # huge: must be MASKED because chunk 1 terminates
    lens = [8, 3]
    terminated = [False, True]

    adv, ret = chunk_gae(
        r, values, final_value, lens, terminated, gamma=gamma, lam=lam, normalize=False
    )

    # R_1 uses ONLY the 3 executed steps, internally discounted gamma^i.
    r1_expected = 1.1 + gamma * 1.2 + gamma**2 * 1.3
    # Terminated epoch: no bootstrap, no recursion -> A_1 = R_1 - V_1 exactly.
    a1_expected = r1_expected - values[1]
    np.testing.assert_allclose(adv[1], a1_expected, atol=1e-12)

    # Preceding alive epoch recurses into A_1 with (gamma*lam)^8 (its own
    # terminated flag is False) and bootstraps V_1 = values[1] with gamma^8.
    r0_expected = sum(gamma**i * ri for i, ri in enumerate(r[:8]))
    a0_expected = (r0_expected + gamma**8 * values[1] - values[0]) + (
        gamma * lam
    ) ** 8 * a1_expected
    np.testing.assert_allclose(adv[0], a0_expected, atol=1e-12)

    np.testing.assert_allclose(ret, adv + values, atol=1e-12)


def test_discount_compounding_gamma_once_per_chunk():
    """gamma is applied ONCE per chunk transition (gamma^len_k on the
    bootstrap, (gamma*lam)^len_k on the recursion); R_k is internally
    discounted gamma^i."""
    gamma, lam = GAMMA, LAM
    rng = np.random.default_rng(11)
    r = rng.normal(0.0, 1.0, size=16)
    values = np.array([0.3, -0.7])
    final_value = 2.2

    # Fixed-length variant: both chunks len 8, none terminated -> the last
    # chunk bootstraps the stream-end value.
    adv, _ = chunk_gae(
        r,
        values,
        final_value,
        [8, 8],
        [False, False],
        gamma=gamma,
        lam=lam,
        normalize=False,
    )
    r0 = sum(gamma**i * ri for i, ri in enumerate(r[:8]))
    r1 = sum(gamma**i * ri for i, ri in enumerate(r[8:16]))
    a1 = r1 + gamma**8 * final_value - values[1]
    a0 = r0 + gamma**8 * values[1] - values[0] + (gamma * lam) ** 8 * a1
    np.testing.assert_allclose(adv, [a0, a1], atol=1e-12)

    # Variable-length variant: first chunk executes only 5 steps ->
    # multipliers become gamma^5 / (gamma*lam)^5 for THAT transition.
    r5 = rng.normal(0.0, 1.0, size=13)
    adv5, _ = chunk_gae(
        r5,
        values,
        final_value,
        [5, 8],
        [False, False],
        gamma=gamma,
        lam=lam,
        normalize=False,
    )
    r0_5 = sum(gamma**i * ri for i, ri in enumerate(r5[:5]))
    r1_5 = sum(gamma**i * ri for i, ri in enumerate(r5[5:13]))
    a1_5 = r1_5 + gamma**8 * final_value - values[1]
    a0_5 = r0_5 + gamma**5 * values[1] - values[0] + (gamma * lam) ** 5 * a1_5
    np.testing.assert_allclose(adv5[0], a0_5, atol=1e-12)
    np.testing.assert_allclose(adv5[1], a1_5, atol=1e-12)

    # R_k really is gamma^i-discounted internally: with V_0 = bootstrap = 0,
    # A_0 == R_0 and must DIFFER from the undiscounted reward sum.
    zeros_v = np.zeros(1)
    adv_disc, _ = chunk_gae(
        np.array([1.0, 1.0, 1.0, 1.0]),
        zeros_v,
        0.0,
        [4],
        [True],
        gamma=gamma,
        lam=lam,
        normalize=False,
    )
    np.testing.assert_allclose(adv_disc[0], sum(gamma**i for i in range(4)), atol=1e-12)
    assert abs(adv_disc[0] - 4.0) > 1e-3, "discounting inside R_k not applied"


def test_truncated_bootstraps_terminated_does_not():
    """Truncated/ongoing chunks DO bootstrap from V(next obs); terminated do
    not (a huge bootstrap candidate must stay fully masked there)."""
    gamma, lam = GAMMA, LAM
    r = np.array(
        [
            0.5,
            -0.3,
            0.2,
            0.9,
            0.4,
            -0.6,
            0.1,
            0.7,
            0.2,
            -0.1,
            0.3,
            0.5,
            0.8,
            -0.4,
            0.6,
            0.0,
        ]
    )
    values = np.array([0.1, 0.4])
    big_final = 1e6  # any leakage of this into terminated paths explodes the diff
    common = dict(gamma=gamma, lam=lam, normalize=False)
    lens = [8, 8]

    # --- second chunk TERMINATES: no stream-end bootstrap, recursion cut ---
    adv_term, _ = chunk_gae(r, values, big_final, lens, [False, True], **common)
    r0 = sum(gamma**i * ri for i, ri in enumerate(r[:8]))
    r1 = sum(gamma**i * ri for i, ri in enumerate(r[8:16]))
    # chunk 1 terminated: A_1 = R_1 - V_1 despite final_value = 1e6
    np.testing.assert_allclose(adv_term[1], r1 - values[1], atol=1e-9)
    # chunk 0 recursed into A_1 (alive transition) and bootstrapped V_1
    a1_term = r1 - values[1]
    np.testing.assert_allclose(
        adv_term[0],
        r0 + gamma**8 * values[1] - values[0] + (gamma * lam) ** 8 * a1_term,
        atol=1e-9,
    )

    # --- same numbers, second chunk TRUNCATED instead: bootstrap flows ---
    adv_trunc, _ = chunk_gae(r, values, big_final, lens, [False, False], **common)
    a1_trunc = r1 + gamma**8 * big_final - values[1]
    np.testing.assert_allclose(adv_trunc[1], a1_trunc, atol=1e-6)
    assert abs(adv_trunc[1] - adv_term[1]) > 1.0, "truncated chunk did not bootstrap"
    # predecessor sees the difference too (recursion continues through truncation)
    assert abs(adv_trunc[0] - adv_term[0]) > 1.0


def test_advantages_centered_afterward():
    """Advantages are standardized (mean ~= 0, std ~= 1) by default;
    returns are unaffected by normalization."""
    rewards, values, final_value, n_action_steps, terminated = random_stream(seed=3)

    adv_c, ret_c = chunk_gae(
        rewards,
        values,
        final_value,
        n_action_steps,
        terminated,
        gamma=GAMMA,
        lam=LAM,
        normalize=True,
    )
    assert abs(float(np.mean(adv_c))) < 1e-9
    assert abs(float(np.std(adv_c)) - 1.0) < 1e-6

    adv_r, ret_r = chunk_gae(
        rewards,
        values,
        final_value,
        n_action_steps,
        terminated,
        gamma=GAMMA,
        lam=LAM,
        normalize=False,
    )
    # Normalization standardizes raw advantages; returns identical.
    np.testing.assert_allclose(
        adv_c, (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8), atol=1e-12
    )
    np.testing.assert_allclose(ret_c, ret_r, atol=1e-12)
    assert abs(float(np.mean(adv_r))) > 1e-6  # unnormalized stream is off-center


def test_input_validation():
    """Basic contract checks: mismatched lengths / bad chunk lens raise."""
    with pytest.raises(ValueError):
        chunk_gae(np.zeros(10), np.zeros(2), 0.0, [8, 8], [False, False])
    with pytest.raises(ValueError):
        chunk_gae(np.zeros(16), np.zeros(2), 0.0, [8, 0], [False, False])
    with pytest.raises(ValueError):
        chunk_gae(np.zeros(16), np.zeros(2), 0.0, [8, 8], [False, True, False])
