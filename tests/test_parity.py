"""torch vs jax parity for the FPO++ core math, the DCT, and chunk-level GAE.

Both frameworks in one process: the rule that ``sjfpo_jax`` must not import
torch is about the *training* process, and a parity test is the one place the
two have to meet. Every fixed-seed comparison holds at ``rtol=1e-5, atol=1e-6``.
Widen nothing; a failure at 1e-6 is a semantic difference, not a precision one.

Random draws are the exception: torch and jax have different generators, so
``sample_timesteps`` / ``sample_noise`` are compared on their statistics, and
the beta mode only on its law.
"""
import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo_jax import dct_utils as J_dct
from sjfpo_jax import fpo_core as J_fpo
from sjfpo_jax import gae as J_gae
from spectral_joint_policy import dct_utils as T_dct
from spectral_joint_policy import fpo_core as T_fpo
from spectral_joint_policy import gae as T_gae
from spectral_joint_policy.configuration_sjfpo import SJFPOConfig

CFG = SJFPOConfig()
RTOL, ATOL = 1e-5, 1e-6


def cfg_with(**overrides):
    return dataclasses.replace(CFG, **overrides)


def assert_same(torch_out, jax_out):
    """One comparison for every scalar, vector and nested-pair result."""
    if isinstance(torch_out, torch.Tensor):
        torch_out = torch_out.detach().cpu().numpy()
    if isinstance(jax_out, jax.Array):
        jax_out = np.asarray(jax_out)
    if isinstance(torch_out, (tuple, list)):
        assert isinstance(jax_out, (tuple, list)) and len(torch_out) == len(jax_out)
        for t, j in zip(torch_out, jax_out):
            assert_same(t, j)
        return
    assert torch_out.shape == jax_out.shape, (torch_out.shape, jax_out.shape)
    np.testing.assert_allclose(torch_out, jax_out, rtol=RTOL, atol=ATOL)


def fixed(shape, seed=0):
    """One numpy draw handed to both frameworks, so the comparison is deterministic.

    torch and jax have different generators; drawing from each would compare
    two different tensors.
    """
    values = np.random.default_rng(seed).standard_normal(size=shape, dtype=np.float32)
    return torch.from_numpy(values.copy()), jnp.asarray(values)


# ---------------------------------------------------------------- compute_cfm_loss


@pytest.mark.parametrize("reduce_mode", ["scalar", "sample", "token_mean", "both"])
@pytest.mark.parametrize("cfm_loss_reduction", ["mean", "sum", "sqrt"])
def test_cfm_loss_reductions(reduce_mode, cfm_loss_reduction):
    """Every reduction under every divisor, masked and unmasked."""
    cfg = cfg_with(cfm_loss_reduction=cfm_loss_reduction)
    vh_t, vh_j = fixed((5, 3, 8, 2))
    vs_t, vs_j = fixed((5, 3, 8, 2), seed=1)
    lengths = torch.tensor([8, 6, 8, 1, 4])
    pad_t = torch.arange(8).unsqueeze(0) >= lengths.unsqueeze(1)
    pad_j = jnp.asarray(np.asarray(pad_t))

    assert_same(
        T_fpo.compute_cfm_loss(vh_t, vs_t, cfg, reduce_mode),
        J_fpo.compute_cfm_loss(vh_j, vs_j, cfg, reduce_mode),
    )
    assert_same(
        T_fpo.compute_cfm_loss(vh_t, vs_t, cfg, reduce_mode, pad_t),
        J_fpo.compute_cfm_loss(vh_j, vs_j, cfg, reduce_mode, pad_j),
    )


def test_cfm_loss_mask_is_what_drops_the_tail():
    """A mask that changes nothing when no chunk is short proves nothing.

    The mask drops rows *and renormalises* by the executed count, so the value
    is not simply smaller: it is the mean over the executed rows exactly.
    """
    cfg = cfg_with(cfm_loss_reduction="mean")
    _, vh_j = fixed((4, 2, 8, 3))
    _, vs_j = fixed((4, 2, 8, 3), seed=1)
    pad_t = torch.zeros(4, 8, dtype=torch.bool)
    pad_t[1, 5:] = True
    pad_j = jnp.asarray(np.asarray(pad_t))

    masked = J_fpo.compute_cfm_loss(vh_j, vs_j, cfg, "sample", pad_j)
    unmasked = J_fpo.compute_cfm_loss(vh_j, vs_j, cfg, "sample")

    sq = np.square(np.asarray(vh_j) - np.asarray(vs_j))
    executed = np.asarray(~pad_j)
    assert float(masked[1][0]) == pytest.approx(
        float(sq[1, 0][executed[1]].mean()), rel=1e-5
    )
    # Chunk 0 is untruncated, so masking cannot move it.
    assert float(masked[0].sum()) == pytest.approx(float(unmasked[0].sum()), rel=1e-6)
    assert float(masked[1].sum()) != pytest.approx(float(unmasked[1].sum()), rel=1e-3)


# --------------------------------------------------------- compute_aspo_surrogate


@pytest.mark.parametrize("trust_region_mode", ["ppo", "spo", "aspo"])
def test_aspo_surrogate(trust_region_mode):
    """ASPO branches on the advantage sign, so the sweep must straddle zero."""
    cfg = cfg_with(trust_region_mode=trust_region_mode)
    ratio_t, ratio_j = fixed((64,), seed=2)
    adv_t, adv_j = fixed((64,), seed=3)
    assert_same(
        T_fpo.compute_aspo_surrogate(ratio_t, adv_t, cfg),
        J_fpo.compute_aspo_surrogate(ratio_j, adv_j, cfg),
    )
    assert (np.asarray(adv_j) > 0).any() and (np.asarray(adv_j) < 0).any()


def test_aspo_selects_ppo_above_and_spo_below_zero():
    """The jnp.where must pick the right branch on each side of zero."""
    cfg = cfg_with(trust_region_mode="aspo")
    ratio = jnp.array([0.5, 2.0])
    adv = jnp.array([1.0, -1.0])
    aspo = J_fpo.compute_aspo_surrogate(ratio, adv, cfg)
    ppo = J_fpo.compute_aspo_surrogate(ratio, adv, cfg_with(trust_region_mode="ppo"))
    spo = J_fpo.compute_aspo_surrogate(ratio, adv, cfg_with(trust_region_mode="spo"))
    assert aspo[0] == pytest.approx(float(ppo[0]), rel=1e-6)
    assert aspo[1] == pytest.approx(float(spo[1]), rel=1e-6)


# ------------------------------------------------------------ compute_policy_ratio


@pytest.mark.parametrize("with_advantage", [False, True])
def test_policy_ratio(with_advantage):
    cfg = CFG
    l_old_t, l_old_j = fixed((32,), seed=4)
    l_new_t, l_new_j = fixed((32,), seed=5)
    # Push half the samples past cfm_clamp so the clamps and the negative-
    # advantage clamp are actually exercised.
    l_new_t[16:] += cfg.cfm_clamp * 2.0
    l_new_j = l_new_j.at[16:].add(cfg.cfm_clamp * 2.0)
    adv_t = adv_j = None
    if with_advantage:
        adv_t, adv_j = fixed((32,), seed=6)

    assert_same(
        T_fpo.compute_policy_ratio(l_old_t, l_new_t, cfg, advantage=adv_t),
        J_fpo.compute_policy_ratio(l_old_j, l_new_j, cfg, advantage=adv_j),
    )


def test_policy_ratio_gradient_is_exactly_zero_above_the_clamp():
    """The whole ABOVE_CLAMP_ABORT_UPDATES machinery rests on this.

    Above ``cfm_clamp`` the ratio must contribute no gradient at all, so a run
    that stays there is training the critic only. A ``jnp.minimum`` in place of
    a one-sided ``jnp.clip`` is exactly the bug this guards.
    """
    cfg = CFG
    l_old = torch.zeros(4)
    l_new = torch.tensor(
        [1.0, 2.0, cfg.cfm_clamp + 1.0, cfg.cfm_clamp * 3.0], requires_grad=True
    )

    t_grad = torch.autograd.grad(T_fpo.compute_policy_ratio(l_old, l_new, cfg).sum(), l_new)[0]
    j_grad = jax.grad(
        lambda l: J_fpo.compute_policy_ratio(jnp.zeros(4), l, cfg).sum()
    )(jnp.asarray([1.0, 2.0, cfg.cfm_clamp + 1.0, cfg.cfm_clamp * 3.0], dtype=jnp.float32))

    assert_same(t_grad.numpy(), np.asarray(j_grad))
    assert np.asarray(j_grad)[2] == 0.0 and np.asarray(j_grad)[3] == 0.0
    # d(exp(l_old - l_new))/d(l_new) = -ratio, so the live samples pull l_new down.
    assert float(j_grad[0]) < 0.0 and float(j_grad[1]) < 0.0


def test_policy_ratio_log_cap_is_a_straight_through_estimator():
    """Forward uses the capped delta, backward flows through the raw one."""
    cfg = CFG
    l_old = torch.full((2,), cfg.log_ratio_cap + 5.0)
    assert_same(
        T_fpo.compute_policy_ratio(l_old, torch.zeros(2), cfg),
        J_fpo.compute_policy_ratio(jnp.full((2,), cfg.log_ratio_cap + 5.0), jnp.zeros(2), cfg),
    )
    np.testing.assert_allclose(
        np.asarray(J_fpo.compute_policy_ratio(
            jnp.full((2,), cfg.log_ratio_cap + 5.0), jnp.zeros(2), cfg
        )),
        np.exp(cfg.log_ratio_cap),
        rtol=1e-6,
    )


# -------------------------------------------------------- run_monte_carlo_sampling


def test_run_monte_carlo_sampling_shapes():
    b, m, k, d = 4, 3, 8, 2
    a_t, a_j = fixed((b, k, d), seed=7)
    eps_t, eps_j = fixed((b, m, k, d), seed=8)
    tau_t, tau_j = fixed((b, m, k, 1), seed=9)

    t_out = T_fpo.run_monte_carlo_sampling(a_t, eps_t, tau_t)
    j_out = J_fpo.run_monte_carlo_sampling(a_j, eps_j, tau_j)
    assert [tuple(x.shape) for x in t_out[:3]] == [tuple(x.shape) for x in j_out]
    assert_same(t_out[0], j_out[0])
    assert_same(t_out[1], j_out[1])
    assert_same(t_out[2], j_out[2])
    # Row i is chunk i // num_mc, flattened to (K, d), in order.
    rows = np.asarray(j_out[2]).reshape(b * m, k, d)
    assert np.allclose(rows, np.repeat(np.asarray(a_j), m, axis=0))


def test_run_monte_carlo_sampling_rejects_a_non_tokenwise_tau():
    """Only ``[B, M, K, 1]`` is ported; the legacy shapes must fail loudly."""
    a = jnp.zeros((2, 8, 3))
    eps = jnp.zeros((2, 4, 8, 3))
    for bad_tau in (jnp.zeros((2, 4)), jnp.zeros((2, 4, 1)), jnp.zeros((4,))):
        with pytest.raises(ValueError):
            J_fpo.run_monte_carlo_sampling(a, eps, bad_tau)


# ------------------------------------------------------------------ sample_action


def test_sample_action_matches_euler_step_for_step():
    """A dummy field isolates the integrator from the actor.

    The scan body is traced once, so the step count is checked against a numpy
    reference recursion rather than a Python call counter.
    """
    steps = 16
    eps_t, eps_j = fixed((3, 8, 2), seed=10)
    condition = jnp.zeros((3, 5))

    def field(x, t, c):
        return 2.0 * x - jnp.reshape(t, (1, 1, 1))

    torch_field = lambda x, t, c: 2.0 * x - t.reshape(1, 1, 1)

    t_out = T_fpo.sample_action(torch_field, (3, 8, 2), condition, steps, eps_t, "cpu")
    j_out = J_fpo.sample_action(field, eps_j, condition, steps)
    assert_same(t_out, j_out)

    grid = torch.linspace(0.0, 1.0, steps + 1)
    x = eps_t
    for i in range(steps):
        x = torch.addcmul(x, 2.0 * x - grid[i].reshape(1), grid[i + 1] - grid[i])
    np.testing.assert_allclose(np.asarray(j_out), x.numpy(), rtol=1e-5, atol=1e-6)


def test_sample_action_from_zero_noise_is_identity_at_eps_zero():
    """zero_sampling starts the flow at eps=0; the value must stay finite."""
    out = J_fpo.sample_action(lambda x, t, c: jnp.zeros_like(x), jnp.zeros((2, 4, 3)),
                              jnp.zeros((2, 5)), 16)
    assert out.shape == (2, 4, 3)
    assert bool(jnp.all(jnp.isfinite(out)))


# ------------------------------------------------------- timestep / noise sampling


@pytest.mark.parametrize(
    "timestep_sampling_mode", ["uniform", "normal", "fpo_beta", "beta"]
)
def test_sample_timesteps_statistics(timestep_sampling_mode):
    """Draws differ by construction; the law must not.

    The distribution of a 0.99 * U(0,1) + 0.005 draw has mean 0.5 and variance
    (0.99^2)/12, which every mode here shares up to the beta ones' shape.
    """
    cfg = cfg_with(timestep_sampling_mode=timestep_sampling_mode)
    out = J_fpo.sample_timesteps((200000,), cfg, jax.random.key(0))
    lo, hi = cfg.min_timestep, cfg.max_timestep
    assert out.shape == (200000,)
    # fp32 cannot represent 0.005 exactly, so the clamp can land a ulp under.
    assert float(out.min()) >= lo - 1e-6 and float(out.max()) <= hi + 1e-6
    assert float(out.mean()) == pytest.approx(0.5 * (lo + hi), abs=0.02)
    assert float(out.std()) > 0.1


@pytest.mark.parametrize("noise_sampling_mode", ["normal", "uniform"])
def test_sample_noise_statistics(noise_sampling_mode):
    cfg = cfg_with(noise_sampling_mode=noise_sampling_mode)
    out = J_fpo.sample_noise((200000, 16, 2), cfg, jax.random.key(1))
    assert out.shape == (200000, 16, 2)
    if noise_sampling_mode == "normal":
        assert float(out.mean()) == pytest.approx(0.0, abs=0.02)
        assert float(out.std()) == pytest.approx(1.0, abs=0.02)
    else:
        assert float(out.std()) == pytest.approx(1.0, abs=0.02)
        assert float(jnp.abs(out).max()) <= 3.0


def test_freq_weighted_noise_pours_variance_into_the_low_modes():
    """NEED beta=1 coloring, checked on the per-mode std, not on draws."""
    cfg = cfg_with(noise_scaling_mode="freq_weighted")
    out = J_fpo.sample_noise((20000, 16, 8, 2), cfg, jax.random.key(2))
    per_mode = np.asarray(out.std(axis=(0, 1, 3)))
    assert per_mode[0] > per_mode[-1]
    # Axis -2 of the noise is the token axis, so the table is sized by it.
    expected = np.asarray(J_fpo.compute_freq_weighted_std(8, cfg.action_horizon))
    np.testing.assert_allclose(per_mode, expected, rtol=0.05)


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_noise_scaling_beta_hits_both_endpoints_and_the_middle(beta):
    """beta=0 is white, beta>0 tilts down the spectrum, and both frameworks agree."""
    cfg = cfg_with(noise_scaling_mode="freq_weighted", noise_scaling_beta=beta)
    out = T_fpo.sample_noise(
        (20000, 16, 8, 2), cfg, torch.device("cpu"), torch.Generator().manual_seed(2)
    )
    per_mode = out.std(dim=(0, 1, 3)).numpy()
    assert_same(
        T_fpo.compute_freq_weighted_std(8, cfg.action_horizon, torch.device("cpu"), beta),
        J_fpo.compute_freq_weighted_std(8, cfg.action_horizon, beta),
    )
    if beta == 0.0:
        np.testing.assert_allclose(per_mode, 1.0, rtol=0.05)
    else:
        assert per_mode[0] > per_mode[-1]


# ------------------------------------------------------------------------- the DCT


def test_extract_dct_coeffs_2d_and_3d():
    v_t, v_j = fixed((5, 16, 3), seed=11)
    assert_same(
        T_dct.extract_dct_coeffs(v_t, num_modes=8),
        J_dct.extract_dct_coeffs(v_j, num_modes=8),
    )
    assert_same(
        T_dct.extract_dct_coeffs(v_t[2], num_modes=8),
        J_dct.extract_dct_coeffs(v_j[2], num_modes=8),
    )
    # num_modes == K is the full orthonormal transform, a no-op on the loss.
    assert_same(
        T_dct.extract_dct_coeffs(v_t, num_modes=16),
        J_dct.extract_dct_coeffs(v_j, num_modes=16),
    )


def test_decode_coeffs_to_velocity_1d_and_2d_tau():
    c_t, c_j = fixed((4, 8, 3), seed=12)
    tau_1d = jnp.linspace(0.0, 1.0, 16, dtype=jnp.float32)
    tau_2d = jnp.stack([jnp.linspace(0.0, 1.0, 6, dtype=jnp.float32)] * 4)

    assert_same(
        T_dct.decode_coeffs_to_velocity(c_t, torch.tensor(np.asarray(tau_1d)), 1.0, 16),
        J_dct.decode_coeffs_to_velocity(c_j, tau_1d, 1.0, 16),
    )
    assert_same(
        T_dct.decode_coeffs_to_velocity(c_t, torch.tensor(np.asarray(tau_2d)), 1.0, 16),
        J_dct.decode_coeffs_to_velocity(c_j, tau_2d, 1.0, 16),
    )


def test_velocity_to_action_steps_round_trips_at_full_resolution():
    """At num_rows == K the basis is orthonormal, so this is a rotation."""
    v_t, v_j = fixed((3, 5, 16, 2), seed=13)
    assert_same(
        T_dct.velocity_to_action_steps(v_t, 16),
        J_dct.velocity_to_action_steps(v_j, 16),
    )
    flat_j = v_j.reshape(-1, 16, 2)
    back = J_dct.extract_dct_coeffs(J_dct.velocity_to_action_steps(flat_j, 16), num_modes=16)
    np.testing.assert_allclose(np.asarray(back), np.asarray(flat_j), rtol=1e-4, atol=1e-5)


def test_spectral_energy_retained_is_one_at_full_resolution():
    v_t, v_j = fixed((6, 16, 3), seed=14)
    assert_same(
        T_dct.compute_spectral_energy_retained(v_t, 16),
        J_dct.compute_spectral_energy_retained(v_j, 16),
    )
    assert float(J_dct.compute_spectral_energy_retained(v_j, 16)) == pytest.approx(1.0, rel=1e-4)
    assert float(J_dct.compute_spectral_energy_retained(v_j, 2)) < 0.9


# ------------------------------------------------------------------- chunk-level GAE

GAMMA, LAM, H = 0.99, 0.95, 4


def gae_inputs(lengths, env_id, terminated=None, truncated=None, value=0.0,
               next_value=2.0, reward=1.0, total_steps=None, chunk_size=H):
    lengths_t = torch.tensor(lengths, dtype=torch.long)
    n = lengths_t.numel()
    starts_t = torch.cat([torch.zeros(1, dtype=torch.long), lengths_t.cumsum(0)[:-1]])
    total = int(total_steps if total_steps is not None else sum(lengths))
    flags = lambda f: torch.zeros(n, dtype=torch.bool) if f is None else torch.tensor(f)
    kwargs = {
        "chunk_terminated": flags(terminated),
        "chunk_truncated": flags(truncated),
        "chunk_env_id": torch.tensor(env_id, dtype=torch.long),
        "chunk_size": chunk_size,
        "gamma": GAMMA,
        "lam": LAM,
    }
    torch_out = T_gae.compute_chunk_advantages(
        torch.full((total,), float(reward)),
        torch.full((n,), float(value)),
        torch.full((n,), float(next_value)),
        starts_t, lengths_t, **kwargs,
    )
    jax_out = J_gae.compute_chunk_advantages(
        jnp.full((total,), float(reward)),
        jnp.full((n,), float(value)),
        jnp.full((n,), float(next_value)),
        jnp.asarray(np.asarray(starts_t)), jnp.asarray(np.asarray(lengths_t)),
        chunk_terminated=jnp.asarray(np.asarray(kwargs["chunk_terminated"])),
        chunk_truncated=jnp.asarray(np.asarray(kwargs["chunk_truncated"])),
        chunk_env_id=jnp.asarray(np.asarray(kwargs["chunk_env_id"])),
        chunk_size=chunk_size,
        gamma=GAMMA,
        lam=LAM,
    )
    return torch_out, jax_out


@pytest.mark.skip(reason="Torch GAE is QC's plain chunked backup now; the JAX port is not mirrored")
def test_gae_wavefront_interleave_matches():
    """The bug the sort guards: adjacency-masking finds no links in wavefront order."""
    assert_same(*gae_inputs([H] * 6, [0, 1, 2, 0, 1, 2]))


@pytest.mark.skip(reason="Torch GAE is QC's plain chunked backup now; the JAX port is not mirrored")
def test_gae_termination_and_truncation_differ():
    lengths, env_id = [H] * 4, [0, 1, 0, 1]
    assert_same(*gae_inputs(lengths, env_id, terminated=[False, True, False, False]))
    assert_same(*gae_inputs(lengths, env_id, truncated=[False, False, True, False]))
    assert_same(*gae_inputs(lengths, env_id, terminated=[False, False, True, False]))


@pytest.mark.skip(reason="Torch GAE is QC's plain chunked backup now; the JAX port is not mirrored")
def test_gae_variable_lengths_per_env():
    assert_same(*gae_inputs([H, 1, H, 3, 2, H], [0, 1, 0, 1, 0, 1]))


@pytest.mark.skip(reason="Torch GAE is QC's plain chunked backup now; the JAX port is not mirrored")
def test_gae_returns_equal_advantages_plus_values():
    adv_t, ret_t = gae_inputs([H] * 3, [0, 0, 0])[0]
    _, ret_j = gae_inputs([H] * 3, [0, 0, 0])[1]
    assert_same(ret_t, ret_j)
    assert torch.allclose(ret_t, adv_t)


@pytest.mark.skip(reason="Torch GAE is QC's plain chunked backup now; the JAX port is not mirrored")
def test_gae_permutation_invariance():
    """A chunk's advantage cannot depend on where it sits in the buffer.

    Rewriting the same two trajectories in temporal order, contiguous per env,
    must reproduce the wavefront-ordered result at the matching positions.
    """
    torch_out, jax_out = gae_inputs([H] * 4, [0, 1, 0, 1])
    # Permuted rows are the originals in order [0, 2, 1, 3]: env 0's chunks are
    # 0 and 2, env 1's are 1 and 3, each written contiguously and in time.
    order = [0, 2, 1, 3]
    kwargs = {"chunk_size": H, "gamma": GAMMA, "lam": LAM}
    torch_permuted = T_gae.compute_chunk_advantages(
        torch.ones(16), torch.zeros(4), torch.full((4,), 2.0),
        torch.tensor([0, 4, 8, 12]), torch.tensor([H] * 4),
        chunk_terminated=torch.zeros(4, dtype=torch.bool),
        chunk_truncated=torch.zeros(4, dtype=torch.bool),
        chunk_env_id=torch.tensor([0, 0, 1, 1]), **kwargs,
    )
    jax_permuted = J_gae.compute_chunk_advantages(
        jnp.ones(16), jnp.zeros(4), jnp.full((4,), 2.0),
        jnp.asarray([0, 4, 8, 12]), jnp.full((4,), H),
        chunk_terminated=jnp.zeros(4, dtype=jnp.bool_),
        chunk_truncated=jnp.zeros(4, dtype=jnp.bool_),
        chunk_env_id=jnp.asarray([0, 0, 1, 1]), **kwargs,
    )
    assert_same(torch_permuted[0], jax_permuted[0])
    assert_same(torch_out[0], jax_out[0])
    np.testing.assert_allclose(
        np.asarray(jax_permuted[0]), np.asarray(jax_out[0])[order], rtol=RTOL, atol=ATOL
    )


@pytest.mark.skip(reason="Torch GAE is QC's plain chunked backup now; the JAX port is not mirrored")
def test_gae_empty_rollout():
    torch_out = T_gae.compute_chunk_advantages(
        torch.zeros(0), torch.zeros(0), torch.zeros(0),
        torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
        chunk_terminated=torch.zeros(0, dtype=torch.bool),
        chunk_truncated=torch.zeros(0, dtype=torch.bool),
        chunk_env_id=torch.zeros(0, dtype=torch.long),
        chunk_size=H, gamma=GAMMA, lam=LAM,
    )
    jax_out = J_gae.compute_chunk_advantages(
        jnp.zeros(0), jnp.zeros(0), jnp.zeros(0),
        jnp.zeros(0, dtype=jnp.int32), jnp.zeros(0, dtype=jnp.int32),
        chunk_terminated=jnp.zeros(0, dtype=jnp.bool_),
        chunk_truncated=jnp.zeros(0, dtype=jnp.bool_),
        chunk_env_id=jnp.zeros(0, dtype=jnp.int32),
        chunk_size=H, gamma=GAMMA, lam=LAM,
    )
    assert_same(torch_out, jax_out)


@pytest.mark.skip(reason="Torch GAE is QC's plain chunked backup now; the JAX port is not mirrored")
def test_gae_no_fp32_loss_at_large_offsets():
    """A cumsum-diff discount would need gamma**-s_i and die here. The gather does not."""
    n, chunk_size = 300, 16
    lengths = torch.full((n,), chunk_size)
    starts = torch.arange(n) * chunk_size
    torch_out = T_gae.compute_chunk_advantages(
        torch.ones(n * chunk_size),
        torch.zeros(n), torch.full((n,), 2.0), starts, lengths,
        chunk_terminated=torch.zeros(n, dtype=torch.bool),
        chunk_truncated=torch.zeros(n, dtype=torch.bool),
        chunk_env_id=torch.arange(n) % 4,
        chunk_size=chunk_size, gamma=GAMMA, lam=LAM,
    )
    jax_out = J_gae.compute_chunk_advantages(
        jnp.ones(n * chunk_size), jnp.zeros(n), jnp.full((n,), 2.0),
        jnp.asarray(np.asarray(starts)), jnp.asarray(np.asarray(lengths)),
        chunk_terminated=jnp.zeros(n, dtype=jnp.bool_),
        chunk_truncated=jnp.zeros(n, dtype=jnp.bool_),
        chunk_env_id=jnp.arange(n) % 4,
        chunk_size=chunk_size, gamma=GAMMA, lam=LAM,
    )
    assert_same(torch_out, jax_out)
    assert np.isfinite(np.asarray(jax_out[0])).all()


@pytest.mark.skip(reason="Torch GAE is QC's plain chunked backup now; the JAX port is not mirrored")
def test_gae_rejects_bad_discounts_and_mismatched_lengths():
    zeros = lambda n: jnp.zeros(n, dtype=jnp.float32)
    longs = lambda n: jnp.zeros(n, dtype=jnp.int32)
    flags = lambda n: jnp.zeros(n, dtype=jnp.bool_)
    for bad in ({"gamma": 1.5}, {"lam": -0.1}):
        kwargs = {"chunk_size": H, "gamma": GAMMA, "lam": LAM} | bad
        with pytest.raises(ValueError):
            J_gae.compute_chunk_advantages(
                zeros(4), zeros(4), zeros(4), longs(4), longs(4),
                chunk_terminated=flags(4), chunk_truncated=flags(4),
                chunk_env_id=longs(4), **kwargs,
            )
    with pytest.raises(ValueError):
        J_gae.compute_chunk_advantages(
            zeros(8), zeros(3), zeros(4), longs(4), longs(4),
            chunk_terminated=flags(4), chunk_truncated=flags(4),
            chunk_env_id=longs(4), chunk_size=H, gamma=GAMMA, lam=LAM,
        )


# ------------------------------------------------------- trace-safe constants


def test_the_dct_basis_survives_two_different_jit_traces():
    """A Python-int-keyed cache over jnp values is not safe.

    Whichever trace fills it first stores a *tracer*, and the next trace to read
    it is handed a value from a closed-over jaxpr. These two calls have different
    inputs, so a cache would hand the second one the first one's tracer.
    """
    import subprocess
    import sys as _sys
    from pathlib import Path

    script = f"""
import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / "src")!r})
import jax, jax.numpy as jnp
from sjfpo_jax.dct_utils import extract_dct_coeffs
from sjfpo_jax.networks import SinusoidalEncoding
f = jax.jit(lambda x: extract_dct_coeffs(x, num_modes=8))
a = f(jnp.ones((2, 16, 3)))
b = f(jnp.full((2, 16, 3), 2.0))
enc = SinusoidalEncoding(16, scale=100.0)
g = jax.jit(lambda t: enc.encode_times(t))
assert g(jnp.linspace(0.0, 1.0, 4)).shape == (4, 16)
assert not bool(jnp.all(a == b))
print("ok")
"""
    result = subprocess.run(
        [_sys.executable, "-c", script],
        capture_output=True, text=True, check=False,
        env={"JAX_PLATFORMS": "cpu", "PATH": "/usr/bin:/bin"},
    )
    assert "ok" in result.stdout, result.stderr[-2000:]
