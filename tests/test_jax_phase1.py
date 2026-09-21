"""Phase 1 parity: sjfpo_jax pure functions vs torch/numpy originals (1e-5)."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo import dct_utils as torch_dct  # noqa: E402
from sjfpo import fpo_core as torch_fpo  # noqa: E402
from sjfpo import gae as np_gae  # noqa: E402
from sjfpo_jax import dct as jax_dct  # noqa: E402
from sjfpo_jax import fpo as jax_fpo  # noqa: E402
from sjfpo_jax import gae as jax_gae  # noqa: E402

def test_dct_matches_torch():
    x = torch.randn(4, 16, 6, dtype=torch.float32)
    xj = jnp.asarray(x.numpy())
    np.testing.assert_allclose(
        np.asarray(jax_dct.dct(xj, dim=1)), torch_dct.dct(x, dim=1).numpy(),
        rtol=1e-5, atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(jax_dct.idct(jax_dct.dct(xj, dim=1), dim=1)), x.numpy(),
        rtol=1e-5, atol=1e-5,
    )


def test_discounted_chunk_reward_matches():
    r = torch.randn(5, 8, dtype=torch.float32)
    np.testing.assert_allclose(
        np.asarray(jax_dct.discounted_chunk_reward(jnp.asarray(r.numpy()), 0.99)),
        torch_dct.discounted_chunk_reward(r, 0.99).numpy(),
        rtol=1e-5, atol=1e-5,
    )


def test_gae_matches_numpy_oracle():
    rng = np.random.default_rng(0)
    lengths = np.array([8, 8, 4])
    rew = rng.normal(size=int(lengths.sum()))
    val, nxt = rng.normal(size=3), rng.normal(size=3)
    term = np.array([False, False, True])
    trunc = np.array([False, True, False])
    a1, r1 = np_gae.compute_advantages(
        rew, val, nxt, lengths, term, gamma=0.999, lam=0.995,
        normalize=True, truncated=trunc,
    )
    a2, r2 = jax_gae.compute_advantages(
        jnp.asarray(rew), jnp.asarray(val), jnp.asarray(nxt),
        jnp.asarray(lengths), jnp.asarray(term),
        gamma=0.999, lam=0.995, normalize=True,
        truncated=jnp.asarray(trunc),
    )
    np.testing.assert_allclose(np.asarray(a2), a1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(r2), r1, rtol=1e-5, atol=1e-5)


def test_gae_jitted():
    f = jax.jit(jax_gae.compute_advantages, static_argnames=("normalize",))
    a, r = f(
        jnp.ones(20), jnp.ones(3), jnp.ones(3), jnp.array([8, 8, 4]),
        jnp.zeros(3, dtype=bool), 0.99, 0.95, True, None, 100.0,
    )
    assert a.shape == (3,) and r.shape == (3,)


def test_cfm_aspo_ratio_match_torch():
    torch.manual_seed(0)
    old = torch.randn(6, 4, 5, 3)
    new = torch.randn(6, 4, 5, 3)
    star = torch.randn(6, 4, 5, 3)
    adv = torch.randn(6)
    oj, nj, sj = (jnp.asarray(t.numpy()) for t in (old, new, star))
    for mode in ("sample", "bin"):
        np.testing.assert_allclose(
            np.asarray(jax_fpo.compute_cfm_loss(oj, sj, mode)),
            torch_fpo.compute_cfm_loss(old, star, mode).numpy(),
            rtol=1e-5, atol=1e-5,
        )
    aj = jnp.asarray(adv.numpy())[:, None]  # [B, 1] like adv_mc in forward()
    lo = torch_fpo.compute_cfm_loss(old, star, "sample")
    ln = torch_fpo.compute_cfm_loss(new, star, "sample")
    np.testing.assert_allclose(
        np.asarray(jax_fpo.compute_policy_ratio(
            jnp.asarray(lo.numpy()), jnp.asarray(ln.numpy()))),
        torch_fpo.compute_policy_ratio(lo, ln).numpy(),
        rtol=1e-5, atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(jax_fpo.compute_policy_loss(
            jnp.asarray(lo.numpy()), jnp.asarray(ln.numpy()), aj)),
        torch_fpo.compute_policy_loss(lo, ln, adv.unsqueeze(1)).numpy(),
        rtol=1e-5, atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(jax_fpo.compute_aspo_surrogate(
            jnp.asarray(torch_fpo.compute_policy_ratio(lo, ln).numpy()), aj)),
        torch_fpo.compute_aspo_surrogate(
            torch_fpo.compute_policy_ratio(lo, ln), adv.unsqueeze(1)).numpy(),
        rtol=1e-5, atol=1e-5,
    )


def test_validate_stream_rejects_bad_streams():
    with pytest.raises(ValueError):
        jax_gae.validate_stream(jnp.ones(10), jnp.ones(2), jnp.array([8, 8]))
    with pytest.raises(ValueError):
        jax_gae.validate_stream(jnp.ones(16), jnp.ones(2), jnp.array([8, 0]))
    jax_gae.validate_stream(jnp.ones(16), jnp.ones(2), jnp.array([8, 8]))


def test_mc_simulation_shapes():
    a = jnp.zeros((5, 4, 3))
    eps = jnp.zeros((5, 7, 4, 3))
    tau, ef, ar, shape, m = jax_fpo.run_mc_simulation(a, eps, jnp.zeros(7))
    assert (tuple(tau.shape), tuple(ef.shape), tuple(ar.shape), shape, m) == (
        (35,), (35, 12), (35, 12), (4, 3), 7,
    )
    with pytest.raises(ValueError):
        jax_fpo.run_mc_simulation(a, eps, jnp.zeros(6))


def test_sample_action_euler_constant_field():
    eps = jnp.zeros((2, 3))
    cond = jnp.zeros((2, 4))
    out = jax_fpo.sample_action(
        lambda x, t, c: jnp.ones_like(x), (2, 3), cond, 4, eps=eps,
    )
    np.testing.assert_allclose(np.asarray(out), np.ones((2, 3)), atol=1e-6)
