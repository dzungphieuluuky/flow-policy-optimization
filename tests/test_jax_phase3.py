"""Phase 3: JAX loss/update vs torch forward + optimizer behavior (CPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from lerobot.configs.types import FeatureType, PolicyFeature

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.configuration_sjfpo import SJFPOConfig  # noqa: E402
from sjfpo.modeling_sjfpo import SJFPOPolicy  # noqa: E402
from sjfpo_jax import policy as jpol  # noqa: E402
from sjfpo_jax.networks import load_torch_weights  # noqa: E402

OBS, ACT, H, T, M = 6, 3, 8, 4, 2


def _torch_policy():
    cfg = SJFPOConfig(
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (OBS,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT,))},
        action_horizon=H,
        n_obs_steps=T,
        top_p_spectral_tokens=1.0,
        hidden_dim=8,  # only hidden_dim reaches submodules; dit_* stay default
        num_mc_samples=M,
        num_euler_steps=2,
        num_inference_steps=2,
    )
    pol = SJFPOPolicy(cfg)
    pol.set_action_bounds(np.full(ACT, -1.0), np.full(ACT, 1.0))
    return pol


def _jax_pair(tpol):
    # NOTE: SJFPOPolicy forwards only hidden_dim; dit_* resolve to bare
    # SJFPOConfig defaults (depth=4, heads=4, ratio=4.0, time_dim=128).
    pj, bj = jpol.build_policy(
        OBS, ACT, seed=0, hidden_dim=8, dit_depth=4, dit_heads=4,
        dit_mlp_ratio=4.0, dit_time_dim=128,
    )
    for name in ("actor", "conditioner", "critic"):
        load_torch_weights(getattr(pj, name), getattr(tpol, name).state_dict())
        load_torch_weights(getattr(bj, name), getattr(tpol, name).state_dict())
    pj.set_action_bounds(jnp.full((ACT,), -1.0), jnp.full((ACT,), 1.0))
    bj.set_action_bounds(jnp.full((ACT,), -1.0), jnp.full((ACT,), 1.0))
    return pj, bj


def _batch(n=3):
    torch.manual_seed(0)
    tau = torch.rand(M) * (0.995 - 0.005) + 0.005
    return {
        "observation.state": torch.randn(n, T, OBS),
        "action": torch.randn(n, H, ACT).clamp(-1.0, 1.0),
        "action_is_pad": torch.zeros(n, H, dtype=torch.bool),
        "advantages": torch.randn(n) * 0.1,
        "returns": torch.randn(n) * 10.0,
        "values": torch.randn(n),
        "tau": tau.unsqueeze(0).expand(n, -1).contiguous(),
        "eps": torch.randn(n, M, H, ACT),
    }


def _aligned(tpol, pj, bj, tb):
    """Same normalizer stats + snapshot on both sides, eval mode."""
    tpol.train()
    tpol.preprocessor.update_stats(tb["observation.state"])
    tpol.eval()
    pj.normalizer.update(jnp.asarray(tb["observation.state"].numpy()))
    tpol.snapshot_behavior_policy()
    jpol.snapshot_behavior_policy(pj, bj)
    jb = {k: jnp.asarray(v.numpy()) if torch.is_tensor(v) else v for k, v in tb.items()}
    return jb


def test_loss_matches_torch_forward():
    tpol = _torch_policy()
    pj, bj = _jax_pair(tpol)
    tb = _batch()
    jb = _aligned(tpol, pj, bj, tb)
    with torch.no_grad():
        tpol.eval()
        t_loss, t_info = tpol(tb)
    prep = jpol.prepare_batch(
        pj, bj, jb, n_obs_steps=T, action_horizon=H,
        top_p_spectral_tokens=1.0, num_mc_samples=M,
    )
    j_loss, j_info = jpol.compute_loss(pj, prep)
    assert float(t_loss) == float(t_loss)  # finite torch side
    np.testing.assert_allclose(float(j_loss), float(t_loss), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        j_info["policy_loss"], t_info["policy_loss"], rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        j_info["value_loss"], t_info["value_loss"], rtol=1e-4, atol=1e-4
    )


def test_train_step_updates_live_freezes_snapshot():
    tpol = _torch_policy()
    pj, bj = _jax_pair(tpol)
    tb = _batch()
    jb = _aligned(tpol, pj, bj, tb)
    opt = jpol.build_optimizer(pj)
    before = jnp.concatenate([v.reshape(-1) for v in jax.tree.leaves(nnx.state(pj, nnx.Param))])
    old_before = jnp.concatenate([v.reshape(-1) for v in jax.tree.leaves(nnx.state(bj, nnx.Param))])
    prep = jpol.prepare_batch(
        pj, bj, jb, n_obs_steps=T, action_horizon=H, num_mc_samples=M,
    )
    loss, info = jpol.train_step(pj, opt, prep)
    assert np.isfinite(float(loss))
    after = jnp.concatenate([v.reshape(-1) for v in jax.tree.leaves(nnx.state(pj, nnx.Param))])
    old_after = jnp.concatenate([v.reshape(-1) for v in jax.tree.leaves(nnx.state(bj, nnx.Param))])
    assert not np.allclose(np.asarray(after), np.asarray(before))  # live moved
    np.testing.assert_array_equal(np.asarray(old_after), np.asarray(old_before))  # frozen
    assert set(info) >= {"policy_loss", "value_loss", "explained_var"}


def test_decay_mask_structure():
    pj, _ = jpol.build_policy(OBS, ACT, seed=0, hidden_dim=8, dit_depth=1, dit_heads=2)
    mask = jpol.decay_mask(pj)
    pairs = dict(jax.tree.flatten_with_path(mask)[0])

    from sjfpo_jax.policy import _path_names

    ada = [k for k in pairs if any("adaLN" in n for n in _path_names(k))]
    assert ada and not any(bool(pairs[k]) for k in ada)
    names_of = _path_names
    out = [k for k in pairs if names_of(k)[:2] == ["actor", "out"]]
    assert out and not any(bool(pairs[k]) for k in out)
    # Conditioner attn out-projections still decay (torch parity).
    cond_attn_out = [
        k for k in pairs
        if "conditioner" in names_of(k) and "out" in names_of(k)
    ]
    assert cond_attn_out and all(bool(pairs[k]) for k in cond_attn_out)
    live = [k for k in pairs if "conditioner" in _path_names(k)]
    assert live and all(bool(pairs[k]) for k in live)


def test_predict_chunk_matches_torch():
    tpol = _torch_policy()
    pj, bj = _jax_pair(tpol)
    _ = bj
    tpol.eval()
    states = torch.randn(2, T, OBS)
    with torch.inference_mode():
        t_chunk = tpol.predict_action_chunk({"observation.state": states}, noise=False)
    j_chunk = jpol.predict_action_chunk(
        pj, jnp.asarray(states.numpy()), jax.random.PRNGKey(0),
        action_horizon=H, num_euler_steps=2, num_inference_steps=2, noise=False,
    )
    np.testing.assert_allclose(
        np.asarray(j_chunk), t_chunk.numpy(), rtol=1e-4, atol=1e-4
    )
