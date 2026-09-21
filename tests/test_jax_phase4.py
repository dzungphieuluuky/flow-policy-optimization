"""Phase 4: Orbax checkpoint roundtrip on CPU (tmp dirs)."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo_jax import policy as jpol  # noqa: E402
from sjfpo_jax.checkpointing import (  # noqa: E402
    export_safetensors,
    load_checkpoint,
    load_safetensors,
    save_checkpoint,
)


def _leaves(obj):
    return [np.asarray(v) for v in jax.tree.leaves(nnx.state(obj, nnx.Param))]


def test_roundtrip_restores_params_and_counters(tmp_path):
    pj, bj = jpol.build_policy(6, 3, seed=0, hidden_dim=8, dit_depth=1, dit_heads=2)
    opt = jpol.build_optimizer(pj)
    pj.normalizer.update(jnp.ones((4, 4, 6)))
    # One real step so optimizer moments are non-trivial.
    _batch = {
        "observation.state": jnp.ones((2, 4, 6)),
        "action": jnp.zeros((2, 8, 3)),
        "advantages": jnp.ones(2),
        "returns": jnp.ones(2),
        "tau": jnp.full((2, 2), 0.5),
        "eps": jnp.zeros((2, 2, 8, 3)),
    }
    _prep = jpol.prepare_batch(
        pj, bj, _batch, n_obs_steps=4, action_horizon=8, num_mc_samples=2,
    )
    jpol.train_step(pj, opt, _prep)
    out = save_checkpoint(tmp_path / "ckpt_10", policy=pj, optimizer=opt,
                          global_step=10, env_name="HalfCheetah-v5")
    assert Path(out["meta"]).exists()
    # Restore into differently-initialized objects; everything must match.
    fresh, _ = jpol.build_policy(6, 3, seed=999, hidden_dim=8, dit_depth=1, dit_heads=2)
    fresh_opt = jpol.build_optimizer(fresh)
    meta = load_checkpoint(tmp_path / "ckpt_10", policy=fresh, optimizer=fresh_opt)
    assert meta["global_step"] == 10
    assert meta["env_name"] == "HalfCheetah-v5"
    for a, b in zip(_leaves(pj), _leaves(fresh)):
        np.testing.assert_array_equal(a, b)
    # Normalizer stats (non-param variables) ride along too.
    assert float(fresh.normalizer.count[...]) == 16.0
    # Full optimizer state roundtrips (fresh_opt was stepped never).
    for a, b in zip(
        jax.tree.leaves(nnx.state(opt)), jax.tree.leaves(nnx.state(fresh_opt))
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _torch_policy_6x3():
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature

    from sjfpo.configuration_sjfpo import SJFPOConfig
    from sjfpo.modeling_sjfpo import SJFPOPolicy

    cfg = SJFPOConfig(
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (6,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (3,))},
        hidden_dim=8,  # only hidden_dim reaches submodules; dit_* stay default
    )
    return SJFPOPolicy(cfg)


def test_safetensors_roundtrip(tmp_path):
    pj, _ = jpol.build_policy(6, 3, seed=0, hidden_dim=8)
    pj.normalizer.update(jnp.ones((4, 4, 6)))
    path = export_safetensors(pj, tmp_path / "model.safetensors")
    fresh, _ = jpol.build_policy(6, 3, seed=999, hidden_dim=8)
    loaded = load_safetensors(fresh, path)
    assert len(loaded) > 100  # full policy incl. normalizer stats
    for a, b in zip(_leaves(pj), _leaves(fresh)):
        np.testing.assert_array_equal(a, b)
    assert float(fresh.normalizer.count[...]) == 16.0


def test_safetensors_jax_to_torch_strict(tmp_path):
    import torch
    from safetensors.torch import load_file as torch_load

    pj, bj = jpol.build_policy(6, 3, seed=0, hidden_dim=8)
    path = export_safetensors(pj, tmp_path / "model.safetensors", behavior=bj)
    tpol = _torch_policy_6x3()
    tpol.load_state_dict({k: torch.as_tensor(v) for k, v in torch_load(path).items()},
                         strict=True)
    x = torch.randn(2, 8, 3)
    t = torch.rand(2)
    c = torch.randn(2, 8)
    tpol.eval()
    with torch.no_grad():
        expected = tpol.actor(x, t, c).detach().numpy()
    got = pj.actor(jnp.asarray(x.numpy()), jnp.asarray(t.numpy()),
                       jnp.asarray(c.numpy()))
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-5, atol=1e-5)


def test_safetensors_torch_to_jax(tmp_path):
    import torch
    from safetensors.torch import save_file as torch_save

    tpol = _torch_policy_6x3()
    path = str(tmp_path / "torch.safetensors")
    torch_save(tpol.state_dict(), path)
    pj, _ = jpol.build_policy(6, 3, seed=999, hidden_dim=8)
    loaded = load_safetensors(pj, path)
    assert len(loaded) > 100
    x = torch.randn(2, 8, 3)
    t = torch.rand(2)
    c = torch.randn(2, 8)
    tpol.eval()
    with torch.no_grad():
        expected = tpol.actor(x, t, c).detach().numpy()
    got = pj.actor(jnp.asarray(x.numpy()), jnp.asarray(t.numpy()),
                       jnp.asarray(c.numpy()))
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-5, atol=1e-5)


def test_save_overwrites_and_missing_meta_raises(tmp_path):
    import pytest

    pj, _ = jpol.build_policy(6, 3, seed=1, hidden_dim=8, dit_depth=1, dit_heads=2)
    opt = jpol.build_optimizer(pj)
    save_checkpoint(tmp_path / "c", policy=pj, optimizer=opt, global_step=1)
    save_checkpoint(tmp_path / "c", policy=pj, optimizer=opt, global_step=2)
    assert load_checkpoint(tmp_path / "c", policy=pj, optimizer=opt)["global_step"] == 2
    (tmp_path / "c" / "meta.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "c", policy=pj, optimizer=opt)
