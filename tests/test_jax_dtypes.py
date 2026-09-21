"""bf16 param dtype: detection defaults + explicit-bf16 parity (CPU-safe)."""

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
    load_safetensors,
    to_torch_dict,
)
from sjfpo_jax.dtypes import is_blackwell, resolve_param_dtype  # noqa: E402


def test_cpu_box_defaults_to_fp32():
    assert is_blackwell() is False
    assert resolve_param_dtype() == jnp.float32
    pj, _ = jpol.build_policy(6, 3, seed=0, hidden_dim=8, dit_depth=1, dit_heads=2)
    dtypes = {np.dtype(jnp.asarray(v).dtype) for v in jax.tree.leaves(nnx.state(pj, nnx.Param))}
    assert dtypes == {np.dtype(np.float32)}


def test_explicit_bf16_storage_with_fp32_stats():
    pj, _ = jpol.build_policy(6, 3, seed=0, hidden_dim=8, dit_depth=1, dit_heads=2,
                              param_dtype=jnp.bfloat16)
    dtypes = {np.dtype(jnp.asarray(v).dtype) for v in jax.tree.leaves(nnx.state(pj, nnx.Param))}
    assert dtypes == {np.dtype('bfloat16')}
    # Running stats stay fp32 for stability.
    assert np.dtype(jnp.asarray(pj.normalizer.mean[...]).dtype) == np.dtype(np.float32)
    x = jnp.zeros((2, 4, 6))
    out, _ = pj.actor(jnp.zeros((2, 8, 3)), jnp.full((2,), 0.5),
                      pj.conditioner(pj.normalizer(x)))
    assert bool(jnp.all(jnp.isfinite(out)))


def test_bf16_loss_parity_with_fp32(tmp_path):
    pf, _ = jpol.build_policy(6, 3, seed=0, hidden_dim=8, dit_depth=1, dit_heads=2)
    pb, _ = jpol.build_policy(6, 3, seed=0, hidden_dim=8, dit_depth=1, dit_heads=2,
                              param_dtype=jnp.bfloat16)
    d = to_torch_dict(pf)
    import tempfile, os

    with tempfile.TemporaryDirectory() as t:
        p = export_safetensors(pf, os.path.join(t, "f.safetensors"))
        load_safetensors(pb, p)
    assert d  # export exercised
    batch = {
        "observation.state": jnp.zeros((2, 4, 6)),
        "action": jnp.zeros((2, 8, 3)),
        "advantages": jnp.ones(2) * 0.1,
        "returns": jnp.ones(2),
        "tau": jnp.full((2, 2), 0.5),
        "eps": jnp.zeros((2, 2, 8, 3)),
    }
    prep_kw = {"n_obs_steps": 4, "action_horizon": 8, "num_mc_samples": 2}
    _, bf = jpol.build_policy(6, 3, seed=0, hidden_dim=8, dit_depth=1, dit_heads=2)
    lf, _ = jpol.compute_loss(pf, jpol.prepare_batch(pf, bf, batch, **prep_kw))
    lb, _ = jpol.compute_loss(pb, jpol.prepare_batch(pb, bf, batch, **prep_kw))
    np.testing.assert_allclose(float(lb), float(lf), rtol=1e-2, atol=1e-2)
