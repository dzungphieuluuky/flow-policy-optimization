"""Phase 2 parity: sjfpo_jax NNX models vs torch originals (1e-5, fp32)."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo import networks as torch_nets  # noqa: E402
from sjfpo_jax import networks as jax_nets  # noqa: E402

OBS, ACT, H, T, K = 4, 3, 8, 4, 4
N_HEADS = 2


def _load(tm: torch.nn.Module, jm: nnx.Module):
    jax_nets.load_torch_weights(jm, tm.state_dict())


def test_window_encoder_matches():
    tm = torch_nets.WindowEncoder(OBS, hidden_dim=H, n_heads=N_HEADS).eval()
    jm = jax_nets.WindowEncoder(OBS, hidden_dim=H, n_heads=N_HEADS, rngs=nnx.Rngs(0))
    _load(tm, jm)
    x = torch.randn(2, T, OBS)
    xj = jnp.asarray(x.numpy())
    np.testing.assert_allclose(
        np.asarray(jm(xj)), tm(x).detach().numpy(), rtol=1e-5, atol=1e-5
    )
    mask = torch.tensor([[False] * T, [True] + [False] * (T - 1)])
    np.testing.assert_allclose(
        np.asarray(jm(xj, jnp.asarray(mask.numpy()))),
        tm(x, mask).detach().numpy(), rtol=1e-5, atol=1e-5,
    )


def test_running_mean_std_matches():
    tm = torch_nets.RunningMeanStd(OBS)
    jm = jax_nets.RunningMeanStd(OBS)
    data = torch.randn(32, T, OBS)
    tm.update(data)
    jm.update(jnp.asarray(data.numpy()))
    x = torch.randn(2, T, OBS)
    np.testing.assert_allclose(
        np.asarray(jm(jnp.asarray(x.numpy()))), tm(x).detach().numpy(),
        rtol=1e-5, atol=1e-5,
    )
    np.testing.assert_allclose(np.asarray(jm.mean.value), tm.mean.numpy(), atol=1e-6)
    np.testing.assert_allclose(np.asarray(jm.var.value), tm.var.numpy(), atol=1e-6)


def test_conditioner_matches():
    tm = torch_nets.StateConditioner(OBS, hidden_dim=H).eval()
    jm = jax_nets.StateConditioner(OBS, hidden_dim=H, rngs=nnx.Rngs(0))
    _load(tm, jm)
    x = torch.randn(2, T, OBS)
    np.testing.assert_allclose(
        np.asarray(jm(jnp.asarray(x.numpy()))), tm(x).detach().numpy(),
        rtol=1e-5, atol=1e-5,
    )


def test_flow_model_matches():
    kwargs = dict(hidden_dim=H, dit_depth=1, dit_heads=N_HEADS,
                  dit_mlp_ratio=2.0, dit_time_dim=H)
    tm = torch_nets.SJFPOFlowModel(ACT, **kwargs).eval()
    jm = jax_nets.SJFPOFlowModel(ACT, **kwargs, rngs=nnx.Rngs(0))
    _load(tm, jm)
    x = torch.randn(2, K, ACT)
    t = torch.rand(2)
    c = torch.randn(2, H)
    out_t = tm(x, t, c).detach().numpy()
    out_j = np.asarray(jm(jnp.asarray(x.numpy()), jnp.asarray(t.numpy()),
                          jnp.asarray(c.numpy())))
    np.testing.assert_allclose(out_j, out_t, rtol=1e-5, atol=1e-5)
    # Zero-init: untrained velocity is ~0.
    assert abs(out_j).max() < 1e-4


def test_critic_matches():
    tm = torch_nets.Critic(OBS, hidden_dim=H).eval()
    jm = jax_nets.Critic(OBS, hidden_dim=H, rngs=nnx.Rngs(0))
    _load(tm, jm)
    x = torch.randn(2, T, OBS)
    np.testing.assert_allclose(
        np.asarray(jm(jnp.asarray(x.numpy()))), tm(x).detach().numpy(),
        rtol=1e-5, atol=1e-5,
    )


def test_zero_init_preserved():
    jm = jax_nets.SJFPOFlowModel(
        ACT, hidden_dim=H, dit_depth=1, dit_heads=N_HEADS,
        dit_mlp_ratio=2.0, dit_time_dim=H, rngs=nnx.Rngs(0),
    )
    assert float(jnp.abs(jm.out.kernel.value).max()) == 0.0
    assert float(jnp.abs(jm.adaLN_out.kernel.value).max()) == 0.0
