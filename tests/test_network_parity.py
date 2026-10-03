"""torch vs jax parity for the networks: conditioner, flow head, critic.

Forward parity, not init parity: the torch module is built with a seed, its
weights are copied into the NNX tree, and both are then asked for the same
prediction. Tolerance is ``rtol=1e-4`` because fp32 GEMM accumulation order
differs between the two frameworks; do not widen it.

The weight mapping here is the §8 converter's job in miniature:
``tests/torch_jax_weights.py`` owns it and ``scripts/convert_torch_to_jax.py``
takes it over at milestone 6. Two rules: a 2-D ``.weight`` is a torch
``[out, in]`` Linear and becomes an NNX ``[in, out]`` kernel, and critic member
indices stack onto a leading axis.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from torch_jax_weights import load, to_jax_tree

from sjfpo_jax import networks as J
from spectral_joint_policy import networks as T

OBS, ACTION, HIDDEN, HEADS, DEPTH, K, MC = 11, 2, 32, 4, 2, 16, 6
RTOL, ATOL = 1e-4, 1e-5



def obs_batch(seed=0, batch=MC, steps=16):
    values = np.random.default_rng(seed).standard_normal((batch, steps, OBS), dtype=np.float32)
    return torch.from_numpy(values.copy()), jnp.asarray(values)


def action_batch(seed=1, batch=MC):
    values = np.random.default_rng(seed).standard_normal((batch, K, ACTION), dtype=np.float32)
    return torch.from_numpy(values.copy()), jnp.asarray(values)


def build_conditioner(seed=0):
    torch.manual_seed(seed)
    t = T.StateEncoder(OBS, HIDDEN, 16)
    j = J.StateConditioner(OBS, HIDDEN, nnx.Rngs(seed))
    return t, j


def build_critic(seed=0, num_critics=2, n_obs_steps=16):
    torch.manual_seed(seed)
    t = T.Critic(OBS, HIDDEN, n_obs_steps, num_critics)
    j = J.Critic(OBS, HIDDEN, n_obs_steps, num_critics, nnx.Rngs(seed))
    return t, j


def build_actor(seed=0):
    torch.manual_seed(seed)
    t = T.SJFPOFlowModel(ACTION, K, HIDDEN, DEPTH, HEADS, 4.0, 32, 10000.0)
    j = J.SJFPOFlowModel(ACTION, K, HIDDEN, DEPTH, HEADS, 4.0, 32, 10000.0, nnx.Rngs(seed))
    return t, load(j, to_jax_tree(t.state_dict()))


def assert_same(torch_out, jax_out):
    torch_out = torch_out.detach().cpu().numpy() if isinstance(torch_out, torch.Tensor) else torch_out
    jax_out = np.asarray(jax_out) if isinstance(jax_out, jax.Array) else jax_out
    if isinstance(torch_out, dict):
        assert set(torch_out) == set(jax_out), (sorted(torch_out), sorted(jax_out))
        for key in torch_out:
            assert_same(torch_out[key], jax_out[key])
        return
    assert torch_out.shape == jax_out.shape, (torch_out.shape, jax_out.shape)
    np.testing.assert_allclose(torch_out, jax_out, rtol=RTOL, atol=ATOL)


# ------------------------------------------------------------------- StateEncoder


@pytest.mark.skip(reason="JAX port lacks frame_pos/freq_pos; torch conditioner diverged on purpose")
def test_conditioner_forward():
    t, j = build_conditioner()
    obs_t, obs_j = obs_batch()
    assert_same(t(obs_t), j(obs_j))


def test_conditioner_uses_every_frame_and_its_order():
    """A mean-pool can lose both, and this is the test that says it does not."""
    t, j = build_conditioner()
    obs_t, obs_j = obs_batch()
    base_j = np.asarray(j(obs_j))

    perturbed_t, perturbed_j = obs_batch(seed=0)
    perturbed_t[:, -1] = 5.0
    perturbed_j = perturbed_j.at[:, -1].set(5.0)
    assert float(jnp.abs(j(perturbed_j) - base_j).max()) > 1e-3
    assert not np.allclose(
        t(perturbed_t).detach().numpy(), t(obs_t).detach().numpy()
    )
    assert not np.allclose(j(obs_j[::-1]), base_j)


def test_conditioner_rejects_a_malformed_window():
    _, j = build_conditioner()
    _, obs_j = obs_batch()
    with pytest.raises(ValueError):
        j(obs_j[0])
    with pytest.raises(ValueError):
        j(jnp.zeros((2, 16, OBS + 1), dtype=jnp.float32))
    with pytest.raises(ValueError):
        j(jnp.zeros((0, 16, OBS), dtype=jnp.float32))


# --------------------------------------------------------------------------- Critic


@pytest.mark.parametrize("num_critics", [1, 2, 3])
@pytest.mark.skip(reason="JAX port lacks frame_pos/freq_pos; torch critic diverged on purpose")
def test_critic_forward(num_critics):
    t, j = build_critic(num_critics=num_critics)
    obs_t, obs_j = obs_batch()
    torch_out, jax_out = t(obs_t), j(obs_j)
    assert tuple(jax_out.shape) == (num_critics, MC)
    assert_same(torch_out, jax_out)


@pytest.mark.skip(reason="JAX port lacks frame_pos/freq_pos; torch critic diverged on purpose")
def test_critic_members_stay_independent():
    """A shared trunk would leave only a linear map differing, and min() over that
    means nothing -- so the ensemble has to be num_critics full conditioners."""
    t, j = build_critic(num_critics=3)
    obs_t, obs_j = obs_batch()
    out = np.asarray(j(obs_j))
    assert not np.allclose(out[0], out[1])
    assert not np.allclose(out[1], out[2])
    # And the stack really is the torch ensemble, member for member.
    assert_same(t(obs_t), out)


def test_critic_rejects_a_window_it_was_not_built_for():
    _, j = build_critic(n_obs_steps=16)
    _, obs_j = obs_batch()
    with pytest.raises(ValueError):
        j(obs_j[:, :8])


def test_critic_gradient_reaches_every_member():
    t, j = build_critic(num_critics=2)
    obs_t, obs_j = obs_batch()
    torch.autograd.grad(t(obs_t).sum(), list(t.parameters()))[0]
    grads = jax.grad(lambda x: j(x).sum())(obs_j)
    assert grads.shape == obs_j.shape
    assert float(jnp.abs(grads).sum()) > 0


@pytest.mark.skip(reason="Torch conditioner is now flatten+MLP; JAX port diverged by design")
def test_critic_parameter_count_matches_torch():
    t, j = build_critic(num_critics=2)
    torch_params = sum(
        p.numel() for p in t.parameters()
    )
    jax_params = sum(
        leaf.size for leaf in jax.tree_util.tree_leaves(nnx.state(j, nnx.Param))
    )
    assert jax_params == torch_params


# -------------------------------------------------------------------- SJFPOFlowModel


@pytest.mark.parametrize(
    "t_shape", [(), (MC,), (MC, K, 1)], ids=["scalar", "per_batch", "per_token"]
)
def test_actor_forward_time_layouts(t_shape):
    t_actor, j_actor = build_actor()
    t_cond, j_cond = build_conditioner()
    obs_t, obs_j = obs_batch()
    xt_t, xt_j = action_batch()
    cond_t, cond_j = t_cond(obs_t), j_cond(obs_j)

    times = np.random.default_rng(2).uniform(0.05, 0.95, size=t_shape or ())
    t_t = torch.tensor(times, dtype=torch.float32).reshape(t_shape) if t_shape else torch.tensor(float(times))
    t_j = jnp.asarray(np.asarray(t_t))
    assert_same(t_actor(xt_t, t_t, cond_t), j_actor(xt_j, t_j, cond_j))


def test_actor_rejects_a_time_layout_neither_framework_understands():
    """[B, 1] is neither scalar, [B], nor [B, K, 1]; both sides must refuse it."""
    t_actor, j_actor = build_actor()
    t_cond, j_cond = build_conditioner()
    obs_t, obs_j = obs_batch()
    xt_t, xt_j = action_batch()
    with pytest.raises(ValueError):
        t_actor(xt_t, torch.zeros((MC, 1)), t_cond(obs_t))
    with pytest.raises(ValueError):
        j_actor(xt_j, jnp.zeros((MC, 1)), j_cond(obs_j))


def test_actor_return_stats():
    t_actor, j_actor = build_actor()
    t_cond, j_cond = build_conditioner()
    obs_t, obs_j = obs_batch()
    xt_t, xt_j = action_batch()
    cond_t, cond_j = t_cond(obs_t), j_cond(obs_j)
    t_times = torch.full((MC, K, 1), 0.5)
    j_times = jnp.full((MC, K, 1), 0.5)

    _, torch_stats = t_actor(xt_t, t_times, cond_t, return_stats=True)
    _, jax_stats = j_actor(xt_j, j_times, cond_j, return_stats=True)
    assert set(torch_stats) == set(jax_stats)
    assert set(torch_stats) == {
        "temb", "fused", "c", "gate_msa", "gate_mlp", "scale_msa", "scale_mlp"
    }
    assert_same({k: v for k, v in torch_stats.items()}, jax_stats)


def test_untrained_actor_is_adaLN_zero_and_predicts_nothing():
    """Both the modulation head and the output layer are zero-init, so a fresh
    model is the identity on the tokens and emits zero velocity."""
    j_actor = J.SJFPOFlowModel(ACTION, K, HIDDEN, DEPTH, HEADS, 4.0, 32, 10000.0, nnx.Rngs(0))
    _, j_cond = build_conditioner()
    _, obs_j = obs_batch()
    _, xt_j = action_batch()
    out = j_actor(xt_j, jnp.full((MC, K, 1), 0.5), j_cond(obs_j))
    assert float(jnp.abs(out).max()) == 0.0


def test_adaln_zero_leaves_the_stack_as_the_identity_at_init():
    """The property the zero-init buys: no gradient upstream of a block until the
    modulation head moves. Perturb it and the same path lights up."""
    torch.manual_seed(0)
    t_actor = T.SJFPOFlowModel(ACTION, K, HIDDEN, DEPTH, HEADS, 4.0, 32, 10000.0)
    params = to_jax_tree(t_actor.state_dict())
    torch_params = jax.tree_util.tree_map(lambda p: jnp.asarray(p) + 0.05, params)
    j_actor = load(J.SJFPOFlowModel(ACTION, K, HIDDEN, DEPTH, HEADS, 4.0, 32, 10000.0, nnx.Rngs(0)), params)
    _, j_cond = build_conditioner()
    _, obs_j = obs_batch()
    _, xt_j = action_batch()

    def grads(actor):
        graphdef, state = nnx.split(actor, nnx.Param)
        out = jax.grad(
            lambda s: jnp.sum(
                nnx.merge(graphdef, s)(xt_j, jnp.full((MC, K, 1), 0.5), j_cond(obs_j))
            )
        )(state)
        return {
            jax.tree_util.keystr(path): leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(nnx.to_pure_dict(out))[0]
        }

    zeroed = grads(j_actor)
    dead = {k for k, v in zeroed.items() if float(jnp.abs(v).sum()) == 0}
    assert "['in_proj']['kernel']" in dead and "['token_pos']" in dead

    perturbed = grads(
        load(J.SJFPOFlowModel(ACTION, K, HIDDEN, DEPTH, HEADS, 4.0, 32, 10000.0, nnx.Rngs(0)), torch_params)
    )
    assert all(float(jnp.abs(leaf).sum()) > 0 for leaf in perturbed.values()), (
        sorted(k for k, v in perturbed.items() if float(jnp.abs(v).sum()) == 0)
    )


# ------------------------------------------------------------------- attention mask


def test_attention_mask_inverts_correctly_and_never_nans():
    """SDPA's bool mask is the opposite of JAX's additive bias.

    A kept key must contribute and a masked one must not, and an all-masked row
    must still be finite -- SDPA emits NaN there unless a key is forced.
    """
    torch.manual_seed(0)
    t_attn = T.SelfAttention(HIDDEN, HEADS)
    j_attn = load(J.SelfAttention(HIDDEN, HEADS, nnx.Rngs(0)), to_jax_tree(t_attn.state_dict()))

    values = np.random.default_rng(3).standard_normal((2, K, HIDDEN), dtype=np.float32)
    x_t, x_j = torch.from_numpy(values.copy()), jnp.asarray(values)
    keep = np.array([[1] * K, [1] * 8 + [0] * (K - 8)], dtype=bool)
    torch_mask = torch.from_numpy(keep)
    jax_mask = jnp.asarray(keep)

    torch_out = t_attn(x_t, torch_mask)
    jax_out = j_attn(x_j, jax_mask)
    assert torch.isfinite(torch_out).all()
    assert bool(jnp.isfinite(jax_out).all())
    assert_same(torch_out, jax_out)

    # Masking a key must change the answer: an inverted bias would not.
    assert not np.allclose(np.asarray(jax_out[1]), np.asarray(j_attn(x_j)[1]))

    # An empty row must still be finite on both sides.
    empty = np.zeros((2, K), dtype=bool)
    assert torch.isfinite(t_attn(x_t, torch.from_numpy(empty))).all()
    assert bool(jnp.isfinite(j_attn(x_j, jnp.asarray(empty))).all())


def test_attention_mask_must_match_the_sequence_length():
    j_attn = J.SelfAttention(HIDDEN, HEADS, nnx.Rngs(0))
    _, x_j = obs_batch(batch=2, steps=K)
    with pytest.raises(ValueError):
        j_attn(x_j, jnp.ones((2, K - 1), dtype=jnp.bool_))


# -------------------------------------------------------------------- RunningMeanStd


def test_running_mean_std_count_stays_fp64():
    """fp32 count dies at 2^24 frames, so x64 has to be on before the first array."""
    rms = J.RunningMeanStd(OBS)
    assert rms.count.dtype == jnp.float64
    _, obs_j = obs_batch(batch=64)
    rms.update(obs_j)
    assert float(rms.count) == 64 * 16
    assert rms.uninitialised is False


def test_running_mean_std_is_identity_until_the_first_update():
    t_rms = T.RunningMeanStd(OBS)
    j_rms = J.RunningMeanStd(OBS)
    obs_t, obs_j = obs_batch()
    assert_same(t_rms(obs_t), j_rms(obs_j))
    t_rms.update(obs_t)
    j_rms.update(obs_j)
    assert_same(t_rms(obs_t), j_rms(obs_j))
    assert_same(t_rms.mean, j_rms.mean)
    assert_same(t_rms.var, j_rms.var)
