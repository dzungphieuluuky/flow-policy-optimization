"""The converter produces a JAX checkpoint that reproduces the torch one.

The input is a real ``save_policy_checkpoint`` output with a warmed AdamW, so the
moments being converted are not the trivial zeros a fresh optimizer writes. The
gate is that the restored JAX state has the same weights **and** the same Adam
moments, and that the converted policy's forward matches the torch policy's.
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from convert_torch_to_jax import convert, to_jax_tree
from test_optimizer_parity import apply_torch_grads, torch_grads
from test_policy_parity import CFG, OBS, pair
from torch_jax_weights import to_jax_tree as mapper_in_tests

from sjfpo_jax import checkpointing as JCK
from sjfpo_jax.state import optimizer_params
from spectral_joint_policy.checkpointing import save_policy_checkpoint
from spectral_joint_policy.modeling_sjfpo import (
    build_optimizer as torch_build_optimizer,
)

ADAMW_AT = 1
# One call per module, module prefix stripped, exactly as convert() does it:
# ``stack`` names path segments *inside* a module.
MODULES = (("actor", ()), ("conditioner", ()), ("critic", ("conditioners", "heads")))


def write_torch_checkpoint(tmp_path, steps=3, ema=True):
    t_policy, optimizer, path = write_torch_checkpoint_plain(tmp_path, steps)
    if ema:
        # A real ema.pt, not None: the converter's shadow path keys off this file
        # existing, so writing None leaves the whole conversion untested.
        t_policy._ema_shadow = {
            name: torch.full_like(param, 0.125)
            for name, param in t_policy.actor.named_parameters()
        }
        t_policy._ema_initialised = True
        torch.save(t_policy.ema_state_dict(), path / "ema.pt")
    return t_policy, optimizer, path


def write_torch_checkpoint_plain(tmp_path, steps=3):
    t_policy, _ = pair(CFG, seed=0)
    optimizer = torch_build_optimizer(t_policy)
    t_grads = torch_grads(t_policy)
    for _ in range(steps):
        apply_torch_grads(t_policy, t_grads)
        optimizer.step()
    path = tmp_path / "torch_ckpt"
    save_policy_checkpoint(path, policy=t_policy, optimizer=optimizer, global_step=103, ema_state=None)
    return t_policy, optimizer, path


def per_module(weights, mapper):
    """``{module: nested tree}`` using ``mapper`` once per module."""
    return {
        module: mapper(
            {k[len(module) + 1:]: v for k, v in weights.items() if k.startswith(f"{module}.")},
            stack=stack,
        )
        for module, stack in MODULES
    }


def test_the_script_mapper_agrees_with_the_test_mapper(tmp_path):
    """The weight mapping is written down twice -- once in the script, once in the
    test helper. If they ever diverge the converter is silently wrong, so pin them."""
    weights = load_file(str(write_torch_checkpoint(tmp_path)[2] / "model.safetensors"))
    model_weights = {k: v for k, v in weights.items() if not k.startswith("obs_normalizer.")}
    from_script = per_module(model_weights, lambda sd, stack: to_jax_tree(sd, stack=stack)[0])
    from_test = per_module(model_weights, lambda sd, stack: mapper_in_tests(sd, stack=stack))
    for module, _ in MODULES:
        a, b = JCK.flatten(from_script[module]), JCK.flatten(from_test[module])
        assert set(a) == set(b), module
        for key in b:
            np.testing.assert_allclose(np.asarray(a[key]), np.asarray(b[key]), rtol=1e-6, atol=1e-8)


def test_the_name_table_covers_every_optimizer_parameter(tmp_path):
    """optimizer.pt refers to params by torch name; the table is what bridges them."""
    t_policy, optimizer, path = write_torch_checkpoint(tmp_path)
    weights = load_file(str(path / "model.safetensors"))
    model_weights = {k: v for k, v in weights.items() if not k.startswith("obs_normalizer.")}
    names = {}
    for module, stack in MODULES:
        prefix = f"{module}."
        sub = {k[len(prefix):]: v for k, v in model_weights.items() if k.startswith(prefix)}
        _, sub_names = to_jax_tree(sub, stack=stack)
        names.update({f"{prefix}{n}": entry for n, entry in sub_names.items()})
    optimizer_names = [
        n_from_group(t_policy, group, p)
        for group in optimizer.param_groups
        for p in group["params"]
    ]
    assert set(optimizer_names) <= set(names), sorted(set(optimizer_names) - set(names))[:3]
    # 80 safetensors leaves, and 80 names out: the critic's 28 torch params stack
    # onto 14 JAX leaves but the table is keyed by torch name, so it stays 1:1.
    assert len(names) == 80, len(names)
    # Every torch name must address a (key, slot) pair, and the critic's members
    # must land on distinct slots of the same key.
    critic = {n: v for n, v in names.items() if n.startswith("critic.conditioners.0.")}
    twin = {n.replace("conditioners.0.", "conditioners.1."): v for n, v in critic.items()}
    for zero, one in zip(critic, twin):
        assert names[zero][0] == names[one][0], zero
        assert names[zero][1] == 0 and names[one][1] == 1, (zero, names[zero], names[one])


def n_from_group(policy, group, param):
    by_id = {id(p): n for n, p in policy.named_parameters()}
    return by_id[id(param)]


def test_conversion_reproduces_the_weights(tmp_path):
    _, _, src = write_torch_checkpoint(tmp_path)
    dst = tmp_path / "jax_ckpt"
    assert convert(src, dst) == 0
    state, _, cfg, counters = JCK.load_checkpoint(dst)
    assert cfg.obs_dim == OBS and cfg.action_dim == 2
    assert counters["global_step"] == 103 and state.global_step == 103

    weights = load_file(str(src / "model.safetensors"))
    expected = per_module(
        {k: v for k, v in weights.items() if not k.startswith("obs_normalizer.")},
        lambda sd, stack: mapper_in_tests(sd, stack=stack),
    )
    restored = JCK.flatten(JCK.params_to_pure(state))
    for module, _ in MODULES:
        for key, value in JCK.flatten(expected[module]).items():
            # params_to_pure is keyed by module, so the module prefix belongs on
            # the lookup side: "['actor']['adaLN_out']['bias']", not
            # "['adaLN_out']['bias']".
            np.testing.assert_allclose(
                np.asarray(restored[f"['{module}']{key}"]),
                np.asarray(value), rtol=1e-6, atol=1e-8,
            )


def test_conversion_reproduces_the_adam_moments(tmp_path):
    _, _, src = write_torch_checkpoint(tmp_path)
    dst = tmp_path / "jax_ckpt"
    assert convert(src, dst) == 0
    state, opt_state, _, _ = JCK.load_checkpoint(dst)

    adam = opt_state[ADAMW_AT][0]
    assert int(np.asarray(adam.count)) == 3, "three torch steps should land as count == 3"
    mu, nu = JCK.flatten(adam.mu), JCK.flatten(adam.nu)
    # Every moment is non-zero: a fresh optax state would be all zeros, so this
    # is what proves the moments were actually converted.
    assert all(float(jnp.abs(v).sum()) > 0 for v in mu.values())
    assert all(float(jnp.abs(v).sum()) > 0 for v in nu.values())
    # Each moment mirrors the parameter it belongs to, so compare against the
    # *optimizer's* regrouping. params_to_pure is keyed by module, which is a
    # different keyspace: '["actor"]["actor"][...]' vs '["actor"][...]'.
    params = JCK.flatten(optimizer_params(state))
    for key in mu:
        assert mu[key].shape == params[key].shape, key
    # The critic's moments are one stacked leaf over both members, not two
    # half-overwritten ones -- the failure this test exists to catch.
    critic_bias = mu["['critic']['critic']['conditioners']['frame_mlp']['down']['bias']"]
    assert critic_bias.shape == (CFG.num_critics, CFG.hidden_dim), critic_bias.shape
    assert np.asarray(critic_bias)[0].any() and np.asarray(critic_bias)[1].any()


def test_the_converted_policy_agrees_with_torch(tmp_path):
    """The end the whole converter exists for: same weights, same prediction."""
    t_policy, _, src = write_torch_checkpoint(tmp_path)
    dst = tmp_path / "jax_ckpt"
    assert convert(src, dst) == 0
    state, _, cfg, _ = JCK.load_checkpoint(dst)

    from sjfpo_jax.policy import module

    obs = jnp.asarray(
        np.random.default_rng(5).standard_normal((3, cfg.n_obs_steps, cfg.obs_dim), dtype=np.float32)
    )
    # Both sides get the *normalised* window. The torch side goes through the
    # policy's preprocessor; the jax side through the restored running stats,
    # which is the same object. Comparing torch-on-normalised against
    # jax-on-raw measures the normaliser, not the conversion.
    with torch.no_grad():
        normalised = t_policy.preprocessor({"observation.state": torch.from_numpy(np.asarray(obs))})["observation.state"]
        t_cond = t_policy.conditioner(normalised)
        t_critic = t_policy.critic(normalised)
    normalised_j = state.obs_normalizer(obs)
    j_cond = module(state.conditioner)(normalised_j)
    j_critic = module(state.critic)(normalised_j)
    np.testing.assert_allclose(np.asarray(j_cond), t_cond.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(j_critic), t_critic.numpy(), rtol=1e-5, atol=1e-6)


def test_the_converted_checkpoint_carries_the_ema_shadow(tmp_path):
    """The shadow is keyed by the actor module's own names, already unprefixed.

    The converter used to filter on an ``actor.`` prefix that torch never writes,
    which matched nothing and produced an empty shadow. ``write_torch_checkpoint``
    also used to pass ``ema_state=None``, so the file the converter looks for was
    absent and this whole path never ran.
    """
    _, _, src = write_torch_checkpoint(tmp_path)
    assert (src / "ema.pt").is_file(), "the converter only looks for ema.pt if it exists"
    dst = tmp_path / "jax_ckpt"
    assert convert(src, dst) == 0
    state, _, _, _ = JCK.load_checkpoint(dst)

    assert state.ema_initialised is True
    assert state.ema_shadow is not None
    leaves = JCK.flatten(nnx.to_pure_dict(state.ema_shadow))
    assert leaves, "the shadow must not come back empty"
    # torch's shadow was filled with a constant, so every leaf must be that value.
    for key, value in leaves.items():
        np.testing.assert_allclose(np.asarray(value), 0.125, rtol=1e-6, atol=1e-8, err_msg=key)


def test_conversion_without_an_ema_file_leaves_the_shadow_absent(tmp_path):
    """The other direction: no ema.pt means no shadow, not an empty one."""
    _, _, src = write_torch_checkpoint_plain(tmp_path)
    assert not (src / "ema.pt").exists()
    dst = tmp_path / "jax_ckpt"
    assert convert(src, dst) == 0
    state, _, _, _ = JCK.load_checkpoint(dst)
    assert state.ema_shadow is None
    assert state.ema_initialised is False


def test_conversion_reports_a_missing_checkpoint(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(Exception, match="model.safetensors"):
        convert(empty, tmp_path / "out")
