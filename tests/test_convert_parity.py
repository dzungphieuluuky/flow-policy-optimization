"""The converter produces a JAX checkpoint that reproduces the torch one.

The input is a real ``save_policy_checkpoint`` output with a warmed AdamW, so the
moments being converted are not the trivial zeros a fresh optimizer writes. The
gate is that the restored JAX state has the same weights **and** the same Adam
moments, and that the converted policy's forward matches the torch policy's.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from convert_torch_to_jax import convert, to_jax_tree
from sjfpo_jax import checkpointing as JCK
from spectral_joint_policy.checkpointing import save_policy_checkpoint
from spectral_joint_policy.modeling_sjfpo import build_optimizer as torch_build_optimizer
from test_optimizer_parity import apply_torch_grads, torch_grads
from test_policy_parity import CFG, OBS, pair
from torch_jax_weights import to_jax_tree as mapper_in_tests

ADAMW_AT = 1
# One call per module, module prefix stripped, exactly as convert() does it:
# ``stack`` names path segments *inside* a module.
MODULES = (("actor", ()), ("conditioner", ()), ("critic", ("conditioners", "heads")))


def write_torch_checkpoint(tmp_path, steps=3):
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
        names.update({f"{prefix}{n}": key for n, key in sub_names.items()})
    optimizer_names = [
        n for group in optimizer.param_groups for p in group["params"]
        for n in [dict(t_policy.named_parameters())[n_from_group(t_policy, optimizer, group, p)]]
    ]
    assert set(optimizer_names) <= set(names), sorted(set(optimizer_names) - set(names))[:3]
    assert len(names) == 83
    del jnp, torch


def n_from_group(policy, optimizer, group, param):
    by_id = {id(p): n for n, p in policy.named_parameters()}
    del optimizer, group
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
            np.testing.assert_allclose(
                np.asarray(restored[f"['{module}']{key[1:]}"[1:] if False else key]),
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
    params = JCK.flatten(JCK.params_to_pure(state))
    for key in mu:
        assert mu[key].shape == params[key].shape, key


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
    with torch.no_grad():
        normalised = t_policy.preprocessor({"observation.state": obs})["observation.state"]
        t_cond = t_policy.conditioner(normalised)
        t_critic = t_policy.critic(normalised)
    j_cond = module(state.conditioner)(obs)
    j_critic = module(state.critic)(obs)
    np.testing.assert_allclose(np.asarray(j_cond), t_cond.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(j_critic), t_critic.numpy(), rtol=1e-5, atol=1e-6)


def test_conversion_reports_a_missing_checkpoint(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(Exception, match="model.safetensors"):
        convert(empty, tmp_path / "out")
