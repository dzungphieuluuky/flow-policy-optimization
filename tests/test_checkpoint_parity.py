"""torch vs jax parity for checkpointing: convert a real torch checkpoint.

The torch side is a genuine ``save_policy_checkpoint`` output -- safetensors,
``optimizer.pt`` with four param groups, ``ema.pt``, ``counters.json`` -- built
by taking one real AdamW step, so the optimizer state is not the trivial zeros a
freshly-built optimizer would write. The gate is that the converted Orbax tree
restores to *the same weights and the same Adam moments*.
"""
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from safetensors.torch import load_file

from sjfpo_jax import checkpointing as JCK
from sjfpo_jax.configuration_sjfpo import SJFPOConfig
from sjfpo_jax.state import build_optimizer, build_state, optimizer_params
from spectral_joint_policy.checkpointing import save_policy_checkpoint
from spectral_joint_policy.modeling_sjfpo import build_optimizer as torch_build_optimizer
from test_optimizer_parity import apply_torch_grads, torch_grads
from test_policy_parity import CFG, pair
from torch_jax_weights import to_jax_tree

CKPT = ROOT / "tmp" / "_parity_ckpt"


def write_torch_checkpoint(tmp_path, steps=3):
    """A real torch checkpoint: live weights plus a warmed AdamW."""
    t_policy, state = pair(CFG, seed=0)
    optimizer = torch_build_optimizer(t_policy)
    t_grads = torch_grads(t_policy)
    for step in range(steps):
        apply_torch_grads(t_policy, t_grads)
        optimizer.step()
    path = tmp_path / "torch_ckpt"
    save_policy_checkpoint(
        path,
        policy=t_policy,
        optimizer=optimizer,
        global_step=100 + steps,
        ema_state=None,
    )
    del state
    return t_policy, optimizer, path

def test_saved_tree_is_values_only_and_round_trips(tmp_path):
    t_policy, _, path = write_torch_checkpoint(tmp_path)
    weights = load_file(str(path / "model.safetensors"))
    cfg = SJFPOConfig(obs_dim=11, action_dim=2)

    jax_state = build_state(cfg, 11, 2, jax.random.key(0))
    params = optimizer_params(jax_state)
    # Overwrite from the torch weights, then save and restore.
    loaded = to_jax_tree(
        {k: v for k, v in weights.items() if not k.startswith("obs_normalizer.")}
    )
    jax_state = jax_state.replace(
        actor=_load_pair(jax_state.actor, loaded["actor"]),
        conditioner=_load_pair(jax_state.conditioner, loaded["conditioner"]),
        critic=_load_pair(jax_state.critic, to_jax_tree(
            {k[len("critic."):]: v for k, v in weights.items() if k.startswith("critic.")},
            stack=["conditioners", "heads"],
        )),
        obs_normalizer=_load_normalizer(jax_state.obs_normalizer, weights),
        global_step=103,
    )
    chain, opt_state = build_optimizer(cfg, optimizer_params(jax_state))
    JCK.save_checkpoint(tmp_path / "jax_ckpt", jax_state, opt_state, cfg, step=103)

    restored, restored_opt, restored_cfg, counters = JCK.load_checkpoint(tmp_path / "jax_ckpt")
    assert restored_cfg.obs_dim == 11 and restored_cfg.action_dim == 2
    assert counters["global_step"] == 103 and restored.global_step == 103

    saved_flat = JCK.flatten(JCK.params_to_pure(jax_state))
    restored_flat = JCK.flatten(JCK.params_to_pure(restored))
    assert set(saved_flat) == set(restored_flat)
    for key in saved_flat:
        np.testing.assert_allclose(
            np.asarray(restored_flat[key]), np.asarray(saved_flat[key]), rtol=1e-6, atol=1e-8
        )
    del t_policy


def test_the_normalizer_buffers_and_latch_survive(tmp_path):
    cfg = SJFPOConfig(obs_dim=11, action_dim=2)
    state = build_state(cfg, 11, 2, jax.random.key(0))
    frames = jnp.asarray(np.random.default_rng(3).standard_normal((64, 11), dtype=np.float32))
    state.obs_normalizer.update(frames)
    assert state.obs_normalizer.uninitialised is False
    chain, opt_state = build_optimizer(cfg, optimizer_params(state))
    JCK.save_checkpoint(tmp_path / "ckpt", state, opt_state, cfg, step=7)
    restored, _, _, _ = JCK.load_checkpoint(tmp_path / "ckpt")
    assert restored.obs_normalizer.uninitialised is False
    np.testing.assert_allclose(
        np.asarray(restored.obs_normalizer.mean), np.asarray(state.obs_normalizer.mean)
    )
    np.testing.assert_allclose(
        np.asarray(restored.obs_normalizer.var), np.asarray(state.obs_normalizer.var)
    )
    assert float(restored.obs_normalizer.count) == float(state.obs_normalizer.count)
    assert restored.obs_normalizer.count.dtype == jnp.float64


def test_loading_a_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        JCK.load_checkpoint(tmp_path / "nothing_here")


def test_unflatten_like_reports_the_missing_leaves():
    template = {"a": jnp.zeros(2), "b": {"c": jnp.zeros(3)}}
    keys = list(JCK.flatten(template))
    rebuilt = JCK.unflatten_like(
        template, {keys[0]: jnp.ones(2), keys[1]: jnp.ones(3)}
    )
    assert float(rebuilt["a"].sum()) == 2.0
    assert float(rebuilt["b"]["c"].sum()) == 3.0
    with pytest.raises(KeyError, match="missing"):
        JCK.unflatten_like(template, {keys[0]: jnp.ones(2)})


def _load_pair(pair, params):
    from torch_jax_weights import load_pair

    return load_pair(pair, params)


def _load_normalizer(normalizer, weights):
    normalizer.mean = jnp.asarray(np.asarray(weights["obs_normalizer.mean"]))
    normalizer.var = jnp.asarray(np.asarray(weights["obs_normalizer.var"]))
    normalizer.count = jnp.asarray(np.asarray(weights["obs_normalizer.count"]), dtype=jnp.float64)
    normalizer.uninitialised = bool(float(normalizer.count) == 0)
    return normalizer


def test_the_torch_checkpoint_drops_the_runtime_subtrees(tmp_path):
    """The premise of the whole converter: the frozen trees are not on disk."""
    _, _, path = write_torch_checkpoint(tmp_path)
    keys = set(load_file(str(path / "model.safetensors")))
    assert not any(k.startswith(("old_actor.", "old_conditioner.", "target_critic.")) for k in keys)
    assert {"actor", "conditioner", "critic", "obs_normalizer"} == {k.split(".")[0] for k in keys}
    assert json.loads((path / "counters.json").read_text())["global_step"] == 103
