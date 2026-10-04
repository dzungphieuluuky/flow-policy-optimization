#!/usr/bin/env python3
"""Convert a torch SJFPO checkpoint to the JAX learner's Orbax format.

    python scripts/convert_torch_to_jax.py --to-jax SRC DST

    SRC  a torch checkpoint dir: model.safetensors, optimizer.pt, ema.pt,
         counters.json, config.json
    DST  where to write the JAX checkpoint

What it reads, and why:

* ``model.safetensors`` -- live ``actor.*`` / ``encoder.*`` / ``critic.*`` /
  ``obs_normalizer.*``. The frozen ``old_actor``, ``old_encoder`` and
  ``target_critic`` are **not** in the file and are not needed: the snapshot is
  refreshed once per rollout, and the target critic is a Polyak shadow that
  hard-copies from the critic on its first update.
* ``optimizer.pt`` -- one AdamW ``state_dict`` over four param groups
  (actor-decay, actor-no-decay, encoder, critic), with ``state`` keyed by a
  **flat index** that runs through the groups in order. The moments are mapped
  onto the optax tree, so a resumed run continues the trajectory rather than
  restarting every moment from zero.
* ``ema.pt`` -- decay, latch and shadow; runtime state, restored as such.
* ``config.json`` / ``counters.json``.

``--to-torch`` is deliberately absent: nothing needs to read a JAX checkpoint
back until a JAX run exists to convert.

This script may import torch -- it is a torch-to-JAX converter. The
``sjfpo_jax`` package may not.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from safetensors.torch import load_file

from sjfpo_jax import checkpointing as JCK
from sjfpo_jax.configuration_sjfpo import SJFPOConfig
from sjfpo_jax.state import build_optimizer, build_state, optimizer_params

# The four torch param groups, in the order build_optimizer emits them
# (modeling_sjfpo.py:460-495). Only the ordering matters here.
GROUP_ACTOR_DECAY, GROUP_ACTOR_NO_DECAY, GROUP_CONDITIONER, GROUP_CRITIC = 0, 1, 2, 3
# The chain built by build_optimizer is
# [per_role_clip, adamw, add_decayed_weights(masked), scale_by_lr].
ADAMW_AT = 1


# ----------------------------------------------------------------- the weight map
#
# Two rules and no others: a 2-D ``.weight`` is a torch ``[out, in]`` Linear and
# becomes an NNX ``[in, out]`` kernel, and critic member indices stack onto one
# leading axis. The head is the one case stacking needs more than a transpose --
# torch holds ``[1, H]``, NNX wants ``[H, 1]``.
#
# Numeric path segments become list indices, because a ModuleList really is a
# list and a dict keyed ``'0'`` would flatten differently from the module it
# describes. This is the one place the torch/JAX layout difference is written
# down; ``tests/torch_jax_weights.py`` is the same function and a test asserts
# the two agree.

STACKED = {"encoders": "conditioners", "heads": "head"}


def rename_leaf(path: str, array) -> str:
    """torch leaf name -> NNX leaf name, given the leaf's shape."""
    name = path.rsplit(".", 1)[-1]
    if name == "weight" and array.ndim == 2:
        return path[: -len("weight")] + "kernel"
    if name == "weight" and "norm" in path:
        return path[: -len("weight")] + "scale"
    return path


def _empty_for(parts):
    return [] if parts[0].isdigit() else {}


def _insert(node, parts, value):
    head, rest = parts[0], parts[1:]
    if head.isdigit():
        node = node if isinstance(node, list) else []
        index = int(head)
        while len(node) <= index:
            node.append(None)
        if rest:
            child = node[index] if isinstance(node[index], (dict, list)) else _empty_for(rest)
            node[index] = _insert(child, rest, value)
        else:
            node[index] = value
        return node
    node = node if isinstance(node, dict) else {}
    if rest:
        node[head] = _insert(node.get(head), rest, value)
    else:
        node[head] = value
    return node


def nest(flat: dict) -> dict:
    """``{'a.0.b': x}`` -> ``{'a': [ {...} ]}``."""
    root: dict = {}
    for key in sorted(flat):
        root = _insert(root, key.split("."), flat[key])
    return root


def assign(node, tree) -> None:
    """Write a nested pure dict over a module's parameters, in place.

    ``nnx.replace_by_pure_dict`` is the documented way and it silently no-ops on
    an ``nnx.state(module, nnx.Param)`` view, so the attribute tree is walked.
    Dicts *and* lists recurse: the mapper builds real lists for the ModuleList
    indices, so ``blocks`` arrives as a list of block dicts.

    Merge-then-assign, rather than ``nnx.merge(graphdef, plain_dict)``: the
    latter unflattens against the graphdef's *node* types, and a hand-nested dict
    does not always match them even when every leaf key does.
    """
    if isinstance(tree, list):
        for index, item in enumerate(tree):
            assign(node[index], item)
        return
    for key, value in tree.items():
        index = int(key) if isinstance(node, list) else key
        child = node[index] if isinstance(node, list) else (
            node[key] if isinstance(node, dict) else getattr(node, key)
        )
        if isinstance(value, (dict, list)):
            assign(child, value)
        else:
            param = nnx.Param(jnp.asarray(value))
            if isinstance(node, list):
                node[index] = param
            elif isinstance(node, dict):
                node[key] = param
            else:
                setattr(node, key, param)


def load_pair(pair, params):
    """Apply ``params`` to a ``(graphdef, state)`` pair and re-split it."""
    graphdef, state = pair
    module = nnx.merge(graphdef, state)
    load(module, params)
    return nnx.split(module, nnx.Param)


def load(module, params):
    """Apply ``params`` to a module in place, and return it."""
    assign(module, params)
    return module


def to_jax_tree(sd, stack=()) -> tuple[dict, dict]:
    """torch state_dict -> ``(nested NNX tree, {torch name: flat jax key})``.

    The name table is what lets the optimizer converter address a torch param by
    name: it is built from leaf identity, so it cannot drift from the tree.
    """
    members = {name: STACKED[name] for name in stack}
    leaves: dict = {}
    originals: dict = {}
    for key, value in sd.items():
        array = np.asarray(value.detach().cpu().numpy()) if hasattr(value, "detach") else np.asarray(value)
        head, _, rest = key.partition(".")
        if head in members and rest.split(".")[0].isdigit():
            target = f"{members[head]}.{rename_leaf(rest.split('.', 1)[1], array)}"
            leaves.setdefault(target, []).append(array)
            originals.setdefault(target, []).append(key)
        else:
            target = rename_leaf(key, array)
            leaves[target] = array.T if array.ndim == 2 and key.endswith("weight") else array
            originals.setdefault(target, []).append(key)
    tree = {}
    for key, arrays in leaves.items():
        if isinstance(arrays, list):
            stacked = np.stack(arrays)
            # [num, out, in] -> [num, in, out]: transpose the last two axes only.
            # Biases and norm scales stack as they are.
            tree[key] = jnp.asarray(stacked.transpose(0, 2, 1) if stacked.ndim == 3 else stacked)
        else:
            tree[key] = jnp.asarray(arrays)
    nested = nest(tree)
    # Keystr per leaf. `tree_flatten_with_path` may hand back a different object
    # for the same leaf, so its paths are zipped onto `tree_leaves` by traversal
    # order rather than paired by identity.
    leaves = jax.tree_util.tree_leaves(nested)
    paths = [jax.tree_util.keystr(p).removesuffix(".value")
             for p, _ in jax.tree_util.tree_flatten_with_path(nested)[0]]
    flat_key_of = {id(leaf): key for leaf, key in zip(leaves, paths)}
    name_to_flat = {}
    for renamed, originals_list in originals.items():
        dotted = _dotted_leaf(nested, renamed)
        key = flat_key_of[id(dotted)]
        for original in originals_list:
            name_to_flat[original] = key
    return nested, name_to_flat


def _dotted_leaf(nested, dotted: str):
    """The leaf that ``dotted`` addresses, reached through real lists."""
    node = nested
    for part in dotted.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


# ------------------------------------------------------------------ the converter


def read_config(path: Path, obs_dim: int, action_dim: int) -> SJFPOConfig:
    """``config.json`` into the JAX dataclass, with the two dims filled in.

    torch's ``config.json`` is a LeRobot ``PreTrainedConfig`` dump, so it carries
    keys the JAX config has no field for (``type``, the hub settings). Filter to
    the fields we actually declare rather than blacklisting names: a new LeRobot
    field would otherwise crash the converter. The dims are not in the file at
    all -- they live on the PolicyFeature -- so they come from the safetensors.
    """
    import dataclasses

    raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
    known = {f.name for f in dataclasses.fields(SJFPOConfig)}
    fields = {k: v for k, v in raw.items() if k in known}
    fields["obs_dim"] = int(obs_dim)
    fields["action_dim"] = int(action_dim)
    return SJFPOConfig(**fields)


def observed_dims(weights: dict) -> tuple[int, int]:
    """``(obs_dim, action_dim)`` read off the shapes, since the torch config has
    neither: they live on the PolicyFeature, which the safetensors no longer
    carries.

    ``frame_proj`` is ``obs_dim -> hidden_dim``, so torch's ``[hidden, obs]``
    weight puts the obs width on the *input* axis. ``out`` is
    ``hidden_dim -> action_dim``, so the action width is its output axis.
    """
    obs_dim = int(weights["critic.encoders.0.frame_proj.weight"].shape[1])
    action_dim = int(weights["actor.out.weight"].shape[0])
    return obs_dim, action_dim


def convert(src: Path, dst: Path) -> int:
    from spectral_joint_policy.checkpointing import load_policy_checkpoint

    weights = load_file(str(src / "model.safetensors"))
    obs_dim, action_dim = observed_dims(weights)
    cfg = read_config(src, obs_dim, action_dim)

    # Rebuilding the torch policy is the only way to recover the param names in
    # group order; load_policy_checkpoint does it and validates the artifacts.
    policy, optimizer, counters = load_policy_checkpoint(src, device="cpu")
    name_by_id = {id(param): name for name, param in policy.named_parameters()}
    names = [
        name_by_id[id(param)] for group in optimizer.param_groups for param in group["params"]
    ]

    torch_opt = torch.load(src / "optimizer.pt", map_location="cpu", weights_only=True)

    model_sd = {k: v for k, v in weights.items() if not k.startswith("obs_normalizer.")}
    # One call per module, with the module prefix stripped, because ``stack``
    # names path segments *inside* a module: "critic.encoders.0.x" has to
    # arrive as "conditioners.0.x" or the member index is never dropped and the
    # critic ends up unstacked.
    trees, name_to_flat = {}, {}
    for module, stack in (
        ("actor", ()),
        ("encoder", ()),
        ("critic", ("encoders", "heads")),
    ):
        prefix = f"{module}."
        sub_sd = {k[len(prefix):]: v for k, v in model_sd.items() if k.startswith(prefix)}
        sub_tree, sub_names = to_jax_tree(sub_sd, stack=stack)
        trees[module] = sub_tree
        # Re-add the prefix: optimizer.pt names the parameters "actor.…", while
        # the mapper saw them with it stripped.
        name_to_flat.update({f"{prefix}{name}": key for name, key in sub_names.items()})
    tree = trees

    state = build_state(cfg, obs_dim, action_dim, jax.random.key(0))
    state = _load_from_tree(state, tree, weights)
    state = state.replace(global_step=int(counters["global_step"]))
    state = _restore_target_critic_latch(state, src)

    chain, opt_state = build_optimizer(cfg, optimizer_params(state))
    opt_state = _convert_optimizer(torch_opt, names, name_to_flat, opt_state)

    ema = torch.load(src / "ema.pt", map_location="cpu", weights_only=True) if (src / "ema.pt").is_file() else None
    if ema is not None:
        # The shadow is keyed by the *actor's* parameter names, so the same
        # prefix stripping applies.
        shadow_sd = {
            k[len("actor."):]: v
            for k, v in ema["shadow_params"].items()
            if k.startswith("actor.")
        }
        shadow_tree, _ = to_jax_tree(shadow_sd)
        from sjfpo_jax.policy import load_ema_state_dict

        state = load_ema_state_dict(state, {**ema, "shadow_params": shadow_tree})
    dst.mkdir(parents=True, exist_ok=True)
    JCK.save_checkpoint(dst, state, opt_state, cfg, step=int(counters["global_step"]))
    print(f"wrote {dst} at step {counters['global_step']} ({len(names)} params)")
    return 0


def _load_from_tree(state, tree: dict, weights: dict):
    for name in ("actor", "encoder", "critic"):
        state = state.replace(**{name: load_pair(getattr(state, name), tree[name])})
    normalizer = state.obs_normalizer
    saved = {k[len("obs_normalizer."):]: v for k, v in weights.items() if k.startswith("obs_normalizer.")}
    normalizer.mean = jnp.asarray(np.asarray(saved["mean"]), dtype=jnp.float32)
    normalizer.var = jnp.asarray(np.asarray(saved["var"]), dtype=jnp.float32)
    normalizer.count = jnp.asarray(np.asarray(saved["count"]), dtype=jnp.float64)
    normalizer.uninitialised = bool(float(normalizer.count) == 0)
    return state


def _restore_target_critic_latch(state, src: Path):
    """The target critic is a Polyak shadow, not a checkpointed artifact.

    It starts uninitialised so its first update hard-copies from the restored
    critic rather than blending into whatever the fresh module happened to be
    built with (checkpointing.py:214-217).
    """
    del src
    return state.replace(target_critic_initialised=False)


def _convert_optimizer(torch_opt: dict, names: list[str], name_to_flat: dict, opt_state):
    """Copy exp_avg / exp_avg_sq / step onto the optax AdamW state.

    optax 0.2.8's adamw state is a ``(count, mu, nu)`` triple at index
    ``ADAMW_AT`` of the chain, with ``mu``/``nu`` as NNX States (the chain is
    built from NNX params, so the moments follow their structure).
    """
    count, mu, nu = opt_state[ADAMW_AT][0].count, opt_state[ADAMW_AT][0].mu, opt_state[ADAMW_AT][0].nu
    # mu/nu mirror the parameter tree, which is plain nested dicts of States, so
    # flatten them directly -- to_pure_dict only accepts a single State.
    flat_mu = JCK.flatten(mu)
    flat_nu = JCK.flatten(nu)
    steps = []
    for position, name in enumerate(names):
        entry = torch_opt["state"].get(position)
        if entry is None:
            continue
        key = name_to_flat.get(name)
        if key is None:
            raise KeyError(f"optimizer param {name!r} has no JAX counterpart")
        flat_mu[key] = jnp.asarray(entry["exp_avg"].numpy(), dtype=jnp.float32)
        flat_nu[key] = jnp.asarray(entry["exp_avg_sq"].numpy(), dtype=jnp.float32)
        steps.append(int(entry["step"]))
    if not steps:
        return opt_state
    new_adamw_state = opt_state[ADAMW_AT][0]._replace(
        count=jnp.asarray(max(steps), dtype=jnp.asarray(count).dtype),
        mu=JCK.unflatten_like(mu, flat_mu),
        nu=JCK.unflatten_like(nu, flat_nu),
    )
    return (
        opt_state[:ADAMW_AT]
        + ((new_adamw_state,) + opt_state[ADAMW_AT][1:],)
        + opt_state[ADAMW_AT + 1:]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a torch SJFPO checkpoint to the JAX format.")
    parser.add_argument("--to-jax", required=True, nargs=2, metavar=("SRC", "DST"))
    args = parser.parse_args()
    src, dst = Path(args.to_jax[0]), Path(args.to_jax[1])
    if not (src / "model.safetensors").is_file():
        print(f"ERROR: {src} has no model.safetensors", file=sys.stderr)
        return 1
    return convert(src, dst)


if __name__ == "__main__":
    raise SystemExit(main())
