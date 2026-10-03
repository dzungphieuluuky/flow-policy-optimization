"""torch ``state_dict`` -> the NNX parameter layout, for the parity tests.

Two rules and no others: a 2-D ``.weight`` is a torch ``[out, in]`` Linear and
becomes an NNX ``[in, out]`` kernel, and critic member indices stack onto one
leading axis. The head is the one case stacking needs more than a transpose:
torch holds ``[1, H]``, NNX wants ``[H, 1]``.

``scripts/convert_torch_to_jax.py`` takes this over at milestone 6; until then it
is the only place the layout difference is written down.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

STACKED = {"encoders": "conditioners", "heads": "head"}


def rename_leaf(path: str, array) -> str:
    """torch leaf name -> NNX leaf name, given the leaf's shape."""
    name = path.rsplit(".", 1)[-1]
    if name == "weight" and array.ndim == 2:
        return path[: -len("weight")] + "kernel"
    if name == "weight" and "norm" in path:
        return path[: -len("weight")] + "scale"
    return path


def nest(flat: dict) -> dict:
    """``{'a.0.b': x}`` -> ``{'a': [ {...} ]}``, the shape the module really has.

    Numeric path segments become **list indices**, not string keys. A torch
    ``state_dict`` addresses a ModuleList as ``blocks.0.adaLN.fc1.weight``, and
    the NNX module holds a real list there, so building a dict with the key
    ``'0'`` would produce a tree that flattens differently from the one it is
    supposed to describe.
    """
    root: dict = {}
    for key in sorted(flat):
        root = _insert(root, key.split("."), flat[key])
    return root


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


def to_jax_tree(sd, stack=()) -> dict:
    """torch ``state_dict`` (or grad dict) -> the nested NNX pure-dict layout."""
    members = {name: STACKED[name] for name in stack}
    leaves: dict = {}
    for key, value in sd.items():
        array = np.asarray(value.detach().cpu().numpy()) if hasattr(value, "detach") else np.asarray(value)
        head, _, rest = key.partition(".")
        if head in members and rest.split(".")[0].isdigit():
            leaves.setdefault(f"{members[head]}.{rename_leaf(rest.split('.', 1)[1], array)}", []).append(array)
        else:
            leaves[rename_leaf(key, array)] = array.T if array.ndim == 2 and key.endswith("weight") else array
    tree = {}
    for key, arrays in leaves.items():
        if not isinstance(arrays, list):
            tree[key] = jnp.asarray(arrays)
            continue
        stacked = np.stack(arrays)
        # A stacked kernel is [num, out, in] and wants [num, in, out]: transpose
        # the last two axes only. Biases and norm scales stack as they are.
        tree[key] = jnp.asarray(stacked.transpose(0, 2, 1) if stacked.ndim == 3 else stacked)
    return nest(tree)


def assign(node, tree) -> None:
    """Write a nested pure dict over a module's existing parameters, in place.

    ``nnx.replace_by_pure_dict`` is the documented way to do this and it silently
    no-ops on an ``nnx.state(module, nnx.Param)`` view, so walk the attribute tree.
    Dicts *and* lists recurse: the mapper builds real lists for the ModuleList
    indices, so ``blocks`` arrives as a list of two block dicts.
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


def load(module, params):
    """Overwrite an NNX module's parameters in place from a nested pure dict."""
    assign(module, params)
    return module


def load_pair(pair, params):
    """Same, for a ``(graphdef, state)`` pair: re-split and hand the pair back."""
    graphdef, state = pair
    module = nnx.merge(graphdef, state)
    load(module, params)
    return nnx.split(module, nnx.Param)


def flatten(tree, prefix: str = "") -> dict:
    """Any pytree -> ``{'a.b.c': array}``, string keys throughout.

    Uses jax's own flattening so NNX ``State`` objects, plain dicts and lists all
    work, and normalises the keys to strings because NNX keys list indices with
    ``int`` and dict keys with ``str`` -- the two trees are structurally identical
    but not key-identical.
    """
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        # An NNX State flattens *through* Param, so its path ends in `.value`;
        # a plain dict of the same arrays does not. Dropping the suffix makes
        # the two forms comparable.
        key = jax.tree_util.keystr(path)
        out[key.removesuffix(".value")] = leaf
    return out
