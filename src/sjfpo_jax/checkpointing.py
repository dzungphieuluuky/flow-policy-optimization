"""Orbax save/restore for the JAX learner.

The tree that goes on disk is deliberately **values only**: an NNX ``graphdef``
is structural and rebuildable from the config, so it is not saved. Load rebuilds
the modules from a freshly built state's graphdef and merges the stored values
into them -- the same trick :func:`policy.load_ema_state_dict` uses.

What the torch checkpoint holds, and therefore what has to be converted
(``checkpointing.py:66-145``):

| torch artifact | contents | note |
|---|---|---|
| ``model.safetensors`` | live ``actor.*``, ``conditioner.*``, ``critic.*``, ``obs_normalizer.*`` | ``old_actor.*``, ``old_conditioner.*`` and ``target_critic.*`` are **dropped**; the target critic is a Polyak shadow, rebuilt from the critic on its first update |
| ``ema.pt`` | decay, ``initialised``, shadow params by name | a separate file, not in the safetensors |
| ``optimizer.pt`` | one AdamW ``state_dict``, four param groups | group sizes 28/12/12/28 at the default config |
| ``counters.json`` | ``{global_step, step_count}`` | |

**Donation hazard.** If the optimizer's params were passed with
``donate_argnums``, the donated buffer would be invalid after the call and
Orbax's save would still hold a reference to it. Nothing here donates, and the
params are small enough (~0.5-1M) that copying them is free.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

from sjfpo_jax.configuration_sjfpo import SJFPOConfig
from sjfpo_jax.state import SJFPOState, build_optimizer, build_state, optimizer_params

# Subtrees that are trainable parameters, and the one that is a buffer.
PARAM_FIELDS = ("actor", "conditioner", "critic")


def params_to_pure(state: SJFPOState) -> dict:
    """The values to save: parameters plus the normaliser's buffers."""
    out = {name: nnx.to_pure_dict(getattr(state, name)[1]) for name in PARAM_FIELDS}
    out["obs_normalizer"] = {
        "mean": state.obs_normalizer.mean,
        "var": state.obs_normalizer.var,
        "count": state.obs_normalizer.count,
    }
    return out


def params_into_state(state: SJFPOState, saved: dict) -> SJFPOState:
    """The inverse of :func:`params_to_pure`, against a freshly built state."""
    for name in PARAM_FIELDS:
        graphdef = getattr(state, name)[0]
        merged = nnx.split(nnx.merge(graphdef, saved[name]), nnx.Param)
        state = state.replace(**{name: merged})
    normalizer = state.obs_normalizer
    saved_normalizer = saved["obs_normalizer"]
    normalizer.mean = saved_normalizer["mean"]
    normalizer.var = saved_normalizer["var"]
    normalizer.count = saved_normalizer["count"]
    # count == 0 means the latch is off, which is how the torch original
    # re-derives it in _load_from_state_dict.
    normalizer.uninitialised = bool(saved_normalizer["count"] == 0)
    return state


CONFIG_FILE = "config.json"


def checkpoint_tree(state: SJFPOState, opt_state, cfg: SJFPOConfig, step: int) -> dict:
    """The numeric half of one checkpoint.

    No config here: Orbax's StandardSave cannot serialise the config's string
    fields, so the config goes out as a sibling ``config.json``, which is also
    where the torch checkpoint keeps it.

    ``opt_state`` is stored **flattened**, and that is not a stylistic choice.
    Orbax round-trips a namedtuple into a plain dict, turns optax's ``EmptyState``
    into ``None``, and hands the outer tuple back as a list -- a 133-leaf Adam
    state came back as 66 leaves, so a resumed run's optimizer was unusable. The
    container types are all rebuildable from ``chain.init(params)``, so only the
    leaves are worth keeping; :func:`restore_from_tree` unflattens onto a fresh
    template.
    """
    return {
        "step": int(step),
        "counters": {"global_step": int(state.global_step), "step_count": 0},
        "params": params_to_pure(state),
        "opt_state": flatten(opt_state),
        "ema": {
            "decay": float(cfg.ema_decay),
            "initialised": bool(state.ema_initialised),
            "target_critic_initialised": bool(state.target_critic_initialised),
            # Keyed ``shadow_params`` because that is what
            # ``policy.load_ema_state_dict`` reads. It used to be ``shadow``, and
            # the lookup silently missed, so a resumed run came back with no
            # shadow and a cleared latch -- and re-hard-copied, throwing the
            # running average away.
            "shadow_params": None
            if state.ema_shadow is None
            else nnx.to_pure_dict(state.ema_shadow),
        },
    }


def restore_from_tree(tree: dict, cfg: SJFPOConfig) -> tuple[SJFPOState, Any, dict]:
    """Rebuild ``(state, opt_state, counters)`` from a tree plus its config."""
    state = build_state(cfg, cfg.obs_dim, cfg.action_dim, jax.random.key(0))
    state = params_into_state(state, tree["params"])
    state = state.replace(
        global_step=int(tree["counters"]["global_step"]),
        target_critic_initialised=bool(tree["ema"]["target_critic_initialised"]),
    )
    if tree["ema"]["shadow_params"] is not None:
        from sjfpo_jax.policy import load_ema_state_dict

        state = load_ema_state_dict(state, tree["ema"])
    _, template = build_optimizer(cfg, optimizer_params(state))
    return state, unflatten_like(template, tree["opt_state"]), tree["counters"]


def _manager(path: str | Path, keep: int = 2) -> ocp.CheckpointManager:
    # orbax 0.12.6 exposes retention as max_to_keep on the manager options (there
    # is no ocp.PreserveLatest to pass), and the save args go on manager.save.
    return ocp.CheckpointManager(
        # Orbax rejects a relative path outright, and the CLI is often given one.
        Path(path).resolve(),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=keep,
            enable_async_checkpointing=False,
        ),
    )


def save_checkpoint(
    path: str | Path, state: SJFPOState, opt_state, cfg: SJFPOConfig, step: int
) -> None:
    """Write one checkpoint.

    Synchronous on purpose: Orbax's async save plus ``donate_argnums`` is the
    wrong pair, and the tree is small enough that waiting for it costs nothing.
    """
    manager = _manager(path)
    manager.save(int(step), args=ocp.args.StandardSave(checkpoint_tree(state, opt_state, cfg, step)))
    manager.close()
    (Path(path) / CONFIG_FILE).write_text(
        json.dumps(dataclasses.asdict(cfg), indent=2), encoding="utf-8"
    )


def load_checkpoint(
    path: str | Path, step: int | None = None
) -> tuple[SJFPOState, Any, SJFPOConfig, dict]:
    """Restore ``(state, opt_state, cfg, counters)``. ``step=None`` takes the latest."""
    manager = _manager(path)
    if step is None:
        step = manager.latest_step()
        if step is None:
            raise FileNotFoundError(f"no checkpoint under {path}")
    tree = manager.restore(int(step))
    manager.close()
    cfg = SJFPOConfig(**json.loads((Path(path) / CONFIG_FILE).read_text(encoding="utf-8")))
    state, opt_state, counters = restore_from_tree(tree, cfg)
    return state, opt_state, cfg, counters


def flatten(tree) -> dict:
    """Any pytree -> ``{flat key: array}``, Param wrappers stripped.

    NNX States flatten *through* Param, so a State and an equal plain dict
    produce the same keys once the trailing ``.value`` is dropped. That is what
    lets the converter address a torch parameter by name.
    """
    return {
        jax.tree_util.keystr(path).removesuffix(".value"): leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
    }


def unflatten_like(template, flat: dict):
    """Rebuild ``template``'s structure from a flat ``{key: array}`` dict.

    Shape-checked, because a wrong-rank leaf is exactly the failure this port
    made twice: the converter used to write one critic member's moments over the
    whole stacked array, and a kernel's moments were never transposed to
    ``[in, out]``. Both passed a pure key lookup and only showed up as a
    corrupted optimizer state much later.
    """
    entries = jax.tree_util.tree_flatten_with_path(template)[0]
    paths = [jax.tree_util.keystr(path).removesuffix(".value") for path, _ in entries]
    missing = [k for k in paths if k not in flat]
    if missing:
        raise KeyError(f"flat tree is missing {len(missing)} leaves, e.g. {missing[:3]}")
    mismatched = [
        (key, tuple(jnp.shape(flat[key])), tuple(leaf.shape))
        for key, (_, leaf) in zip(paths, entries)
        if tuple(jnp.shape(flat[key])) != tuple(leaf.shape)
    ]
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} leaves have the wrong shape, e.g. "
            f"{mismatched[0][0]}: got {mismatched[0][1]}, template wants {mismatched[0][2]}"
        )
    return jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(template), [flat[path] for path in paths]
    )
