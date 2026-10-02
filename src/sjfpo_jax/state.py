"""SJFPOState: the pytree the loss reads. No optimizer here -- that is §5.

The networks are stored as ``(graphdef, state)`` pairs rather than as modules. A
module captured in a closure is the flax pitfall in JAX_MIGRATION.md §6(b); a
pair is an ordinary pytree, so ``jax.grad`` and ``jax.jit`` work on the whole
state with no plumbing at the boundaries. ``policy.py`` rebuilds a module at
trace time with ``nnx.merge(*pair)``.

``obs_normalizer`` and the action bounds are state rather than code: both are
mutated once per rollout and read inside the loss. They live and die with the
step, not with the process.

The PRNG key is deliberately **not** here. It is threaded as its own argument
(``update_step(state, batch, key)``, §10) because a typed key is not a
differentiable leaf: keeping it in the state makes ``jax.grad(state)`` refuse to
run. Nothing in the loss needs randomness -- the MC draws arrive in the batch.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.struct import dataclass, field

from sjfpo_jax.networks import Critic, RunningMeanStd, SJFPOFlowModel, StateConditioner


@dataclass
class SJFPOState:
    """Everything ``forward_loss`` reads, and nothing more yet.

    ``old_actor`` / ``old_conditioner`` / ``target_critic`` are runtime-only: the
    torch checkpoint drops them (``checkpointing.py:101-113``) and so will the
    Orbax one, so they are not part of the saved state either.
    """

    actor: tuple
    conditioner: tuple
    critic: tuple
    old_actor: tuple
    old_conditioner: tuple
    target_critic: tuple
    obs_normalizer: RunningMeanStd
    action_low: jax.Array
    action_high: jax.Array
    # The EMA shadow is an NNX State like the networks it mirrors, so the Polyak
    # blend is a plain tree_map between two States. None means "never started",
    # which is also the state of an inert (decay == 0) EMA.
    ema_shadow: object | None = None
    # Static latches: a Python bool as a leaf would make the graph a function of
    # it. Both flip at most once per rollout, so the retrace is free.
    ema_initialised: bool = field(pytree_node=False, default=False)
    target_critic_initialised: bool = field(pytree_node=False, default=False)
    # Static, so it is not a pytree child: a step counter that traced as a leaf
    # would make the whole graph a function of a Python int.
    global_step: int = field(pytree_node=False, default=0)


def build_state(cfg, obs_dim: int, action_dim: int, key: jax.Array) -> SJFPOState:
    """Assemble the networks, the normalizer and the bounds.

    ``cfg`` is duck-typed -- anything carrying the architecture fields. The JAX
    ``SJFPOConfig`` dataclass arrives with the trainer (milestone 4); until then
    the parity tests hand this the torch config. ``key`` seeds the
    initialisation only; it is not stored on the state.

    The frozen snapshots are built from *different* keys rather than deep-copied,
    so they can never alias the live trees. Both are overwritten by
    ``snapshot_behavior_policy`` before the first rollout anyway.

    The normalizer's ``uninitialised`` latch is a Python bool, so flipping it on
    the first update changes the pytree structure. That is fine here -- the update
    happens once per rollout, outside ``jit`` -- but it means the first update
    costs one retrace.
    """
    def build_flow(name):
        module = SJFPOFlowModel(
            action_dim,
            cfg.action_horizon,
            cfg.hidden_dim,
            cfg.dit_depth,
            cfg.dit_heads,
            cfg.dit_mlp_ratio,
            cfg.time_embed_dim,
            cfg.timestep_scale,
            nnx.Rngs(jax.random.fold_in(key, name)),
        )
        return nnx.split(module, nnx.Param)

    actor = build_flow(0)
    old_actor = build_flow(1)
    conditioner = nnx.split(StateConditioner(obs_dim, cfg.hidden_dim, nnx.Rngs(key)), nnx.Param)
    old_conditioner = nnx.split(
        StateConditioner(obs_dim, cfg.hidden_dim, nnx.Rngs(jax.random.fold_in(key, 2))), nnx.Param
    )
    critic = nnx.split(
        Critic(obs_dim, cfg.hidden_dim, cfg.n_obs_steps, cfg.num_critics, nnx.Rngs(jax.random.fold_in(key, 3))),
        nnx.Param,
    )
    target_critic = nnx.split(
        Critic(obs_dim, cfg.hidden_dim, cfg.n_obs_steps, cfg.num_critics, nnx.Rngs(jax.random.fold_in(key, 4))),
        nnx.Param,
    )
    return SJFPOState(
        actor=actor,
        conditioner=conditioner,
        critic=critic,
        old_actor=old_actor,
        old_conditioner=old_conditioner,
        target_critic=target_critic,
        obs_normalizer=RunningMeanStd(obs_dim),
        action_low=jnp.full((action_dim,), -1.0, dtype=jnp.float32),
        action_high=jnp.full((action_dim,), 1.0, dtype=jnp.float32),
        global_step=0,
    )


def update_normalizer(state: SJFPOState, frames: jax.Array) -> SJFPOState:
    """Fold raw frames into the running obs stats. Once per rollout, unjitted.

    Functional, not in-place: the counts are fp64 and the merge is a read-modify-
    write over device arrays, which is exactly the shape that breaks under
    ``jit`` without donation.
    """
    state.obs_normalizer.update(frames)
    return state


# ----------------------------------------------------------------- the optimizer
#
# One AdamW over the whole tree, with per-role lr, per-role decay and a per-role
# clip. The tree is regrouped to ``{"actor": ..., "critic": ...}`` so a single
# transform can clip each role by its own threshold -- the critic's grad norm
# runs 45-170x the actor's, and a shared clip would spend the whole budget on it
# (modeling_sjfpo.py:160-176).
#
# ``optax.add_decays`` is ``add_decayed_weights`` in optax 0.2.8.


class ClipState(NamedTuple):
    """Pre-clip global norm of the subtree, kept only to log it.

    ``optax.clip_by_global_norm`` discards the norm it computed, and the trainer
    logs actor and critic norms separately (``train/*_grad_norm_preclip``).
    """

    pre_clip_norm: jax.Array


def global_norm_and_clip(max_norm):
    """``optax.clip_by_global_norm``, but stashes the pre-clip norm in the state.

    ``max_norm=None`` means measure and do not clip, which is a supported config:
    ``critic_max_grad_norm=None`` logs the norm without touching the step.
    """
    if max_norm is None:
        # Identity, but the state still records the norm.
        def init_fn(params):
            return ClipState(pre_clip_norm=jnp.zeros(()))

        def update_fn(updates, state, params, extra_args=None):
            return updates, ClipState(pre_clip_norm=optax.global_norm(updates))

        return optax.GradientTransformation(init_fn, update_fn)

    inner = optax.clip_by_global_norm(max_norm)

    def init_fn(params):
        return ClipState(pre_clip_norm=jnp.zeros(()))

    def update_fn(updates, state, params, extra_args=None):
        # optax 0.2.8's clip_by_global_norm takes (updates, state, params) only --
        # no extra_args slot -- so it is called without the fourth argument.
        clipped, _ = inner.update(updates, state, params)
        return clipped, ClipState(pre_clip_norm=optax.global_norm(updates))

    return optax.GradientTransformation(init_fn, update_fn)


def per_role_clip(actor_max, critic_max):
    """``global_norm_and_clip`` on each half of a ``{actor, critic}`` tree."""
    actor = global_norm_and_clip(actor_max)
    critic = global_norm_and_clip(critic_max)

    def init_fn(params):
        return {"actor": actor.init(params["actor"]), "critic": critic.init(params["critic"])}

    def update_fn(grads, state, params, extra_args=None):
        a_updates, a_state = actor.update(grads["actor"], state["actor"], params["actor"])
        c_updates, c_state = critic.update(grads["critic"], state["critic"], params["critic"])
        return {"actor": a_updates, "critic": c_updates}, {"actor": a_state, "critic": c_state}

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_lr(lr_tree):
    """Multiply every leaf's update by that leaf's learning rate.

    AdamW runs at ``learning_rate=1.0`` and the rate is applied here, *after*
    the decay term, so the composed update is exactly
    ``p -= lr*wd*p + lr*m/(sqrt(v)+eps)``. optax's ``scale`` takes one scalar or
    array, not a pytree, hence this.
    """

    def init_fn(params):
        return ()

    def update_fn(updates, state, params):
        return jax.tree_util.tree_map(lambda u, lr: u * lr, updates, lr_tree), state

    return optax.GradientTransformation(init_fn, update_fn)


def _keys_of(path):
    # Dict keys come back as plain strings, list indices and attribute names as
    # KeyEntry wrappers, so unwrap the ones that carry a name.
    return tuple(k.key if hasattr(k, "key") else str(k) for k in path)


def _map_by_path(params, fn):
    """``tree_map`` that can see the key path, which the masks need."""
    flat, treedef = jax.tree_util.tree_flatten_with_path(params)
    return jax.tree_util.tree_unflatten(treedef, [fn(_keys_of(p), leaf) for p, leaf in flat])


def decay_mask(params):
    """True where weight decay applies, per torch's rule.

    Actor: skip anything under ``adaLN`` and the output layer, which are the
    modulation and projection terms (modeling_sjfpo.py:465-472). Critic: decay is
    off entirely -- pulling the value head toward zero shows up as a drifting
    value scale and a wandering grad norm (``:481-495``).

    The path is ``(role, model, ...)``, so the rule reads from ``path[2:]``: the
    role and the model are both called "actor", and torch's rule is written
    against names *inside* the actor module.
    """

    def allowed(path, _leaf):
        if path[1] == "critic":
            return False
        inside = path[2:]
        # torch tests a *substring* of the parameter name ("adaLN" in name), not
        # an exact path segment -- which is why adaLN_out is covered too. Path
        # segments may be ints (list indices), so coerce before substring.
        return not any("adaLN" in str(segment) for segment in inside) and inside[:1] != ("out",)

    return _map_by_path(params, allowed)


def build_optimizer(cfg, params):
    """One AdamW over the trainable tree. Returns ``(update, init_state)``.

    ``params`` is ``{"actor": {...flow model + conditioner...}, "critic": {...}}``,
    which is what :func:`optimizer_params` produces. Two learning rates and two
    decay masks ride along as pytrees, so this is still a single chain and a
    single set of Adam moments -- the property torch's one-AdamW-over-four-groups
    has and two separate optimizers would not.
    """
    lr_by_role = {"actor": cfg.policy_lr, "critic": cfg.critic_lr}
    lr_tree = _map_by_path(
        params, lambda path, leaf: jnp.full_like(leaf, lr_by_role[path[0]])
    )
    decay = optax.masked(
        optax.add_decayed_weights(-cfg.optimizer_weight_decay), decay_mask(params)
    )
    chain = optax.chain(
        per_role_clip(cfg.max_grad_norm, cfg.critic_max_grad_norm),
        # Moments only. The decay and the rate are separate steps so they can be
        # masked and per-leaf respectively; optax's own `mask` argument would
        # zero the whole update and freeze the no-decay parameters instead.
        optax.adamw(learning_rate=1.0, b1=0.9, b2=0.999, eps=1e-8),
        decay,
        scale_by_lr(lr_tree),
    )
    return chain, chain.init(params)


def pre_clip_norms(opt_state):
    """The two pre-clip grad norms, for the trainer's two log lines."""
    return {role: float(state.pre_clip_norm) for role, state in opt_state[0].items()}


def optimizer_params(state: SJFPOState):
    """The trainable subtrees, regrouped by role. Frozen trees are excluded."""
    return {
        "actor": {"actor": state.actor[1], "conditioner": state.conditioner[1]},
        "critic": {"critic": state.critic[1]},
    }
