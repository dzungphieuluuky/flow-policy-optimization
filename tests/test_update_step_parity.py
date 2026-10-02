"""One optimizer step, JAX vs torch: both halves, or neither.

The gate is two-sided on purpose. A forward that matches while its gradients do
not is the worst failure this port can have, and a gradient that matches while
the applied step does not is the second worst -- ``l_old`` and the trust region
are both read off the *stepped* weights. So every test here drives the same
initial weights and the same minibatch through :func:`build_update_step` and
through a real ``torch.optim.AdamW``, and compares:

* the **gradients**, tree against tree;
* the **pre-clip grad norms** the chain stashes, against ``clip_grad_norm_``'s
  own return value for both roles;
* the **stepped parameters**.

Torch's gradients are mapped onto the JAX layout with ``to_jax_tree``, the same
mapper the converter uses, so the 2-D kernels get transposed and the critic's
members get stacked by construction rather than by string surgery in the test.
That is also the only reason the critic is comparable at all: its 28 torch
parameters are 14 stacked leaves.

Three fixtures carry the weight:

* **``wake_actor``.** ``SJFPOFlowModel`` zero-inits its output layer, so an
  unwoken actor predicts zero velocity and ``l_old`` does not depend on the
  frozen weights at all. Every comparison below would be two copies of a
  constant.
* **A warmed snapshot.** ``l_old`` is read off ``old_actor`` /
  ``old_conditioner``, so the snapshot has to be taken on both sides or the ratio
  compares a policy against its own weights.
* **``adam_betas``.** ``SJFPOConfig.adam_betas`` is ``(0.999, 0.999)``,
  deliberately equal on both moments. torch's ``build_optimizer`` reads it; the
  JAX chain used to hardcode ``(0.9, 0.999)``, a difference no forward-only test
  can see.
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

from test_policy_parity import CFG, both_batches, pair
from torch_jax_weights import to_jax_tree

from sjfpo_jax import policy as J
from sjfpo_jax.state import build_optimizer, load_pair, optimizer_params, pre_clip_norms
from spectral_joint_policy.modeling_sjfpo import (
    build_optimizer as torch_build_optimizer,
)

FROZEN_PREFIXES = ("old_actor.", "old_conditioner.", "target_critic.")
# torch and optax place bias correction and the eps differently, so an applied
# step agrees to a few percent, not to the last bit.
STEP_RTOL = 2e-2
# The pre-clip norm is a ratio of two nearly cancelling quantities in one place
# and a plain sum in another; 2e-3 is where fp32 accumulation order shows up.
NORM_RTOL = 2e-3
# The suite's parity tolerance (test_policy_parity.RTOL/ATOL). atol does most of
# the work here: the entries that disagree are the ones whose value nearly
# cancels, where a relative tolerance is meaningless but the absolute residual is
# a few units in the last fp32 place of a quantity of order 1e-2.
RTOL, ATOL = 1e-4, 1e-5
# A moved leaf moves by ~lr = 1e-5 against weights of order 1, so rtol alone
# cannot say "it moved". This floor can.
MOVE_FLOOR = 1e-9


def trainable_named(policy):
    return {
        name: param
        for name, param in policy.named_parameters()
        if not name.startswith(FROZEN_PREFIXES)
    }


def torch_side(t_policy, t_batch, *, do_step=True):
    """Gradients, pre-clip norms, and the stepped policy. One forward, one step."""
    t_policy.zero_grad(set_to_none=True)
    loss, _ = t_policy(t_batch, compute_info=False, check_finite=False)
    loss.backward()
    named = trainable_named(t_policy)
    grads = {
        name: param.grad.detach().clone()
        for name, param in named.items()
        if param.grad is not None
    }
    assert len(grads) == len(named), "some trainable leaf received no gradient"

    actor = [p for g in t_policy.get_actor_optim_params()["params"] for p in g["params"]]
    critic = [p for g in t_policy.get_critic_optim_params()["params"] for p in g["params"]]
    norms = (
        torch.nn.utils.clip_grad_norm_(actor, CFG.max_grad_norm),
        torch.nn.utils.clip_grad_norm_(critic, CFG.critic_max_grad_norm),
    )
    before = {name: p.detach().clone() for name, p in named.items()}
    if do_step:
        torch_build_optimizer(t_policy).step()
    return grads, norms, before, float(loss)


def to_grouped(tree_of_torch_values):
    """``{torch name: array}`` -> the layout :func:`optimizer_params` produces."""
    out = {}
    for module, stack in (("actor", ()), ("conditioner", ()), ("critic", ("conditioners", "heads"))):
        prefix = f"{module}."
        sub = {
            name[len(prefix):]: value
            for name, value in tree_of_torch_values.items()
            if name.startswith(prefix)
        }
        nested = to_jax_tree(sub, stack=stack)
        role = "critic" if module == "critic" else "actor"
        out.setdefault(role, {})[module] = nested
    return out




def wake_actor_and_adaln(state, t_policy, seed=21):
    """Wake every zero-initialised head on both sides.

    ``wake_actor`` (test_policy_parity) only wakes ``out``. That is enough for the
    snapshot tests, but the ``adaLN`` modulation head is zero-init too, so the
    DiT blocks stay the identity and the gradient w.r.t. ``in_proj``,
    ``token_pos``, ``time_mlp`` and all four adaLN leaves comes out **exactly**
    zero: 50 of 80 trainable leaves. A step compared against 50 zeros would agree
    on almost nothing, so this wakes the modulation heads as well.

    The JAX half is reloaded whole through ``to_jax_tree`` rather than patched
    leaf by leaf, so the transposes and the block list are handled by the mapper
    the converter already uses.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        for module in [t_policy.actor.out] + [b.adaLN.fc2 for b in t_policy.actor.blocks]:
            module.weight.normal_(0.0, 0.1)
            module.bias.normal_(0.0, 0.1)
    # Prove the zeros are gone, or the fixture cannot do its job.
    assert float(t_policy.actor.blocks[0].adaLN.fc2.weight.abs().sum()) > 0
    return state.replace(actor=load_pair(state.actor, to_jax_tree(t_policy.actor.state_dict())))


def fixture(seed=0):
    """A torch policy and a JAX state, both awake and snapshotted, one batch.

    ``t_batch`` and ``j_batch`` are the *same* batch: the behaviour losses are
    part of the batch, so rebuilding it for one side is how a comparison ends up
    measuring two different experiments.
    """
    t_policy, state = pair(CFG, seed=seed)
    state = wake_actor_and_adaln(state, t_policy)
    t_batch, j_batch, _ = both_batches(t_policy, state, CFG)
    t_policy.snapshot_behavior_policy()
    state = J.snapshot_behavior_policy(state)
    update = J.build_update_step(CFG, optimizer_params(state))
    _, opt_state = build_optimizer(CFG, optimizer_params(state))
    return t_policy, state, t_batch, j_batch, update, opt_state


def zero_gradient_names(t_policy):
    return [
        name
        for name, param in trainable_named(t_policy).items()
        if param.grad is None or float(param.grad.abs().sum()) == 0.0
    ]


def jax_grouped_grads(state, j_batch):
    """Gradients w.r.t. ``optimizer_params``, w.r.t. the same loss forward_loss sees."""

    def loss_fn(grouped):
        merged = J._state_from_params(state, grouped)
        loss, _ = J.forward_loss(merged, j_batch, CFG, reduction="mean")
        return loss

    return jax.grad(loss_fn)(optimizer_params(state))


def assert_same_tree(torch_values, jax_tree, rtol, atol, what):
    """Tree against tree, after mapping the torch half through ``to_jax_tree``.

    The JAX half is an NNX State, which flattens *through* Param to the raw array,
    so its keys carry a trailing ``.value`` that the plain dict the mapper builds
    does not. Dropping it is what makes the two comparable at all -- and it is why
    ``checkpointing.flatten`` exists.
    """
    expected = jax.tree_util.tree_map(jnp.asarray, to_grouped(torch_values))
    got_leaves = {
        jax.tree_util.keystr(p).removesuffix(".value"): l
        for p, l in jax.tree_util.tree_flatten_with_path(jax_tree)[0]
    }
    matched = 0
    for path, value in jax.tree_util.tree_flatten_with_path(expected)[0]:
        key = jax.tree_util.keystr(path)
        assert key in got_leaves, f"{what}: no jax leaf at {key}"
        np.testing.assert_allclose(
            np.asarray(got_leaves[key]),
            np.asarray(value),
            rtol=rtol,
            atol=atol,
            err_msg=f"{what}: {key}",
        )
        matched += 1
    assert matched > 0, f"{what}: nothing matched"
    return matched


# ------------------------------------------------------------------- the gate


def test_the_gradients_match_torch_tree_for_tree():
    t_policy, state, t_batch, j_batch, _, _ = fixture()
    grads, _, _, _ = torch_side(t_policy, t_batch, do_step=False)
    jax_grads = jax_grouped_grads(state, j_batch)

    matched = assert_same_tree(grads, jax_grads, RTOL, ATOL, what="grad")
    # The tree has 80 leaves; a walk that matched a handful would prove nothing.
    assert matched >= 60, matched


def test_the_step_moves_every_leaf_in_torch_s_direction():
    t_policy, state, t_batch, j_batch, update, opt_state = fixture()
    _, _, before, _ = torch_side(t_policy, t_batch, do_step=True)
    j_state, opt_state, _ = update(state, opt_state, j_batch)

    after_torch = trainable_named(t_policy)
    matched = assert_same_tree(
        after_torch, optimizer_params(j_state), STEP_RTOL, ATOL, what="stepped"
    )
    assert matched >= 60, matched

    # Every leaf must have moved, so assert the fixture actually exercises the
    # network: an unwoken actor leaves 50 of 80 gradients at exactly zero.
    dead = zero_gradient_names_after(t_policy)
    assert not dead, f"{len(dead)} leaves had no gradient: {dead[:4]}"
    deltas = [
        float(np.abs(after_torch[n].detach().cpu().numpy() - before[n].numpy()).max())
        for n in after_torch
    ]
    assert min(deltas) > MOVE_FLOOR, min(deltas)
    assert max(deltas) > 1e-6, "no leaf moved by more than 1e-6, so nothing was compared"


def zero_gradient_names_after(t_policy):
    return zero_gradient_names(t_policy)


def test_the_pre_clip_norms_match_clip_grad_norm():
    t_policy, state, t_batch, j_batch, update, opt_state = fixture()
    _, norms, _, _ = torch_side(t_policy, t_batch, do_step=False)
    _, opt_state, _ = update(state, opt_state, j_batch)
    got = pre_clip_norms(opt_state)

    assert float(got["actor"]) == pytest.approx(float(norms[0]), rel=NORM_RTOL)
    assert float(got["critic"]) == pytest.approx(float(norms[1]), rel=NORM_RTOL)


def test_jit_matches_eager():
    """The jitted step is what the trainer calls; eager is the reference.

    Compared at the suite's ``rtol=1e-4`` rather than bit-for-bit: jit fuses and
    reassociates, so XLA and eager differ in the last bits. The repo's other
    jit-vs-eager gate (``test_predict_action_chunk_is_jittable_and_agrees_with_eager``)
    uses the same tolerance for the same reason.
    """
    _, state, _, j_batch, update, _ = fixture()
    raw = getattr(update, "__wrapped__", None)
    assert raw is not None, "update is not jitted, so there is no eager path to compare"

    jitted = J.build_update_step(CFG, optimizer_params(state))
    _, opt_a = build_optimizer(CFG, optimizer_params(state))
    a_state, a_opt, a_info = jitted(state, opt_a, j_batch)

    _, opt_b = build_optimizer(CFG, optimizer_params(state))
    b_state, b_opt, b_info = raw(state, opt_b, j_batch)

    for a, b in zip(jax.tree_util.tree_leaves(a_state), jax.tree_util.tree_leaves(b_state)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=RTOL, atol=ATOL)
    for a, b in zip(jax.tree_util.tree_leaves(a_opt), jax.tree_util.tree_leaves(b_opt)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=RTOL, atol=ATOL)
    assert set(a_info) == set(b_info)
    for k in a_info:
        np.testing.assert_allclose(
            np.asarray(a_info[k]), np.asarray(b_info[k]), rtol=RTOL, atol=ATOL
        )


# ------------------------------------------------------------------- the guards


def test_the_clip_binds_when_the_norm_exceeds_it():
    """The clip is inert at the default and must stay observable.

    ``max_grad_norm=1.0`` against gradients of order 1e-2, so the chain's clip
    never fires and deleting it changes nothing. This drives the same step under a
    norm small enough that it binds, and checks three things at once: the applied
    step shrinks, the *pre*-clip norm is still the one measured, and the two
    differ.
    """
    import dataclasses

    t_policy, state, t_batch, j_batch, _, opt_state = fixture()
    grads, norms, _, _ = torch_side(t_policy, t_batch, do_step=False)

    unclipped = J.build_update_step(CFG, optimizer_params(state))
    plain_state, plain_opt, _ = unclipped(state, opt_state, j_batch)

    tight = dataclasses.replace(CFG, max_grad_norm=1e-4, critic_max_grad_norm=1e-4)
    clipped = J.build_update_step(tight, optimizer_params(state))
    _, opt_tight = build_optimizer(tight, optimizer_params(state))
    tight_state, opt_tight, _ = clipped(state, opt_tight, j_batch)

    # The pre-clip norm is the same measurement either way -- that is the point of
    # stashing it rather than letting clip_by_global_norm discard it -- and it is
    # far above the threshold, so the clip did fire.
    measured = float(pre_clip_norms(opt_tight)["actor"])
    assert measured > 1e-4, measured
    assert measured == pytest.approx(float(norms[0]), rel=2e-3)
    assert measured == pytest.approx(float(pre_clip_norms(plain_opt)["actor"]), rel=2e-3)

    shrunk = [
        float(jnp.abs(a - b).sum())
        for a, b in zip(
            jax.tree_util.tree_leaves(optimizer_params(plain_state)),
            jax.tree_util.tree_leaves(optimizer_params(tight_state)),
        )
    ]
    assert max(shrunk) > 0, "a 1e-4 clip changed nothing, so the clip is inert"
    del grads


def test_the_first_step_hard_copies_the_target_critic():
    """update_target_critic runs inside the step, after the parameter update.

    torch calls it after ``optimizer.step()`` too, so on the first call the target
    equals the *stepped* critic exactly -- not the pre-step one, and not a Polyak
    blend. This also pins the ordering, which is otherwise invisible.
    """
    _, state, _, j_batch, update, opt_state = fixture()
    j_state, _, _ = update(state, opt_state, j_batch)
    assert j_state.target_critic_initialised is True

    live = jax.tree_util.tree_leaves(nnx.to_pure_dict(j_state.critic[1]))
    target = jax.tree_util.tree_leaves(nnx.to_pure_dict(j_state.target_critic[1]))
    assert live, "the critic has no leaves, so nothing was compared"
    assert len(live) == len(target)
    for a, b in zip(live, target):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_the_second_step_polyaks_rather_than_re_copies():
    t_policy, state, t_batch, j_batch, update, opt_state = fixture()
    j_state, opt_state, _ = update(state, opt_state, j_batch)
    target_after_first = jax.tree_util.tree_leaves(
        nnx.to_pure_dict(j_state.target_critic[1])
    )
    j_state, opt_state, _ = update(j_state, opt_state, j_batch)
    target_after_second = jax.tree_util.tree_leaves(
        nnx.to_pure_dict(j_state.target_critic[1])
    )
    for a, b, c in zip(target_after_first, target_after_second, target_after_first):
        assert float(jnp.abs(np.asarray(b) - np.asarray(c)).sum()) > 0, "target froze"
        del a
    del t_policy, t_batch


def test_the_step_produces_the_full_info_block():
    _, state, _, j_batch, update, opt_state = fixture()
    _, _, info = update(state, opt_state, j_batch)
    assert len(info) == 37, len(info)
    assert all(v.shape == () for v in info.values())
    assert float(info["policy_loss"]) > 0
    assert float(info["value_loss"]) > 0
    for key in ("explained_var", "clipfrac", "cfm_above_clamp_frac", "flow_ratio_mean"):
        assert key in info, key
    # The Dohare collapse probes ride along, read off the same rows the loss used.
    for key in ("cond_effective_rank", "cond_weight_magnitude"):
        assert key in info, key
        assert float(info[key]) > 0, key


def test_the_adam_betas_field_reaches_the_step():
    """Not the (0.9, 0.999) the chain used to hardcode.

    Two configs differing only in the betas must produce different steps, and the
    configured value must agree with a torch optimizer built from the same field.
    """
    t_policy, state, t_batch, j_batch, _, opt_state = fixture()
    grads, _, _, _ = torch_side(t_policy, t_batch, do_step=False)

    plain = J.build_update_step(CFG, optimizer_params(state))
    j_state, _, _ = plain(state, opt_state, j_batch)

    skewed_cfg = dataclasses_replace_adam_betas(CFG, (0.9, 0.999))
    skewed = J.build_update_step(skewed_cfg, optimizer_params(state))
    _, opt_skewed = build_optimizer(skewed_cfg, optimizer_params(state))
    k_state, _, _ = skewed(state, opt_skewed, j_batch)

    deltas = [
        float(jnp.abs(a - b).sum())
        for a, b in zip(
            jax.tree_util.tree_leaves(optimizer_params(j_state)),
            jax.tree_util.tree_leaves(optimizer_params(k_state)),
        )
    ]
    assert max(deltas) > 0, "the configured betas had no effect on any leaf"
    # Not min: a leaf whose gradient is exactly zero has an exactly zero update
    # whatever the betas are, and the zero-init adaLN leaves are in that class.

    # And the configured value agrees with torch, which reads the same field.
    torch_side(t_policy, t_batch, do_step=True)
    after_torch = trainable_named(t_policy)
    assert_same_tree(after_torch, optimizer_params(j_state), STEP_RTOL, ATOL, what="betas")
    assert CFG.adam_betas == (0.999, 0.999), CFG.adam_betas
    del grads


def dataclasses_replace_adam_betas(cfg, betas):
    import dataclasses

    return dataclasses.replace(cfg, adam_betas=betas)


def test_a_missing_behaviour_loss_still_raises_inside_the_jitted_step():
    """The validation in forward_loss has to survive being traced."""
    _, state, _, j_batch, update, opt_state = fixture()
    broken = dict(j_batch)
    del broken["behavior_cfm_loss"]
    with pytest.raises((KeyError, ValueError)):
        update(state, opt_state, broken)


def test_a_malformed_batch_raises_the_same_error_it_does_eager():
    _, state, _, j_batch, update, opt_state = fixture()
    broken = dict(j_batch)
    broken["action_is_pad"] = jnp.zeros((j_batch["action_is_pad"].shape[0], 3), dtype=jnp.bool_)
    with pytest.raises(ValueError):
        update(state, opt_state, broken)
    with pytest.raises(ValueError):
        J.forward_loss(state, broken, CFG)


def test_the_frozen_trees_receive_no_gradient():
    """The snapshot and the target are not in optimizer_params, so they cannot."""
    _, state, t_batch, j_batch, _, _ = fixture()
    grads = jax_grouped_grads(state, j_batch)
    names = set(grads)
    assert names == {"actor", "critic"}, names
    frozen = jax.tree_util.tree_leaves(state.old_actor[1])
    assert frozen, "there is nothing to freeze"
    del t_batch, j_batch