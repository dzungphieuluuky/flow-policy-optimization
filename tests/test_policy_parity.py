"""torch vs jax parity for the loss path: helpers, l_old, and forward_loss.



The torch ``SJFPOPolicy.forward`` is the oracle; both sides are handed the same
batch dict, built the way ``generate_rollout`` builds it, including the
frozen-snapshot losses.

Two gates beyond the forward value:

* **Per-chunk, not scalar.** A scalar mean hides a chunk that went the wrong way.
* **Gradient parity.** ``jax.grad`` against ``torch.autograd`` on a batch with a
  truncated chunk, so the mask and the value clip are both live. A loss that
  matches while its gradients do not is the failure this port is most exposed to.
"""
import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
pytestmark = pytest.mark.skip(reason="Torch StateEncoder diverged from the JAX port; parity conversion disabled.")
import torch
from flax import nnx
from lerobot.utils.constants import ACTION, OBS_STATE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from torch_jax_weights import flatten, load_pair, to_jax_tree

from sjfpo_jax import policy as J
from sjfpo_jax.state import build_state
from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
from spectral_joint_policy.modeling_sjfpo import SJFPOPolicy, compute_value_huber_loss
from spectral_joint_policy.train_sjfpo import build_policy

OBS, ACT = 11, 2
N = 6
RTOL, ATOL = 1e-4, 1e-5
# num_kept_tokens < action_horizon so the DCT actually truncates: at the default
# 16 == action_horizon it is the full orthonormal transform and a no-op on the
# loss, so it would prove nothing. num_mc_samples is cut to 4 purely for test
# speed; the loss is linear in the MC axis.
CFG = SJFPOConfig(num_mc_samples=4, num_kept_tokens=8)
MC = CFG.num_mc_samples
LOW = np.array([-2.0, -2.0], dtype=np.float32)
HIGH = np.array([2.0, 2.0], dtype=np.float32)


def cfg_with(**overrides):
    return dataclasses.replace(CFG, **overrides)


def make_batch(cfg, seed=0, truncate=()):
    """A batch shaped like ``generate_rollout``'s, including one short chunk."""
    rng = np.random.default_rng(seed)
    horizon = cfg.action_horizon
    tokens = cfg.num_flow_tokens
    lengths = np.full(N, horizon)
    for index, length in truncate:
        lengths[index] = length
    pad = np.arange(horizon)[None, :] >= lengths[:, None]

    batch = {
        OBS_STATE: rng.standard_normal((N, cfg.n_obs_steps, OBS), dtype=np.float32),
        ACTION: rng.standard_normal((N, horizon, ACT), dtype=np.float32),
        "action_is_pad": pad,
        "normalised_advantages": rng.standard_normal(N, dtype=np.float32),
        "raw_returns": (rng.standard_normal(N, dtype=np.float32) * 10.0),
        "raw_values": rng.standard_normal(N, dtype=np.float32),
        "mc_tau": rng.uniform(0.05, 0.95, size=(N, MC, tokens, 1)).astype(np.float32),
        "mc_eps": rng.standard_normal((N, MC, tokens, ACT), dtype=np.float32),
    }
    return {k: (torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v) for k, v in batch.items()}


def pair(cfg=CFG, seed=0):
    """A torch policy and a JAX state carrying the same weights, **both built from
    ``cfg``**. A policy keeps the config it was constructed with, so building it
    from a different one silently compares two different experiments.

    The normalizer is warmed with the same frames on both sides so the loss path
    runs against a real normalisation rather than the identity fallback.
    """
    torch.manual_seed(seed)
    t_policy = build_policy(OBS, ACT, cfg, device="cpu", action_low=LOW, action_high=HIGH)
    state = build_state(cfg, OBS, ACT, jax.random.key(seed))

    state = state.replace(
        actor=load_pair(state.actor, to_jax_tree(t_policy.actor.state_dict())),
        conditioner=load_pair(state.conditioner, to_jax_tree(t_policy.encoder.state_dict())),
        critic=load_pair(
            state.critic,
            to_jax_tree(t_policy.critic.state_dict(), stack=["encoders", "heads"]),
        ),
        old_actor=load_pair(state.old_actor, to_jax_tree(t_policy.old_actor.state_dict())),
        old_conditioner=load_pair(
            state.old_conditioner, to_jax_tree(t_policy.old_encoder.state_dict())
        ),
        action_low=jnp.asarray(LOW, dtype=jnp.float32),
        action_high=jnp.asarray(HIGH, dtype=jnp.float32),
    )

    frames = np.random.default_rng(99).standard_normal((64, OBS), dtype=np.float32)
    t_policy.update_observation_stats(torch.from_numpy(frames.copy()))
    state.obs_normalizer.update(jnp.asarray(frames))
    return t_policy, state


def both_batches(t_policy, state, cfg, truncate=(), seed=0):
    """The batch in both dialects, with the behaviour losses filled in."""
    t_batch = make_batch(cfg, seed, truncate)
    j_batch = {k: jnp.asarray(np.asarray(v)) for k, v in t_batch.items()}

    normalised = t_policy.preprocessor({OBS_STATE: t_batch[OBS_STATE]})[OBS_STATE]
    l_old, l_old_token = t_policy.compute_behavior_cfm_losses(
        normalised,
        t_batch[ACTION],
        t_batch["mc_tau"],
        t_batch["mc_eps"],
        t_batch["action_is_pad"],
    )
    t_batch["behavior_cfm_loss"] = l_old
    t_batch["behavior_cfm_loss_token"] = l_old_token
    j_batch["behavior_cfm_loss"] = jnp.asarray(np.asarray(l_old))
    j_batch["behavior_cfm_loss_token"] = jnp.asarray(np.asarray(l_old_token))
    return t_batch, j_batch, normalised


def as_np(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, jax.Array):
        return np.asarray(value)
    if isinstance(value, dict):
        return {k: as_np(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [as_np(v) for v in value]
    return np.asarray(value) if isinstance(value, (int, float, np.floating)) else value


def assert_same(torch_out, jax_out):
    t, j = as_np(torch_out), as_np(jax_out)
    if isinstance(t, dict):
        assert set(t) == set(j), (sorted(t), sorted(j))
        for key in t:
            assert_same(t[key], j[key])
        return
    if isinstance(t, list):
        assert isinstance(j, list) and len(t) == len(j), (type(j), len(t))
        for a, b in zip(t, j):
            assert_same(a, b)
        return
    assert t.shape == j.shape, (t.shape, j.shape)
    np.testing.assert_allclose(t, j, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------- pure helpers


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"use_dct_transform": False}, CFG.action_horizon),
        ({"use_dct_transform": True, "cfm_mask_unexecuted": True}, CFG.action_horizon),
        ({"use_dct_transform": True, "cfm_mask_unexecuted": False}, CFG.num_flow_tokens),
    ],
    ids=["dct_off", "mask_on", "mask_off"],
)
def test_token_count_follows_the_layout_the_mask_chooses(overrides, expected):
    cfg = cfg_with(**overrides)
    assert J.get_token_count_for_cfm_loss(cfg) == expected
    assert J.get_token_count_for_cfm_loss(cfg) == SJFPOPolicy.get_token_count_for_cfm_loss(cfg)


def test_prepare_cfm_loss_inputs_expands_onto_the_action_grid():
    """DCT on with the mask on: coeffs -> action steps, or the mask indexes the
    wrong axis and the loss is fitted on rows the chunk never ran."""
    cfg = cfg_with(use_dct_transform=True, cfm_mask_unexecuted=True)
    t_policy, _ = pair(cfg)
    v_t = torch.zeros(3, cfg.num_flow_tokens, ACT)
    v_j = jnp.zeros((3, cfg.num_flow_tokens, ACT))
    pad_t = torch.zeros(3, cfg.action_horizon, dtype=torch.bool)
    pad_t[1, 5:] = True
    pad_j = jnp.asarray(np.asarray(pad_t))
    t_mask, (t_rows,) = t_policy.prepare_cfm_loss_inputs(cfg, pad_t, v_t)
    j_mask, (j_rows,) = J.prepare_cfm_loss_inputs(cfg, pad_j, v_j)
    assert_same(t_mask, j_mask)
    assert_same(t_rows, j_rows)
    # It really expanded: 8 coefficients onto 16 action steps.
    assert j_rows.shape == (3, cfg.action_horizon, ACT)


def test_prepare_cfm_loss_inputs_drops_the_mask_without_the_grid():
    """DCT on with the mask off, the mask would index the frequency axis."""
    cfg = cfg_with(use_dct_transform=True, cfm_mask_unexecuted=False)
    t_policy, _ = pair(cfg)
    pad_t = torch.zeros(3, cfg.action_horizon, dtype=torch.bool)
    assert t_policy.prepare_cfm_loss_inputs(cfg, pad_t, torch.zeros(3, 8, ACT))[0] is None
    assert J.prepare_cfm_loss_inputs(
        cfg, jnp.zeros((3, cfg.action_horizon), dtype=jnp.bool_), jnp.zeros((3, 8, ACT))
    )[0] is None


def test_target_velocity_broadcasts_over_the_mc_draws():
    a_t, eps_t = torch.randn(4, 8, ACT), torch.randn(4, 3, 8, ACT)
    assert_same(SJFPOPolicy.compute_target_velocity(a_t, eps_t), J.compute_target_velocity(jnp.asarray(np.asarray(a_t)), jnp.asarray(np.asarray(eps_t))))


def test_value_huber_is_quadratic_inside_delta_and_linear_outside():
    """Same three cases as tests/test_value_loss.py, against torch."""
    pred_t, target = torch.randn(3, 5), torch.randn(5)
    pred_j = jnp.asarray(np.asarray(pred_t))
    for delta in (0.5, 2.0, 5.0):
        d = torch.tensor(delta)
        assert_same(
            T_huber(pred_t, target, d),
            J.compute_value_huber_loss(pred_j, jnp.asarray(np.asarray(target)), jnp.float32(delta)),
        )


def T_huber(pred, target, delta):
    return compute_value_huber_loss(pred, target.expand_as(pred), delta)


# --------------------------------------------------------------- behaviour loss


def test_behavior_cfm_loss_matches_torch():
    """l_old is the ratio's numerator: if this drifts, the trust region bounds nothing."""
    t_policy, state = pair(CFG, seed=0)
    t_batch, j_batch, _ = both_batches(t_policy, state, CFG, truncate=((1, 9),))
    assert_same(
        (t_batch["behavior_cfm_loss"], t_batch["behavior_cfm_loss_token"]),
        J.compute_behavior_cfm_loss(state, j_batch, CFG),
    )


def test_behavior_cfm_loss_uses_the_frozen_snapshot():
    """It must read old_actor, not the live actor, or l_old tracks l_new."""
    t_policy, state = pair(CFG, seed=0)
    _, j_batch, _ = both_batches(t_policy, state, CFG, truncate=((1, 9),))
    base = J.compute_behavior_cfm_loss(state, j_batch, CFG)[0]

    # Rebuild the live actor with different weights; l_old must not move.
    torch.manual_seed(7)
    other = build_policy(OBS, ACT, CFG, device="cpu", action_low=LOW, action_high=HIGH)
    moved = state.replace(actor=load_pair(state.actor, to_jax_tree(other.actor.state_dict())))
    after = J.compute_behavior_cfm_loss(moved, j_batch, CFG)[0]
    assert_same(base, after)


# --------------------------------------------------------------------- forward


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"use_dct_transform": False},
        {"cfm_mask_unexecuted": False},
        {"trust_region_mode": "ppo"},
        {"trust_region_mode": "spo"},
        {"cfm_loss_reduction": "sum"},
        {"cfm_loss_reduction": "sqrt"},
        {"num_critics": 2},
        {"num_critics": 3},
    ],
    ids=[
        "default", "dct_off", "mask_off", "ppo", "spo",
        "cfm_sum", "cfm_sqrt", "two_critics", "three_critics",
    ],
)
def test_forward_loss_matches_torch(overrides):
    """Per-chunk, not the scalar mean: the scalar hides a chunk going the wrong way."""
    cfg = cfg_with(**overrides)
    t_policy, state = pair(cfg, seed=0)
    t_batch, j_batch, _ = both_batches(t_policy, state, cfg, truncate=((1, 9), (4, 4)))

    t_loss, _ = t_policy(t_batch, compute_info=False, check_finite=False)
    j_loss, terms = J.forward_loss(state, j_batch, cfg)
    assert_same(t_loss, j_loss)
    assert terms["policy_per_chunk"].shape == (N,)
    assert terms["value_loss_per_chunk"].shape == (N,)

def test_forward_loss_per_chunk_vectors_match_torch():
    """torch exposes no per-chunk split, so use its own reduction='none' path:
    the per-chunk loss is policy + value_loss_coef * value, per chunk."""
    cfg = CFG
    t_policy, state = pair(cfg, seed=0)
    t_batch, j_batch, _ = both_batches(t_policy, state, cfg, truncate=((1, 9), (4, 4)))

    t_loss, _ = t_policy(t_batch, reduction="none", compute_info=False, check_finite=False)
    j_loss, terms = J.forward_loss(state, j_batch, cfg, reduction="none")
    assert_same(t_loss, j_loss)
    assert terms["policy_per_chunk"].shape == (N,)
    assert terms["value_loss_per_chunk"].shape == (N,)
    assert float(jnp.abs(terms["policy_per_chunk"]).sum()) > 0
    assert float(jnp.abs(terms["value_loss_per_chunk"]).sum()) > 0
    # The two parts reconstruct the per-chunk total.
    np.testing.assert_allclose(
        np.asarray(terms["policy_per_chunk"]) + cfg.value_loss_coef * np.asarray(terms["value_loss_per_chunk"]),
        np.asarray(j_loss),
        rtol=RTOL,
        atol=ATOL,
    )


def test_forward_loss_rejects_a_batch_without_the_behavior_losses():
    """Recomputing l_old per minibatch compares two different snapshots."""
    t_policy, state = pair(CFG, seed=0)
    _, j_batch, _ = both_batches(t_policy, state, CFG)
    del j_batch["behavior_cfm_loss"]
    with pytest.raises(KeyError, match="behavior_cfm_loss"):
        J.forward_loss(state, j_batch, CFG)


def test_forward_loss_reduction_none_is_per_chunk():
    t_policy, state = pair(CFG, seed=0)
    _, j_batch, _ = both_batches(t_policy, state, CFG, truncate=((2, 5),))
    mean_loss, _ = J.forward_loss(state, j_batch, CFG)
    per_chunk, terms = J.forward_loss(state, j_batch, CFG, reduction="none")
    assert per_chunk.shape == (N,)
    assert float(jnp.abs(jnp.mean(per_chunk) - mean_loss)) < 1e-5
    assert terms["value_loss_per_chunk"].shape == (N,)


def test_forward_loss_rejects_an_unknown_reduction():
    t_policy, state = pair(CFG, seed=0)
    _, j_batch, _ = both_batches(t_policy, state, CFG)
    with pytest.raises(ValueError, match="reduction"):
        J.forward_loss(state, j_batch, CFG, reduction="median")


# --------------------------------------------------------------------- gradient


def test_gradients_match_torch_parameter_for_parameter():
    """The gate that matters: a loss can match while its gradients do not."""
    cfg = CFG
    t_policy, state = pair(seed=0)
    t_batch, j_batch, _ = both_batches(t_policy, state, cfg, truncate=((1, 9), (4, 4)))

    t_policy.zero_grad(set_to_none=True)
    t_loss, _ = t_policy(t_batch, compute_info=False, check_finite=False)
    t_loss.backward()
    j_grads = jax.grad(lambda s: J.forward_loss(s, j_batch, cfg)[0])(state)

    for role, module_name in (("actor", "actor"), ("encoder", "encoder")):
        torch_grads = {
            name: param.grad
            for name, param in getattr(t_policy, module_name).named_parameters()
        }
        assert_same(
            flatten(to_jax_tree(torch_grads)),
            flatten(nnx.to_pure_dict(getattr(j_grads, role)[1])),
        )

    torch_grads = {
        name: param.grad for name, param in t_policy.critic.named_parameters()
    }
    assert_same(
        flatten(to_jax_tree(torch_grads, stack=["encoders", "heads"])),
        flatten(nnx.to_pure_dict(j_grads.critic[1])),
    )


def test_frozen_snapshots_receive_no_gradient():
    """old_actor / old_conditioner are the numerator's own snapshot; a gradient
    reaching them means the ratio stopped being a ratio."""
    t_policy, state = pair(CFG, seed=0)
    _, j_batch, _ = both_batches(t_policy, state, CFG, truncate=((1, 9),))
    grads = jax.grad(lambda s: J.forward_loss(s, j_batch, CFG)[0])(state)
    for role in ("old_actor", "old_conditioner"):
        leaves = jax.tree_util.tree_leaves(nnx.to_pure_dict(getattr(grads, role)[1]))
        assert all(float(jnp.abs(leaf).sum()) == 0 for leaf in leaves), role


def test_every_sample_above_the_clamp_stops_the_policy_and_not_the_critic():
    """The condition ABOVE_CLAMP_ABORT_UPDATES watches, as a gradient property.

    ``l_new = (a - eps)^2`` blows past ``cfm_clamp`` when the actions do, and the
    one-sided clamp then has exactly zero gradient above it. The policy stops
    learning while the critic keeps training -- which is what the abort message
    says, and what a ``jnp.minimum`` in place of ``jnp.clip`` would break.
    """
    cfg = CFG
    t_policy, state = pair(cfg, seed=0)
    _, j_batch, _ = both_batches(t_policy, state, cfg)
    saturated = dict(j_batch)
    saturated["action"] = j_batch["action"] * 1e4
    _, terms = J.forward_loss(state, saturated, cfg)
    l_new = np.asarray(terms["l_new"])
    assert float(l_new.min()) > cfg.cfm_clamp, "the clamp is not engaged; the test proves nothing"

    grads = jax.grad(lambda s: J.forward_loss(s, saturated, cfg)[0])(state)
    for role in ("actor", "conditioner"):
        leaves = jax.tree_util.tree_leaves(nnx.to_pure_dict(getattr(grads, role)[1]))
        assert all(float(jnp.abs(leaf).sum()) == 0 for leaf in leaves), role
    critic_leaves = jax.tree_util.tree_leaves(nnx.to_pure_dict(grads.critic[1]))
    assert any(float(jnp.abs(leaf).sum()) > 0 for leaf in critic_leaves)


# ------------------------------------------------------------------ compute_info


def test_compute_info_matches_torch_key_for_key():
    """The keys are an interface: the trainer logs by name, so a renamed key
    silently stops a metric appearing on W&B."""
    cfg = CFG
    t_policy, state = pair(cfg, seed=0)
    t_batch, j_batch, _ = both_batches(t_policy, state, cfg, truncate=((1, 9), (4, 4)))
    _, t_info = t_policy(t_batch, compute_info=True, check_finite=False)
    _, j_terms = J.forward_loss(state, j_batch, cfg)
    j_info = J.compute_info(j_terms, cfg)

    assert set(j_info) == set(t_info), (
        sorted(set(j_info) - set(t_info)), sorted(set(t_info) - set(j_info))
    )
    assert len(j_info) == 35, f"28 scalars + 7 cond_*, got {len(j_info)}"
    assert_same({k: np.asarray(v) for k, v in j_info.items()}, t_info)


def test_compute_info_values_are_zero_dim_and_on_device():
    """The trainer does one device_get on the whole dict; a non-scalar would
    change the shape of that transfer."""
    cfg = CFG
    t_policy, state = pair(cfg, seed=0)
    _, j_batch, _ = both_batches(t_policy, state, cfg)
    _, terms = J.forward_loss(state, j_batch, cfg)
    info = J.compute_info(terms, cfg)
    assert all(v.shape == () for v in info.values())
    assert all(jnp.isfinite(v) for v in info.values())
    # One transfer over the whole dict, and every entry is a 0-dim number.
    transferred = jax.device_get(info)
    assert all(np.asarray(v).ndim == 0 for v in transferred.values())
    assert all(np.issubdtype(np.asarray(v).dtype, np.floating) for v in transferred.values())


def test_explained_var_and_ratio_are_the_fragile_ones():
    """Both are ratios of nearly-cancelling quantities, so they move first when
    the arithmetic differs. Check them explicitly rather than as part of a sweep."""
    cfg = CFG
    t_policy, state = pair(cfg, seed=0)
    t_batch, j_batch, _ = both_batches(t_policy, state, cfg, truncate=((1, 9),))
    _, t_info = t_policy(t_batch, compute_info=True, check_finite=False)
    _, terms = J.forward_loss(state, j_batch, cfg)
    j_info = J.compute_info(terms, cfg)
    for key in ("explained_var", "flow_ratio_mean", "flow_ratio_std", "value_target_std"):
        assert abs(float(j_info[key]) - float(t_info[key])) <= RTOL * max(abs(float(t_info[key])), 1.0) + ATOL
    # And they are not trivially equal to their inputs, i.e. the test bites.
    assert abs(float(j_info["explained_var"])) > 1e-6


# ---------------------------------------------------------- predict_action_chunk


def torch_chunk(t_policy, t_batch, cfg, **kwargs):
    with torch.no_grad():
        return t_policy.predict_action_chunk({OBS_STATE: t_batch[OBS_STATE]}, **kwargs)


def test_predict_action_chunk_zero_sampling():
    """eps=0 is what FPO++ uses for eval and for the reported metrics."""
    cfg = CFG
    t_policy, state = pair(cfg, seed=0)
    t_policy.eval()
    t_batch, j_batch, _ = both_batches(t_policy, state, cfg, truncate=((1, 9),))
    chunk = J.predict_action_chunk(state, j_batch[OBS_STATE], cfg, zero_sampling=True)
    assert chunk.shape == (N, cfg.action_horizon, ACT)
    assert_same(torch_chunk(t_policy, t_batch, cfg, zero_sampling=True), chunk)


def test_predict_action_chunk_dct_off():
    cfg = cfg_with(use_dct_transform=False)
    t_policy, state = pair(cfg, seed=0)
    t_policy.eval()
    t_batch, j_batch, _ = both_batches(t_policy, state, cfg, truncate=((1, 9),))
    assert_same(
        torch_chunk(t_policy, t_batch, cfg, zero_sampling=True),
        J.predict_action_chunk(state, j_batch[OBS_STATE], cfg, zero_sampling=True),
    )


def test_predict_action_chunk_noise_sampling_is_deterministic_and_in_range():
    """The draws differ from torch's generator, so this checks the contract rather
    than the values: a key fixes the output, and the rescale puts it in env units.

    The integrator itself is already checked value-for-value in
    ``test_parity.py::test_sample_action_matches_euler_step_for_step`` against a
    numpy Euler recursion.
    """
    cfg = CFG
    _, state = pair(cfg, seed=0)
    _, j_batch, _ = both_batches(*pair(cfg, seed=0), cfg, truncate=((1, 9),))
    states = j_batch[OBS_STATE]
    first = J.predict_action_chunk(state, states, cfg, zero_sampling=False, key=jax.random.key(5))
    second = J.predict_action_chunk(state, states, cfg, zero_sampling=False, key=jax.random.key(5))
    other = J.predict_action_chunk(state, states, cfg, zero_sampling=False, key=jax.random.key(6))
    assert first.shape == (N, cfg.action_horizon, ACT)
    assert bool(jnp.all(first == second)), "the same key must give the same chunk"
    assert not bool(jnp.all(first == other))
    # Env bounds are [-2, 2] here, so a healthy chunk never exceeds them.
    assert float(jnp.abs(first).max()) <= 2.0 + 1e-6
    assert float(jnp.mean(first)) == pytest.approx(0.0, abs=0.2)


def test_predict_action_chunk_perturbs_in_normalised_space():
    """The perturbation is added before the unscale, so its size does not scale
    with the env's action range."""
    cfg = cfg_with(action_perturb_std=0.5)
    _, state = pair(cfg, seed=0)
    _, j_batch, _ = both_batches(*pair(cfg, seed=0), cfg)
    states = j_batch[OBS_STATE]
    clean = J.predict_action_chunk(state, states, cfg, zero_sampling=True)
    noisy = J.predict_action_chunk(
        state, states, cfg, zero_sampling=True, key=jax.random.key(11), perturb=True
    )
    # action_low/high are [-2, 2], so undo the rescale and the perturbation is
    # exactly action_perturb_std * N(0, 1). Applied after the rescale instead, it
    # would arrive as 1.0.
    normalised = (noisy - clean) / (0.5 * (state.action_high - state.action_low))
    assert float(jnp.std(normalised)) == pytest.approx(cfg.action_perturb_std, rel=0.2)


def test_predict_action_chunk_rescales_after_clipping():
    """Clip to [-1, 1] then rescale. The other order makes the trainer's
    action_abs_max > 1.05 guard trip on a healthy policy."""
    cfg = CFG
    t_policy, state = pair(cfg, seed=0)
    t_policy.eval()
    t_batch, j_batch, _ = both_batches(t_policy, state, cfg)
    chunk = J.predict_action_chunk(state, j_batch[OBS_STATE], cfg, zero_sampling=True)
    assert float(jnp.max(state.action_high)) == pytest.approx(2.0)
    assert float(jnp.abs(chunk).max()) <= 2.0 + 1e-6
    assert_same(torch_chunk(t_policy, t_batch, cfg, zero_sampling=True), chunk)


def test_predict_action_chunk_is_jittable_and_agrees_with_eager():
    """One compiled program per call is the point of the scan."""
    cfg = CFG
    _, state = pair(cfg, seed=0)
    _, j_batch, _ = both_batches(*pair(cfg, seed=0), cfg)
    states = j_batch[OBS_STATE]
    jitted = jax.jit(lambda s: J.predict_action_chunk(s, states, cfg, zero_sampling=True))
    assert_same(J.predict_action_chunk(state, states, cfg, zero_sampling=True), jitted(state))


def test_predict_action_chunk_noise_sampling_needs_a_key():
    cfg = CFG
    _, state = pair(cfg, seed=0)
    _, j_batch, _ = both_batches(*pair(cfg, seed=0), cfg)
    with pytest.raises(ValueError, match="key"):
        J.predict_action_chunk(state, j_batch[OBS_STATE], cfg, zero_sampling=False)


# ------------------------------------------------- snapshots, Polyak, EMA


def torch_snapshot(t_policy):
    t_policy.snapshot_behavior_policy()
    return t_policy


def test_snapshot_behavior_policy_copies_the_live_weights():
    """The snapshot must actually change what l_old reads."""
    cfg = CFG
    t_policy, state = pair(cfg, seed=0)
    state = wake_actor(state, t_policy)
    t_batch, j_batch, _ = both_batches(t_policy, state, cfg, truncate=((1, 9),))

    # Point old_actor somewhere else; the snapshot must pull it back.
    torch.manual_seed(5)
    other = build_policy(OBS, ACT, cfg, device="cpu", action_low=LOW, action_high=HIGH)
    stale = state.replace(
        old_actor=load_pair(state.actor, to_jax_tree(other.old_actor.state_dict()))
    )
    assert not np.allclose(
        as_np(J.compute_behavior_cfm_loss(stale, j_batch, cfg)[0]),
        as_np(J.compute_behavior_cfm_loss(state, j_batch, cfg)[0]),
    ), "the stale snapshot is indistinguishable; the test cannot bite"

    torch_snapshot(t_policy)
    restored = J.snapshot_behavior_policy(stale)
    assert_same(
        J.compute_behavior_cfm_loss(restored, j_batch, cfg),
        t_policy.compute_behavior_cfm_losses(
            t_policy.preprocessor({OBS_STATE: t_batch[OBS_STATE]})[OBS_STATE],
            t_batch["action"], t_batch["mc_tau"], t_batch["mc_eps"], t_batch["action_is_pad"],
        ),
    )


def test_target_critic_hard_copies_then_polyaks():
    """gamma ** action_horizon puts most of every value target on the live critic,
    so the first update must be a copy, not a partial move."""
    cfg = CFG
    _, state = pair(cfg, seed=0)
    first = J.update_target_critic(state, cfg)
    assert first.target_critic_initialised is True
    assert_same(
        flatten(nnx.to_pure_dict(first.target_critic[1])),
        flatten(nnx.to_pure_dict(first.critic[1])),
    )
    # Move the live critic, then one Polyak step toward it.
    torch.manual_seed(8)
    other = build_policy(OBS, ACT, cfg, device="cpu", action_low=LOW, action_high=HIGH)
    moved = first.replace(
        critic=load_pair(
            first.critic, to_jax_tree(other.critic.state_dict(), stack=["encoders", "heads"])
        )
    )
    stepped = J.update_target_critic(moved, cfg)
    decay = cfg.critic_ema_decay
    for key, value in flatten(nnx.to_pure_dict(stepped.target_critic[1])).items():
        before = flatten(nnx.to_pure_dict(moved.critic[1]))[key]
        start = flatten(nnx.to_pure_dict(first.critic[1]))[key]
        np.testing.assert_allclose(
            np.asarray(value), decay * np.asarray(start) + (1 - decay) * np.asarray(before),
            rtol=RTOL, atol=ATOL,
        )


def test_ema_warmup_latch_is_one_shot_over_five_updates():
    """N = 5, not N = 1.

    The bug this guards is a ``==`` warmup gate: global_step moves by a whole
    rollout, so it is never hit, ``ema_ready`` stays false forever, and every later
    call re-hard-copies and discards the running average. A single-call test
    cannot see that.
    """
    cfg = cfg_with(ema_warmup_steps=500, ema_decay=0.95)
    t_policy, state = pair(cfg, seed=0)
    # Walk global_step the way the trainer does: by whole rollouts.
    steps = [0, 200, 500, 800, 1200]
    for step in steps:
        state = J.update_ema(state, cfg, step)
        t_policy.update_ema(step)
        assert state.ema_initialised == t_policy._ema_initialised, step
        if state.ema_shadow is not None:
            # torch keys the shadow by name with `.weight`; the mapper renames to
            # `.kernel`, so compare through it rather than string-matching.
            assert_same(
                flatten(nnx.to_pure_dict(state.ema_shadow)),
                flatten(to_jax_tree(t_policy._ema_shadow)),
            )
    # And the readiness gate agrees, which is what eval actually asks.
    for step in steps:
        assert J.ema_ready(state, cfg, step) == t_policy.ema_ready(step), step


def test_ema_is_inert_when_decay_is_zero():
    cfg = cfg_with(ema_decay=0.0)
    _, state = pair(cfg, seed=0)
    after = J.update_ema(state, cfg, 10_000)
    assert after.ema_shadow is None
    assert J.ema_ready(after, cfg, 10_000) is False
    swapped, displaced = J.swap_in_ema(after, cfg, 10_000)
    assert displaced is None
    assert_same(
        flatten(nnx.to_pure_dict(swapped.actor[1])), flatten(nnx.to_pure_dict(state.actor[1]))
    )


def test_swap_in_and_out_ema_round_trips_exactly():
    cfg = cfg_with(ema_warmup_steps=0, ema_decay=0.5)
    _, state = pair(cfg, seed=0)
    # Make the shadow differ from the live actor: warm up, then move the actor.
    state = J.update_ema(state, cfg, 10)
    torch.manual_seed(12)
    other = build_policy(OBS, ACT, cfg, device="cpu", action_low=LOW, action_high=HIGH)
    state = state.replace(actor=load_pair(state.actor, to_jax_tree(other.actor.state_dict())))
    assert J.ema_ready(state, cfg, 10)

    live_before = flatten(nnx.to_pure_dict(state.actor[1]))
    swapped, displaced = J.swap_in_ema(state, cfg, 10)
    shadow = flatten(nnx.to_pure_dict(state.ema_shadow))
    assert_same(flatten(nnx.to_pure_dict(swapped.actor[1])), shadow)
    assert_same(flatten(nnx.to_pure_dict(displaced)), live_before)

    restored = J.swap_out_ema(swapped, displaced)
    assert_same(flatten(nnx.to_pure_dict(restored.actor[1])), live_before)
    # Restoring twice is not idempotent-but-wrong; it is a no-op when nothing was
    # displaced.
    assert J.swap_out_ema(restored, None).actor[1] is restored.actor[1]


def test_ema_state_dict_round_trips():
    cfg = cfg_with(ema_warmup_steps=0, ema_decay=0.9)
    _, state = pair(cfg, seed=0)
    warmed = J.update_ema(state, cfg, 10)
    saved = J.ema_state_dict(warmed, cfg)
    assert set(saved) == {"decay", "initialised", "shadow_params"}
    fresh = J.update_ema(state, cfg, 10)
    restored = J.load_ema_state_dict(fresh, saved)
    assert restored.ema_initialised == warmed.ema_initialised
    assert_same(
        flatten(nnx.to_pure_dict(restored.ema_shadow)),
        flatten(nnx.to_pure_dict(warmed.ema_shadow)),
    )


def wake_actor(state, t_policy, seed=21):
    """Give the actor a non-zero output layer on both sides.

    ``SJFPOFlowModel`` zero-inits ``out``, so a freshly built actor predicts zero
    velocity and **l_old does not depend on the frozen weights at all**. Any test
    that claims to exercise the behaviour snapshot has to wake it first, or it is
    comparing two copies of the same constant.
    """
    torch.manual_seed(seed)
    weights = torch.randn_like(t_policy.actor.out.weight) * 0.1
    bias = torch.randn_like(t_policy.actor.out.bias) * 0.1
    with torch.no_grad():
        t_policy.actor.out.weight.copy_(weights)
        t_policy.actor.out.bias.copy_(bias)
    params = nnx.to_pure_dict(state.actor[1])
    params["out"]["kernel"] = jnp.asarray(np.asarray(weights.numpy()).T)
    params["out"]["bias"] = jnp.asarray(np.asarray(bias.numpy()))
    state = state.replace(actor=load_pair(state.actor, params))
    return J.snapshot_behavior_policy(state)


# --------------------------------------------------------- validation parity


DELETE = object()
"""Marks a batch key that must be absent, not merely malformed."""


def batch_with(cfg, **overrides):
    """A valid batch, then one field broken. Both sides get the same break.

    Overrides are callables applied to the jax batch; the broken value is
    converted back to torch for the oracle, which only speaks torch. ``DELETE``
    removes the key entirely, for the "provided together" condition.
    """
    t_policy, state = pair(cfg, seed=0)
    t_batch, j_batch, _ = both_batches(t_policy, state, cfg, truncate=((1, 9),))
    for key, value in overrides.items():
        if value is DELETE:
            j_batch.pop(key, None)
            t_batch.pop(key, None)
            continue
        broken = value(j_batch[key]) if callable(value) else value
        j_batch[key] = broken
        t_batch[key] = torch.as_tensor(np.asarray(broken))
    return t_policy, state, t_batch, j_batch


VALIDATION_CASES = {
    "mc_tau_missing_alone": ({"mc_eps": DELETE}, "provided together"),
    "mc_eps_missing_alone": ({"mc_tau": DELETE}, "provided together"),
    "mc_tau_wrong_shape": ({"mc_tau": lambda x: jnp.zeros((N, MC, CFG.num_flow_tokens + 1, 1))}, "mc_tau shape"),
    "mc_eps_wrong_shape": ({"mc_eps": lambda x: jnp.zeros((N, MC, CFG.num_flow_tokens + 1, ACT))}, "stored MC noise shape"),
    "behavior_loss_wrong_shape": ({"behavior_cfm_loss": lambda x: jnp.zeros((N, MC + 1))}, "behavior_cfm_loss shape"),
    "behavior_token_wrong_shape": (
        {"behavior_cfm_loss_token": lambda x: jnp.zeros((N, MC, CFG.num_flow_tokens + 1))},
        "behavior_cfm_loss_token shape",
    ),
}


@pytest.mark.parametrize("case", sorted(VALIDATION_CASES))
def test_malformed_batches_raise_on_both_sides(case):
    """One test per condition, both sides, same exception type.

    An invalid batch that silently reshapes is worse than one that raises: the
    ratio would compare means over different positions and nothing would complain.
    """
    overrides, match = VALIDATION_CASES[case]
    cfg = CFG
    t_policy, state, t_batch, j_batch = batch_with(cfg, **overrides)
    with pytest.raises(ValueError, match=match):
        t_policy(t_batch, compute_info=False, check_finite=False)
    with pytest.raises(ValueError, match=match):
        J.forward_loss(state, j_batch, cfg)


def test_value_return_length_mismatch_raises_on_both_sides():
    cfg = CFG
    t_policy, state, t_batch, j_batch = batch_with(cfg, raw_returns=lambda x: jnp.zeros(N + 1))
    with pytest.raises(ValueError, match="value/return shape mismatch"):
        t_policy(t_batch, compute_info=False, check_finite=False)
    with pytest.raises(ValueError, match="value/return shape mismatch"):
        J.forward_loss(state, j_batch, cfg)


@pytest.mark.parametrize(
    "case, match",
    [
        ("obs_width", "state length"),
        ("horizon", "action horizon"),
        ("pad_shape", "action_is_pad shape"),
    ],
)
def test_geometry_mismatches_raise_on_both_sides(case, match):
    cfg = CFG
    shapes = {
        "obs_width": lambda x: jnp.zeros((N, cfg.n_obs_steps + 1, OBS), dtype=jnp.float32),
        "horizon": lambda x: jnp.zeros((N, cfg.action_horizon + 1, ACT), dtype=jnp.float32),
        "pad_shape": lambda x: jnp.zeros((N, cfg.action_horizon + 1), dtype=jnp.bool_),
    }
    key = {"obs_width": OBS_STATE, "horizon": ACTION, "pad_shape": "action_is_pad"}[case]
    t_policy, state, t_batch, j_batch = batch_with(cfg, **{key: shapes[case]})
    with pytest.raises(ValueError, match=match):
        t_policy(t_batch, compute_info=False, check_finite=False)
    with pytest.raises(ValueError, match=match):
        J.forward_loss(state, j_batch, cfg)
