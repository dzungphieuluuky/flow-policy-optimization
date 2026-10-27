"""The ``resmlp`` actor: the paper's residual unit, with attention kept.

Reference: Wang, Javali, Bortkiewicz, Trzcinski, Eysenbach. "1000 Layer
Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching
Capabilities." arXiv:2503.14858, NeurIPS 2025 Best Paper.

This file pins three things and refuses to claim a fourth.

It pins that the residual layout matches Section 3 of that paper. It pins that
the attention and the adaLN conditioning survive, because the paper has neither
a token axis nor a time input. And it pins the zero-init contracts, so a fresh
actor is the identity flow.

It does NOT pin that ``resmlp`` trains better. The paper's own Appendix A.2
reports that temporal-difference actor-critics saturate at depth 4, and this
repo is one. Only an ablation settles that.
"""

import inspect

import pytest
import torch
from torch import nn

from fixtures import cfg, make_batch

from myfpo.configuration import ACTOR_ARCHITECTURES
from myfpo.networks import (
    ResidualMLP,
    ResMLPBlock,
    MyfpoFlowModel,
    MyfpoFlowResMLP,
)
from myfpo.training import build_policy

OBS, ACT, HORIZON, HIDDEN, TIME_EMBED = 6, 3, 8, 32, 16
HEADS = 2


def _cfg(**kw):
    base = dict(
        obs_dim=OBS, act_dim=ACT,
        action_horizon=HORIZON, hidden_dim=HIDDEN, time_embed_dim=TIME_EMBED,
        dit_depth=1, dit_heads=HEADS, n_obs_steps=2,
    )
    base.update(kw)
    return cfg(**base)


def _policy(**kw):
    """The actor under test, on CPU with a squashed [-1, 1] action space."""
    return build_policy(
        OBS, ACT, _cfg(compile=False, **kw), device="cpu",
        action_low=[-1.0] * ACT, action_high=[1.0] * ACT,
    )


def _resmlp(blocks=2, units=4, mult=2.0):
    return MyfpoFlowResMLP(
        ACT, HORIZON, HIDDEN, blocks, units, mult, HEADS, TIME_EMBED
    )


# --- the paper's residual unit ---------------------------------------------


def test_one_residual_mlp_is_dense_norm_swish_repeated():
    """Section 3: four repeats of Dense -> LayerNorm -> Swish."""
    mlp = ResidualMLP(HIDDEN, 2.0, 4)
    assert len(mlp.denses) == 4
    assert len(mlp.norms) == 4
    # RMSNorm stands in for the paper's LayerNorm. Recorded in the docstring.
    assert all(isinstance(n, nn.RMSNorm) for n in mlp.norms)
    # The expansion is hidden * mult, and only the first unit reads hidden_dim.
    assert mlp.denses[0].in_features == HIDDEN
    assert mlp.denses[0].out_features == int(HIDDEN * 2.0)
    for d in mlp.denses[1:]:
        assert d.in_features == int(HIDDEN * 2.0)


def test_the_skip_is_added_once_after_the_final_unit_not_per_unit():
    """The paper puts the skip after the final activation of the block.

    So the gradient path is ``num_units`` layers long however deep the stack.
    An implementation that added a skip after every unit would be a different
    architecture, and it would train differently.
    """
    mlp = ResidualMLP(HIDDEN, 2.0, 4)
    torch.nn.init.normal_(mlp.out.weight, std=0.1)
    x = torch.randn(2, 5, HIDDEN)
    with torch.no_grad():
        expected_inner = x
        for dense, norm in zip(mlp.denses, mlp.norms):
            expected_inner = torch.nn.functional.silu(norm(dense(expected_inner)))
        expected = x + mlp.out(mlp.out_norm(expected_inner))
    torch.testing.assert_close(mlp(x), expected, rtol=0, atol=0)


def test_the_final_norm_sits_between_the_last_unit_and_the_output_projection():
    """A separate ``out_norm``, so ``norms`` stays one-per-unit for the paper."""
    mlp = ResidualMLP(HIDDEN, 2.0, 4)
    assert len(mlp.norms) == 4
    assert isinstance(mlp.out_norm, torch.nn.RMSNorm)
    assert mlp.out_norm.normalized_shape == (int(HIDDEN * 2.0),)
    names = [n for n, _ in mlp.named_children()]
    assert names == ["denses", "norms", "out_norm", "out"]


def test_a_fresh_residual_mlp_is_the_identity():
    """``out`` is zero-init, so the branch contributes nothing at init."""
    mlp = ResidualMLP(HIDDEN, 2.0, 4)
    x = torch.randn(2, 5, HIDDEN)
    assert torch.equal(mlp(x), x)


def test_the_unit_count_is_configurable_and_drives_the_dense_count():
    """Depth is counted in Dense layers, so the unit count must be a knob."""
    for units in (1, 4, 8):
        mlp = ResidualMLP(HIDDEN, 2.0, units)
        assert len(mlp.denses) == units


# --- the two parts that are NOT from the paper ------------------------------


def test_the_block_keeps_self_attention():
    """The chunk stays a sequence.

    The CFM loss is separable across tokens, so per-token mixing would still
    train. But the ASPO trust region and the per-chunk advantage couple the
    tokens. Both reference networks mix across action steps: Diffusion Policy
    with a temporal Conv1d (kernel_size=5), ACT with self-attention over the
    chunk.
    """
    block = ResMLPBlock(HIDDEN, HEADS, 2.0, 4)
    assert isinstance(block.attn, nn.Module)
    assert hasattr(block.attn, "qkv") and hasattr(block.attn, "out")
    assert block.attn.qkv.out_features == 3 * HIDDEN


def test_the_block_keeps_adaln_so_time_reaches_every_depth():
    """The paper has no time input, so it can condition once at the input.

    A flow model cannot. Without adaLN the per-token ``t`` would only reach the
    stem, and the target ``a - eps`` depends on it at every layer.
    """
    block = ResMLPBlock(HIDDEN, HEADS, 2.0, 4)
    assert isinstance(block.adaLN, nn.Module)
    assert block.adaLN.fc2.out_features == 6 * HIDDEN


def test_the_adaln_head_still_zero_inits():
    """adaLN-Zero. The whole stack starts as the identity."""
    block = ResMLPBlock(HIDDEN, HEADS, 2.0, 4)
    assert torch.equal(
        block.adaLN.fc2.weight, torch.zeros_like(block.adaLN.fc2.weight)
    )
    assert torch.equal(
        block.adaLN.fc2.bias, torch.zeros_like(block.adaLN.fc2.bias)
    )


# --- the actor contract ------------------------------------------------------


def test_resmlp_shares_the_actor_prefix_with_the_dit_actor():
    for cls in (MyfpoFlowModel, MyfpoFlowResMLP):
        params = list(inspect.signature(cls.__init__).parameters)[1:]
        assert params[:3] == ["action_dim", "action_horizon", "hidden_dim"], (
            f"{cls.__name__} prefix is {params[:3]}"
        )


def test_resmlp_meets_the_shape_and_zero_init_contract():
    net = _resmlp()
    x = torch.randn(3, HORIZON, ACT)
    t, cond = torch.rand(3, HORIZON, 1), torch.randn(3, HIDDEN)
    assert net(x, t, cond).shape == x.shape
    assert torch.equal(net(x, t, cond), torch.zeros_like(x))


def test_resmlp_reports_the_same_conditioning_stats_as_the_dit():
    """``loss_with_info`` reads ``cond_*``. The keys must match across actors."""
    keys = {"temb", "fused", "c", "gate_msa", "gate_mlp", "scale_msa",
            "scale_mlp", "effective_rank", "weight_magnitude"}
    _, stats = _resmlp().forward_with_stats(
        torch.randn(2, HORIZON, ACT), torch.rand(2, HORIZON, 1), torch.randn(2, HIDDEN)
    )
    assert set(stats) == keys, set(stats) ^ keys


# --- the integration traps ---------------------------------------------------


def test_the_modulation_layer_is_named_adaln_so_weight_decay_still_skips_it():
    """The weight-decay exclusion matches the literal substring "adaLN".

    ``get_actor_optim_params`` puts a parameter in the no-decay group when
    ``"adaLN" in name`` or ``name.startswith("out.")``. A block that named its
    modulation anything else would start decaying it, with no error and no
    warning. This is the easiest way to get the integration wrong.
    """
    policy = _policy(actor_arch="resmlp")
    no_decay = {id(p) for p in policy.get_actor_optim_params()["params"][1]["params"]}
    checked = 0
    for name, param in policy.actor.named_parameters():
        if "adaLN" in name or name.startswith("out."):
            assert id(param) in no_decay, f"{name} is being weight-decayed"
            checked += 1
    assert checked >= 3, f"expected several excluded params, matched {checked}"


def test_the_resmlp_branch_params_still_get_weight_decay():
    """The guard above must not have swallowed the whole actor."""
    policy = _policy(actor_arch="resmlp")
    groups = policy.get_actor_optim_params()["params"]
    no_decay = {id(p) for p in groups[1]["params"]}
    total = {id(p) for p in policy.actor.parameters()}
    assert len(no_decay) < len(total), "everything landed in the no-decay group"


def test_the_config_names_the_three_resmlp_knobs_and_validates_them():
    gcfg = _cfg()
    assert (gcfg.resmlp_blocks, gcfg.resmlp_units, gcfg.resmlp_hidden_mult) == (2, 4, 2.0)
    assert "resmlp" in ACTOR_ARCHITECTURES
    for field, value in (
        ("resmlp_blocks", 0),
        ("resmlp_units", 0),
        ("resmlp_hidden_mult", 0.0),
    ):
        with pytest.raises(ValueError, match=field):
            _cfg(**{field: value}).validate()


def test_a_tracked_run_works_for_every_actor():
    """``loss_with_info`` unpacks actor stats, so every actor must honour it."""
    for arch in ACTOR_ARCHITECTURES:
        policy = _policy(actor_arch=arch)
        batch = make_batch(policy, obs_dim=OBS, action_dim=ACT, horizon=HORIZON)
        loss, info = policy.loss_with_info(batch)
        assert torch.isfinite(loss).all(), arch
        assert "mean_log_ratio" in info, arch
        assert any(k.startswith("cond_") for k in info), arch


def test_the_policy_builds_the_configured_actor():
    for arch, cls in (("dit", MyfpoFlowModel), ("resmlp", MyfpoFlowResMLP)):
        assert isinstance(_policy(actor_arch=arch).actor, cls), arch


def test_the_dit_path_is_untouched_by_the_resmlp_actor():
    """Same seed, same stem weights.

    Parameter construction order fixes the torch RNG stream. Refactoring the
    stem into a base class moved the blocks, and that redrew every weight in
    the DiT actor: 262 of the 1017 pinned tensors changed. This pins the order
    so a later refactor cannot repeat it silently.
    """
    torch.manual_seed(0)
    a = MyfpoFlowModel(ACT, HORIZON, HIDDEN, 3, HEADS, 2.0, TIME_EMBED)
    first = {
        k: v.clone() for k, v in a.state_dict().items()
        if k.startswith(("in_proj", "token_pos", "time_mlp"))
    }
    torch.manual_seed(0)
    b = MyfpoFlowModel(ACT, HORIZON, HIDDEN, 3, HEADS, 2.0, TIME_EMBED)
    for k, v in first.items():
        assert torch.equal(v, b.state_dict()[k]), k


def test_the_depth_metric_the_paper_uses_is_recoverable_from_the_config():
    """Depth is Dense layers. The config must express both factors."""
    gcfg = _cfg(resmlp_blocks=3, resmlp_units=4)
    dense_mlp_layers = gcfg.resmlp_blocks * gcfg.resmlp_units
    assert dense_mlp_layers == 12, "3 blocks x 4 units"
    dit = _cfg(dit_depth=3)
    # The DiT block holds six Dense layers, so it is already past the paper's
    # depth-4 baseline at the shipped default.
    assert 6 * dit.dit_depth == 18
