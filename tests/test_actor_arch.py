"""The ``actor_arch`` swap point, and the contract every actor must honour.

The value ``mlp`` used to live here. It flattened the chunk into one vector, so
a per-token timestep could not modulate per token. It also had three hidden
layers, Mish, and no residual and no normalisation. Both reference networks for
chunked action prediction mix across action steps: Diffusion Policy with a
temporal ``Conv1d`` (``kernel_size=5``), ACT with self-attention over the chunk.
The flattened trunk was the odd one out on both counts. The attention here is
causal, unlike ACT; ``SelfAttention`` records why.
"""

import dataclasses

import pytest
import torch

from fixtures import cfg
from fixtures import make_batch

from myfpo.configuration import ACTOR_ARCHITECTURES, MyfpoConfig
from myfpo.flow import MyfpoFlowModel
from myfpo.training import build_policy

OBS, ACT, HORIZON, HIDDEN, TIME_EMBED = 6, 3, 8, 32, 16


def _cfg(**kw):
    base = dict(
        obs_dim=OBS, act_dim=ACT,
        action_horizon=HORIZON, hidden_dim=HIDDEN, time_embed_dim=TIME_EMBED,
        dit_depth=1, dit_heads=2, n_obs_steps=2,
    )
    base.update(kw)
    return cfg(**base)



def _dit():
    return MyfpoFlowModel(ACT, HORIZON, HIDDEN, 1, 2, 2.0, TIME_EMBED)


def test_the_dit_first_layer_projects_the_action_only():
    """The chunk enters through one ``action_dim -> hidden_dim`` projection.

    The conditioning and the time enter later, through ``adaLN``. So no single
    layer sees the concatenation of chunk, time and observation.
    """
    assert _dit().in_proj.in_features == ACT


def test_shapes_follow_the_actor_contract():
    """``[B, K, action_dim]`` in, ``[B, K, action_dim]`` velocity out."""
    dit = _dit()
    x = torch.randn(3, HORIZON, ACT)
    t = torch.rand(3, HORIZON, 1)
    cond = torch.randn(3, HIDDEN)
    assert dit(x, t, cond).shape == x.shape


def test_untrained_predicts_zero_velocity():
    """The output layer is zero-init, so a fresh policy is the identity flow."""
    out = _dit()(torch.randn(3, HORIZON, ACT), torch.rand(3, HORIZON, 1),
                 torch.randn(3, HIDDEN))
    assert torch.equal(out, torch.zeros_like(out))


def test_every_form_of_time_is_accepted():
    """Scalar, ``[B]``, and ``[B, K, 1]``. Per-token is the one training uses."""
    dit = _dit()
    x, cond = torch.randn(3, HORIZON, ACT), torch.randn(3, HIDDEN)
    assert dit(x, torch.tensor(0.4), cond).shape == x.shape
    assert dit(x, 0.4, cond).shape == x.shape
    assert dit(x, torch.rand(3), cond).shape == x.shape
    assert dit(x, torch.rand(3, HORIZON, 1), cond).shape == x.shape


def test_a_fresh_actor_ignores_the_time_entirely():
    """adaLN-Zero makes this true on purpose, so it must be pinned.

    Every ``adaLN.fc2`` and ``adaLN_out`` is zero-initialised. So the whole
    conditioning path is zero at init: the blocks are the identity and the
    output modulation is the identity. A fresh actor reads only ``x_t`` and the
    token positions. Together with the zero output layer this is what makes a
    fresh policy the identity flow.
    """
    dit = _dit()
    x, cond = torch.randn(2, HORIZON, ACT), torch.randn(2, HIDDEN)
    early = dit(x, torch.full((2,), 0.1), cond)
    late = dit(x, torch.full((2,), 0.9), cond)
    assert torch.equal(early, late), (
        "a fresh actor responded to t; the zero-init path is not intact"
    )


def test_a_per_token_time_reaches_every_token_once_the_path_is_broken():
    """Two tokens, two times, and the adaLN path opened.

    The CFM loss is separable across tokens and its target is per token, so an
    actor that dropped the per-token time would still train. Only this pins that
    the time arrives. The zero-init contract is pinned by the test above.
    """
    torch.manual_seed(0)
    dit = _dit()
    torch.nn.init.normal_(dit.out.weight, std=0.05)
    torch.nn.init.normal_(dit.adaLN_out.weight, std=0.05)
    for block in dit.blocks:
        torch.nn.init.normal_(block.adaLN.fc2.weight, std=0.05)

    x, cond = torch.randn(1, 2, ACT), torch.randn(1, HIDDEN)
    uniform = dit(x, torch.tensor([[[0.1], [0.1]]]), cond)
    mixed = dit(x, torch.tensor([[[0.1], [0.9]]]), cond)
    # The two tokens of the mixed call disagree, so each token read its own t.
    assert not torch.allclose(mixed[0, 0], mixed[0, 1], atol=1e-6)
    # And the tokens are not a fixed pattern: moving t swaps which one differs.
    swapped = dit(x, torch.tensor([[[0.9], [0.1]]]), cond)
    assert not torch.allclose(mixed, swapped, atol=1e-6)
    assert uniform.shape == mixed.shape


def test_the_policy_builds_the_configured_actor():
    policy = build_policy(
        OBS, ACT, _cfg(actor_arch="dit", compile=False), device="cpu",
        action_low=[-1.0] * ACT, action_high=[1.0] * ACT,
    )
    assert isinstance(policy.actor, MyfpoFlowModel)


def test_a_tracked_run_works():
    """``loss_with_info`` unpacks actor stats, so the actor must honour it."""
    policy = build_policy(
        OBS, ACT, _cfg(actor_arch="dit", compile=False), device="cpu",
        action_low=[-1.0] * ACT, action_high=[1.0] * ACT,
    )
    batch = make_batch(policy, obs_dim=OBS, action_dim=ACT, horizon=HORIZON)
    loss, info = policy.loss_with_info(batch)
    assert torch.isfinite(loss).all()
    assert "mean_log_ratio" in info
    assert any(k.startswith("cond_") for k in info)


def test_the_actor_output_layer_skips_weight_decay_when_compiled():
    """``torch.compile`` renames params, so the ``out.`` test must still match."""
    policy = build_policy(
        OBS, ACT, _cfg(compile=True), device="cpu",
        action_low=[-1.0] * ACT, action_high=[1.0] * ACT,
    )
    assert hasattr(policy.actor, "_orig_mod"), "the actor must be compiled"
    no_decay = {id(p) for p in policy.get_actor_optim_params()["params"][1]["params"]}
    out = policy.actor._orig_mod.out
    assert id(out.weight) in no_decay
    assert id(out.bias) in no_decay


def test_an_unknown_arch_is_rejected():
    with pytest.raises(ValueError, match="actor_arch"):
        _cfg(actor_arch="unet").validate()


def test_the_removed_arch_is_rejected():
    """``mlp`` is gone. A config still carrying it must fail loudly."""
    with pytest.raises(ValueError, match="actor_arch"):
        _cfg(actor_arch="mlp").validate()


def test_no_actor_field_remains_for_the_removed_arch():
    """``mlp_hidden_dims`` had no meaning left, so the field is gone too."""
    assert "mlp_hidden_dims" not in {f.name for f in
                                     dataclasses.fields(MyfpoConfig)}
    assert "mlp" not in ACTOR_ARCHITECTURES
