"""The critic head: two token streams that condition each other, no mask.

The state window enters as one token per frame. The action chunk enters as one
token per step. Each stream queries the other, then the last state token and
the mean action token go to the same MLP the concat head used. The action
tokens carry no norm, so a constant chunk keeps its offset.
"""

from __future__ import annotations


import torch
from fixtures import cfg

from myfpo.critic import Critic, CriticMember
from myfpo.layers import CrossAttention
from myfpo.training import build_policy

OBS, HIDDEN, STEPS, ACT, H = 11, 32, 8, 3, 6


def _member(**kw) -> CriticMember:
    return CriticMember(OBS, ACT, HIDDEN, STEPS, H, **kw)


def _pool(member: CriticMember, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """The vector the MLP reads, captured with a hook."""
    seen: dict[str, torch.Tensor] = {}
    handle = member.mlp[0].register_forward_pre_hook(
        lambda _m, args: seen.__setitem__("x", args[0].detach().clone())
    )
    with torch.no_grad():
        member(obs, act, torch.full((obs.shape[0],), 0.5))
    handle.remove()
    return seen["x"]


def test_cross_attention_keeps_the_query_length():
    attn = CrossAttention(32, 4)
    out = attn(torch.randn(2, 7, 32), torch.randn(2, 3, 32))
    assert out.shape == (2, 7, 32)


def test_a_late_key_reaches_an_early_query():
    """No causal mask: query 0 reads key 7."""
    torch.manual_seed(0)
    attn = CrossAttention(32, 4).eval()
    query = torch.randn(1, 1, 32)
    context = torch.randn(1, 8, 32)
    moved = context.clone()
    moved[:, 7] += 5.0
    with torch.no_grad():
        base = attn(query, context)
        other = attn(query, moved)
    assert not torch.allclose(base, other)


def test_the_two_directions_are_parallel():
    """Zero one output projection: that stream is the identity, and the other
    stream still carries the cross information. Both read the same input."""
    torch.manual_seed(0)
    obs = torch.randn(2, STEPS, OBS)
    act = torch.randn(2, H, ACT)

    def _reference(m: CriticMember) -> tuple[torch.Tensor, torch.Tensor]:
        window = m.norm_state(obs.flatten(1)).view(obs.shape)
        state_last = m.state_encoder.out_norm(
            m.state_encoder.encode_tokens(window)
        )[:, -1]
        tokens = m.act_in(m.norm_action(act.flatten(1)).view(act.shape))
        tokens = tokens + m.act_attn(m.act_attn_norm(tokens))
        return state_last, tokens.mean(dim=1)

    m = _member(dropout=0.0).eval()
    with torch.no_grad():
        state_last, act_mean = _reference(m)
        m.act_to_obs.out.weight.zero_()
        m.act_to_obs.out.bias.zero_()
        pooled = _pool(m, obs, act)
    assert torch.allclose(pooled[:, HIDDEN : 2 * HIDDEN], act_mean)
    assert not torch.allclose(pooled[:, :HIDDEN], state_last)

    m = _member(dropout=0.0).eval()
    with torch.no_grad():
        state_last, act_mean = _reference(m)
        m.obs_to_act.out.weight.zero_()
        m.obs_to_act.out.bias.zero_()
        pooled = _pool(m, obs, act)
    assert torch.equal(pooled[:, :HIDDEN], state_last)
    assert not torch.allclose(pooled[:, HIDDEN : 2 * HIDDEN], act_mean)


def test_the_action_tokens_keep_the_affine():
    torch.manual_seed(0)
    m = _member(dropout=0.0).eval()
    with torch.no_grad():
        m.norm_action.weight.copy_(torch.linspace(0.5, 2.0, H * ACT))
        m.norm_action.bias.copy_(torch.linspace(-0.3, 0.4, H * ACT))
    seen: dict[str, torch.Tensor] = {}
    handle = m.act_in.register_forward_pre_hook(
        lambda _m, args: seen.__setitem__("x", args[0].detach().clone())
    )
    act = torch.randn(2, H, ACT)
    with torch.no_grad():
        m(torch.randn(2, STEPS, OBS), act, torch.full((2,), 0.5))
    handle.remove()
    expected = (act.flatten(1) * m.norm_action.weight + m.norm_action.bias).view(act.shape)
    assert torch.equal(seen["x"], expected)


def test_a_constant_chunk_survives():
    """A chunk of all +5 and a chunk of all 0 must not reach the critic equal."""
    torch.manual_seed(0)
    m = _member(dropout=0.0).eval()
    obs = torch.randn(1, STEPS, OBS)
    with torch.no_grad():
        high = _pool(m, obs, torch.full((1, H, ACT), 5.0))
        zero = _pool(m, obs, torch.zeros(1, H, ACT))
    assert not torch.allclose(high[:, HIDDEN : 2 * HIDDEN], zero[:, HIDDEN : 2 * HIDDEN])


def test_the_shared_encoder_branch_runs_one_token():
    m = CriticMember(OBS, ACT, HIDDEN, 1, H, condition_dim=HIDDEN)
    assert m.norm_state is None
    assert m.state_encoder is None
    assert m.input_dim == 2 * HIDDEN + 1
    with torch.no_grad():
        q = m(torch.randn(3, HIDDEN), torch.randn(3, H, ACT), torch.rand(3))
    assert q.shape == (3,)


def test_the_chunk_attention_is_causal():
    """A later step cannot change an earlier step's token."""
    torch.manual_seed(0)
    m = _member(dropout=0.0).eval()
    obs = torch.randn(2, STEPS, OBS)
    act = torch.randn(2, H, ACT)
    seen: dict[str, torch.Tensor] = {}
    handle = m.act_attn.register_forward_hook(
        lambda _m, _args, out: seen.__setitem__("y", out.detach().clone())
    )
    with torch.no_grad():
        m(obs, act, torch.full((2,), 0.5))
        first = seen["y"]
        moved = act.clone()
        moved[:, -1] += 5.0
        m(obs, moved, torch.full((2,), 0.5))
    handle.remove()
    assert torch.allclose(first[:, :-1], seen["y"][:, :-1])
    assert not torch.allclose(first[:, -1], seen["y"][:, -1])


def test_the_head_count_is_four():
    assert _member().attn_heads == 4

def test_the_pool_is_twice_the_state_width():
    m = _member()
    assert m.input_dim == 2 * HIDDEN + 1
    assert m.mlp[0].in_features == m.input_dim


def test_a_narrow_pool_resizes_into_the_mlp():
    """state_hidden sets the attention width, hidden_dim sets the MLP width."""
    m = CriticMember(OBS, ACT, HIDDEN, STEPS, H, state_hidden=16)
    assert m.input_dim == 2 * 16 + 1
    with torch.no_grad():
        q = m(torch.randn(2, STEPS, OBS), torch.randn(2, H, ACT), torch.rand(2))
    assert q.shape == (2,)


def test_the_param_count_identity_holds():
    m = _member(dropout=0.0)
    parts = sum(p.numel() for p in m.parameters())
    named = {
        "mlp": sum(p.numel() for p in m.mlp.parameters()),
        "encoder": sum(p.numel() for p in m.state_encoder.parameters()),
        "act_in": sum(p.numel() for p in m.act_in.parameters()),
        "act_attn": sum(p.numel() for p in m.act_attn.parameters()),
        "obs_to_act": sum(p.numel() for p in m.obs_to_act.parameters()),
        "act_to_obs": sum(p.numel() for p in m.act_to_obs.parameters()),
        "norms": sum(p.numel() for p in m.cross_norm_obs.parameters())
        + sum(p.numel() for p in m.cross_norm_act.parameters())
        + sum(p.numel() for p in m.act_attn_norm.parameters()),
    }
    fixed = 2 * (STEPS * OBS) + 2 * (H * ACT)
    assert parts == sum(named.values()) + fixed


def test_the_ensemble_shares_the_head_count():
    c = Critic(OBS, HIDDEN, ACT, STEPS, H, 2)
    assert all(net.attn_heads == 4 for net in c.networks)
    with torch.no_grad():
        q = c(torch.randn(4, STEPS, OBS), torch.randn(4, H, ACT), torch.rand(4))
    assert q.shape == (2, 4)


def test_the_policy_builds_the_cross_critic():
    policy = build_policy(
        11,
        3,
        cfg(),
        device="cpu",
        action_low=[-1.0] * 3,
        action_high=[1.0] * 3,
    )
    first = policy.critic.networks[0].mlp[0]
    assert first.in_features == 2 * policy.config.hidden_dim + 1
    assert hasattr(policy.critic.networks[0], "obs_to_act")
