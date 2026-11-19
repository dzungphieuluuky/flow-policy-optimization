"""The action-token attention is causal, and the config rejects odd head_dim.

The module docstring used to say "the attention is maskless" while the call
passed ``is_causal=True``. The mask is deliberate (see ``SelfAttention``); this
module pins the behavior so the docstring and the code cannot drift again.
"""

from __future__ import annotations


import pytest
import torch

from flow_policy_optimization.networks import SelfAttention
from flow_policy_optimization.config import FlowPolicyOptimizationConfig


def test_a_later_token_cannot_change_an_earlier_one():
    attn = SelfAttention(16, 2)
    x = torch.randn(1, 6, 16)
    perturbed = x.clone()
    perturbed[:, -1] = torch.randn(1, 16)
    assert torch.equal(attn(x)[:, :-1], attn(perturbed)[:, :-1])


def test_an_earlier_token_does_change_a_later_one():
    attn = SelfAttention(16, 2)
    x = torch.randn(1, 6, 16)
    perturbed = x.clone()
    perturbed[:, 0] = torch.randn(1, 16)
    assert not torch.equal(attn(x)[:, 1:], attn(perturbed)[:, 1:])


def test_the_docstring_does_not_claim_the_mask_is_off():
    assert "maskless" not in SelfAttention.__doc__


def test_odd_head_dim_is_rejected_at_construction():
    with pytest.raises(ValueError, match=r"head_dim=3 must be even for RoPE"):
        SelfAttention(6, 2)


def test_even_head_dim_is_accepted():
    assert SelfAttention(12, 2).head_dim == 6


def test_config_accepts_any_width_the_critic_can_split():
    cfg = FlowPolicyOptimizationConfig(hidden_dim=24)
    assert cfg.hidden_dim % 8 == 0