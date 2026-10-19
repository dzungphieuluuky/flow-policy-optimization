"""The shared adaLN block: one implementation, two MLPs.

``DiTBlock`` (a SwiGLU) and ``ResMLPBlock`` (the paper's residual MLP) used to
be the same block written twice. They are one class now, so the parameter
counts and the state_dict keys are pinned here: the counts are the ones
``docs/architecture.md`` quotes, and the keys are the checkpoint contract.
"""

import pytest

from myfpo.attention import AdaLNBlock, DiTBlock
from myfpo.configuration import MyfpoConfig
from myfpo.flow import ResMLPBlock


def _params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_both_blocks_are_the_same_block_with_a_different_mlp():
    """One implementation. A fix to the adaLN path now reaches both actors."""
    assert issubclass(DiTBlock, AdaLNBlock)
    assert issubclass(ResMLPBlock, AdaLNBlock)


def test_the_block_parameter_counts_are_pinned():
    """The docs quote 2.36M parameters and 280k per DiT block."""
    cfg = MyfpoConfig()
    dit = DiTBlock(cfg.hidden_dim, cfg.dit_heads, cfg.dit_mlp_ratio)
    res = ResMLPBlock(
        cfg.hidden_dim, cfg.dit_heads, cfg.resmlp_hidden_mult, cfg.resmlp_units
    )
    assert _params(dit) == 280_832
    assert _params(res) == 446_464


def test_the_block_state_dict_keys_are_the_checkpoint_contract():
    """A shared base must not move a weight. Old checkpoints keep loading."""
    cfg = MyfpoConfig()
    dit = DiTBlock(cfg.hidden_dim, cfg.dit_heads, cfg.dit_mlp_ratio)
    res = ResMLPBlock(
        cfg.hidden_dim, cfg.dit_heads, cfg.resmlp_hidden_mult, cfg.resmlp_units
    )
    assert set(dit.state_dict()) == {
        "norm1.weight",
        "attn.qkv.weight",
        "attn.qkv.bias",
        "attn.out.weight",
        "attn.out.bias",
        "norm2.weight",
        "mlp.gate_up.weight",
        "mlp.gate_up.bias",
        "mlp.down.weight",
        "mlp.down.bias",
        "adaLN.fc1.weight",
        "adaLN.fc1.bias",
        "adaLN.fc2.weight",
        "adaLN.fc2.bias",
    }
    # The resmlp block has the same five parts; only the MLP inside differs.
    assert {k.split(".")[0] for k in res.state_dict()} == {
        "norm1",
        "attn",
        "norm2",
        "mlp",
        "adaLN",
    }


def test_both_blocks_reject_a_width_of_one():
    """The width guard is the block's, so the resmlp actor gets it too."""
    with pytest.raises(ValueError, match="hidden_dim must be at least 2"):
        DiTBlock(1, 2, 2.0)
    with pytest.raises(ValueError, match="hidden_dim must be at least 2"):
        ResMLPBlock(1, 2, 2.0, 4)
