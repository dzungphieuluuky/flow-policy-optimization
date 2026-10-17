"""The two actor classes must agree at every argument position.

``actor_arch`` swaps one for the other. ``SJFPOPolicy`` builds them with
positional arguments. So a name that differs at the same index means one class
is called with a value the other would call something else.
"""

import inspect

import pytest
import torch

from myfpo.flow import SJFPOFlowMLP, SJFPOFlowModel


def _params(cls):
    return list(inspect.signature(cls.__init__).parameters)[1:]


def test_the_first_three_positions_have_the_same_names_in_both_actors():
    """action_dim, action_horizon, hidden_dim.

    The MLP used to call the third one ``condition_dim``. Nothing caught it,
    because the value passed was ``config.hidden_dim`` in both cases, so the
    code ran. Only a reader comparing the two signatures could see the clash.
    """
    assert _params(SJFPOFlowMLP)[:3] == ["action_dim", "action_horizon", "hidden_dim"]
    assert _params(SJFPOFlowModel)[:3] == ["action_dim", "action_horizon", "hidden_dim"]


def test_the_time_embedding_position_is_named_the_same():
    assert "time_embed_dim" in _params(SJFPOFlowMLP)
    assert "time_embed_dim" in _params(SJFPOFlowModel)


def test_the_shared_prefix_matches_and_the_rest_is_architecture_specific():
    """The contract is the prefix, not the whole signature.

    The two actors legitimately differ after the third argument: the MLP takes
    ``hidden_dims``, the DiT takes ``dit_depth``, ``dit_heads`` and
    ``dit_mlp_ratio``. So the test compares the prefix and records where it
    stops.
    """
    mlp, dit = _params(SJFPOFlowMLP), _params(SJFPOFlowModel)
    shared_prefix = []
    for a, b in zip(mlp, dit):
        if a != b:
            break
        shared_prefix.append(a)
    assert shared_prefix == ["action_dim", "action_horizon", "hidden_dim"], (
        f"the shared prefix is {shared_prefix}"
    )
    # The divergence is the architecture knobs, and it starts right after.
    assert mlp[3] == "time_embed_dim" and dit[3] == "dit_depth"
    assert "hidden_dims" in mlp and "hidden_dims" not in dit
    assert {"dit_depth", "dit_heads", "dit_mlp_ratio"} <= set(dit)


def test_both_actors_take_the_same_conditioning_vector():
    """The value is the encoder output width, so both must accept it."""
    hidden, tokens, action_dim = 16, 4, 3
    mlp = SJFPOFlowMLP(action_dim, tokens, hidden, 8, (32,))
    dit = SJFPOFlowModel(action_dim, tokens, hidden, 2, 2, 2.0, 8)

    x_t = torch.zeros(2, tokens, action_dim)
    t = torch.tensor([0.5])
    condition = torch.zeros(2, hidden)
    for actor in (mlp, dit):
        out = actor(x_t, t, condition)
        assert out.shape == (2, tokens, action_dim), type(actor).__name__


def test_the_encoder_width_is_what_the_actor_receives():
    """cfg.hidden_dim is the value at the third position for both actors."""
    from myfpo.configuration_sjfpo import SJFPOConfig
    from myfpo.encoder import StateEncoder

    cfg = SJFPOConfig()
    encoder = StateEncoder(3, cfg.hidden_dim, cfg.n_obs_steps)
    out = encoder(torch.zeros(1, cfg.n_obs_steps, 3))
    assert out.shape[-1] == cfg.hidden_dim, (
        "the actor's third argument is the encoder output width"
    )


def test_renaming_a_position_breaks_this_test():
    """Guard: the comparison must read the live signatures, not a copy."""
    src = inspect.getsource(SJFPOFlowMLP.__init__)
    assert "condition_dim" not in src, "the MLP took condition_dim again"
    assert "hidden_dim" in src
    with pytest.raises(AssertionError):
        assert _params(SJFPOFlowMLP)[:3] == ["action_dim", "action_horizon", "nope"]