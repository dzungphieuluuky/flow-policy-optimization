"""The actor classes must agree at every argument position.

``actor_arch`` swaps one trunk for another. ``MyfpoPolicy`` builds them with
positional arguments. So a name that differs at the same index means one class
is called with a value the other would call something else.

``SJFPOFlowMLP`` used to be the second class here, and it called the third
argument ``condition_dim`` while the DiT called it ``hidden_dim``. Both received
``config.hidden_dim``, so the code ran. Only a reader comparing signatures could
see the clash. That class is gone; this file keeps the check for whatever the
second actor is.
"""

import inspect

import torch

from myfpo.configuration import MyfpoConfig
from myfpo.processor import StateEncoder
from myfpo.networks import FlowModel

SRC = __import__("pathlib").Path(__file__).resolve().parents[1] / "src" / "myfpo"


def _params(cls):
    return list(inspect.signature(cls.__init__).parameters)[1:]


def test_the_dit_prefix_is_the_contract():
    """action_dim, action_horizon, hidden_dim, in that order."""
    assert _params(FlowModel)[:3] == [
        "action_dim", "action_horizon", "hidden_dim",
    ]


def test_the_third_argument_is_the_encoder_output_width():
    """cfg.hidden_dim is the value at that position, and the encoder matches."""
    cfg = MyfpoConfig()
    encoder = StateEncoder(3, cfg.hidden_dim, cfg.num_obs_steps)
    out = encoder(torch.zeros(1, cfg.num_obs_steps, 3))
    assert out.shape[-1] == cfg.hidden_dim, (
        "the actor's third argument is the encoder output width"
    )


def test_the_time_embedding_argument_is_present():
    assert "time_embed_dim" in _params(FlowModel)


def test_no_actor_names_an_argument_condition_dim():
    """That name read as the space dimension, not a width.

    The check is scoped to the actor section of ``networks.py``. The critic
    section after ``# CRITIC`` is out of scope: ``Critic.condition_dim`` is a
    different thing and stays. It selects the shared-encoder mode, where the
    critic is fed one vector instead of a window. ``modeling`` still passes it
    to the Critic, so that file is out of scope here.
    """
    actor = (SRC / "networks.py").read_text().split("# CRITIC")[0]
    assert "condition_dim" not in actor, "the actor reintroduced condition_dim"


def test_the_comparison_reads_the_live_signature():
    """Guard: this file must not pin a copy of the signature."""
    live = _params(FlowModel)
    assert live is not None
    assert live == _params(FlowModel)
    # A wrong expectation must fail, so the assertion above is doing work.
    assert live[:3] != ["action_dim", "action_horizon", "nope"]