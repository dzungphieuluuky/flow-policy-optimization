"""The checkpoint is one payload, and an old layout fails with a clear reason."""

import pytest
import torch

from myfpo.checkpointing import (
    CHECKPOINT_FILE,
    ResumeError,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from myfpo.modeling import make_optimizer
from fixtures import make_test_policy


def test_a_round_trip_restores_weights_counters_and_ema(tmp_path):
    policy = make_test_policy()
    policy.update_ema(0)
    optimizer = make_optimizer(policy)
    policy.step_count = 7
    save_policy_checkpoint(
        tmp_path,
        policy=policy,
        optimizer=optimizer,
        global_step=42,
        ema_state=policy.ema_state_dict(),
    )

    loaded, loaded_optimizer, counters = load_policy_checkpoint(tmp_path)
    assert counters == {"global_step": 42, "step_count": 7}
    for name, value in policy.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name]), name
    assert len(loaded_optimizer.param_groups) == len(optimizer.param_groups)


def test_an_old_style_directory_says_what_is_missing(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ResumeError, match=CHECKPOINT_FILE):
        load_policy_checkpoint(tmp_path)
