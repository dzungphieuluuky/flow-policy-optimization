"""Tests for the LeRobot-native policy checkpoint (config.json + safetensors)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402

from sjfpo.checkpointing import (  # noqa: E402
    ResumeError,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from sjfpo.configuration_sjfpo import SJFPOConfig  # noqa: E402
from sjfpo.modeling_sjfpo import SJFPOPolicy  # noqa: E402

OBS_DIM, ACT_DIM = 4, 2


def make_policy(**overrides) -> SJFPOPolicy:
    base = dict(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (OBS_DIM,))
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT_DIM,))},
        num_mc_samples=2,
        dit_depth=2,
    )
    base.update(overrides)
    return SJFPOPolicy(SJFPOConfig(**base))


def make_batch(n: int = 2) -> dict:
    return {
        "observation.state": torch.randn(n, 16, OBS_DIM),
        "action": torch.randn(n, 16, ACT_DIM).clamp(-1.0, 1.0),
        "action_is_pad": torch.zeros(n, 16, dtype=torch.bool),
        "advantages": torch.randn(n) * 0.1,
        "returns": torch.randn(n) * 10.0,
    }


def test_save_writes_hub_files_plus_training_state(tmp_path):
    policy = make_policy()
    opt = torch.optim.AdamW(policy.get_optim_params()["params"])
    d = tmp_path / "ckpt"
    save_policy_checkpoint(d, policy=policy, optimizer=opt, global_step=123)
    names = {p.name for p in d.iterdir()}
    assert {
        "config.json",
        "model.safetensors",
        "optimizer.pt",
        "counters.json",
    } <= names
    saved = json.loads((d / "config.json").read_text())
    assert saved["type"] == "sjfpo"
    assert saved["action_horizon"] == 16
    assert json.loads((d / "counters.json").read_text()) == {
        "global_step": 123,
        "step_count": 0,
    }


def test_roundtrip_restores_weights_optimizer_and_counters(tmp_path):
    torch.manual_seed(0)
    policy = make_policy()
    opt = policy.config.get_optimizer_preset().build(
        policy.get_optim_params()["params"]
    )
    loss, _ = policy(make_batch())
    opt.zero_grad()
    loss.backward()
    opt.step()  # populates AdamW state (exp_avg buffers)
    policy.step_count = 7
    d = tmp_path / "ckpt"
    save_policy_checkpoint(d, policy=policy, optimizer=opt, global_step=123)

    loaded, loaded_opt, counters = load_policy_checkpoint(d)
    assert counters == {"global_step": 123, "step_count": 7}
    assert loaded.step_count == 7
    assert loaded.config.action_horizon == 16 and loaded.config.type == "sjfpo"
    for (n1, p1), (n2, p2) in zip(policy.named_parameters(), loaded.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), n1
    live_state, loaded_state = opt.state_dict(), loaded_opt.state_dict()
    assert live_state["param_groups"] == loaded_state["param_groups"]
    assert set(live_state["state"]) == set(loaded_state["state"])
    for k in live_state["state"]:
        for sk, v in live_state["state"][k].items():
            w = loaded_state["state"][k][sk]
            if isinstance(v, torch.Tensor):
                assert torch.equal(v, w), (k, sk)
            else:
                assert v == w, (k, sk)


def test_load_missing_or_partial_dir_raises(tmp_path):
    with pytest.raises(ResumeError, match="missing"):
        load_policy_checkpoint(tmp_path / "absent")
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "config.json").write_text("{}")
    with pytest.raises(ResumeError, match="missing"):
        load_policy_checkpoint(partial)


def test_save_rejects_non_sjfpo_policy(tmp_path):
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    with pytest.raises(ValueError, match="SJFPOPolicy"):
        save_policy_checkpoint(
            tmp_path, policy=torch.nn.Linear(2, 2), optimizer=opt, global_step=0
        )


def test_compiled_policy_saves_eager_keys_and_loads_cross_mode(tmp_path):
    """Checkpoints hold the real inner model: no _orig_mod keys on disk, and
    a compile-wrapped saver reloads exactly into an eager policy.

    On CPU (no CUDA) maybe_compile auto-disables, so this becomes an eager
    roundtrip check — the unwrapping logic is still exercised via
    unwrapped_policy (no-op when not wrapped).
    """
    from safetensors.torch import load_file  # noqa: E402

    from sjfpo.utils import maybe_compile  # noqa: E402

    torch.manual_seed(3)
    policy = make_policy()
    policy.actor = maybe_compile(policy.actor)
    policy.old_actor = maybe_compile(policy.old_actor)
    policy.critic = maybe_compile(policy.critic)
    is_compiled = hasattr(policy.actor, "_orig_mod")
    opt = torch.optim.AdamW(policy.get_optim_params()["params"])
    d = tmp_path / "ckpt"
    save_policy_checkpoint(d, policy=policy, optimizer=opt, global_step=5)
    # Wrappers restored after save: training continues compiled (no-op on CPU).
    assert hasattr(policy.actor, "_orig_mod") == is_compiled

    keys = load_file(d / "model.safetensors")
    assert keys, "empty safetensors"
    assert all(not k.startswith("_orig_mod.") for k in keys), "wrapper leaked to disk"
    assert any(k.startswith("actor.") for k in keys)

    loaded, _, _ = load_policy_checkpoint(d)
    assert not hasattr(loaded.actor, "_orig_mod")  # from_pretrained builds eager
    orig_state = policy.actor._orig_mod.state_dict() if is_compiled else policy.actor.state_dict()
    for (kn, vn), (kl, vl) in zip(
        sorted(orig_state.items()),
        sorted(loaded.actor.state_dict().items()),
    ):
        assert kn == kl
        assert torch.equal(vn.cpu(), vl.cpu())
