"""Tests for scripts/train.py --resume and train_sjfpo_run(resume_from_checkpoint=)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402

from sjfpo.checkpointing import save_policy_checkpoint  # noqa: E402
from sjfpo.configuration_sjfpo import SJFPOConfig  # noqa: E402
from sjfpo.modeling_sjfpo import SJFPOPolicy  # noqa: E402
from sjfpo.train_sjfpo import train_sjfpo_run  # noqa: E402
from tests.fixtures.fake_env import FakeGymEnv  # noqa: E402


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_cli", ROOT / "scripts" / "train.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_cli"] = mod  # dataclasses resolve via sys.modules
    spec.loader.exec_module(mod)
    return mod


def test_resolve_run_dir_plain_and_resume_dedup(tmp_path):
    train = load_train_module()
    assert train.resolve_run_dir(tmp_path, "sjfpo", "Hopper-v5", 0, None) == (
        tmp_path / "sjfpo_Hopper-v5_seed0"
    )
    first = train.resolve_run_dir(tmp_path, "sjfpo", "Hopper-v5", 0, "ckpt")
    assert first == tmp_path / "sjfpo_Hopper-v5_seed0_resume"
    first.mkdir()
    second = train.resolve_run_dir(tmp_path, "sjfpo", "Hopper-v5", 0, "ckpt")
    assert second == tmp_path / "sjfpo_Hopper-v5_seed0_resume2"


def test_main_rejects_resume_for_baselines_and_missing_dir(tmp_path):
    train = load_train_module()
    assert train.main(["--method", "ppo", "--env", "Hopper-v5", "--resume", "x"]) == 1
    assert (
        train.main(
            [
                "--method",
                "sjfpo",
                "--env",
                "Hopper-v5",
                "--resume",
                str(tmp_path / "nope"),
            ]
        )
        == 1
    )


def test_resume_from_continues_training(tmp_path):
    torch.manual_seed(0)
    cfg = SJFPOConfig(
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (4,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        num_mc_samples=2,
        dit_depth=2,
        chunks_per_update=2,
        num_epochs=1,
        num_minibatches=1,
    )
    policy = SJFPOPolicy(cfg)
    optimizer = cfg.get_optimizer_preset().build(policy.get_optim_params()["params"])
    ckpt_dir = tmp_path / "source" / "checkpoint_0"
    save_policy_checkpoint(ckpt_dir, policy=policy, optimizer=optimizer, global_step=0)

    out_dir = tmp_path / "resumed"
    result = train_sjfpo_run(
        FakeGymEnv(obs_dim=4, action_dim=2, max_steps=20),
        out_dir,
        cfg=cfg,
        seed=0,
        total_steps=32,
        eval_steps=1000,
        save_steps=1000,
        resume_from_checkpoint=ckpt_dir,
    )
    assert result["global_step"] >= 32  # absolute budget continued past restore
    info = json.loads((out_dir / "resume_info.json").read_text(encoding="utf-8"))
    assert info == {"resume_from_checkpoint": str(ckpt_dir), "restored_global_step": 0}
    assert (out_dir / "checkpoint_final" / "model.safetensors").is_file()
