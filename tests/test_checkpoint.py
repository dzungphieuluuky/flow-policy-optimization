"""Tests for src/sjfpo/checkpointing.py (plan task 9).

TDD: written BEFORE the implementation exists — the first pytest run must
fail RED on the missing module, then go GREEN once checkpointing.py lands.

Contract under test (generic; the runner consumes this exact API with the
policy bundle):

* ``save_checkpoint(dir_path, state)`` serializes a nested ``state`` mapping
  whose leaves are torch modules / optimizers / tensors / plain python
  objects into a torch.save zip blob plus a manifest.json sidecar listing
  EVERY key (dotted paths), written via tmp-file + os.replace.
* ``load_checkpoint(dir_path, expected_keys=None, state=None)`` validates
  manifest completeness against ``expected_keys`` (iterable of dotted paths
  or a nested mapping), raising ResumeError listing missing keys; restores
  in-place into live objects found in ``state`` and returns the deserialized
  payload.

Test (b) is the plan-mandated SAME-STACK KILL-RESUME harness: parent runs 50
scripted steps (seed 0), snapshots, continues 20 steps; a subprocess worker
restores from the snapshot and replays the same 20 steps; rewards must be
BITWISE equal (==).
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.checkpointing import (  # noqa: E402
    ResumeError,
    capture_env_fingerprint,
    load_checkpoint,
    save_checkpoint,
)
from tests.resume_worker import (  # noqa: E402
    build_bundle,
    build_state,
    configure_determinism,
    run_segment,
    seed_everything,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Explicit enumeration of every manifest field required by plan 9(a).
EXPECTED_KEYS = [
    "models.encoder",
    "models.policy_head",
    "models.aux_head",
    "models.critic",
    "models.ema_proj",
    "optimizers.policy_opt",
    "optimizers.critic_opt",
    "frozen_stats.mu",
    "frozen_stats.std",
    "frame_buffer",
    "rng_python",
    "rng_numpy",
    "rng_torch",
    "counters.global_step",
    "counters.chunks",
    "eval_history_ptr",
    "flags.diverged",
    "env_fingerprint.gpu_name",
    "env_fingerprint.torch_version",
    "env_fingerprint.cuda_version",
]


@pytest.fixture(autouse=True)
def deterministic_cpu():
    configure_determinism()
    yield
    torch.use_deterministic_algorithms(False)


def assert_optimizer_state_equal(a: dict, b: dict, ctx: str) -> None:
    assert set(a["state"]) == set(b["state"]), ctx
    for pid in a["state"]:
        sa, sb = a["state"][pid], b["state"][pid]
        assert set(sa) == set(sb), f"{ctx}[{pid}]"
        for field in sa:
            va, vb = sa[field], sb[field]
            if torch.is_tensor(va):
                assert torch.equal(va, vb), f"{ctx}[{pid}].{field}"
            else:
                assert va == vb, f"{ctx}[{pid}].{field}"
    assert a["param_groups"] == b["param_groups"], f"{ctx}.param_groups"


def assert_numpy_rng_state_equal(a: tuple, b: tuple) -> None:
    assert a[0] == b[0]
    assert np.array_equal(a[1], b[1])
    assert a[2:] == b[2:]


def test_roundtrip_restores_every_manifest_field(tmp_path):
    seed_everything(42)
    models, optimizers = build_bundle(seed=42)
    run_segment(models, optimizers, start=0, n=3)  # populate optimizer state

    frame_buffer = deque([float(i) ** 2 for i in range(10)], maxlen=10)
    counters = {"global_step": 1234, "chunks": 155}
    eval_history_ptr = 7
    flags = {"diverged": False, "clip_warned": True}
    fingerprint = capture_env_fingerprint()
    assert set(fingerprint) == {"gpu_name", "torch_version", "cuda_version"}

    state = {
        "models": models,
        "optimizers": optimizers,
        "frozen_stats": {
            "mu": torch.linspace(-1.0, 1.0, 16),
            "std": torch.linspace(0.5, 2.0, 16),
        },
        "frame_buffer": frame_buffer,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "counters": counters,
        "eval_history_ptr": eval_history_ptr,
        "flags": flags,
        "env_fingerprint": fingerprint,
    }
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, state)

    # Manifest sidecar lists EVERY key.
    manifest = json.loads((ckpt / "manifest.json").read_text(encoding="utf-8"))
    saved_keys = set(manifest["keys"])
    for dotted in EXPECTED_KEYS:
        assert dotted in saved_keys, dotted

        # Fresh bundle with DIFFERENT init + empty containers as restore targets.
    models2, optimizers2 = build_bundle(seed=999)
    frame_buffer2 = deque(maxlen=10)
    template = {
        "models": models2,
        "optimizers": optimizers2,
        "frozen_stats": {"mu": torch.zeros(16), "std": torch.ones(16)},
        "frame_buffer": frame_buffer2,
        "rng_python": None,
        "rng_numpy": None,
        "rng_torch": None,
        "counters": {},
        "eval_history_ptr": 0,
        "flags": {},
        "env_fingerprint": {},
    }
    restored = load_checkpoint(ckpt, expected_keys=EXPECTED_KEYS, state=template)

    for name in ("encoder", "policy_head", "aux_head", "critic", "ema_proj"):
        ref = state["models"][name].state_dict()
        live = models2[name].state_dict()
        payload = restored["models"][name]
        for k in ref:
            assert torch.equal(ref[k], live[k]), f"live restore {name}.{k}"
            assert torch.equal(ref[k], payload[k]), f"payload restore {name}.{k}"

    for name in ("policy_opt", "critic_opt"):
        assert_optimizer_state_equal(
            state["optimizers"][name].state_dict(),
            optimizers2[name].state_dict(),
            name,
        )

    assert torch.equal(template["frozen_stats"]["mu"], state["frozen_stats"]["mu"])
    assert torch.equal(template["frozen_stats"]["std"], state["frozen_stats"]["std"])
    assert isinstance(restored["frame_buffer"], deque)
    assert list(restored["frame_buffer"]) == list(frame_buffer)
    assert restored["rng_python"] == state["rng_python"]
    assert_numpy_rng_state_equal(restored["rng_numpy"], state["rng_numpy"])
    assert torch.equal(restored["rng_torch"], state["rng_torch"])
    assert restored["counters"] == counters
    assert restored["eval_history_ptr"] == eval_history_ptr
    assert restored["flags"] == flags
    assert restored["env_fingerprint"] == fingerprint


def test_same_stack_kill_resume_bitwise_rewards(tmp_path):
    configure_determinism()
    seed_everything(0)
    models, optimizers = build_bundle(seed=0)
    run_segment(models, optimizers, start=0, n=50)

    snapshot = tmp_path / "snap"
    save_checkpoint(snapshot, build_state(models, optimizers))

    # Parent continues 20 steps with its LIVE (post-snapshot) RNG evolution.
    rewards_r0 = run_segment(models, optimizers, start=50, n=20)

    out_json = tmp_path / "worker_rewards.json"
    subprocess.run(
        [sys.executable, "tests/resume_worker.py", str(snapshot), str(out_json)],
        cwd=str(REPO_ROOT),
        check=True,
        timeout=300,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    worker_rewards = json.loads(out_json.read_text(encoding="utf-8"))

    assert len(worker_rewards) == 20
    assert rewards_r0 == worker_rewards, (
        f"resume diverged\nparent={rewards_r0!r}\nworker={worker_rewards!r}"
    )
    print("RESUME DETERMINISTIC: 20/20 rewards equal")


def test_corrupt_manifest_missing_key_raises_resume_error(tmp_path):
    seed_everything(0)
    models, optimizers = build_bundle(seed=0)
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, build_state(models, optimizers))

    manifest = json.loads((ckpt / "manifest.json").read_text(encoding="utf-8"))
    manifest["keys"].pop("counters.global_step")
    manifest["keys"].pop("env_fingerprint.gpu_name")
    (ckpt / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ResumeError) as excinfo:
        load_checkpoint(ckpt, expected_keys=EXPECTED_KEYS)
    msg = str(excinfo.value)
    assert "counters.global_step" in msg
    assert "env_fingerprint.gpu_name" in msg

    # Missing manifest entirely is also a ResumeError, never a bare OSError.
    with pytest.raises(ResumeError):
        load_checkpoint(tmp_path / "does_not_exist", expected_keys=EXPECTED_KEYS)


def test_atomicity_killed_mid_write_leaves_previous_checkpoint_loadable(
    tmp_path, monkeypatch
):
    seed_everything(0)
    models, optimizers = build_bundle(seed=0)
    ckpt = tmp_path / "ckpt"
    v1_counters = {"global_step": 100, "chunks": 12}
    save_checkpoint(ckpt, build_state(models, optimizers, counters=v1_counters))

    # Phase 1: stale partial tmp files left behind by a killed writer are ignored.
    (ckpt / "checkpoint.pt.tmp").write_bytes(b"\x80\x05truncated-partial-blob")
    (ckpt / "manifest.json.tmp").write_text(
        '{"format": 1, "keys": {"mod', encoding="utf-8"
    )
    restored = load_checkpoint(ckpt, expected_keys=EXPECTED_KEYS)
    assert restored["counters"] == v1_counters

    # Phase 2: crash mid-save (os.replace interrupted) -> v1 still intact.
    import sjfpo.checkpointing as ckpt_mod

    def die(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(ckpt_mod.os, "replace", die)
    v2_counters = {"global_step": 200, "chunks": 24}
    with pytest.raises(KeyboardInterrupt):
        save_checkpoint(ckpt, build_state(models, optimizers, counters=v2_counters))
    monkeypatch.undo()

    assert (ckpt / "checkpoint.pt.tmp").exists(), "partial tmp must be left behind"
    restored_v1 = load_checkpoint(ckpt, expected_keys=EXPECTED_KEYS)
    assert restored_v1["counters"] == v1_counters


def test_truncated_blob_raises_resume_error(tmp_path):
    seed_everything(0)
    models, optimizers = build_bundle(seed=0)
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, build_state(models, optimizers))

    blob = ckpt / "checkpoint.pt"
    data = blob.read_bytes()
    blob.write_bytes(data[: len(data) // 2])

    with pytest.raises(ResumeError):
        load_checkpoint(ckpt, expected_keys=EXPECTED_KEYS)
