"""Standalone resume worker for the plan task-9 same-stack kill-resume proof.

Exact harness (plan todo 9b):

    python tests/resume_worker.py <snapshot_path> <out_json>

Builds a fresh trainer-like bundle, loads the checkpoint produced by
``sjfpo.checkpointing.save_checkpoint`` into it (restoring module weights,
optimizer states and every other manifest field), re-seeds the
python/numpy/torch RNGs from the RESTORED bit-states recorded in the
snapshot, replays env steps 50..69 of the shared scripted environment, and
writes the 20 rewards as a JSON list to ``<out_json>``.

This module is ALSO imported by ``tests/test_checkpoint.py`` so that parent
pytest process and child subprocess share ONE definition of the bundle
builder, the scripted environment and the determinism configuration.
"""

from __future__ import annotations

import json
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sjfpo.checkpointing import capture_env_fingerprint, load_checkpoint  # noqa: E402


def configure_determinism() -> None:
    """Same-stack determinism flags; MUST be called by parent AND worker."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_bundle(seed: int = 0):
    """Minimal trainer-like bundle: five modules + two Adam optimizers.

    ``torch.manual_seed(seed)`` pins the parameter init so both processes
    construct identically-shaped (and identically-initialized, pre-restore)
    networks.
    """
    torch.manual_seed(seed)
    models = {
        "encoder": torch.nn.Sequential(
            torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 16)
        ),
        "policy_head": torch.nn.Linear(16, 4),
        "aux_head": torch.nn.Linear(16, 16),
        "critic": torch.nn.Sequential(
            torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)
        ),
        "ema_proj": torch.nn.Linear(16, 16),
    }
    optimizers = {
        "policy_opt": torch.optim.Adam(
            list(models["encoder"].parameters())
            + list(models["policy_head"].parameters())
            + list(models["aux_head"].parameters()),
            lr=1e-3,
        ),
        "critic_opt": torch.optim.Adam(models["critic"].parameters(), lr=1e-3),
    }
    return models, optimizers


def scripted_step(models: dict, optimizers: dict, t: int) -> float:
    """One env step + one train update; consumes ALL THREE global RNG streams.

    The observation is a pure function of ``t``; the reward mixes the policy
    output with python/numpy/torch RNG draws, and each step performs real
    optimizer updates so weights AND optimizer state must survive a resume.
    """
    obs = torch.sin(torch.arange(16, dtype=torch.float32) * 0.1 + t)
    z = models["encoder"](obs)
    action = models["policy_head"](z)
    noise = torch.randn(4)  # torch RNG draw #1
    acted = action + 0.1 * noise
    r_py = random.random()  # python RNG
    r_np = float(np.random.rand())  # numpy RNG
    r_t = float(torch.rand(()))  # torch RNG draw #2
    reward = float(acted.sum()) + 0.01 * r_py + 0.01 * r_np + 0.01 * r_t

    value = models["critic"](z.detach())
    closs = torch.nn.functional.mse_loss(value, torch.tensor([reward]))
    optimizers["critic_opt"].zero_grad()
    closs.backward()
    optimizers["critic_opt"].step()

    surrogate = -(acted.sum() * models["aux_head"](z).sum())
    optimizers["policy_opt"].zero_grad()
    surrogate.backward()
    optimizers["policy_opt"].step()
    return reward


def run_segment(models: dict, optimizers: dict, start: int, n: int) -> list[float]:
    return [scripted_step(models, optimizers, t) for t in range(start, start + n)]


def build_state(
    models: dict, optimizers: dict, *, counters: dict | None = None
) -> dict:
    """Full checkpoint state: every manifest field enumerated in plan 9(a)."""
    return {
        "models": models,
        "optimizers": optimizers,
        "frozen_stats": {"mu": torch.zeros(16), "std": torch.ones(16)},
        "frame_buffer": deque([float(i) for i in range(8)], maxlen=8),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "counters": counters
        if counters is not None
        else {"global_step": 50, "chunks": 6},
        "eval_history_ptr": 1,
        "flags": {"diverged": False},
        "env_fingerprint": capture_env_fingerprint(),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: python tests/resume_worker.py <snapshot_path> <out_json>",
            file=sys.stderr,
        )
        return 2
    snapshot_path, out_json = argv[1], argv[2]

    configure_determinism()
    seed_everything(0)  # baseline only; overwritten by restored bit-states below
    models, optimizers = build_bundle(seed=0)
    template = build_state(models, optimizers)

    # expected_keys must be dotted paths matching the manifest, not top-level
    # dict keys.  Pass the list of dotted paths that the manifest enumerates.
    dotted_keys = [
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
    restored = load_checkpoint(snapshot_path, expected_keys=dotted_keys, state=template)

    # Replay uses the RESTORED RNG bit-states, not the freshly seeded ones.
    random.setstate(restored["rng_python"])
    np.random.set_state(restored["rng_numpy"])
    torch.set_rng_state(restored["rng_torch"])

    rewards = run_segment(
        models, optimizers, start=int(restored["counters"]["global_step"]), n=20
    )
    Path(out_json).write_text(json.dumps(rewards), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
