#!/usr/bin/env python3
"""Print the metrics that only exist on a CUDA device, per update.

scripts/train.py's summary JSON carries only the final reward, and the diagnostics
live in the W&B info dict, so a CUDA run needs wandb offline mode plus a wrapper
on wandb.log to read them out. Nothing in the repo is modified.

Environment:
  ENV     gymnasium id                      (default Hopper-v5)
  SEED    int                               (default 3407)
  DCT     true|false                        (default true)
  STEPS   int                               (default 32768)
  KEPT    num_kept_tokens when DCT is true  (default 16)
  OUT_DIR path                              (default /tmp/cuda_probe)
  EXTRA   extra --sjfpo.* flags, space separated

Every metric below is a per-update training metric. explained_var and
advantage_std_fraction are the two that decide whether the critic is wired up;
see BUGS.md.
"""

import os
import runpy
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
os.environ["WANDB_MODE"] = "offline"

import torch  # noqa: E402
import wandb  # noqa: E402

from sjfpo.utils import is_rtx_1000_pro_blackwell, resolve_storage_dtype  # noqa: E402

KEEP = [
    # critic health: the primary signal
    "explained_var",
    "advantage_std_fraction",
    "value_loss",
    "value_target_std",
    "value_pred_std",
    "returns_rms",
    # gradient health: B4 has no guard on the gradients
    "actor_grad_norm_preclip",
    "critic_grad_norm_preclip",
    # policy health
    "policy_loss",
    "clipfrac",
    "flow_ratio_mean",
    "cfm_above_clamp_frac",
    # rollout shape
    "truncated_chunk_frac",
    "budget_truncated_chunk_frac",
    "gae_linked_frac",
    "action_magnitude",
]

_orig_log = None


def _install(data, step=None, **kw):
    if any(k.startswith("train/") for k in data):
        parts = []
        for k in KEEP:
            v = data.get("train/" + k)
            parts.append(f"{k}={'--' if v is None else format(v, '.5g')}")
        print(f"[{step:>7}] " + " ".join(parts), flush=True)
    if any(k.startswith("eval/") for k in data):
        parts = [
            f"{k}={format(v, '.1f')}"
            for k, v in data.items()
            if k in ("eval/mean_return", "eval/best_mean_return")
        ]
        if parts:
            print(f"[{step:>7}] EVAL " + " ".join(parts), flush=True)
    return _orig_log(data, step=step, **kw)


_orig_init = wandb.init


def _init(*a, **kw):
    # wandb.init rebinds wandb.log, so the wrapper has to go on afterwards.
    global _orig_log
    r = _orig_init(*a, **kw)
    _orig_log = wandb.log
    wandb.log = _install
    return r


wandb.init = _init


def _preflight() -> None:
    print("=" * 72)
    print(f"torch           {torch.__version__}")
    print(f"cuda available  {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("device count    0")
        print("NO CUDA DEVICE. This probe is only meaningful on a GPU box.")
        print("=" * 72)
        return
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"device {i}         {p.name}  sm_{p.major}{p.minor}  {p.total_memory / 2**30:.1f} GiB")
    print(f"is_rtx_1000_pro_blackwell  {is_rtx_1000_pro_blackwell()}")
    print(f"resolve_storage_dtype      {resolve_storage_dtype('cuda')}")
    if resolve_storage_dtype("cuda") is not torch.bfloat16:
        print("  ^ fp32 storage, so BUGS.md E1 and B1 do not apply to this run.")
    print("=" * 72, flush=True)


def main() -> None:
    _preflight()
    env = os.environ.get("ENV", "Hopper-v5")
    dct = os.environ.get("DCT", "true")
    extra = os.environ.get("EXTRA", "").split()
    args = [
        "train.py",
        "--method", "sjfpo",
        "--env", env,
        "--seed", os.environ.get("SEED", "3407"),
        "--total_steps", os.environ.get("STEPS", "32768"),
        "--eval_steps", os.environ.get("STEPS", "32768"),
        "--save_steps", os.environ.get("STEPS", "32768"),
        "--wandb_enabled", "true",
        "--out_dir", os.environ.get("OUT_DIR", "/tmp/cuda_probe"),
        "--sjfpo.chunks_per_update", "256",
        "--sjfpo.num_envs", "8",
        "--sjfpo.batch_size", "64",
        "--sjfpo.num_learning_epochs", "4",
        "--sjfpo.use_dct_transform", dct,
    ]
    if dct == "true":
        args += ["--sjfpo.num_kept_tokens", os.environ.get("KEPT", "16")]
    args += extra
    print(f"args: {' '.join(args)}\n", flush=True)
    sys.argv = args
    try:
        runpy.run_path("scripts/train.py", run_name="__main__")
    finally:
        if torch.cuda.is_available():
            print(f"\npeak VRAM reserved  {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")
            print(f"peak VRAM allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    sys.exit(main())
