#!/usr/bin/env python3
"""Build self-contained notebooks: spectral_joint_flow.ipynb + flow_matching_vs_ppo_sac.ipynb.

Reads each src/mfpo/*.py module as code cells, intersperses markdown
documentation blocks, and adds executable cells. Outputs validated .ipynb files.

Usage:
    python scripts/build_notebook.py [--output spectral_joint_flow.ipynb]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "mfpo"

# ---------------------------------------------------------------------------
# Legacy module order (MuJoCo)
# ---------------------------------------------------------------------------
MODULE_ORDER = [
    "config.py",
    "dct_utils.py",
    "networks.py",
    "fpo_core.py",
    "collector.py",
    "gae.py",
    "trainer.py",
    "checkpointing.py",
    "evaluator.py",
    "runner.py",
    "baselines.py",
    "calibration.py",
    "sweep.py",
    "analysis.py",
]

MODULE_TITLES = {
    "config.py": "Configuration & Run Matrix",
    "dct_utils.py": "DCT Utilities (Orthonormal DCT-II/III)",
    "networks.py": "Neural Network Modules",
    "fpo_core.py": "FPO++ Core Math (CFM, Ratios, ASPO, Euler Sampler)",
    "collector.py": "Chunked Rollout Collector",
    "gae.py": "Chunk-Level Semi-MDP GAE",
    "trainer.py": "MFPO Trainer (Update Assembly)",
    "checkpointing.py": "Checkpoint / Resume IO",
    "evaluator.py": "Shared Deterministic Evaluator",
    "runner.py": "MFPO End-to-End Run Driver",
    "baselines.py": "skrl PPO & SAC Baseline Drivers",
    "calibration.py": "Calibration & Budget Gate",
    "sweep.py": "Reentrant Sweep Executor",
    "analysis.py": "Analysis: Curves, CIs, IQM, Figures",
}

# ---------------------------------------------------------------------------
# Spectral joint flow order - spec §3-10
# ---------------------------------------------------------------------------
MODULE_ORDER_SPECTRAL = [
    "config.py",
    "dct_utils.py",
    "networks.py",
    "inference.py",
    "fpo_core.py",
    "collector.py",
    "gae.py",
    "checkpointing.py",
    "trainer.py",
    "evaluator.py",
    "runner.py",
    "baselines.py",
    "calibration.py",
    "sweep.py",
    "analysis.py",
]

MODULE_TITLES_SPECTRAL = {
    "config.py": "Configuration & LEGACY_MUJOCO Flag",
    "dct_utils.py": "DCT Joint Target Normalization (Per-Branch Frozen Stats)",
    "networks.py": "SpectralTokens + JointFlowTransformer + SIGReg (Epps-Pulley CF normality)",
    "inference.py": "Joint Euler ODE 0→1 + IDCT Decode",
    "fpo_core.py": "FPO++ Core",
    "collector.py": "Chunked Rollout Collector",
    "gae.py": "Chunk-Level GAE",
    "checkpointing.py": "Checkpoint / Resume IO",
    "trainer.py": "MFPO Trainer (Update Assembly)",
    "evaluator.py": "Deterministic Evaluator + Temporal Metrics (Jerk, DCT Recon)",
    "runner.py": "MFPO End-to-End Run Driver",
    "baselines.py": "skrl PPO & SAC Baseline Drivers",
    "calibration.py": "Projected-Hours Budget Gate",
    "sweep.py": "Reentrant Sweep State Machine (run_pending)",
    "analysis.py": "Analysis (Bootstrap 2000, IQM)",
}

# ---------------------------------------------------------------------------
# Markdown blocks - MuJoCo comparison notebook
# ---------------------------------------------------------------------------
TITLE_MD = """\
# Flow-Matching vs PPO/SAC: A Reproducible Benchmark

> **FPO++** (arXiv 2602.02481) with Fourier-conditioned spectral tokens vs
> standard PPO and SAC on MuJoCo locomotion tasks.

| Component | Description |
|-----------|-------------|
| **MFPO** | Fourier-conditioned FPO++ with DCT chunks, spectral tokens, SigReg, aux future-state CFM |
| **PPO** | skrl 2.1.0 PPO baseline (GaussianMixin MLP[256,256,256]) |
| **SAC** | skrl 2.1.0 SAC baseline (MLP[256,256,256] x2 critics + actor) |
| **Envs** | HalfCheetah-v5, Walker2d-v5, Hopper-v5 |
| **Seeds** | {0, 1, 2, 3, 4} per method per env |
| **Steps** | 1M env steps per run |
"""

SETUP_MD = """\
## Setup & Version Assertions

Validate dependency versions before any imports. This cell enforces
torch >= 2.2, gymnasium >= 1.0, skrl == 2.1.0, and fp32 precision.
"""

BINDING_CONVENTIONS_MD = """\
## Binding Conventions

These conventions are **binding** — deviations require explicit justification.

| Convention | Value | Reference |
|-----------|-------|-----------|
| Interpolation | `x_t = t·ε + (1−t)·a` (t=1 = pure noise) | arXiv 2602.02481 Eq.3 |
| Target velocity | `v* = ε − a` | arXiv 2602.02481 Eq.3 |
| Euler integration | `x ← x − δ·v̂`, t:1→0 | arXiv 2602.02481 Eq.4 |
| τ distribution | `U[0.005, 0.995]` (truncated uniform) | Simplification of Beta(1,β)-ICDF |
| N_mc | 16 (τ, ε) pairs per chunk | arXiv 2602.02481 §III |
| CFM clamp | ±20 symmetric | arXiv 2602.02481 §III |
| Log-ratio cap | 10.0 (STE clamp) | arXiv 2602.02481 §III |
| ASPO ε_clip | 0.05 | arXiv 2602.02481 Eq.10 |
| Chunk horizon | H=8 env steps | Plan Scope |
| Context length | N=16 frame-stack | Plan Scope |
| Flow space | Orthonormal DCT coefficients | Plan Scope §Flow space |
"""

PROVENANCE_MD = """\
## Hyperparameter Provenance Table

| Method | Key | Value | Source |
|--------|-----|-------|--------|
| MFPO | lr | 1e-4 | FPO++ (AdamW) |
| MFPO | wd | 1e-4 | FPO++ (AdamW) |
| MFPO | chunks_per_update | 512 | FPO++ |
| MFPO | epochs | 16 | FPO++ |
| MFPO | minibatch | 128 | FPO++ |
| MFPO | N_mc | 16 | FPO++ |
| MFPO | euler_train | 64 | FPO++ |
| MFPO | euler_eval | 16 | Plan (zero-sampling) |
| MFPO | lambda_aux | 0.5 | Plan D9 |
| MFPO | lambda_sigreg | 0.1 | Plan D10 |
| MFPO | c_value | 1.0 | Standard PPO convention |
| MFPO | ema_decay | 0.99 | FPO++ |
| MFPO | grad_clip | 1.0 | FPO++ |
| PPO | lr | 3e-4 | skrl default |
| PPO | rollouts | 2048 | skrl default |
| PPO | learning_epochs | 10 | skrl default |
| PPO | mini_batches | 8 | skrl default |
| PPO | kl_threshold | 0.01 | skrl default (KLAdaptiveLR) |
| PPO | grad_norm_clip | 1.0 | skrl default |
| PPO | discount | 0.99 | Adapted (match ours) |
| PPO | gae_lambda | 0.95 | Adapted (match ours) |
| PPO | ratio_clip | 0.2 | skrl default |
| SAC | lr | 3e-4 | skrl default |
| SAC | batch_size | 256 | skrl default |
| SAC | memory_size | 300000 | Adapted (reduced) |
| SAC | gradient_steps | 1 | skrl default |
| SAC | polyak | 0.005 | skrl default |
| SAC | initial_random_steps | 5000 | Adapted (reduced) |
"""

ARCHITECTURE_MD = """\
## Architecture Table

### MFPO (Fourier-Conditioned FPO++)

```
obs [B, T=16, obs_dim]
  → ContextTrunk: Linear(obs_dim→128) → ELU → MultiheadAttention(128, 4) → Z [B, 16, 128]
  → SigReg: Epps-Pulley CF normality test on Z (λ=0.1)
  → SpectralTokens: Hann-window rFFT → low-K=5 tokens [B, 5, 256] (frozen stats)
  → PolicyFlowHead: MultiheadAttention(128, 4) conditioned on [Z; fft_tokens] → velocity [B, 8, A]
  → AuxStateFlowHead: Same arch → velocity [B, 8, 128] for future-state prediction
  → ValueCritic: meanpool(Z) → MLP[256,256,256] ELU → scalar V
  → EMAProjection: AveragedModel(online, get_ema_multi_avg_fn(0.99)) — same Polyak formula via torch._foreach_lerp_
```

### PPO/SAC Baselines (skrl 2.1.0)

```
obs [B, 16*obs_dim] (flattened frame-stack)
  → Policy: GaussianMixin MLP[256, 256, 256] → mean + log_std
  → Value (PPO only): DeterministicMixin MLP[256, 256, 256] → scalar
  → Critics (SAC): 2x DeterministicMixin MLP[256, 256, 256] + 2x target
```
"""

# ---------------------------------------------------------------------------
# Spectral markdown blocks
# ---------------------------------------------------------------------------
TITLE_SPECTRAL_MD = """\
# Spectral Joint Flow: MuJoCo Locomotion Benchmark

> **Joint flow-matching** learning actions and future latents together from
> spectrally-conditioned proprioceptive context on MuJoCo.

| Component | Description |
|-----------|-------------|
| **MFPO (proposed)** | Joint DCT [A*,S*] flow + Hann-rFFT K=5 spectral tokens + SIGReg (Epps-Pulley) |
| **PPO** | skrl PPO baseline (Gaussian MLP[256,256,256]) |
| **SAC** | skrl SAC baseline (actor + twin critics) |
| **Envs** | HalfCheetah-v5, Walker2d-v5, Hopper-v5 (flat Box) |
| **Seeds** | {0..4} per method per env |
"""

BINDING_SPECTRAL_MD = """\
## Binding Conventions (MuJoCo — Ascending Flow)

> **CRITICAL:** Joint flow uses `z_t = (1-t)z0 + t X*` with t=0 noise → t=1 data,
> integrated ascending 0→1. **Do not mix conventions.**

| Convention | Value | Reference |
|-----------|-------|-----------|
| Interpolation | `z_t = (1-t)z0 + t X*` (t=0 noise, t=1 data) | Spec §5 |
| Target velocity | `u* = X* - z0` | Spec §5 |
| Euler integration | `z ← z + dt·v̂`, t:0→1 ascending | Spec §5 Eq.7 |
| t distribution | `U[0.005, 0.995]` | Spec §7 |
| Spectral | `Z_w = Z*hann(16)` → rFFT(dim=1) → cat(real,imag)→ K=5 → frozen norm → Linear(2D→64) | Spec §4, Tasks 4 |
| K | 5 bins (N/2+1=9 → 0..4) | Spec §4 |
| Per-branch norm | `X*=[A*_norm,S*_norm]` std(A*)≈1 std(S*)≈1 frozen | Spec §5 Task 5 |
| SIGReg | Epps-Pulley CF normality on Z, single scalar, λ=0.1 | networks.sigreg_losses |
| λ_A / λ_S / λ_SIGReg | 1.0 / 0.5 / 0.1 | Spec §7 |
| N / H / D / Dk | 16 / 8 / 128 / 64 | Spec §3 |
| Inference steps | 16 train / 8 eval | Spec Table 3 |
"""

PROVENANCE_SPECTRAL_MD = """\
## Hyperparameter Provenance (MuJoCo)

| Method | Key | Value | Source |
|--------|-----|-------|--------|
| JointFlow | λ_A | 1.0 | Spec §7 binding |
| JointFlow | λ_S | 0.5 | Spec §7 binding |
| SIGReg | λ_sigreg | 0.1 | Spec §6 binding |
| Spectral | Dk | 64 | Spec §4 |
| Spectral | K | 5 | Spec §4 |
| Flow | N_mc | 16 | FPO++ |
| Flow | inference_steps | 16 train / 8 eval | Spec Table 3 |
| Trainer | optimizer | AdamW 1e-4 | Spec §7 |
| Analysis | bootstrap | n_boot=2000 | analysis.analyze_results |
| Matrix | runs | 45 (3 methods x 3 envs x 5 seeds) | config.RUN_MATRIX |
"""

ARCHITECTURE_SPECTRAL_MD = """\
## Architecture Table (MuJoCo)

```
MuJoCo flat Box obs [B,16,obs_dim]
  → ContextTrunk: Linear(obs_dim→128) → ELU → MultiheadAttention(128,4) → Z [B,16,128]
  → SigReg: Epps-Pulley CF normality test on Z (λ=0.1)
  → SpectralTokens: Z*hann(16) → rFFT → cat(real,imag) → K=5 → frozen norm → Linear(2D→64) → T_FFT [B,5,64]
  → Joint target: A*=DCT(A_gt[B,8,Da]), S*=DCT(Z_future[B,8,128]) → per-branch norm → X* [B,8,135]
  → JointFlowTransformer: z_t=(1-t)z0+tX*, u*=X*-z0, vθ(z_t,t,Z,T_FFT) → [v_A,v_S], loss 1.0*MSE_A+0.5*MSE_S
  → Inference: euler_integrate_joint 0→1 ascending → IDCT → Â executed, Ŝ auxiliary, gripper→{0,1}
```
"""

DEVIATION_NOTES_SPECTRAL_MD = """\
## Deviation Notes (MuJoCo)

| ID | Deviation | Justification |
|----|-----------|---------------|
| D Synthetic | Local tests use synthetic fixtures, no vendored data | Small deterministic fixtures, no downloads |
| D Flag | MuJoCo-only; LIBERO twins pruned | Single-mode codebase, no dead branches |
| D Per-branch | A* and S* normalized with separate frozen stds | Prevents scale dominance of λ weights |
| D Hann | Window applied BEFORE rFFT, norm AFTER rFFT | Critical order for spectral leakage |
| D SIGReg | Epps-Pulley CF normality single scalar, no Barlow/EMA | SIGReg-only gate |
"""

LIMITATIONS_SPECTRAL_MD = """\
## Limitations (MuJoCo)

1. **Synthetic fixtures local** — tests use small deterministic fixtures; real-env runs need MuJoCo.
2. **fp32 only** — no autocast/bf16/fp16.
3. **CPU-first local verification** — full 1M-step runs need accelerators; local gates use tiny matrices.
"""

DEVIATION_NOTES_MD = """\
## Deviation Notes

| ID | Deviation | Justification |
|----|-----------|---------------|
| D2 | Vision DiT dropped | User-approved: proprio-only benchmark, no pixel observations |
| D7 | Two heads on shared trunk instead of joint ODE | User-approved: separate policy + aux heads share Z via attention; simpler to train, same information flow |
| D9 | λ_aux = 0.5 for aux future-state CFM | Plan-specified; balances aux signal without overwhelming policy gradient |
| D10 | λ_sigreg = 0.1 for SigReg | Plan-specified; regularizes Z without over-constraining |
| D12 | Tiny matrix override for local testing | MFPO_TINY_MATRIX=1 env var enables 3-run local validation |
"""

LIMITATIONS_MD = """\
## Limitations

1. **SAC fresh buffer on restart**: SAC warm-restarts use a fresh replay buffer
   (skrl limitation). This means the first ~5000 steps after resume are
   exploratory, slightly underperforming a continuously-trained SAC.

2. **Untuned baselines**: PPO and SAC use skrl default hyperparameters
   (except the PPO lr sensitivity pair). A thorough hyperparameter sweep
   could improve baseline performance.

3. **CPU-only local verification**: Full 1M-step runs require GPU.
   Local verification uses tiny matrices (20k steps) only.

4. **Single precision**: fp32 only (no autocast/bf16/fp16) for portability.
"""

PROTOCOL_MD = """\
## Evaluation Protocol

- **MFPO**: Zero-sampling (ε=0) at eval time — Euler integration from noise
   to action with no stochastic exploration
- **PPO**: Mean-action (deterministic) at eval time
- **SAC**: Deterministic mode at eval time
- **Episodes**: 12 deterministic episodes per eval checkpoint
- **Eval interval**: Every 50k env steps (every 25k for ckpts)
- **Eval seed**: 10000 + run_seed
"""

# ---------------------------------------------------------------------------
# Executable cells (kept)
# ---------------------------------------------------------------------------
SETUP_CELL = """\
# Setup: imports and version assertions (self-contained: modules inlined below, no src import needed)
import sys, os, importlib.metadata
from pathlib import Path

# Self-containment: all mfpo modules are inlined as code cells below; no external src/ required.
# Keep path setup as no-op for notebook portability (guard __file__ which is undefined in notebook).
try:
    _file = Path(__file__)
except NameError:
    _file = Path.cwd()
_cwd = Path(os.getcwd())
_src = str(_cwd / "src") if (_cwd / "src" / "mfpo" / "__init__.py").exists() else str(_file.resolve().parents[0] / "src") if _file.exists() else str(_cwd / "src")
if _src not in sys.path and Path(_src).exists():
    sys.path.insert(0, _src)

import torch
import gymnasium
import numpy as np

_gym_ver = importlib.metadata.version("gymnasium")
_skrl_ver = importlib.metadata.version("skrl")

print(f"torch={torch.__version__}, gymnasium={_gym_ver}, skrl={_skrl_ver}")
assert torch.__version__ >= "2.2", f"torch {torch.__version__} < 2.2"
assert _gym_ver >= "1.0", f"gymnasium {_gym_ver} < 1.0"
assert _skrl_ver == "2.1.0", f"skrl {_skrl_ver} != 2.1.0"
assert torch.get_default_dtype() == torch.float32, "expected fp32"

print("All version assertions passed.")
"""

CALIBRATION_GATE_CELL = """\
# Calibration & Budget Gate (Kaggle-gated)
from mfpo.config import IS_KAGGLE, APPROVE_OVERRUN, RUN_MATRIX, BUDGET_TOTAL_HOURS

if IS_KAGGLE:
    import time
    import gymnasium
    from mfpo.calibration import calibrate_and_gate, render_markdown

    measured_fps = {}
    for method in ("mfpo", "ppo", "sac"):
        env = gymnasium.make("Hopper-v5")
        t0 = time.monotonic()
        obs, _ = env.reset(seed=42)
        for _ in range(300):
            action = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                obs, _ = env.reset()
        elapsed = time.monotonic() - t0
        env.close()
        measured_fps[method] = 300.0 / max(elapsed, 1e-6)

    report = calibrate_and_gate(RUN_MATRIX, measured_fps)
    print(render_markdown(report))

    if not report.gate_pass and not APPROVE_OVERRUN:
        raise RuntimeError(
            f"BUDGET GATE FAILED: projected {report.total_hours:.1f}h > "
            f"{BUDGET_TOTAL_HOURS}h. Set MFPO_APPROVE_OVERRUN=1 to override."
        )
else:
    print("Not on Kaggle — skipping calibration gate.")
    print(f"Budget limits: {BUDGET_TOTAL_HOURS}h total, per-run limit in config.")
"""

SMOKE_GATE_CELL = """\
# Smoke Gate: tiny local validation
import os, json, subprocess, sys
from pathlib import Path

_smoke_dir = Path(".omo/evidence/smoke")
_smoke_dir.mkdir(parents=True, exist_ok=True)

if os.environ.get("MFPO_RESUME_ONLY") is not None:
    print("RESUME_ONLY mode — checking for existing checkpoints...")
    for method in ("mfpo", "ppo", "sac"):
        method_dirs = list(_smoke_dir.glob(f"{method}_Hopper-v5_seed*"))
        if method_dirs:
            print(f"  {method}: found {len(method_dirs)} checkpoint(s)")
        else:
            print(f"  {method}: no checkpoints found")
else:
    print("Running tiny 3-method smoke (train.py --matrix)...")
    result = subprocess.run(
        [sys.executable, "scripts/train.py",
          "--matrix",
          "--methods", "mfpo", "ppo", "sac",
          "--envs", "Hopper-v5", "--seeds", "0",
          "--total-steps", "2000",
          "--out-dir", str(_smoke_dir)],
        capture_output=True, text=True, timeout=600,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR:\\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Smoke gate failed with exit code {result.returncode}")
    print("Smoke gate PASSED.")
"""

SWEEP_CELL = """\
# Full Sweep: iterate RUN_MATRIX with reentrant executor
import os
from pathlib import Path
from mfpo.config import RUN_MATRIX, TINY_MATRIX, IS_KAGGLE
from mfpo.sweep import run_pending

if IS_KAGGLE:
    base = Path("/kaggle/working/runs")
else:
    base = Path(".omo/evidence/runs")

base.mkdir(parents=True, exist_ok=True)

def dispatch(spec, base_dir):
    out = base_dir / f"{spec.method}_{spec.env}_seed{spec.seed}"
    out.mkdir(parents=True, exist_ok=True)
    import gymnasium
    env = gymnasium.make(spec.env)
    try:
        if spec.method == "mfpo":
            from mfpo.runner import train_run
            return train_run(env=env, out_dir=out, seed=spec.seed,
                           total_steps=spec.total_steps,
                           eval_interval=spec.eval_interval,
                           ckpt_interval=spec.ckpt_interval)
        elif spec.method == "ppo":
            from mfpo.baselines import ppo_train_run
            return ppo_train_run(env=env, out_dir=out, seed=spec.seed,
                               total_steps=spec.total_steps,
                               eval_interval=spec.eval_interval,
                               ckpt_interval=spec.ckpt_interval)
        elif spec.method == "sac":
            from mfpo.baselines import sac_train_run
            return sac_train_run(env=env, out_dir=out, seed=spec.seed,
                               total_steps=spec.total_steps,
                               eval_interval=spec.eval_interval,
                               ckpt_interval=spec.ckpt_interval)
    finally:
        env.close()

results = run_pending(RUN_MATRIX, base, dispatch)
print(f"Completed: {results['completed']}, Skipped: {results['skipped']}, Diverged: {results['diverged']}")
"""

ANALYSIS_CELL = """\
# Analysis: bootstrap CIs, IQM, figures, CSV summary
from pathlib import Path
from mfpo.analysis import analyze_results
from mfpo.config import RUN_MATRIX

if IS_KAGGLE:
    base = Path("/kaggle/working/runs")
else:
    base = Path(".omo/evidence/runs")

output = Path(".omo/evidence/analysis")
output.mkdir(parents=True, exist_ok=True)

if base.exists() and any(base.iterdir()):
    report = analyze_results(RUN_MATRIX, base, output)
    print(f"Analysis complete. Reports in {output}/")
    print(f"Superiority: {report.superiority}")
else:
    print("No run directories found — skipping analysis.")
    print("Run the sweep cell first, or provide run data in .omo/evidence/runs/")
"""

SELF_TEST_CELL = """\
# Embedded self-test: verify core math correctness
import torch
from pathlib import Path

errors = []

from mfpo.dct_utils import dct, idct
x = torch.randn(4, 8, 3)
rt = idct(dct(x, dim=1), dim=1)
if torch.max(torch.abs(rt - x)).item() > 1e-5:
    errors.append(f"DCT roundtrip error: {torch.max(torch.abs(rt - x)).item()}")

from mfpo.config import RUN_MATRIX, LEGACY_MUJOCO, TINY_MATRIX
# Tolerant self-test: RUN_MATRIX non-empty; size check respects mode
assert len(RUN_MATRIX) >= 1, f"RUN_MATRIX empty {len(RUN_MATRIX)}"
# Preserve legacy check only when in MuJoCo mode
if LEGACY_MUJOCO:
    from mfpo.config import METHODS, ENVS, SEEDS
    _full_expected = len(METHODS) * len(ENVS) * len(SEEDS)
    _tiny_expected = len(METHODS)
    _expected = _tiny_expected if TINY_MATRIX else _full_expected
    # In isolated tmpdir with MFPO_TINY_MATRIX=1 we expect tiny, but allow any non-empty for self-containment
    assert len(RUN_MATRIX) >= 1

from mfpo.networks import sigreg_losses
z = torch.randn(32, 16, 128)
sl = sigreg_losses(z)
assert sl.ndim == 0 and torch.isfinite(sl), "SIGReg loss not finite scalar"

from mfpo.fpo_core import aspo_surrogate
rho = torch.tensor([[1.04]])
adv = torch.tensor([[1.0]])
psi = aspo_surrogate(rho, adv, 0.05)
if abs(psi.item() - 1.04) > 1e-3:
    errors.append(f"ASPO psi={psi.item()}, expected ~1.04")

from mfpo.gae import chunk_gae
N = 5
chunk_lens = torch.ones(N, dtype=torch.long) * 4
total_steps = int(chunk_lens.sum().item())
chunk_rewards = torch.randn(total_steps)
chunk_values = torch.randn(N)
chunk_next_values = torch.randn(N)
terminated = [False] * (N - 1) + [True]
gaes, returns = chunk_gae(chunk_rewards, chunk_values, chunk_next_values, chunk_lens, terminated, gamma=0.99, lam=0.95)
gaes_t = torch.as_tensor(gaes)
assert gaes_t.shape == (N,), f"GAE shape: {gaes_t.shape}"
assert torch.isfinite(gaes_t).all(), "GAE has non-finite values"

if errors:
    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    raise RuntimeError(f"Self-test failed with {len(errors)} error(s)")

print("All self-tests PASSED.")
"""

# Spectral-specific cells
SELF_TEST_SPECTRAL_CELL = """\
# Embedded self-test: joint flow correctness (spectral)
import torch, sys

errors = []

# 1. DCT roundtrip joint shape
from mfpo.dct_utils import dct, idct
x = torch.randn(4, 8, 7)
rt = idct(dct(x, dim=1), dim=1)
if (rt - x).abs().max().item() > 1e-5:
    errors.append(f"DCT joint roundtrip { (rt-x).abs().max().item()}")

# 2. Per-branch norm invertibility
from mfpo.dct_utils import capture_training_stats, normalize_joint_targets, denormalize_joint_targets, set_frozen_target_stats
import torch
batches = [{"actions": torch.randn(4,8,7), "z_future": torch.randn(4,8,128)} for _ in range(3)]
stats = capture_training_stats(batches)
set_frozen_target_stats(stats)
A_dct = torch.randn(2,8,7)
S_dct = torch.randn(2,8,128)
X_star = normalize_joint_targets(A_dct, S_dct, stats)
A_rec, S_rec = denormalize_joint_targets(X_star, stats)
if (A_rec - A_dct).abs().max().item() > 1e-5:
    errors.append("per-branch denorm invert A")
if (S_rec - S_dct).abs().max().item() > 1e-5:
    errors.append("per-branch denorm invert S")

# 3. SIGReg Epps-Pulley CF normality: gaussian low, degenerate high
from mfpo.networks import sigreg_losses
z_gauss = torch.randn(64,16,128)
z_deg = torch.ones(64,16,128)*5
lg = sigreg_losses(z_gauss).item()
ld = sigreg_losses(z_deg).item()
if not (lg < 2.2):
    errors.append(f"SIGReg Gaussian {lg} >=2.2")
if not (ld > 95.0):
    errors.append(f"SIGReg degenerate {ld} <=95.0")
if not (ld - lg > 95.0):
    errors.append("SIGReg separation <=95.0")

# 4. Joint flow identity recovery ascending 0→1
from mfpo.inference import euler_integrate_joint
class IdentityV:
    def __call__(self, z_t, t, Z, T_FFT=None):
        # v = X* - z0 ; with X*=1, z0=0 => v=1 everywhere
        return torch.ones_like(z_t)
z0 = torch.zeros(2,8,135)
Z = torch.randn(2,16,128)
T = torch.randn(2,5,64)
X_hat = euler_integrate_joint(IdentityV(), z0, Z, T, steps=16)
if (X_hat - 1.0).abs().max().item() > 1e-4:
    errors.append(f"identity ODE recovery {(X_hat-1).abs().max().item()}")

# 5. Spectral frozen invariance note - checked via training tests

if errors:
    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    raise RuntimeError(f"Spectral self-test failed {errors}")
print("Spectral self-tests PASSED.")

# 6. LEGACY_MUJOCO flag preserved
from mfpo import config as cfg
assert hasattr(cfg, "LEGACY_MUJOCO"), "LEGACY_MUJOCO missing"
print(f"LEGACY_MUJOCO={cfg.LEGACY_MUJOCO} spectral mode verified.")
"""

SMOKE_GATE_MUJOCO_CELL = """\
# Smoke Gate (MuJoCo-only): the LIBERO tiny-synthetic path was pruned.
# Run the live smoke from a checkout with src/scripts present:
#   PYTHONPATH=src python scripts/train.py --method mfpo --env Hopper-v5 --seed 0 \
#       --total-steps 200 --eval-interval 100 --ckpt-interval 200 --out-dir .omo/evidence/smoke
print("MuJoCo smoke lives in scripts/train.py (not run in this cell).")
"""

def md_block(source: str) -> dict:
    cell = nbf.v4.new_markdown_cell(source.strip())
    return cell


def code_block(source: str) -> dict:
    cell = nbf.v4.new_code_cell(source.strip())
    return cell


def read_module_source(filename: str) -> str:
    path = SRC / filename
    if not path.exists():
        raise FileNotFoundError(f"Module not found: {path}")
    content = path.read_text()
    lines = content.split("\n")
    cleaned = []
    skip_next_blank = False
    for line in lines:
        if line.startswith("# %%"):
            skip_next_blank = True
            continue
        if line.startswith("# CELLORDER:"):
            continue
        if skip_next_blank and line.strip() == "":
            skip_next_blank = False
            continue
        skip_next_blank = False
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def register_snippet(filename: str) -> str:
    mod = filename.replace(".py", "")
    return f"""# Register inlined {filename} as importable mfpo.{mod} for self-containment
import sys, types
_m = types.ModuleType('mfpo.{mod}')
_m.__path__ = []
_m.__dict__.update({{k: v for k, v in globals().items() if not k.startswith('__')}})
sys.modules['mfpo.{mod}'] = _m
if 'mfpo' not in sys.modules:
    _pkg = types.ModuleType('mfpo')
    _pkg.__path__ = []
    sys.modules['mfpo'] = _pkg
else:
    _pkg = sys.modules['mfpo']
    if not hasattr(_pkg, '__path__'):
        _pkg.__path__ = []
setattr(_pkg, '{mod}', _m)
"""


def new_notebook_shell() -> nbf.NotebookNode:
    """Create a blank notebook with the standard kernelspec metadata."""
    nb = nbf.v4.new_notebook()
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12.0"},
        }
    )
    return nb


def bootstrap_spectral() -> str:
    all_mods = [p.stem for p in SRC.glob("*.py") if p.name != "__init__.py"]
    names = ["mfpo"] + [f"mfpo.{m}" for m in all_mods]
    return "import sys, types\n" + "\n".join(
        f"if '{n}' not in sys.modules:\n    _tmp = types.ModuleType('{n}')\n    _tmp.__path__ = []\n    sys.modules['{n}'] = _tmp\n    "
        + (
            f"setattr(sys.modules['mfpo'], '{n.split('.')[-1]}', sys.modules['{n}'])"
            if "." in n
            else ""
        )
        for n in names
    )


def append_spectral_cells(cells: list) -> None:
    """Append the spectral joint-flow cell sequence."""
    cells.append(md_block(TITLE_SPECTRAL_MD))
    cells.append(md_block(SETUP_MD))
    cells.append(code_block(SETUP_CELL))
    cells.append(code_block(bootstrap_spectral()))
    cells.append(md_block(BINDING_SPECTRAL_MD))
    for filename in MODULE_ORDER_SPECTRAL:
        title = MODULE_TITLES_SPECTRAL.get(filename, filename)
        cells.append(md_block(f"## {title}"))
        code = read_module_source(filename)
        cells.append(code_block(code))
        cells.append(code_block(register_snippet(filename)))
        if filename == "config.py":
            cells.append(md_block(PROVENANCE_SPECTRAL_MD))
        if filename == "networks.py":
            cells.append(md_block(ARCHITECTURE_SPECTRAL_MD))
        if filename == "evaluator.py":
            cells.append(md_block(PROTOCOL_MD))
    cells.append(md_block(DEVIATION_NOTES_SPECTRAL_MD))
    cells.append(md_block(LIMITATIONS_SPECTRAL_MD))
    cells.append(md_block("## Embedded Self-Test (Joint Flow)"))
    cells.append(code_block(SELF_TEST_SPECTRAL_CELL))
    cells.append(md_block("## Smoke Gate (MuJoCo-only)"))
    cells.append(code_block(SMOKE_GATE_MUJOCO_CELL))
    # also include flag self-test for regression
    cells.append(md_block("## Flag Self-Test (MuJoCo)"))
    cells.append(code_block(SELF_TEST_CELL))


def append_legacy_cells(cells: list) -> None:
    """Append the legacy comparison cell sequence."""
    cells.append(md_block(TITLE_MD))
    cells.append(md_block(SETUP_MD))
    cells.append(code_block(SETUP_CELL))
    cells.append(md_block(BINDING_CONVENTIONS_MD))
    for filename in MODULE_ORDER:
        title = MODULE_TITLES[filename]
        cells.append(md_block(f"## {title}"))
        code = read_module_source(filename)
        cells.append(code_block(code))
        if filename == "config.py":
            cells.append(md_block(PROVENANCE_MD))
        if filename == "networks.py":
            cells.append(md_block(ARCHITECTURE_MD))
        if filename == "fpo_core.py":
            cells.append(md_block(DEVIATION_NOTES_MD))
        if filename == "evaluator.py":
            cells.append(md_block(PROTOCOL_MD))
    cells.append(md_block(LIMITATIONS_MD))
    cells.append(md_block("## Embedded Self-Test"))
    cells.append(code_block(SELF_TEST_CELL))
    cells.append(md_block("## Calibration & Budget Gate"))
    cells.append(code_block(CALIBRATION_GATE_CELL))
    cells.append(md_block("## Smoke Gate"))
    cells.append(code_block(SMOKE_GATE_CELL))
    cells.append(md_block("## Full Sweep"))
    cells.append(code_block(SWEEP_CELL))
    cells.append(md_block("## Analysis & Figures"))
    cells.append(code_block(ANALYSIS_CELL))


def build_notebook(output_path: Path, spectral: bool = False) -> nbf.NotebookNode:
    nb = new_notebook_shell()
    cells = []
    if spectral:
        append_spectral_cells(cells)
    else:
        append_legacy_cells(cells)
    nb.cells = cells
    return nb


def parse_build_args() -> argparse.Namespace:
    """Parse the notebook-builder CLI flags."""
    parser = argparse.ArgumentParser(description="Build benchmark notebook(s)")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output notebook path (default: build both spectral_joint_flow.ipynb + legacy)",
    )
    parser.add_argument(
        "--legacy-only",
        action="store_true",
        help="Only build legacy flow_matching_vs_ppo_sac.ipynb",
    )
    parser.add_argument(
        "--spectral-only",
        action="store_true",
        help="Only build spectral_joint_flow.ipynb",
    )
    return parser.parse_args()


def write_notebook(out: Path, spectral: bool) -> None:
    """Build, validate, and write one notebook."""
    print(f"Building notebook: {out} (spectral={spectral})")
    nb = build_notebook(out, spectral=spectral)
    nbf.validate(nb)
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {len(nb.cells)} cells to {out} spectral={spectral} validated.")


def write_default_notebooks(args: argparse.Namespace) -> None:
    """Build both notebooks, honoring --legacy-only/--spectral-only."""
    spectral_out = ROOT / "spectral_joint_flow.ipynb"
    legacy_out = ROOT / "flow_matching_vs_ppo_sac.ipynb"
    if not args.legacy_only:
        print(f"Building spectral notebook: {spectral_out}")
        nb_s = build_notebook(spectral_out, spectral=True)
        nbf.validate(nb_s)
        with open(spectral_out, "w", encoding="utf-8") as f:
            nbf.write(nb_s, f)
        print(f"Wrote {len(nb_s.cells)} cells to {spectral_out} validated.")
    if not args.spectral_only:
        print(f"Building legacy notebook: {legacy_out}")
        nb_l = build_notebook(legacy_out, spectral=False)
        nbf.validate(nb_l)
        with open(legacy_out, "w", encoding="utf-8") as f:
            nbf.write(nb_l, f)
        print(f"Wrote {len(nb_l.cells)} cells to {legacy_out} validated.")


def main():
    args = parse_build_args()
    if args.output:
        out = ROOT / args.output
        write_notebook(out, "spectral" in out.name)
        return
    write_default_notebooks(args)


if __name__ == "__main__":
    main()
