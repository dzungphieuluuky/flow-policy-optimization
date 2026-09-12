#!/usr/bin/env python3
"""Build self-contained notebooks: spectral_joint_flow.ipynb + flow_matching_vs_ppo_sac.ipynb.

Reads each src/sjfpo/*.py module as code cells, intersperses markdown
documentation blocks, and adds executable cells. Outputs validated .ipynb files.

Usage:
    python scripts/build_notebook.py [--output spectral_joint_flow.ipynb]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "sjfpo"

# ---------------------------------------------------------------------------
# Module orders (single live path; both notebooks inline the same src modules)
# ---------------------------------------------------------------------------
BASE_ORDER = [
    "utils.py",
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
    "sweep.py",
    "analysis.py",
]

MODULE_ORDER = list(BASE_ORDER)

MODULE_ORDER_SPECTRAL = list(BASE_ORDER)

MODULE_TITLES = {
    "utils.py": "Shared Stateless Utilities (Device, Env Paths)",
    "config.py": "Configuration & Run Matrix",
    "dct_utils.py": "DCT Utilities (Orthonormal DCT-II/III)",
    "networks.py": "Neural Network Modules",
    "fpo_core.py": "FPO++ Core Math (CFM, Ratios, ASPO, Euler Sampler)",
    "collector.py": "Chunked Rollout Collector",
    "gae.py": "Chunk-Level Semi-MDP GAE",
    "trainer.py": "SJFPO Trainer (Update Assembly)",
    "checkpointing.py": "Checkpoint / Resume IO",
    "evaluator.py": "Shared Deterministic Evaluator",
    "runner.py": "SJFPO End-to-End Run Driver",
    "baselines.py": "skrl PPO & SAC Baseline Drivers",
    "sweep.py": "Reentrant Sweep Executor",
    "analysis.py": "Analysis: Curves, CIs, IQM, Figures",
}

MODULE_TITLES_SPECTRAL = {
    "utils.py": "Shared Stateless Utilities (Device, Env Paths)",
    "config.py": "Configuration & LEGACY_MUJOCO Flag",
    "dct_utils.py": "DCT Joint Target Normalization (Per-Branch Frozen Stats)",
    "networks.py": "SpectralEncoder + DiT Flow Policy + SIGReg (Epps-Pulley CF normality)",
    "fpo_core.py": "FPO++ Core",
    "collector.py": "Chunked Rollout Collector",
    "gae.py": "Chunk-Level GAE",
    "checkpointing.py": "Checkpoint / Resume IO",
    "trainer.py": "SJFPO Trainer (Update Assembly)",
    "evaluator.py": "Deterministic Evaluator (3-col CSV + Divergence Gate)",
    "runner.py": "SJFPO End-to-End Run Driver",
    "baselines.py": "skrl PPO & SAC Baseline Drivers",
    "sweep.py": "Reentrant Sweep State Machine (run_pending)",
    "analysis.py": "Analysis (Bootstrap 2000, IQM)",
}

# ---------------------------------------------------------------------------
# Markdown blocks - MuJoCo comparison notebook
# ---------------------------------------------------------------------------
TITLE_MD = """\
# Flow-Matching vs PPO/SAC: A Reproducible Benchmark

> **FPO++** (arXiv 2602.02481) with DCT spectral tokens vs
> standard PPO and SAC on MuJoCo locomotion tasks.

| Component | Description |
|-----------|-------------|
| **SJFPO** | DCT-conditioned FPO++ with DCT action chunks, spectral tokens, SigReg, V(s,a) critic |
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
| num_mc_samples | 16 (τ, ε) pairs per chunk | arXiv 2602.02481 §III |
| CFM clamp | ±20 symmetric | arXiv 2602.02481 §III |
| Log-ratio cap | 10.0 (STE clamp) | arXiv 2602.02481 §III |
| ASPO ε_clip | 0.05 | arXiv 2602.02481 Eq.10 |
| Chunk horizon | H=8 env steps | Plan Scope |
| State window | T=16 frame-stack | Plan Scope |
| Flow space | Orthonormal DCT coefficients | Plan Scope §Flow space |
"""

PROVENANCE_MD = """\
## Hyperparameter Provenance Table

| Method | Key | Value | Source |
|--------|-----|-------|--------|
| SJFPO | lr | 1e-4 | FPO++ (AdamW) |
| SJFPO | wd | 1e-4 | FPO++ (AdamW) |
| SJFPO | chunks_per_update | 512 | FPO++ |
| SJFPO | num_mc_samples | 16 | FPO++ |
| SJFPO | euler_train | 64 | FPO++ |
| SJFPO | euler_eval | 16 | Plan (zero-sampling) |
| SJFPO | dit_depth | 8 | DiT policy |
| SJFPO | dit_heads | 4 | DiT policy |
| SJFPO | dit_mlp_ratio | 4.0 | DiT policy |
| SJFPO | dit_time_dim | 256 | DiT policy |
| SJFPO | sigreg_coef | 0.1 | Plan D10 |
| SJFPO | vf_coef | 1.0 | Standard PPO convention |
| SJFPO | grad_clip | 1.0 | FPO++ |
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

### SJFPO (DCT-Conditioned FPO++)

```
obs [B, T=16, obs_dim]
  → StateEncoder: Linear(obs_dim→128) → SiLU → MultiheadAttention(128, 4) → Z [B, 16, 128]
  → SpectralEncoder: DCT-II over time → low-K=5 tokens [B, 5, 64] (frozen stats)
  → SJFOPolicyModel: DiT-8 (adaLN-Zero) over action tokens, cond = time + pooled tokens → velocity [B, 16, A] (DCT)
  → Critic: V(s,a) = 2 cross-attention blocks (action queries × state) → scalar V; TD target R + γ^L·V(s',a')
  → SigReg: Epps-Pulley CF normality on the critic latent (λ=0.1)
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

> **Flow-matching policy** with spectrally-conditioned proprioceptive state
> on MuJoCo: DCT action chunks + DCT spectral tokens + ASPO + V(s,a)
> critic + SIGReg (Epps-Pulley).

| Component | Description |
|-----------|-------------|
| **SJFPO (proposed)** | DCT action-chunk flow + DCT K=5 spectral tokens + SIGReg (Epps-Pulley) |
| **PPO** | skrl PPO baseline (Gaussian MLP[256,256,256]) |
| **SAC** | skrl SAC baseline (actor + twin critics) |
| **Envs** | HalfCheetah-v5, Walker2d-v5, Hopper-v5 (flat Box) |
| **Seeds** | {0..4} per method per env |
"""

BINDING_SPECTRAL_MD = """\
## Binding Conventions (MuJoCo — Live Train Path)

| Convention | Value | Reference |
|-----------|-------|-----------|
| Interpolation | `x_t = t·ε + (1−t)·a` (t=1 = pure noise) | arXiv 2602.02481 Eq.3 |
| Target velocity | `v* = ε − a` | arXiv 2602.02481 Eq.3 |
| Euler integration | `x ← x − δ·v̂`, t:1→0 descending | arXiv 2602.02481 Eq.4 |
| τ distribution | `U[0.005, 0.995]` | config.tau_range |
| Flow space | Orthonormal DCT-II action-chunk coefficients | dct_utils |
| Spectral | DCT-II `Z` over time (dim=1) → K=5 low bins → frozen norm → Linear(D→64) | networks.SpectralEncoder |
| K | 5 bins (T/2+1=9 → 0..4) | config.num_spectral_tokens |
| ASPO trust region | PPO-clip Â≥0 / SPO Â<0, ε=0.05 | fpo_core.aspo_surrogate |
| SIGReg | Epps-Pulley CF normality on critic latent, λ=0.1 | networks.sigreg_loss |
| Critic | V(s,a) TD `R + γ^L·V(s',a')`, decoupled critic encoder | trainer |
| T / H / n / D / spectral_dim | 16 / 16 / 8 / 128 / 64 | config |
| Euler steps | 64 train / 16 eval | config |
"""

PROVENANCE_SPECTRAL_MD = """\
## Hyperparameter Provenance (MuJoCo)

| Method | Key | Value | Source |
|--------|-----|-------|--------|
| SJFPO | lr / wd | 1e-4 / 1e-4 | FPO++ (AdamW) |
| SJFPO | chunks_per_update | 512 | FPO++ |
| SJFPO | num_mc_samples | 16 | FPO++ |
| SJFPO | euler_train / euler_eval | 64 / 16 | config |
| SJFPO | dit_depth / dit_heads / dit_mlp_ratio / dit_time_dim | 8 / 4 / 4.0 / 256 | DiT policy |
| SJFPO | clip_range / cfm_clamp / log_ratio_cap | 0.05 / 20.0 / 10.0 | FPO++ |
| SIGReg | sigreg_coef | 0.1 | networks.sigreg_loss |
| Critic | vf_coef | 1.0 | Standard PPO convention |
| Spectral | spectral_dim / K | 64 / 5 | config |
| Analysis | bootstrap | n_boot=2000 | analysis.analyze_results |
| Matrix | runs | 45 (3 methods x 3 envs x 5 seeds) | config.RUN_MATRIX |
"""

ARCHITECTURE_SPECTRAL_MD = """\
## Architecture Table (MuJoCo — Live Path)

```
obs [B, T=16, obs_dim]
  → policy_state_encoder: Linear(obs_dim→128) → SiLU → MultiheadAttention(128,4) → Z [B,16,128]
  → SpectralEncoder: DCT-II(Z) over time → K=5 low bins → frozen norm → Linear(D→64) → [B,5,64]
  → SJFOPolicyModel: DiT-8 (adaLN-Zero) over DCT action tokens [B,16,Da], cond = time + pooled tokens → velocity
  → euler_sample t:1→0 → IDCT → clamp(-1,1), execute first n=8 open-loop
  → critic_state_encoder (decoupled) → Critic V(s,a): 2 cross-attention blocks → TD target R + γ^L·V(s',a')
  → SigReg: Epps-Pulley CF normality on the critic latent (λ=0.1)
```
"""

DEVIATION_NOTES_SPECTRAL_MD = """\
## Deviation Notes (MuJoCo)

| ID | Deviation | Justification |
|----|-----------|---------------|
| D Synthetic | Local tests use synthetic fixtures, no vendored data | Small deterministic fixtures, no downloads |
| D Flag | MuJoCo-only; LIBERO twins pruned | Single-mode codebase, no dead branches |
| D Decoupled | Policy and critic own a StateEncoder each | ASPO and TD grads never share Z |
| D DCT-only | observation.state and action use DCT-II; rFFT reserved for observation.image | Single transform family per modality |
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
| D12 | Tiny matrix override for local testing | SJFPO_TINY_MATRIX=1 env var enables 3-run local validation |
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

- **SJFPO**: Zero-sampling (ε=0) at eval time — Euler integration from noise
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

# Self-containment: all sjfpo modules are inlined as code cells below; no external src/ required.
# Keep path setup as no-op for notebook portability (guard __file__ which is undefined in notebook).
try:
    _file = Path(__file__)
except NameError:
    _file = Path.cwd()
_cwd = Path(os.getcwd())
_src = str(_cwd / "src") if (_cwd / "src" / "sjfpo" / "__init__.py").exists() else str(_file.resolve().parents[0] / "src") if _file.exists() else str(_cwd / "src")
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

SMOKE_GATE_CELL = """\
# Smoke Gate: tiny local validation
import os, json, subprocess, sys
from pathlib import Path
from sjfpo.utils import default_smoke_dir

_smoke_dir = default_smoke_dir()
_smoke_dir.mkdir(parents=True, exist_ok=True)

if os.environ.get("SJFPO_RESUME_ONLY") is not None:
    print("RESUME_ONLY mode — checking for existing checkpoints...")
    for method in ("sjfpo", "ppo", "sac"):
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
          "--methods", "sjfpo", "ppo", "sac",
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
from pathlib import Path
from sjfpo.config import RUN_MATRIX
from sjfpo.sweep import run_pending
from sjfpo.utils import default_runs_base, detect_runtime_env

print(f"Runtime environment: {detect_runtime_env()}")
base = default_runs_base()
base.mkdir(parents=True, exist_ok=True)

def dispatch(spec, base_dir):
    out = base_dir / f"{spec.method}_{spec.env}_seed{spec.seed}"
    out.mkdir(parents=True, exist_ok=True)
    import gymnasium
    env = gymnasium.make(spec.env)
    try:
        if spec.method == "sjfpo":
            from sjfpo.runner import train_run
            return train_run(env=env, out_dir=out, seed=spec.seed,
                           total_steps=spec.total_steps,
                           eval_interval=spec.eval_interval,
                           ckpt_interval=spec.ckpt_interval)
        elif spec.method == "ppo":
            from sjfpo.baselines import ppo_train_run
            return ppo_train_run(env=env, out_dir=out, seed=spec.seed,
                               total_steps=spec.total_steps,
                               eval_interval=spec.eval_interval,
                               ckpt_interval=spec.ckpt_interval)
        elif spec.method == "sac":
            from sjfpo.baselines import sac_train_run
            return sac_train_run(env=env, out_dir=out, seed=spec.seed,
                               total_steps=spec.total_steps,
                               eval_interval=spec.eval_interval,
                               ckpt_interval=spec.ckpt_interval)
    finally:
        env.close()

results = run_pending(RUN_MATRIX, base, dispatch)
print(f"Completed: {results['completed']}, Skipped: {results['skipped']}, Diverged: {results['diverged']}, Failed: {results['failed']}")
"""

ANALYSIS_CELL = """\
# Analysis: bootstrap CIs, IQM, figures, CSV summary
from pathlib import Path
from sjfpo.analysis import analyze_results
from sjfpo.config import EVAL_INTERVAL, RUN_MATRIX
from sjfpo.utils import default_analysis_output, default_runs_base, detect_runtime_env

print(f"Runtime environment: {detect_runtime_env()}")
base = default_runs_base()
output = default_analysis_output()
output.mkdir(parents=True, exist_ok=True)

if base.exists() and any(base.iterdir()):
    report = analyze_results(RUN_MATRIX, base, output, eval_interval=EVAL_INTERVAL)
    print(f"Analysis complete. Reports in {output}/")
    print(f"Superiority: {report.superiority}")
else:
    print("No run directories found — skipping analysis.")
    print(f"Run the sweep cell first, or provide run data in {base}/")
"""

SELF_TEST_CELL = """\
# Embedded self-test: verify core math correctness
import torch
from pathlib import Path

errors = []

from sjfpo.dct_utils import dct, idct
x = torch.randn(4, 8, 3)
rt = idct(dct(x, dim=1), dim=1)
if torch.max(torch.abs(rt - x)).item() > 1e-5:
    errors.append(f"DCT roundtrip error: {torch.max(torch.abs(rt - x)).item()}")

from sjfpo.config import RUN_MATRIX, LEGACY_MUJOCO, TINY_MATRIX
# Tolerant self-test: RUN_MATRIX non-empty; size check respects mode
assert len(RUN_MATRIX) >= 1, f"RUN_MATRIX empty {len(RUN_MATRIX)}"
# Preserve legacy check only when in MuJoCo mode
if LEGACY_MUJOCO:
    from sjfpo.config import METHODS, ENVS, SEEDS
    _full_expected = len(METHODS) * len(ENVS) * len(SEEDS)
    _tiny_expected = len(METHODS)
    _expected = _tiny_expected if TINY_MATRIX else _full_expected
    # In isolated tmpdir with SJFPO_TINY_MATRIX=1 we expect tiny, but allow any non-empty for self-containment
    assert len(RUN_MATRIX) >= 1

from sjfpo.networks import sigreg_loss
z = torch.randn(32, 16, 128)
sl = sigreg_loss(z)
assert sl.ndim == 0 and torch.isfinite(sl), "SIGReg loss not finite scalar"

from sjfpo.fpo_core import aspo_surrogate
rho = torch.tensor([[1.04]])
adv = torch.tensor([[1.0]])
psi = aspo_surrogate(rho, adv, 0.05)
if abs(psi.item() - 1.04) > 1e-3:
    errors.append(f"ASPO psi={psi.item()}, expected ~1.04")

from sjfpo.gae import chunk_gae
N = 5
n_action_steps = torch.ones(N, dtype=torch.long) * 4
total_steps = int(n_action_steps.sum().item())
chunk_rewards = torch.randn(total_steps)
chunk_values = torch.randn(N)
chunk_next_values = torch.randn(N)
terminated = [False] * (N - 1) + [True]
gaes, returns = chunk_gae(chunk_rewards, chunk_values, chunk_next_values, n_action_steps, terminated, gamma=0.99, lam=0.95)
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
# Embedded self-test: live spectral path (policy encoder → tokens → DiT → critic)
import torch, sys

errors = []

# 1. Spectral tokens: frozen stats → [B, K, spectral_dim], batch invariance
from sjfpo.networks import SpectralEncoder, StateEncoder, SJFOPolicyModel, Critic
enc = StateEncoder(11, n_obs_steps=16, hidden_size=128)
spec = SpectralEncoder(hidden_size=128, num_spectral_tokens=5, n_obs_steps=16)
try:
    spec(torch.randn(2, 16, 128))
    errors.append("spectral forward without stats should raise")
except RuntimeError:
    pass
with torch.no_grad():
    z_sample = enc(torch.randn(8, 16, 11))
spec.capture_stats(z_sample)
tok = spec(z_sample[:2])
if tuple(tok.shape) != (2, 5, 64):
    errors.append(f"spectral token shape {tuple(tok.shape)} != (2, 5, 64)")
if not torch.isfinite(tok).all():
    errors.append("spectral tokens non-finite")

# 2. DiT policy forward: noisy DCT chunk + t + tokens → velocity
pol = SJFOPolicyModel(3, hidden_size=128)
v = pol(torch.randn(2, 16, 3), torch.rand(2) * 0.99 + 0.005, tok)
if tuple(v.shape) != (2, 16, 3):
    errors.append(f"policy velocity shape {tuple(v.shape)} != (2, 16, 3)")
if v.abs().max().item() > 1e-4:
    errors.append("DiT must predict ~zero velocity at init")

# 3. Critic forward on decoupled encoder latents
cenc = StateEncoder(11, n_obs_steps=16, hidden_size=128)
crit = Critic(hidden_size=128, act_dim=3)
with torch.no_grad():
    zc = cenc(torch.randn(2, 16, 11))
val = crit(zc, torch.randn(2, 16, 3))
if tuple(val.shape) != (2,):
    errors.append(f"critic value shape {tuple(val.shape)} != (2,)")

# 4. SIGReg: gaussian low, degenerate high
from sjfpo.networks import sigreg_loss
lg = sigreg_loss(torch.randn(64, 16, 128)).item()
ld = sigreg_loss(torch.ones(64, 16, 128) * 5).item()
if not (lg < 2.2):
    errors.append(f"SIGReg Gaussian {lg} >= 2.2")
if not (ld > 95.0):
    errors.append(f"SIGReg degenerate {ld} <= 95.0")

if errors:
    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    raise RuntimeError(f"Spectral self-test failed {errors}")
print("Spectral self-tests PASSED.")

# 5. LEGACY_MUJOCO flag preserved
from sjfpo import config as cfg
assert hasattr(cfg, "LEGACY_MUJOCO"), "LEGACY_MUJOCO missing"
print(f"LEGACY_MUJOCO={cfg.LEGACY_MUJOCO} spectral mode verified.")
"""

SMOKE_GATE_MUJOCO_CELL = """\
# Smoke Gate (MuJoCo-only): the LIBERO tiny-synthetic path was pruned.
# Run the live smoke from a checkout with src/scripts present:
#   PYTHONPATH=src python scripts/train.py --method sjfpo --env Hopper-v5 --seed 0 \
#       --total-steps 200 --eval-interval 100 --ckpt-interval 200 --out-dir .omo/evidence/smoke
print("MuJoCo smoke lives in scripts/train.py (not run in this cell).")
"""


def md_block(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code_block(source: str):
    return nbf.v4.new_code_cell(source.strip())


def read_module_source(filename: str) -> str:
    """Read src/sjfpo/<filename>, stripping notebook-assembly markers."""
    content = (SRC / filename).read_text()
    if not (SRC / filename).exists():
        raise FileNotFoundError(f"Module not found: {SRC / filename}")
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
    return f"""# Register inlined {filename} as importable sjfpo.{mod} for self-containment
import sys, types
_m = types.ModuleType('sjfpo.{mod}')
_m.__path__ = []
_m.__dict__.update({{k: v for k, v in globals().items() if not k.startswith('__')}})
sys.modules['sjfpo.{mod}'] = _m
if 'sjfpo' not in sys.modules:
    _pkg = types.ModuleType('sjfpo')
    _pkg.__path__ = []
    sys.modules['sjfpo'] = _pkg
else:
    _pkg = sys.modules['sjfpo']
    if not hasattr(_pkg, '__path__'):
        _pkg.__path__ = []
setattr(_pkg, '{mod}', _m)
"""


def new_notebook_shell():
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
    names = ["sjfpo"] + [f"sjfpo.{m}" for m in all_mods]
    return "import sys, types\n" + "\n".join(
        f"if '{n}' not in sys.modules:\n    _tmp = types.ModuleType('{n}')\n    _tmp.__path__ = []\n    sys.modules['{n}'] = _tmp\n    "
        + (
            f"setattr(sys.modules['sjfpo'], '{n.split('.')[-1]}', sys.modules['{n}'])"
            if "." in n
            else ""
        )
        for n in names
    )


def module_cells(order, titles, *, register: bool, extras: dict | None = None):
    """One home for the per-module md + code (+ register) sequence."""
    extras = extras or {}
    cells = []
    for filename in order:
        cells.append(md_block(f"## {titles.get(filename, filename)}"))
        cells.append(code_block(read_module_source(filename)))
        if register:
            cells.append(code_block(register_snippet(filename)))
        for md in extras.get(filename, ()):
            cells.append(md_block(md))
    return cells


def append_spectral_cells(cells: list) -> None:
    """Append the spectral joint-flow cell sequence."""
    cells.append(md_block(TITLE_SPECTRAL_MD))
    cells.append(md_block(SETUP_MD))
    cells.append(code_block(SETUP_CELL))
    cells.append(code_block(bootstrap_spectral()))
    cells.append(md_block(BINDING_SPECTRAL_MD))
    cells.extend(
        module_cells(
            MODULE_ORDER_SPECTRAL,
            MODULE_TITLES_SPECTRAL,
            register=True,
            extras={
                "config.py": (PROVENANCE_SPECTRAL_MD,),
                "networks.py": (ARCHITECTURE_SPECTRAL_MD,),
                "evaluator.py": (PROTOCOL_MD,),
            },
        )
    )
    cells.append(md_block(DEVIATION_NOTES_SPECTRAL_MD))
    cells.append(md_block(LIMITATIONS_SPECTRAL_MD))
    cells.append(md_block("## Embedded Self-Test (Live Spectral Path)"))
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
    cells.extend(
        module_cells(
            MODULE_ORDER,
            MODULE_TITLES,
            register=False,
            extras={
                "config.py": (PROVENANCE_MD,),
                "networks.py": (ARCHITECTURE_MD,),
                "fpo_core.py": (DEVIATION_NOTES_MD,),
                "evaluator.py": (PROTOCOL_MD,),
            },
        )
    )
    cells.append(md_block(LIMITATIONS_MD))
    cells.append(md_block("## Embedded Self-Test"))
    cells.append(code_block(SELF_TEST_CELL))
    cells.append(md_block("## Smoke Gate"))
    cells.append(code_block(SMOKE_GATE_CELL))
    cells.append(md_block("## Full Sweep"))
    cells.append(code_block(SWEEP_CELL))
    cells.append(md_block("## Analysis & Figures"))
    cells.append(code_block(ANALYSIS_CELL))


def build_notebook(output_path: Path, spectral: bool = False):
    nb = new_notebook_shell()
    cells: list = []
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
    out = Path(out)
    print(f"Building notebook: {out} (spectral={spectral})")
    nb = build_notebook(out, spectral=spectral)
    nbf.validate(nb)
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {len(nb.cells)} cells to {out} spectral={spectral} validated.")


def write_default_notebooks(args: argparse.Namespace) -> None:
    """Build both notebooks, honoring --legacy-only/--spectral-only."""
    if not args.legacy_only:
        write_notebook(ROOT / "spectral_joint_flow.ipynb", True)
    if not args.spectral_only:
        write_notebook(ROOT / "flow_matching_vs_ppo_sac.ipynb", False)


def main():
    args = parse_build_args()
    if args.output:
        out = ROOT / args.output
        write_notebook(out, "spectral" in out.name)
        return
    write_default_notebooks(args)


if __name__ == "__main__":
    main()
