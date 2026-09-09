# AGENTS — src/mfpo

## Overview
MuJoCo-only (LEGACY_MUJOCO=True) hotspot (5464 LOC raw, 19 modules, post-deslop -201) — spectral tokens K=5 → joint DCT 135 → joint flow transformer; CPU-first on gfx1103 APU (780M shared RAM); MJX gated `MFPO_USE_MJX=1` stub (`is_stub=True` default, `mjx_adapter.py`). LIBERO twins pruned, VICReg SIGReg single home, fp32 everywhere, einx.id notation (dct_utils/baselines/collector/fpo_core), <20M params, RUN_MATRIX 45.

## Structure
```
networks.py      693L — ContextTrunk + SpectralTokens K5 Dk64 + JFT + PerBranchFlowLoss + VICReg SIGReg (single home) + EMAProjection
trainer.py        320L — MFPOTrainer assemble_update ASPO+value (legacy wrapper + nanGuard inlined)
dct_utils.py      280L — per-branch DCT/IDCT [B,8,135] frozen stats + einx.id (parity-guarded)
utils.py           23L — shared pure helpers (safe_float_value), no mfpo imports
inference.py      151L — euler_integrate_joint ascending 0→1 steps 16/8 (NOTEBOOK-ONLY, default KEEP)
evaluator.py      161L — Evaluator MuJoCo 3-col [episode,return,length] (jerk/DCT metrics deleted)
mjx_adapter.py    170L — MjxObsAdapter/MjxWrapper is_stub=True stub + gated MjxModelWrapper (MFPO_USE_MJX=1)
wandb_logging.py  159L — sole wandb importer (REQUIRED, hard import): 6 fns SDK-direct (wandb_disabled, init_wandb_run_for_method->bool, log_metrics, log_eval_details, log_eval_summary, finish_run; shared runner/baselines)
sweep.py          227L — scan_runs/run_pending state machine
calibration.py    156L — calibrate_and_gate budget gate (injection chain collapsed)
analysis.py       644L — IQM+bootstrap 2000 (risk boundary; wallclock_h column placeholder)
runner.py         418L — train_run flat Box CPU-first, 3-col CSV + wandb (decode_chunk_actions single clamp home)
baselines.py      691L — skrl PPO/SAC, shared GaussianPolicyImpl + helpers + wandb + einx.id
config.py         352L — frozen dataclasses, RUN_MATRIX 45 (TINY 3) + build_filtered_matrix
collector.py/gae.py/fpo_core.py/checkpointing.py — 204/137/432/246L: chunk rollout, GAE, ASPO/CFM + total_update_loss, atomic ckpt (fpo_core/collector use einx.id)
```

## Where to Look
- `networks.py:179` `SpectralTokens`: `Z [B,N,D] ⊙ hann → rFFT → cat(real,imag) [B,9,2D] → K=5 → [B,5,2D] → frozen norm → Linear 2D→64 → [B,5,64]`
- `networks.py:222` `resolve_config_value("Dk", 64)` (wire properly with test if ever configurable)
- `networks.py:563` `JointFlowTransformer`: `z_t=(1-t)z0 + t X*`, `u*=X*-z0`, `L=1.0*MSE_A+0.5*MSE_S`
- `trainer.py` `assemble_update`: ASPO+value; `fpo_core.py` `total_update_loss` (single cfg-weight home)
- `safe_float_value` lives in `utils.py` (imported, no local def)
- `dct_utils.py:223` `normalize_joint_targets` per-branch frozen — `denormalize` atol 1e-6 test gate
- `inference.py:60` `euler_integrate_joint(v_theta, z0, Z, T_FFT, steps)` dt=1/steps ascending, gripper {0,1} via IDCT
- `evaluator.py` `Evaluator.evaluate` only (PolicyAdapter alias + jerk/DCT metrics deleted post-deslop)
- `mjx_adapter.py:19` `_USE_MJX=os.environ.get("MFPO_USE_MJX")=="1"` gated stub, `MjxModelWrapper` with `put_model`+`jax.jit(mjx.step)` exact-dim assert
- `sweep.py:38` `ScanResult(pending,skipped,diverged)` deterministic sort (method,env,seed)

## Code Map
| File | Export | Caller |
|---|---|---|
| `networks.py` | `SpectralTokens` `JointFlowTransformer` `PerBranchFlowLoss` `sigreg_losses` | `trainer.py` `runner.py` |
| `dct_utils.py` | `dct` `idct` `normalize_joint_targets` `denormalize_joint_targets` | `trainer.py` `inference.py` |
| `utils.py` | `safe_float_value` | `trainer.py` |
| `trainer.py` | `MFPOTrainer` `MFPOAborted` | `runner.py` `sweep.py` |
| `inference.py` | `euler_integrate_joint` `decode_joint` | notebook-only: `tests/test_inference.py` + `build_notebook.py` |
| `wandb_logging.py` | `wandb_disabled` `init_wandb_run_for_method`->bool `log_metrics` `log_eval_details` `log_eval_summary` `finish_run` (6 fns SDK-direct) | `runner.py` `baselines.py` (REQUIRED import, MFPO_DISABLE_WANDB=1 gate) |

## Conventions
- Tensor order `B,N,H,K,D,Dk,Da,Ds` — comment dimensions at each transform (see Google docstrings added 2026-08-28)
- Keep `sigreg_losses` VICReg var+cov single home in `networks.py:139` — no Barlow/EMA heads
- einx.id reshapes in dct_utils/baselines/collector/fpo_core must match torch at 1e-6 (`tests/test_einx_migration.py`); ground truth is current torch behavior
- CPU-first: `requirements-local.txt` CPU torch pin; `CUDA_VISIBLE_DEVICES=""` fallback; gfx1103 vs gfx1101 stub documented in `AGENTS.md`
- MJX never imported in runner/baselines; function-local jax/mujoco.mjx only when MFPO_USE_MJX=1
- wandb imported ONLY in `mfpo.wandb_logging` (sole importer, module-level REQUIRED — absent → hard ImportError); runners route init through `init_wandb_run_for_method()->bool` (full-hyperparam config) + `log_eval_details` (episodes Table data=[list(r)...] + reward_improvement), `log_metrics(data, step=global_step)`+`finish_run()`, global `wandb.define_metric("*", step_metric="global_step")` before loop; metrics: train/*, eval/*, gae/*, timing/*; `wandb/` git-ignored
- All modules Google-style `Args/Returns/Raises` + inline `[B,H,Da]` shapes
- No `as any`/`# type: ignore` — f2-quality scan

## Anti-Patterns
- DO NOT reintroduce a standalone SIGReg module — single source in `networks.py`
- DO NOT touch `analysis.py` statistics/plotting (bootstrap, IQM n_boot 2000, CI) without explicit plan amendment — risk boundary
- DO NOT add RL heads to the flow loss — VICReg SIGReg-only gate (ASPO+value live in `assemble_update`, not in the flow)
- DO NOT import `wandb` outside `mfpo.wandb_logging` (sole importer); do NOT log secrets/API keys
- DO NOT change `AnalysisError`/`CalibrationError` hierarchy without updating `build_notebook.py` Analysis fallback

## Commands
```bash
PYTHONPATH=src .venv/bin/python -m pytest -q                               # 263 green (LIBERO suite pruned)
PYTHONPATH=src .venv/bin/python -m pytest tests/test_pipeline_cpu.py -v    # 3-col CSV + 45 + is_stub + wandb invariants
# CPU fallback (gfx1103 APU, no 1103 wheels):
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu --force-reinstall "torch==2.12.0" "torchvision==0.27.0" "torchaudio==2.11.0"
```

## Notes
- Pipeline wired CPU end-to-end (runner flat Box, evaluator 3-col CSV, build_notebook 61/47 cells, wandb offline 3-method smoke); wallclock_h column placeholder acknowledged
- MJX gated stub — shared RAM APU no throughput win for JAX batched MJX; verified via stub guard test
- Todos 3–5 splits landed — line-count gate holds (31-line one-extract waiver rule, rename-only boundary allowlist `analysis.py` + `fpo_core.py`); the file-split roadmap is retired
