# AGENTS — src/sjfpo

## Overview
MuJoCo-only (LEGACY_MUJOCO=True) hotspot — single live train path, chunked flow policy: Transition-dict rollout (H=16/n=8/T=16, next.action chaining) → chunk GAE → ASPO + V(s,a) TD-critic + Epps-Pulley SIGReg; CUDA-default (CPU fallback) fp32, einx.id shapes (collector/fpo_core/trainer + dct_utils/baselines), RUN_MATRIX 45. analysis.py is notebook-invoked (train.py never imports it), everything else is live.

## Structure
```
networks.py      StateEncoder + SpectralEncoder(DCT-II, 5 tokens, dim 64) + DiT-8 SJFOPolicyModel (actions only) + 2-block cross-attn Critic V(s,a) + Epps-Pulley SIGReg on critic latent (single home)
trainer.py        389L — SJFPOTrainer.assemble_update: ASPO + V(s,a) TD-critic (R + γ^L·V(s',a')) + critic-latent SIGReg; optimizer = trunk+policy+critic+spectral; per-module grad-clip; NaN-guard with sink dump
collector.py      241L — Transition TypedDict (9 LeRobot keys incl. next.action chaining); open-loop chunks H=16/n=8; action anchor = DCT of FULL SCALED proposed chunk [H,Da]; terminated kept for GAE, dropped from update
gae.py            137L — chunk semi-MDP GAE (R_k intradiscounted, γ^L bootstrap, (γλ)^L recursion); terminated never bootstraps; numpy fp64, advantages centered
fpo_core.py       432L — CFM/SQL-ℓ [N,m], STE-capped ρ, ASPO piecewise (PPO-clip Â≥0 / SPO Â<0), mean-of-ψ objective, euler_sample t:1→0; total_update_loss (single weight home)
runner.py         428L — train_run: collect→GAE→update→eval→ckpt; decode_chunk_actions (IDCT + single clamp home)
dct_utils.py      orthonormal DCT-II/III [H,Da] action chunks + discounted_chunk_reward + einx.id (parity-guarded)
config.py         360L — frozen dataclasses (SJFPOConfig H=16/n=8/T=16/K=5/D=128, DiT-8/4heads/mlp×4/t256), RUN_MATRIX 45 (TINY 3) + build_run_matrix; SACConfig.polyak is skrl-boundary (keep)
baselines.py      690L — skrl PPO/SAC via baseline_train_loop; FrameStackWrapper shared with evaluator
checkpointing.py  246L — atomic ckpt (models + optimizer + counters)
evaluator.py      162L — Evaluator MuJoCo 3-col [episode,return,length]
sweep.py          227L — scan_runs/run_pending state machine
analysis.py       IQM+bootstrap 2000, notebook-invoked (train.py never imports it; risk boundary)
wandb_logging.py  159L — sole wandb importer (REQUIRED, hard import): 6 fns SDK-direct (wandb_disabled, init_wandb_run_for_method->bool, log_metrics, log_eval_details, log_eval_summary, finish_run; shared runner/baselines)
utils.py           23L — shared pure helpers (safe_float_value), no sjfpo imports
```

## Where to Look
- `collector.py:51` `Transition`: 9 LeRobot keys (`observation.state [T,Do]`, `action [H,Da]` DCT, `next.action [H,Da]` chained/behavior proposal, `next.*`, `n_action_steps` i64, `clip_count` i64); `next.reward` is variable-length `[L_k]`
- `trainer.py:assemble_update`: stack (guards H/T) → policy trunk `Z [N,T,D]` → spectral `cond [N,K,spectral_dim]` → policy ℓ/ρ/ψ `[N,m]` → critic trunk `Z [N,T,D]` → V(s,a) TD-critic → SigReg over critic-h → clip → AdamW (one optimizer, two disjoint grad paths)
- `networks.py:SpectralEncoder`: DCT-II `Z [B,T,D]` over time → keep K=5 low bins → frozen norm → Linear D→64 → `[B,5,64]` (raises until `set_frozen_stats`/`capture_stats`; rFFT reserved for observation.image)
- `gae.py:chunk_gae`: per-chunk `R_k`, `δ_k`, backward `(γλ)^L`; terminated cuts recursion, truncated bootstraps from last next-frame
- `runner.py:build_training_components`: policy_state_encoder/policy/critic/spectral/critic_state_encoder (no target nets, no aux head); `compute_chunk_values` V(s,a) via critic encoder, bootstraps truncated only
- `fpo_core.py:total_update_loss` (single cfg-weight home); `sweep.py:ScanResult(pending,skipped,diverged)` deterministic sort (method,env,seed)

## Code Map
| File | Export | Caller |
|---|---|---|
| `networks.py` | `StateEncoder` `SpectralEncoder` `SJFOPolicyModel` `Critic` `sigreg_loss` `DiTBlock` `CrossAttentionBlock` | `trainer.py` `runner.py` (live) |
| `collector.py` | `Transition` `collect_chunks` | `trainer.py` `runner.py` `tests/test_transition_v4.py` |
| `dct_utils.py` | `dct` `idct` `discounted_chunk_reward` | `collector.py` `fpo_core.py` (DCT) + `tests/test_dct.py` |
| `utils.py` | `safe_float_value` | `trainer.py` |
| `trainer.py` | `SJFPOTrainer` `SJFPOAborted` | `runner.py` `sweep.py` |
| `wandb_logging.py` | `wandb_disabled` `init_wandb_run_for_method`->bool `log_metrics` `log_eval_details` `log_eval_summary` `finish_run` (6 fns SDK-direct) | `runner.py` `baselines.py` (REQUIRED import, SJFPO_DISABLE_WANDB=1 gate) |

## Conventions
- Tensor order `B,N,H,K,D,Da,Ds` (+ spectral projector dim) — comment dimensions at each transform (Google docstrings + inline `[B,H,Da]`)
- Live cond is `[N,K,spectral_dim=64]`; the DiT pools it (`cond.mean(1)`) into the adaLN vector
- Net activations are SiLU everywhere
- Keep `sigreg_loss` Epps-Pulley CF normality single home in `networks.py` — no Barlow heads
- `polyak` survives ONLY at the skrl boundary (`SACConfig`, `build_sac_skrl_config`, `sac.yaml`); no EMA/target nets in the sjfpo path
- einx.id reshapes in collector/fpo_core/trainer (+dct_utils/baselines) must match torch at 1e-6 (`tests/test_einx_migration.py`); ground truth is current torch behavior
- CUDA-default with CPU fallback, fp32-only (no autocast/bf16/`torch.compile`): `SJFPO_DEVICE=cuda|cpu|auto` overrides; storage (transitions/checkpoints/CSV) stays on CPU, batches move to device per update; LSP absent here — `py_compile` + suite is the gate
- wandb imported ONLY in `sjfpo.wandb_logging` (sole importer, module-level REQUIRED — absent → hard ImportError); runners route init through `init_wandb_run_for_method()->bool` + `log_eval_details` + `log_metrics(data, step=global_step)` + `finish_run()`; metrics: train/*, eval/*, gae/*, timing/*; `wandb/` dir NOT in .gitignore
- `# %%` + `# CELLORDER:` markers are load-bearing (`build_notebook.py` strips them into notebook cells) — never delete
- `retained:`-marked helpers are test seams (flow_loglik/policy_loss, discounted_chunk_reward, assert_runtime) — deleting one breaks its consumer test first

## Anti-Patterns
- DO NOT reintroduce a standalone SIGReg module — single source in `networks.py`
- DO NOT reintroduce target/EMA nets (`target_trunk`, `EMAProjection`, `ema_power`) in the sjfpo path — policy and critic trunks are decoupled by design, no third copy
- DO NOT reintroduce `Polyak`/`ema_decay` naming in the sjfpo path
- DO NOT touch `analysis.py` statistics/plotting (bootstrap, IQM n_boot 2000, CI) without explicit plan amendment — risk boundary, off train path
- DO NOT add RL heads to the flow loss — Epps-Pulley SIGReg-only gate (ASPO+value live in `assemble_update`, not in the flow)
- DO NOT import `wandb` outside `sjfpo.wandb_logging` (sole importer); do NOT log secrets/API keys
- DO NOT change `AnalysisError` hierarchy without updating `build_notebook.py` Analysis fallback

## Commands
```bash
PYTHONPATH=src .venv/bin/python -m pytest -q                               # 277 items, ~103s CPU
PYTHONPATH=src .venv/bin/python -m pytest tests/test_pipeline_cpu.py -v    # 3-col CSV + 45 + wandb invariants
PYTHONPATH=src .venv/bin/python scripts/train.py --method sjfpo --env Hopper-v5 --seed 0 --total-steps 2000  # smoke
```

## Notes
- State encoders are decoupled: ASPO grads land on `policy_state_encoder`, value-TD + SigReg grads on `critic_state_encoder`; no gradient crosses between them
- `compute_chunk_values` bootstraps truncated chunks from the full next window + stored `next.action`
