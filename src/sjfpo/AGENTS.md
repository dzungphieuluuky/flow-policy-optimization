# AGENTS — src/sjfpo

## Overview
MuJoCo-only hotspot — single live train path, chunked flow policy: Transition-dict rollout (H=16/n=8/T=16) → step-level GAE over executed prefixes → ASPO + V(s) TD-critic; CUDA-default (CPU fallback) fp32, einx.id shapes (collector/fpo_core/trainer + dct_utils), RUN_MATRIX 15. analysis.py is notebook-invoked (train.py never imports it), everything else is live.

## Structure
```
networks.py      WindowEncoder (shared; conditioner + critic own instances) + RunningMeanStd (obs stats, checkpointed) + StateConditioner(raw window + DCT bins, Flamingo-style cross-attn -> [B,D]) + DiT SJFPOFlowModel (x_t, t, condition[B,D]) + V(s) Critic
modeling_sjfpo.py   SJFPOPolicy.forward: ASPO (truncated Ka coeffs, negated) + V(s) critic on raw GAE returns (0.5 factor, PPO-clipped) + running obs norm; optimizer = live policy weights; global grad-clip; NaN-guard aborts
train_sjfpo.py    online train entry: generate_rollout → forward → AdamW step → eval → Hub ckpt
collector.py      346L — Transition TypedDict (8 LeRobot keys); full-H open-loop chunks; real-space scaled action anchor [H,action_dim] (DCT downstream in forward); all chunks train (trainer zero-pads unexecuted tails)
gae.py            step-level GAE over executed transitions; terminated masks bootstrap and cuts recursion, truncated keeps bootstrap and cuts recursion; first executed step supplies each chunk target
fpo_core.py       230L — CFM/SQL-ℓ [N,m], STE-capped ρ, ASPO piecewise (PPO-clip Â>0 / SPO Â≤0), mean-of-ψ objective, sample(mode=euler) t:0→1; policy + vf_coef·value summed inline in forward
dct_utils.py      orthonormal DCT-II/III [H,action_dim] action chunks + discounted_chunk_reward + einx.id (parity-guarded)
config.py         env registry (envs.yaml) + RUN_MATRIX 15 (TINY 1) + build_run_matrix. SJFPO hyperparams live ONLY in configuration_sjfpo.py::SJFPOConfig (H=16/T=16/top_p=0.3→K=5/D=128, DiT-8/4heads/mlp×4/t256)
checkpointing.py  Hub policy ckpt (config.json + safetensors via save/load_policy_checkpoint)
evaluator.py      159L — Evaluator MuJoCo 3-col [episode,return,length]
sweep.py          196L — scan_runs/run_pending state machine
analysis.py       IQM+bootstrap 2000, notebook-invoked (train.py never imports it; risk boundary)
wandb_logging.py  ~114L — sole wandb importer (REQUIRED, hard import): 5 fns SDK-direct (wandb_disabled, init_wandb_run_for_method->bool, log_metrics, log_eval_summary, finish_run)
utils.py          149L — shared pure helpers (device/dtype/compile), no sjfpo imports
```

## Where to Look
- `collector.py:51` `Transition`: 8 LeRobot keys (`observation.state [T,obs_dim]`, real scaled `action [H,action_dim]`, `next.*`, `n_action_steps` i64, `clip_count` i64); `next.reward` is variable-length `[L_k]`
- `modeling_sjfpo.py:forward`: stack (guards H/T) → StateConditioner fuses states to `condition [N,D]` → policy ℓ/ρ/ψ `[N,m]` → critic `V(s)` → clip → loss (stepped by the training loop)
- `networks.py:StateConditioner`: raw window tokens query DCT-II bins (bin index = frequency order) in one cross-attn block, mean-pool → `[B,D]` (rFFT reserved for observation.image)
- `gae.py:compute_step_advantages_torch`: per-step `δ_t`, backward `γλ`; terminated masks bootstrap and cuts recursion, truncated keeps bootstrap and cuts recursion
- `runner.py:build_training_components`: policy/critic/conditioner (each model owns its state encoder, no target nets, no aux head); `compute_chunk_values` V(s) via Critic, bootstraps truncated only
- trainer sums `policy + vf_coef·value` inline (no helper); `sweep.py:ScanResult(pending,skipped,diverged)` deterministic sort (method,env,seed)

## Code Map
| File | Export | Caller |
|---|---|---|
| `networks.py` | `WindowEncoder` `RunningMeanStd` `StateConditioner` `SJFPOFlowModel` `Critic` `DiTBlock` `CondMLP` `SwiGLU` | `modeling_sjfpo.py` (live) |
| `collector.py` | `Transition` `collect_chunks` (+`collect_chunks_vec` for `num_envs>1`) | `tests/test_transition_v4.py` |
| `dct_utils.py` | `dct` `idct` `discounted_chunk_reward` | `collector.py` `fpo_core.py` (DCT) + `tests/test_dct.py` |
| `utils.py` | `resolve_device` `is_bf16_available` `resolve_compute_dtype` `maybe_compile` | `train_sjfpo.py` |
| `modeling_sjfpo.py` | `SJFPOPolicy` `SJFPOAborted` | `sweep.py` |
| `wandb_logging.py` | `wandb_disabled` `init_wandb_run_for_method`->bool `log_metrics` `log_eval_summary` `finish_run` (5 fns SDK-direct) | `train_sjfpo.py` (REQUIRED import, SJFPO_DISABLE_WANDB=1 gate) |

## Conventions
- Tensor order `B,N,H,K,D,action_dim,Ds` — comment dimensions at each transform (Google docstrings + inline `[B,H,action_dim]`)
- Flow takes `(x_t, t, condition[B,D])`; observation fusion lives in `StateConditioner` (state queries DCT bins, mean-pool). The transformer sees action tokens only — no prefix tokens, no readout slice.
- Block FFNs and window encoders use SwiGLU (gate/up/down linears); the conditioning path (`time_mlp`, block `adaLN`) uses SiLU MLPs (`CondMLP`) per DiT canon — the gate regression stays (near-)linear.
- Spectral-only: no non-spectral regularizers (no Sobolev/SIGReg/Barlow heads) — smoothness comes from DCT truncation structure, not auxiliary losses
- einx.id reshapes in collector/fpo_core/trainer (+dct_utils) must match torch at 1e-6 (`tests/test_einx_migration.py`); ground truth is current torch behavior
- CUDA-default with CPU fallback, fp32-only (no autocast/bf16/`torch.compile`): `SJFPO_DEVICE=cuda|cpu|auto` overrides; storage (transitions/checkpoints/CSV) stays on CPU, batches move to device per update; LSP absent here — `py_compile` + suite is the gate
- wandb imported ONLY in `sjfpo.wandb_logging` (sole importer, module-level REQUIRED — absent → hard ImportError); train_sjfpo routes init through `init_wandb_run_for_method()->bool` + `log_eval_details` + `log_metrics(data, step=global_step)` + `finish_run()`; metrics: train/*, eval/*, gae/*, timing/*; `wandb/` dir NOT in .gitignore
- `# %%` + `# CELLORDER:` markers are vestigial (the builder no longer inlines `src/` into notebooks) — leave them, don't add new ones
- `retained:`-marked helpers are test seams (compute_cfm_loss/compute_policy_loss, discounted_chunk_reward, assert_runtime) — deleting one breaks its consumer test first

## Anti-Patterns
- DO NOT move observation fusion back into the flow head (`SJFPOFlowModel` stays `(x_t, t, condition)`); do NOT reintroduce target/EMA nets (`target_state_encoder`, `EMAProjection`, `ema_power`) beyond the frozen `old_*` behavior snapshots
- DO NOT reintroduce `Polyak`/`ema_decay` naming in the sjfpo path
- DO NOT touch `analysis.py` statistics/plotting (bootstrap, IQM n_boot 2000, CI) without explicit plan amendment — risk boundary, off train path
- DO NOT add RL heads to the flow loss (ASPO+value live in `assemble_update`, not in the flow)
- DO NOT import `wandb` outside `sjfpo.wandb_logging` (sole importer); do NOT log secrets/API keys
- DO NOT change `AnalysisError` hierarchy without updating `build_notebook.py` Analysis fallback

## Commands
```bash
PYTHONPATH=src .venv/bin/python -m pytest -q                               # 277 items, ~103s CPU
PYTHONPATH=src .venv/bin/python -m pytest tests/test_pipeline_cpu.py -v    # 3-col CSV + 45 + wandb invariants
PYTHONPATH=src .venv/bin/python scripts/train.py --method sjfpo --env Hopper-v5 --seed 0 --total-steps 2000  # smoke
```

## Notes
- One `WindowEncoder` class serves all time windows (conditioner state branch, critic state encoder), each owned by its model; ASPO grads land on the policy head (incl. conditioner)
- `compute_chunk_values` bootstraps truncated chunks from the full next window

---

# Optimization Reference (from current code audit)

Write these as tips when implementing new modules.

## Networks
- Use `SwiGLU` (fused gate+up GEMM) instead of 2 separate Linears.
- Cache sinusoidal freq tables (device-keyed dict); never rebuild `exp` per call.
- Use branchless `RunningMeanStd.forward`; avoid `int(self.count)` host sync.
- `SelfAttention`: `view`/`transpose` only; cache SDPA mask when standard (identity/causal).
- `RMSNorm`: store weights in `STORAGE_DTYPE`; cast input only when `dtype != weight.dtype`.

## Flow / Sampling
- `run_mc_simulation`: broadcast `(a, eps, tau)` onto flat `[B*m, ...]` once; no per-sample loops.
- `compute_policy_ratio`: STE cap (`clamp` forward, `detach` backward) — never break gradient.
- `compute_cfm_loss`: prefer `reduce_mode="bin"` (`sum/sqrt(D)`) over `mean` for variance stability.
- `sample_action`: precompute `dts` grid; resolve device once; no per-step sync.

## Policy / Training Loop
- Batch diagnostics into one `.tolist()` device→host transfer; no `float()` syncs per stat.
- `snapshot_behavior_policy`: `deepcopy` + `.eval()`; frozen in train mode.
- Gradient checkpointing: gate with `cfg.use_gradient_checkpointing`; wrap actor call only.
- `zero_grad(set_to_none=True)` between minibatches; `fused=True` AdamW on CUDA.
- Analytical grad-clip (`min(pre, max_norm)`) — no `.item()` per parameter.

## DCT / Spectral
- Cache `DCT` basis matrices per `(n, device)` in fp32; cast at use site.
- `extract_dct_coeffs` / `decode_coeffs_to_velocity`: share `k`/norm computation.
- `sample_freq_weighted_noise`: local `Generator`; never touch global RNG.

## Processor / Data Path
- `ObservationProcessor`: hold live reference to `policy.obs_normalizer` (not a copy).
- `ActionPostprocessor`: per-`(device, dtype)` affine coeff cache (`_scale_cache`); avoid `.to()` per chunk.

## Compile / Autocast / Device
- `autocast_context`: `bf16` only on CUDA+bf16-supported; else `nullcontext`.
- `maybe_compile`: `torch.compile(dynamic=True, scalar_capture=True)`; only if CUDA+bf16/Blackwell; disable with `SJFPO_COMPILE=0`.
- `to_device_async`: `pin_memory()` + `non_blocking=True` for DMA overlap.
- `resolve_storage_dtype`: `bf16` on Blackwell, `fp32` elsewhere; `resolve_compute_dtype`: same.
