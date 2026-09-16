# AGENTS — src/sjfpo

## Overview
MuJoCo-only hotspot — single live train path, chunked flow policy: Transition-dict rollout (H=16/n=8/T=16, next.action chaining) → chunk GAE → ASPO + V(s,a) TD-critic; CUDA-default (CPU fallback) fp32, einx.id shapes (collector/fpo_core/trainer + dct_utils/baselines), RUN_MATRIX 45. analysis.py is notebook-invoked (train.py never imports it), everything else is live.

## Structure
```
networks.py      StateEncoder + ActionEncoder (same definition; policy state encoder inside SJFOPolicyModel, state+action encoders inside Critic) + RunningMeanStd (obs stats, checkpointed) + SpectralEncoder(raw-obs DCT-II, parameter-free) + DiT-8 SJFOPolicyModel ([state;action] tokens) + 1-block cross-attn Critic V(s,a trans real-space +PE query)
modeling_sjfpo.py   SJFPOPolicy.forward: ASPO (truncated Ka coeffs, negated) + V(s,a) critic on executed-masked real actions (frozen GAE returns, PPO-clipped) + running obs norm; optimizer = live policy weights; global grad-clip; NaN-guard aborts
train_sjfpo.py    online train entry: generate_rollout → forward → AdamW step → eval → Hub ckpt
collector.py      241L — Transition TypedDict (9 LeRobot keys incl. next.action chaining); full-H open-loop chunks; real-space scaled action anchor [H,action_dim] (DCT downstream in forward); all chunks train (trainer zero-pads unexecuted tails)
gae.py            137L — chunk semi-MDP GAE (R_k intradiscounted, γ^L bootstrap, (γλ)^L recursion); terminated cuts all, truncated bootstraps δ but cuts recursion; train advantages raw (ASPO sign)
fpo_core.py       432L — CFM/SQL-ℓ [N,m], STE-capped ρ, ASPO piecewise (PPO-clip Â≥0 / SPO Â<0), mean-of-ψ objective, sample(mode=euler) t:0→1; policy + vf_coef·value summed inline in forward
dct_utils.py      orthonormal DCT-II/III [H,action_dim] action chunks + discounted_chunk_reward + einx.id (parity-guarded)
config.py         flags + PPO/SAC dataclasses, RUN_MATRIX 45 (TINY 3) + build_run_matrix; SACConfig.polyak is skrl-boundary (keep). SJFPO hyperparams live ONLY in configuration_sjfpo.py::SJFPOConfig (H=16/T=16/top_p=0.3→K=5/D=128, DiT-8/4heads/mlp×4/t256)
baselines.py      690L — skrl PPO/SAC via baseline_train_loop; FrameStackWrapper shared with evaluator
checkpointing.py  Hub policy ckpt (config.json + safetensors via save/load_policy_checkpoint)
evaluator.py      162L — Evaluator MuJoCo 3-col [episode,return,length]
sweep.py          227L — scan_runs/run_pending state machine
analysis.py       IQM+bootstrap 2000, notebook-invoked (train.py never imports it; risk boundary)
wandb_logging.py  ~120L — sole wandb importer (REQUIRED, hard import): 5 fns SDK-direct (wandb_disabled, init_wandb_run_for_method->bool, log_metrics, log_eval_summary, finish_run; shared baselines)
utils.py           23L — shared pure helpers (safe_float_value), no sjfpo imports
```

## Where to Look
- `collector.py:51` `Transition`: 9 LeRobot keys (`observation.state [T,obs_dim]`, real scaled `action [H,action_dim]`, `next.action [H,action_dim]` chained/behavior proposal, `next.*`, `n_action_steps` i64, `clip_count` i64); `next.reward` is variable-length `[L_k]`
- `modeling_sjfpo.py:forward`: stack (guards H/T) → spectral `condition [N,K,obs]` (raw bins) → policy ℓ/ρ/ψ `[N,m]` → critic `V(s,a)` → clip → loss (stepped by the training loop)
- `networks.py:SpectralEncoder`: DCT-II raw obs `[B,T,obs]` over time → keep K lowest bins (round(top_p*T)) → `[B,K,obs]` raw tokens (parameter-free; rFFT reserved for observation.image)
- `gae.py:compute_advantages`: per-chunk `R_k`, `δ_k`, backward `(γλ)^L`; terminated cuts recursion, truncated bootstraps δ from last next-frame but cuts recursion
- `runner.py:build_training_components`: policy/critic/spectral (each model owns its state encoder, no target nets, no aux head); `compute_chunk_values` V(s,a) via Critic, bootstraps truncated only
- trainer sums `policy + vf_coef·value` inline (no helper); `sweep.py:ScanResult(pending,skipped,diverged)` deterministic sort (method,env,seed)

## Code Map
| File | Export | Caller |
|---|---|---|
| `networks.py` | `WindowEncoder` `RunningMeanStd` `SpectralEncoder` `SJFOPolicyModel` `Critic` `DiTBlock` `CrossAttentionBlock` `CondMLP` | `modeling_sjfpo.py` (live) |
| `collector.py` | `Transition` `collect_chunks` | `tests/test_transition_v4.py` |
| `dct_utils.py` | `dct` `idct` `discounted_chunk_reward` | `collector.py` `fpo_core.py` (DCT) + `tests/test_dct.py` |
| `utils.py` | `safe_float_value` `resolve_device` `is_bf16_available` `resolve_compute_dtype` | `baselines.py` |
| `modeling_sjfpo.py` | `SJFPOPolicy` `SJFPOAborted` | `sweep.py` |
| `wandb_logging.py` | `wandb_disabled` `init_wandb_run_for_method`->bool `log_metrics` `log_eval_summary` `finish_run` (5 fns SDK-direct) | `runner.py` `baselines.py` (REQUIRED import, SJFPO_DISABLE_WANDB=1 gate) |

## Conventions
- Tensor order `B,N,H,K,D,action_dim,Ds` — comment dimensions at each transform (Google docstrings + inline `[B,H,action_dim]`)
- Live condition is `[N,K,obs]` raw bins; pure-DiT conditioning through one adaLN vector `c = time + state + spectral` (`spectral_condition_mode="mean"`: `condition.mean(1)` via `cond_proj`; `"attention"`: `SpectralAttentionPool` query from time-embed, weights in `last_pool_weights`; state branch: `WindowEncoder` mean-pooled). The transformer sees action tokens only — no prefix tokens, no readout slice.
- Block FFNs and window encoders use SwiGLU (gate/up/down linears); the conditioning path (`time_mlp`, block `adaLN`) uses SiLU MLPs (`CondMLP`) per DiT canon — the gate regression stays (near-)linear.
- Spectral-only: no non-spectral regularizers (no Sobolev/SIGReg/Barlow heads) — smoothness comes from DCT truncation structure, not auxiliary losses
- `polyak` survives ONLY at the skrl boundary (`SACConfig`, `build_sac_skrl_config`, `sac.yaml`); no EMA/target nets in the sjfpo path
- einx.id reshapes in collector/fpo_core/trainer (+dct_utils/baselines) must match torch at 1e-6 (`tests/test_einx_migration.py`); ground truth is current torch behavior
- CUDA-default with CPU fallback, fp32-only (no autocast/bf16/`torch.compile`): `SJFPO_DEVICE=cuda|cpu|auto` overrides; storage (transitions/checkpoints/CSV) stays on CPU, batches move to device per update; LSP absent here — `py_compile` + suite is the gate
- wandb imported ONLY in `sjfpo.wandb_logging` (sole importer, module-level REQUIRED — absent → hard ImportError); baselines route init through `init_wandb_run_for_method()->bool` + `log_eval_details` + `log_metrics(data, step=global_step)` + `finish_run()`; metrics: train/*, eval/*, gae/*, timing/*; `wandb/` dir NOT in .gitignore
- `# %%` + `# CELLORDER:` markers are vestigial (the builder no longer inlines `src/` into notebooks) — leave them, don't add new ones
- `retained:`-marked helpers are test seams (compute_cfm_loss/compute_policy_loss, discounted_chunk_reward, assert_runtime) — deleting one breaks its consumer test first

## Anti-Patterns
- DO NOT reintroduce a standalone policy state encoder or target/EMA nets (`target_state_encoder`, `EMAProjection`, `ema_power`) — policy conditioning is raw-obs DCT tokens, no second copy
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
- One `WindowEncoder` class serves all time windows (policy state encoder, critic state + action encoders), each owned by its model; ASPO grads land on the policy head (incl. state branch + cond_proj)
- `compute_chunk_values` bootstraps truncated chunks from the full next window + stored `next.action`
