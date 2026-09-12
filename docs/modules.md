# Module Guide — src/sjfpo & Tests

Use this as a jump table. Each row: what the file owns, the key symbols, and the test that locks it.

## `src/sjfpo` — Production (MuJoCo-only, flat Box, fp32)

| Module | Owns | Key symbols | Thread to follow |
|---|---|---|---|
| `config.py` | Flags, run matrices, budgets, algorithm dataclasses | `SJFPOConfig / PPOConfig / SACConfig`, `RUN_MATRIX` (45), `build_filtered_matrix` | Change sweep size → `--methods/--envs/--seeds` or edit axes |
| `runner.py` | SJFPO train entry | `train_run`, `ChunkPolicy`, `SjfpoEvalWrapper` | `test_runner_sjfpo.py` + `test_pipeline_cpu.py` |
| `trainer.py` | FPO++ ASPO + value update | `SJFPOTrainer.assemble_update`, `SJFPOAborted` | `test_trainer.py` NaN + param count |
| `networks.py` | Encoder + spectral + DiT flow + Epps-Pulley SIGReg | `StateEncoder`, `SpectralEncoder(top_p=0.3→K=5, spectral_dim=64, DCT→low-K→Linear, no norm)`, `SJFOPolicyModel` (DiT-8 adaLN-Zero), `Critic` (2 cross-attention blocks) V(s,a), `sigreg_loss` | `test_networks.py` |
| `dct_utils.py` | DCT orthonormal action chunks | `dct`, `idct`, `discounted_chunk_reward` | `test_dct.py` roundtrip `1e-5` |
| `evaluator.py` | Metrics | `Evaluator` → 3-col CSV `[episode,return,length]` | `test_evaluator.py` schema + `test_pipeline_cpu.py` |
| `baselines.py` | skrl PPO/SAC | `ppo_train_run` / `sac_train_run`, shared `GaussianPolicyImpl`, `seed_and_wrap_env`, `mean_action_eval_policy`, `baseline_train_loop`, `FrameStackWrapper` | `test_baselines.py` provenance |
| `fpo_core.py` | FPO++ core math (CFM, ratios, ASPO, Euler) | `aspo_surrogate`, `policy_ratio`, `euler_sample` descending 1→0, `total_update_loss` | `test_fpo_core.py` |
| `gae.py`, `collector.py`, `checkpointing.py` | Chunk rollout + GAE + checkpoints | `chunk_gae`, `collect_chunks`/`Transition`, `save/load_checkpoint` | `test_gae.py`, `test_collector.py`, `test_checkpoint.py` |
| `sweep.py` | Dispatch | `scan_runs`, `run_pending`, `flush_pending_runs`, `format_run_directory_name` | `test_sweep.py` order + resume |
| `analysis.py` | Stats + plots | `analyze_results`, `bootstrap_ci` (`n_boot=2000`), `compute_iqm`, `is_superior`, `write_summary_csv` (stats: do not touch without plan amendment) | `test_analysis.py` |
| `wandb_logging.py` | Default-on wandb gate (REQUIRED, hard import) | `wandb_disabled`, `init_wandb_run_for_method` (->bool), `log_metrics`, `log_eval_details`, `log_eval_summary` (shared C4 home), `finish_run` — 6 fns, SDK-direct (only file that imports wandb) | `test_wandb_logging.py` |
| `utils.py` | Shared pure helpers | `safe_float_value` | imported by trainer |

## `tests` — What Locks What (273 green)

- `test_pipeline_cpu.py` (7 tests): 3-col CSV schema, RUN_MATRIX axes,
  wandb gate. **Never touch.**
- `test_config.py`: 45-run matrix asserts. `test_einx_migration.py` (27):
  migration scaffolding — quarantine only.
- `tests/fixtures/` — `fake_env.py`, `synthetic_curves*.py` deterministic
  fixtures. Never downloads real data.

## `scripts` / `configs`

| Path | Usage |
|---|---|
| `scripts/train.py` | `... --method sjfpo --env Hopper-v5 --total-steps 200` (single) or `--matrix --methods … --envs … --seeds …` (sweep) + `--tiny`. Exits 0/1/2/4. |
| `scripts/build_notebook.py` | `python scripts/build_notebook.py` → both ipynbs (6 cells each, validated, no source inlined); `--spectral-only` / `--legacy-only`. Never hand-edit ipynbs. |
| `scripts/verify_deps.sh` | Pinned dep check. |
| `configs/{shared,sjfpo,ppo,sac}.yaml` | YAML defaults mirroring `config.py` dataclasses (pyyaml `safe_load` only). |

## Reading Order

1. New to the SJFPO path: `config.py` → `runner.py::train_run` →
   `networks.py` (encoders→spectral) → `dct_utils.py` → `trainer.py` →
   `evaluator.py` → `sweep.py`/`analysis.py`.
2. Debugging spectral: jump to `networks.py::SpectralEncoder` — order is
   DCT→low-K→Linear, no normalization.
3. Touching sweep dispatch: `sweep.py` — read `test_sweep.py` order + resume invariants before changing anything.

## Naming Standard (HuggingFace-cited)

Per fn-consolidation N1. Function, parameter, and attribute names follow the
HuggingFace convention (transformers / diffusers / accelerate):

- Verb-led snake_case functions: `get_constant_schedule`, `compute_f1`
  (transformers) — helpers here are verb-led too (`run_eval_and_record`,
  `check_divergence`, `log_eval_summary`).
- CapWords classes with role suffixes: `BertModel` / `BertConfig`
  (transformers), `DDPMScheduler` (diffusers) — mirrored here
  (`SJFOPolicyModel`, `Critic`, `SJFPOTrainer`).
- File mirrors class: `modeling_bert.py` → `BertModel`; here `sweep.py` →
  `SweepError` / `ScanResult`, `trainer.py` → `SJFPOTrainer` / `SJFPOAborted`.
- UPPER_SNAKE constants: `CHECKPOINT_FILE`, `BASIS_CACHE`.
- `_` prefix banned on this repo's public surface: a leading underscore marks
  private API, but every helper here is imported across modules or asserted by
  tests, so all defs are public (no `def _[a-z]` in `src/sjfpo` / `scripts`;
  test `_` helpers keep pytest convention).
- Stated deviation: `networks.py` stays one file (trunk + spectral + flow +
  SIGReg) instead of HF-style one-class-per-file — the single-home gate wins.
