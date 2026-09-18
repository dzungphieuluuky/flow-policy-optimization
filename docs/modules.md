# Module Guide — src/sjfpo & Tests

Use this as a jump table. Each row: what the file owns, the key symbols, and the test that locks it.

## `src/sjfpo` — Production (MuJoCo-only, flat Box, fp32)

| Module | Owns | Key symbols | Thread to follow |
|---|---|---|---|
| `config.py` | Flags, run matrices, budgets, PPO/SAC dataclasses | `PPOConfig / SACConfig`, `RUN_MATRIX` (45) (SJFPO hyperparams live in `configuration_sjfpo.py::SJFPOConfig` only) | Change sweep size → `METHODS/ENVS/SEEDS` env axes or edit axes |
| `modeling_sjfpo.py` | SJFPO policy (FPO++ ASPO + value update) | `SJFPOPolicy.forward`, `SJFPOAborted` | `test_modeling_sjfpo.py` NaN + param count |
| `networks.py` | WindowEncoder + conditioner + DiT flow + V(s) critic | `WindowEncoder` (shared class, owned instances), `StateConditioner` (raw window queries DCT bins, Flamingo-style → `[B,D]`), `SJFPOFlowModel` (`x_t, t, condition`, DiT adaLN-Zero, action tokens only, `c = time + condition`), `CondMLP`, `SwiGLU`, `Critic` V(s) | `test_networks.py` |
| `dct_utils.py` | DCT orthonormal action chunks | `dct`, `idct`, `discounted_chunk_reward` | `test_dct.py` roundtrip `1e-5` |
| `evaluator.py` | Metrics | `Evaluator` → 3-col CSV `[episode,return,length]` | `test_evaluator.py` schema + `test_pipeline_cpu.py` |
| `baselines.py` | skrl PPO/SAC | `ppo_train_run` / `sac_train_run`, shared `GaussianPolicyImpl`, `seed_and_wrap_env`, `mean_action_eval_policy`, `baseline_train_loop`, `FrameStackWrapper` | `test_baselines.py` provenance |
| `fpo_core.py` | FPO++ core math (CFM, ratios, ASPO, sampling) | `compute_aspo_surrogate`, `compute_policy_ratio`, `sample(mode=euler)` ascending 0→1, losses summed inline in trainer | `test_fpo_core.py` |
| `gae.py`, `collector.py`, `checkpointing.py` | Chunk rollout + GAE + checkpoints | `compute_advantages`, `collect_chunks`/`collect_chunks_vec`/`Transition`, `save/load_policy_checkpoint` (Hub config + safetensors) | `test_gae.py`, `test_collector.py`, `test_policy_checkpoint.py` |
| `sweep.py` | Dispatch | `scan_runs`, `run_pending`, `flush_pending_runs`, `format_run_directory_name` | `test_sweep.py` order + resume |
| `analysis.py` | Stats + plots | `analyze_results`, `bootstrap_ci` (`n_boot=2000`), `compute_iqm`, `is_superior`, `write_summary_csv` (stats: do not touch without plan amendment) | `test_analysis.py` |
| `wandb_logging.py` | Default-on wandb gate (REQUIRED, hard import) | `wandb_disabled`, `init_wandb_run_for_method` (->bool), `log_metrics`, `log_eval_summary`, `finish_run` — 5 fns, SDK-direct (only file that imports wandb) | `test_wandb_logging.py` |
| `utils.py` | Shared pure helpers | `resolve_device`, `resolve_mixed_precision`, `maybe_compile` | device/dtype plumbing |

## `tests` — What Locks What (300 green)

- `test_pipeline_cpu.py` (5 tests): 3-col CSV schema, RUN_MATRIX axes,
  wandb gate. **Never touch.**
- `test_config.py`: 45-run matrix asserts. `test_einx_migration.py` (27):
  migration scaffolding — quarantine only.
- `tests/fixtures/` — `fake_env.py`, `synthetic_curves*.py` deterministic
  fixtures. Never downloads real data.

## `scripts` / `configs`

| Path | Usage |
|---|---|
| `scripts/train.py` | Single run: `... --method sjfpo --env Hopper-v5 --seed 0 --total_steps 2000` (underscores, draccus). Sweeps live in `scripts/sweep.sh` (`METHODS/ENVS/SEEDS` axes, `--tiny`). Exits 0/1/2/4. |
| `scripts/build_notebook.py` | `python scripts/build_notebook.py` → Kaggle ipynb (33 cells: %%writefile sources + offline pip + !train.py, structure-checked; jupytext percent-format, no nbformat). Never hand-edit ipynbs. |
| `scripts/verify_deps.sh` | Pinned dep check. |
| `configs/{shared,sjfpo,ppo,sac}.yaml` | YAML defaults mirroring `config.py` dataclasses (pyyaml `safe_load` only). |

## Reading Order

1. New to the SJFPO path: `configuration_sjfpo.py` → `modeling_sjfpo.py::SJFPOPolicy` →
   `networks.py` (encoders→spectral) → `dct_utils.py` → `modeling_sjfpo.py` →
   `evaluator.py` → `sweep.py`/`analysis.py`.
2. Debugging conditioning: jump to `networks.py::StateConditioner` — order is
   WindowEncoder(states) queries DCT bins → residual cross-attn → mean-pool.
3. Touching sweep dispatch: `sweep.py` — read `test_sweep.py` order + resume invariants before changing anything.

## Naming Standard (HuggingFace-cited)

Per fn-consolidation N1. Function, parameter, and attribute names follow the
HuggingFace convention (transformers / diffusers / accelerate):

- Verb-led snake_case functions: `get_constant_schedule`, `compute_f1`
  (transformers) — helpers here are verb-led too (`check_divergence`,
  `log_eval_summary`).
- CapWords classes with role suffixes: `BertModel` / `BertConfig`
  (transformers), `DDPMScheduler` (diffusers) — mirrored here
  (`SJFPOFlowModel`, `Critic`, `SJFPOPolicy`).
- File mirrors class: `modeling_bert.py` → `BertModel`; here `sweep.py` →
  `SweepError` / `ScanResult`, `modeling_sjfpo.py` → `SJFPOPolicy` / `SJFPOAborted`.
- UPPER_SNAKE constants: `BASIS_CACHE`, `OPTIMIZER_FILE`, `COUNTERS_FILE`.
- `_` prefix banned on this repo's public surface: a leading underscore marks
  private API, but every helper here is imported across modules or asserted by
  tests, so all defs are public (no `def _[a-z]` in `src/sjfpo` / `scripts`;
  test `_` helpers keep pytest convention).
- Stated deviation: `networks.py` stays one file (state encoders + spectral + flow)
  instead of HF-style one-class-per-file — the single-home gate wins.
