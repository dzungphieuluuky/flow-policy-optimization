# Module Guide — src/mfpo & Tests

Use this as a jump table. Each row: what the file owns, the key symbols, and the test that locks it.

## `src/mfpo` — Production (MuJoCo-only, flat Box, fp32)

| Module | Owns | Key symbols | Thread to follow |
|---|---|---|---|
| `config.py` | Flags, run matrices, budgets, algorithm dataclasses | `MFPOConfig / PPOConfig / SACConfig`, `RUN_MATRIX` (45), `build_filtered_matrix`, `BUDGET_TOTAL_HOURS=120` | Change sweep size → `--methods/--envs/--seeds` or edit axes |
| `runner.py` | MFPO train entry | `train_run`, `PolicyCallable`, `MpoEvalWrapper` | `test_runner_mfpo.py` + `test_pipeline_cpu.py` |
| `trainer.py` | FPO++ ASPO + value update | `MFPOTrainer.assemble_update`, `MFPOAborted` | `test_trainer.py` NaN + param count |
| `networks.py` | Trunk + spectral + flow + VICReg | `ContextTrunk`, `SpectralTokens(K=5, Dk=64, hann→rFFT→cat→frozen→Linear)`, `FlowHeadBase`/`PolicyFlowHead`/`AuxStateFlowHead`, `ValueCritic`, `EMAProjection`, `JointFlowTransformer`, `PerBranchFlowLoss`, `sigreg_losses` | `test_networks.py` + `test_joint_flow.py` |
| `dct_utils.py` | DCT orthonormal + joint target norm | `dct`, `idct`, `normalize_joint_targets`, `denormalize_joint_targets`, frozen per-branch stats | `test_dct.py` roundtrip + invariance `atol 1e-6` |
| `inference.py` | MFPO ODE decode | `euler_integrate_joint` ascending 0→1 (16 train / 8 eval), `decode_joint` with IDCT + gripper `{0,1}` | `test_inference.py` identity `1e-4` |
| `evaluator.py` | Metrics | `Evaluator` → 3-col CSV `[episode,return,length]` | `test_evaluator.py` schema + `test_pipeline_cpu.py` |
| `baselines.py` | skrl PPO/SAC | `ppo_train_run` / `sac_train_run`, shared `GaussianPolicyImpl`, `seed_and_wrap_env`, `mean_action_eval_policy`, `baseline_train_loop`, `FrameStackWrapper` | `test_baselines.py` provenance |
| `fpo_core.py` | Legacy FPO math | `aspo_surrogate`, `euler_sample` descending 1→0 | `test_fpo_core.py` |
| `gae.py`, `collector.py`, `checkpointing.py` | Legacy RL loop | GAE, rollout collection, atomic checkpoint I/O | `test_gae.py`, `test_collector.py`, `test_checkpoint.py` |
| `sweep.py` | Dispatch | `scan_runs`, `run_pending`, `flush_pending_runs`, `format_run_directory_name` | `test_sweep.py` order + resume |
| `analysis.py` | Stats + plots | `analyze_results`, `bootstrap_ci` (`n_boot=2000`), `compute_iqm`, `is_superior` (stats: do not touch without plan amendment) | `test_analysis.py` |
| `calibration.py` | Budgets + gating | `calibrate_and_gate`, `project_run_hours`, `build_fallback_ladder`, `render_markdown`, `GateReport` | `test_calibration.py` |
| `wandb_logging.py` | Default-on wandb gate (REQUIRED, hard import) | `wandb_disabled`, `init_wandb_run_for_method` (->bool), `log_metrics`, `log_eval_details`, `log_eval_summary` (shared C4 home), `finish_run` — 6 fns, SDK-direct (only file that imports wandb) | `test_wandb_logging.py` |
| `mjx_adapter.py` | Gated MJX stub | `is_stub=True` default, `MFPO_USE_MJX=1` gate | `test_pipeline_cpu.py` stub invariants |
| `utils.py` | Shared pure helpers | `safe_float_value` | imported by trainer |

## `tests` — What Locks What (263 green)

- `test_pipeline_cpu.py` (9 invariants): 3-col CSV schema, RUN_MATRIX axes,
  `is_stub` surfaces, wandb gate. **Never touch.**
- `test_config.py`: 45-run matrix asserts. `test_einx_migration.py` (27):
  migration scaffolding — quarantine only.
- `tests/fixtures/` — `fake_env.py`, `synthetic_curves*.py` deterministic
  fixtures. Never downloads real data.

## `scripts` / `configs`

| Path | Usage |
|---|---|
| `scripts/train.py` | `... --method mfpo --env Hopper-v5 --total-steps 200` (single) or `--matrix --methods … --envs … --seeds …` (sweep) + `--tiny`. Exits 0/1/2/4. |
| `scripts/build_notebook.py` | `python scripts/build_notebook.py` → both ipynbs (61 + 47 cells, validated); `--spectral-only` / `--legacy-only`. Never hand-edit ipynbs. |
| `scripts/verify_deps.sh` | Pinned dep check. |
| `configs/{shared,mfpo,ppo,sac}.yaml` | YAML defaults mirroring `config.py` dataclasses (pyyaml `safe_load` only). |

## Reading Order

1. New to the MFPO path: `config.py` → `runner.py::train_run` →
   `networks.py` (trunk→spectral) → `dct_utils.py` → `trainer.py` →
   `inference.py` → `evaluator.py` → `sweep.py`/`analysis.py`.
2. Debugging spectral: jump to `networks.py::SpectralTokens` — order is
   Hann→rFFT→cat→frozen→Linear.
3. Touching budgets/sweep: `calibration.py` + `sweep.py` — read
   `test_calibration.py` for the `BUDGET_TOTAL_HOURS` gate before changing anything.

## Naming Standard (HuggingFace-cited)

Per fn-consolidation N1. Function, parameter, and attribute names follow the
HuggingFace convention (transformers / diffusers / accelerate):

- Verb-led snake_case functions: `get_constant_schedule`, `compute_f1`
  (transformers) — helpers here are verb-led too (`setup_training_env`,
  `run_eval_and_record`, `check_divergence`, `log_eval_summary`).
- CapWords classes with role suffixes: `BertModel` / `BertConfig`
  (transformers), `DDPMScheduler` (diffusers) — mirrored here
  (`JointFlowTransformer`, `PerBranchFlowLoss`, `MFPOTrainer`, `GateReport`).
- File mirrors class: `modeling_bert.py` → `BertModel`; here `sweep.py` →
  `SweepError` / `ScanResult`, `calibration.py` → `GateReport`.
- UPPER_SNAKE constants: `BUDGET_TOTAL_HOURS`, `CHECKPOINT_FILE`, `BASIS_CACHE`.
- `_` prefix banned on this repo's public surface: a leading underscore marks
  private API, but every helper here is imported across modules or asserted by
  tests, so all defs are public (no `def _[a-z]` in `src/mfpo` / `scripts`;
  test `_` helpers keep pytest convention).
- Stated deviation: `networks.py` stays one file (trunk + spectral + flow +
  SIGReg) instead of HF-style one-class-per-file — the single-home gate wins.
