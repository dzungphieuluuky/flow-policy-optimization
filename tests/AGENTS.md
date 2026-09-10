# AGENTS — tests

## Overview
263 test functions across 22 test_*.py files, 1:1 per-module gates + pipeline invariants. Full suite ~100s CPU.

## Structure
```
test_analysis.py (13)      analyze_results IQM/bootstrap
test_baselines.py (7)      skrl PPO/SAC drivers
test_calibration.py (10)   budget gate (build_legacy_run_matrix consumer)
test_checkpoint.py (5)     atomic ckpt (uses test-local resume_worker helpers)
test_collector.py (9)      collect_chunks (fake_env)
test_config.py (15)        dataclasses + 45/TINY matrices (module-reload tests)
test_config_yaml_parity.py (1) YAML keys ⊆ dataclass fields (deslop gate)
test_dct.py (12)           DCT/IDCT (imports discounted_chunk_reward)
test_einx_migration.py (27)      torch↔einx 1e-6 parity (QUARANTINE per ledger)
test_evaluator.py (11)     Evaluator 3-col CSV
test_fpo_core.py (9)       ASPO/CFM (imports euler_sample, PolicyFlowHead)
test_gae.py (6)            chunk_gae
test_inference.py (5)      euler/decode (notebook-only surface)
test_joint_flow.py (6)     JointFlowTransformer (blocks JFT-shape changes)
test_networks.py (21)      trunk/spectral/EMA + Epps-Pulley SIGReg
test_pipeline_cpu.py (7)   end-to-end invariants: CSV + 45 + wandb
test_runner_mfpo.py (9)    train_run
test_sweep.py (20)         scan_runs/run_pending (imports RunSpec, RUN_MATRIX)
test_trainer.py (9)        assemble_update (imports ChunkRecord, MFPOAborted)
test_wandb_logging.py (30)      default-on gate (imports FakeGymEnv)
test_todo3_characterization.py (13)  pre-refactor behavior pins (fn-consolidation)
test_todo4_characterization.py (16)  pre-refactor behavior pins (fn-consolidation)
fixtures/fake_env.py       FakeGymEnv shared fixture
resume_worker.py           subprocess worker for checkpoint resume determinism
```

## Where to Look
- Pipeline lock: `test_pipeline_cpu.py` (CSV schema, 45/TINY cardinality, wandb default-on/disabled)
- Parity locks: `test_einx_migration.py`, `test_config_yaml_parity.py`, `test_dct.py` (denormalize atol 1e-6)
- Reload-pattern tests: `test_config.py` (MFPO_TINY_MATRIX swap), `test_pipeline_cpu.py` (tiny reload)
- Characterization guards: `test_todo3_characterization.py`, `test_todo4_characterization.py` (pre-refactor pins)

## Conventions
- Test-only seams are load-bearing: ChunkRecord/MFPOAborted/policy heads/euler_sample/build_legacy_run_matrix imports — deleting src helpers breaks these first
- Deleting a tested src fn requires deleting/rewriting its test (deslop rule, verified 3×)
- Fixed tiny counts per file; full-suite count recomputed on test add/remove (never hardcoded)

## Anti-Patterns
- DO NOT delete failing tests to pass — fix src or amend plan
- DO NOT touch test_einx_migration.py (quarantine) or analysis stats tests without amendment
- DO NOT weaken pipeline invariants (CSV cols, 45, is_stub) silently
- DO NOT remove characterization tests (test_todo3/4) — they are pre-refactor behavior pins

## Commands
```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m pytest tests/test_pipeline_cpu.py -v
```
