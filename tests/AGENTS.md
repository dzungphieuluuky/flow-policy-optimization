# AGENTS — tests

## Overview
292 collected items across 23 test_*.py files. Full suite ~2min CPU. Counts below are collected items (`pytest --collect-only`), recompute on add/remove — never hardcode.

## Structure
```
test_networks.py (29)        encoder/conditioner/DiT + V(s) Critic
test_wandb_logging.py (27)   default-on gate (imports FakeGymEnv)
test_einx_migration.py (27)  torch↔einx 1e-6 parity (QUARANTINE per ledger)
test_sweep.py (21)           scan_runs/run_pending (imports RunSpec, RUN_MATRIX)
test_modeling_sjfpo.py (18)  SJFPOPolicy modules/inference/forward/hub roundtrip
test_config.py (17)          dataclasses + 45/TINY matrices (module-reload tests)
test_todo3_characterization.py (15) behavior pins (evaluator/sweep/collector goldens)
test_dct.py (14)             DCT/IDCT roundtrip + Parseval + DC=mean·√H (no scipy)
test_analysis.py (14)        analyze_results IQM/bootstrap
test_fpo_core.py (13)        ASPO/CFM (imports sample_action, SJFPOFlowModel)
test_utils.py (12)           env detect + paths + notebook builder + bf16/compile
test_collector.py (12)       collect_chunks incl. NEVER-hardcode-horizon guards (fake_env)
test_evaluator.py (10)       Evaluator 3-col CSV
test_transition_v4.py (9)    Transition 8-key/shape/dtype + continuity + no-aux-head lock
test_todo4_characterization.py (9) behavior pins (GAE/trainer/baselines goldens)
test_device.py (9)           device placement + cpu fp32 inference
test_configuration_sjfpo.py (8) LeRobot SJFPOConfig registry/parity/validation
test_baselines.py (7)        skrl PPO/SAC drivers + tiny smokes
test_gae.py (6)              compute_advantages vs zero-shared-code brute-force oracle
test_policy_checkpoint.py (5) Hub save/load roundtrip
test_pipeline_cpu.py (5)     end-to-end invariants: CSV + 45 + wandb
test_processor_sjfpo.py (4)  preprocessor pair shares live normalizer
test_config_yaml_parity.py (1) YAML keys ⊆ dataclass fields (deslop gate)
fixtures/fake_env.py       FakeGymEnv shared fixture
```

## Where to Look
- Pipeline lock: `test_pipeline_cpu.py` (CSV schema, 45/TINY cardinality, wandb default-on/disabled)
- Parity locks: `test_einx_migration.py`, `test_config_yaml_parity.py`, `test_dct.py` (denormalize atol 1e-6)
- Transition lock: `test_transition_v4.py` (exact 8-key set, `n_action_steps == len(next.reward)`, bool dtypes, chunk continuity)
- Horizon guards: `test_collector.py` NEVER-hardcodes n_obs_steps/horizon (uses policy `SJFPOConfig` defaults); executed steps are caller-passed (`N_STEPS = 8` local)
- Reload-pattern tests: `test_config.py` (SJFPO_TINY_MATRIX swap), `test_pipeline_cpu.py` (tiny reload)
- Characterization guards: `test_todo3_characterization.py`, `test_todo4_characterization.py` (behavior pins)

## Conventions
- Test-only seams are load-bearing: Transition/SJFPOAborted/policy heads/sample_action/build_run_matrix imports — deleting src helpers breaks these first
- Deleting a tested src fn requires deleting/rewriting its test (deslop rule, verified 3×)
- Collected-item counts per file (parametrize-inflated); full-suite count recomputed on test add/remove (never hardcoded)
- RED-GREEN headers: most files carry a TDD docstring (`RED phase` / `RED until implemented`) — historical, suite is green
- `sys.path.insert(src)` boilerplate per file (alt: `PYTHONPATH=src`); `conftest.py` autouse-mocks wandb (no test touches network)

## Anti-Patterns
- DO NOT delete failing tests to pass — fix src or amend plan
- DO NOT touch test_einx_migration.py (quarantine) or analysis stats tests without amendment
- DO NOT weaken pipeline invariants (CSV cols, 45, is_stub) silently
- DO NOT remove characterization tests (test_todo3/4) — they are behavior pins

## Commands
```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m pytest tests/test_pipeline_cpu.py -v
```
