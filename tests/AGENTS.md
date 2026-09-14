# AGENTS — tests

## Overview
289 collected items across 22 test_*.py files. Full suite ~103s CPU. Counts below are collected items (`pytest --collect-only`), recompute on add/remove — never hardcode.

## Structure
```
test_wandb_logging.py (30)   default-on gate (imports FakeGymEnv)
test_einx_migration.py (27)  torch↔einx 1e-6 parity (QUARANTINE per ledger)
test_networks.py (24)        encoder/spectral/DiT + cross-attn Critic + Epps-Pulley SIGRef
test_sweep.py (20)           scan_runs/run_pending (imports RunSpec, RUN_MATRIX)
test_todo4_characterization.py (9) behavior pins (GAE/trainer/baselines goldens)
test_config.py (17)          dataclasses + 45/TINY matrices (module-reload tests)
test_dct.py (14)             DCT/IDCT roundtrip + Parseval + DC=mean·√H (no scipy)
test_modeling_sjfpo.py (13)  SJFPOPolicy modules/inference/forward/hub roundtrip
test_todo3_characterization.py (15) behavior pins (evaluator/sweep/collector goldens)
test_analysis.py (13)        analyze_results IQM/bootstrap
test_evaluator.py (11)       Evaluator 3-col CSV
test_transition_v4.py (11)   Transition 9-key/shape/dtype + continuity + next.action chaining + no-aux-head lock
test_runner_sjfpo.py (9)     train_run incl. CLI smoke
test_fpo_core.py (9)         ASPO/CFM (imports euler_sample, SJFOPolicyModel)
test_collector.py (9)        collect_chunks incl. NEVER-hardcode-horizon guards (fake_env)
test_pipeline_cpu.py (7)     end-to-end invariants: CSV + 45 + wandb
test_baselines.py (7)        skrl PPO/SAC drivers + tiny smokes
test_gae.py (6)              compute_advantages vs zero-shared-code brute-force oracle
test_configuration_sjfpo.py (6) LeRobot SJFPOConfig registry/parity/validation
test_config_yaml_parity.py (1) YAML keys ⊆ dataclass fields (deslop gate)
fixtures/fake_env.py       FakeGymEnv shared fixture
```

## Where to Look
- Pipeline lock: `test_pipeline_cpu.py` (CSV schema, 45/TINY cardinality, wandb default-on/disabled)
- Parity locks: `test_einx_migration.py`, `test_config_yaml_parity.py`, `test_dct.py` (denormalize atol 1e-6)
- Transition lock: `test_transition_v4.py` (exact 9-key set incl. next.action, `n_action_steps == len(next.reward)`, bool dtypes, chunk continuity, next.action chaining)
- Horizon guards: `test_collector.py` NEVER-hardcodes n_obs_steps/horizon (uses policy `SJFPOConfig` defaults); executed steps are caller-passed (`N_STEPS = 8` local)
- Reload-pattern tests: `test_config.py` (SJFPO_TINY_MATRIX swap), `test_pipeline_cpu.py` (tiny reload)
- Characterization guards: `test_todo3_characterization.py`, `test_todo4_characterization.py` (behavior pins)

## Conventions
- Test-only seams are load-bearing: Transition/SJFPOAborted/policy heads/euler_sample/build_run_matrix imports — deleting src helpers breaks these first
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
