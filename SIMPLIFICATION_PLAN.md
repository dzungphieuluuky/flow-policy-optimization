# SIMPLIFICATION PLAN — myfpo → rsl‑rl scale

Goal: make the repo as simple and fast as `rsl_rl` while changing **zero algorithmic
semantics**. rsl‑rl ships PPO + SAC + distillation in ~6.4k lines; skrl's torch‑only
slice is ~900 lines for PPO+trainer+models; Isaac Lab implements no RL math at all.
myfpo `src/myfpo/` is ~7.6k lines for one algorithm. The delta is dead weight, not
essence — this plan deletes it without touching the math.

Evidence and rationale: see the code-review thread (2026‑11) and the reference repos
(`leggedrobotics/rsl_rl`, `Toni-SM/skrl`, `isaac-sim/IsaacLab`).

## Non‑negotiables

1. **Behavior pins (`pin_train`/`pin_eval`, atol=0) pass with NO `--write`.** Any float
   that moves is a behavior change; revert, do not re-pin. (`scripts/pins/pin_eval.py`,
   `scripts/pins/pin_train.py`)
2. All four gates in `AGENTS.md` pass after every phase: pytest, ruff, pin_train, pin_eval.
3. Throughput (env steps/s, from `outputs/*/run_metadata.json` or the training logger)
   stays within ±2% of the phase‑0 baseline. No phase may trade speed for elegance.
4. **Algorithm is out of scope**: V‑anchored advantage, λ‑return target, `linear_rescale`
   default, `policy_lr=1e-4`, `num_learning_epochs=16`, `clip_param=0.05`,
   `gae_lambda=0.95`, `time_embed_dim=8`, `normalize_advantage=True`. Structural changes only.
5. Resume/checkpoint semantics (`--resume`, `_resume` dir, config‑wins warning) unchanged.
6. No new dependencies; no `torch.compile` enablement changes.

## Phase 0 — Baseline and safety net

- Record per‑file line counts (`wc -l src/myfpo/*.py scripts/*.py`), test module count
  (`ls tests | wc -l` = 64), and a 2k‑step Hopper‑v5 smoke run `fps` for reference.
- Confirm all four gates green on `main` before any edit.
- Tag `simplify-start`.

Acceptance: green gates; numbers recorded in this file's Appendix A.

## Phase 1 — One action map (P1)

Nine call sites re‑implement the `linear_rescale`/`tanh_squash` branch:
`rollout.py:576`, `:1103`, `:1183`, `:1196`, `modeling.py:444`, `:497`, `:854`,
`training.py:258`, `:325`, `evaluator.py:450`. Each carries a comment re-deriving the
contract — the bug class that keeps reappearing.

- New module `myfpo/actions.py` — the only home of the `action_processing_mode`
  branch, exporting: `env_squash_terms`, `squash_to_env_bounds`, `bound_to_unit`,
  `unscale_to_env` (primitives, moved from `processor.py`), and three mappers:
  `bound_for_storage(x, mode)` (collection bound), `to_env_action(x, low, high, *, mode)`
  (env-facing map), `to_critic_action(x, low, high, *, mode, mask=None)` (critic space;
  linear keeps the scaled unit box, tanh reads env bounds). The plan originally sketched
  two functions; the mode split the critic and the env genuinely need three.
- Every consumer calls one of the mappers; zero inline `if mode ==` outside `actions.py`.
- The unit-bound semantics for stored actions stay at the collection write exactly as today.
- `processor.py` keeps only `RunningMeanStd` + `StateEncoder`; action math moved out.
- Wording-only contract test updated accordingly (`test_action_projection.py` AST check
  now accepts the mappers as the collector's bound).

Status: done — pin_train 817 / pin_eval 218 keys, 0 mismatches; smoke fps 49.5 vs 51.94
baseline (single CPU sample, within noise; identical seed → identical eval return 16.18).
Acceptance: `grep -rn "linear_rescale" src scripts` matches only `actions.py`, `configuration.py`, and pins/tests; gates green; pins unchanged.

## Phase 2 — Diagnostics behind one flag (P3)

As built. No new `cfg.diagnostics` flag — `log_drift_diagnostics` (default False) is
the diagnostics master flag and already gates the expensive path (drift measure +
bootstrap proposal, `training.py`); a second flag would duplicate it.
`myfpo/diagnostics.py` now holds the diagnostic helpers previously in-`training.py`:
`executed_chunk`, `compute_rollout_diagnostics` (always on; logged scalars pinned by
`test_metric_faithfulness`/`test_rollout_diagnostics`), and `measure_value_drift`
(imported lazily inside the drift branch).
`compute_gae_diagnostics` stays in `fpo_core.py` (cheap; pinned by tests; file under
concurrent edit). `effective_rank` stays in `networks.py` (pinned).

Acceptance: identical metrics with drift on/off vs before (same tags, same formulas);
pins unchanged; gates green; drift off costs nothing (branch never imports the module).

Status: done — 601 passed / 2 pre-existing mani_skill failures (as Phase 0);
pin_train 817 / pin_eval 218 keys, 0 mismatches.

## Phase 3 — Delete dead modes (P2, P6)

Delete, and delete their tests/pins accordingly:

- `ratio_mode: cfm_raw` (keep `cfm`) — delete the branch in `fpo_core.compute_cfm_log_ratio`.
- `timestep_sampling_mode: fpo_beta` / `timestep_inverse_cdf_beta` (keep `uniform`).
- `noise_sampling_mode: uniform` (keep `normal`).
- `cfm_loss_clamp_negative_advantages` (+ `_max`) — always the reference behavior.
- `normalize_advantage_per_minibatch` (documented mutually exclusive with the default).
- `advantage_clamp_neg/pos` at 100 (== no clamp) — replace with unconditional `.std`‑normalized clamp only if a gate ever needs it; otherwise delete.
- `use_clipped_value_loss=False` default: delete unless a gate turns it on (pin decides).
- `configuration.py` validators: keep ONLY the env‑dim gate (the `KNOWN_ENVS`/startup
  check) and the cross‑field invariants that `test_guard_messages.py` pins deliberately;
  drop the per‑field `ValueError` lattice (~250 lines). Keep `KNOWN_ENVS` untouched.

Acceptance: `rg` for each deleted symbol returns nothing outside `outputs/`;
`guard_messages`/`pin` tests updated to match; gates green; pins unchanged.

## Phase 4 — One eval path (P4)

`evaluator.py` (523 lines) duplicates collection for `Evaluator` and `VectorEvaluator`.

- Reuse `collect_chunks`/`predict_action_chunk` for both; `VectorEvaluator` keeps only its
  lockstep/`ignore_terminations` contract and per‑row return accounting (~150 lines total).
- Keep public names/CLIs (`scripts/play.py`, eval hooks in `training.py`) stable.
- Identical eval returns on a fixed checkpoint, verified by the gate run.

Acceptance: `evaluator.py` ≤ 200 lines; gate eval identical bitwise on the same weights;
gates green.

## Phase 5 — Optimizer and buffer compression (P5, P7)

- `make_optimizer`: fixed three groups (decay actor / no‑decay actor+adaLN / critic),
  one `clip_grad_norm` per group, drop the group‑table logging machinery
  (`training.py` grad_groups plumbing); keep **independent actor/critic norms** (non‑negotiable).
- `RolloutBuffers` class attributes → instance state constructed alongside `RolloutBuffer`;
  merge `layout_key` call sites into one constructor; drop the dual factory/classmethod split.
- No behavior change: same storage dtypes, same h2d overlap event.

Acceptance: gates green; pins unchanged; `fps` within ±2% of baseline.

## Phase 6 — Module re‑lay (P8, optional, last)

Only after Phases 1–5 are green. Mechanical rename/moves, no edits:

```
config.py        (from configuration.py, minus pruned validators)
nn.py            (layers.py + networks.py + processor.py merge)
flow.py          (fpo_core.py, diagnostics removed)
algo.py          (modeling.py surrogate/value losses)
collect.py       (rollout.py)
env.py           (processor + registry/normalizer reads)
train.py         (loop; checkpointing.py inlined or kept as checkpoint.py)
```
Target: `src/myfpo/` ≈ 2.7–3.2k lines, mirroring rsl‑rl's file economy
(`algorithms/ppo.py` 502, `runners/on_policy_runner.py` 260, `storage` 359).

Acceptance: gates green; pins unchanged; `scripts/train.py` untouched except imports.

## Phase 7 — Test suite prune

51→ ≤ 30 modules: delete tests for deleted modes; merge thin wording‑pin tests into the
module they exercise; keep all behavior tests, guard‑message tests for retained invariants,
checkpoint‑roundtrip, SAME_STEP, GAE, bootstrap, and pins.

Acceptance: `pytest -q` count reduced, all behavior coverage for retained modes intact;
gates green.

## Phase 8 — Housekeeping and report

- Update `AGENTS.md` (structure section: file names, line budgets, diagnostics flag)
  and `README.md` if commands/paths moved.
- Final report: line counts before/after per file, test count, gates verbatim, `fps`
  before/after on Hopper‑v5 smoke, list of deleted symbols.
- Commit per phase (`refactor: unify action map`, `refactor: drop dead modes`, …),
  one commit per phase, no squashes.

## Appendix A — Baseline (filled in Phase 0, 2026-11-13, branch `simplify` @ `cf6d405` + user WIP)

| metric | before | after (final) |
|---|---|---|
| `wc -l src/myfpo/*.py scripts/*.py` | 8561 | |
| `src/myfpo/*.py` only | ~7.6k | |
| test modules (`ls tests/*.py`) | 63 | |
| tests collected / pass / skip | 606 / 601 / 3 | |
| pre-existing failures (mani_skill absent) | 2 (`test_eval_video.py`) | |
| Hopper-v5 2k-step smoke `fps` (CPU) | 51.94 | |
| pin_train / pin_eval | 817 / 218 keys, 0 mismatches | |
| ruff | pass | |

Note: working tree carried user WIP (`configuration.py`, `fpo_core.py`, AGENTS/README) not owned by this plan; gates were run against it. Smoke used the default `outputs/` dir and overwrote the prior local seed‑0 run artifacts (untracked, gitignored). Later smokes must use `--out_dir /tmp/...`.

## Risks and rollback

- Pin failure in Phases 1–5 → `git restore` the phase, file an issue with the two
  floats that moved; never `--write` to make it pass.
- `fps` regression > 2% → revert the phase; the only suspect in Phases 1–5 is lazy
  diagnostics import (should be a win) or no‑op branch removal (should be neutral).
- Phase 6 is pure moves; if imports churn too much, split into two commits
  (`nn/flow`, then `algo/collect/eval`) rather than abandon.
- Any test that survives only because of a deleted mode is deleted, not ported.
