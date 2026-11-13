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

As built.

Deleted knobs (default reachable only as no-ops; each deleted with its tests/guard entries):
`ratio_mode: cfm_raw` (RATIO_MODES gone), `timestep_sampling_mode: fpo_beta` +
`timestep_inverse_cdf_beta`, `noise_sampling_mode: uniform`,
`cfm_loss_clamp_negative_advantages(_max)` (collapses into the cfm_clamp bound at the
reference default), `normalize_advantage_per_minibatch` (rollout-level kept),
`advantage_clamp_neg/pos` (fixed at 100), `use_clipped_value_loss` + `value_clip_param`
(default-off; took the `old_values` plumbing with it).
Also deleted: the taken-chunk critic read (`q_current_chunk`) — its only consumer was
`old_values`/clipped value loss; bootstrap_values no longer reads the critic there.
Deliberate deviation: the plan's "drop the per-field ValueError lattice" was NOT done
wholesale — validators for deleted fields are gone; surviving fields keep theirs
(they catch real misconfig and are pinned by test_guard_messages).
Tests: `test_ratio_modes.py`, `test_timestep_law.py` deleted; guard tables pruned
(73→66 cases); `test_advantage_normalization` keeps only the rollout-level arm;
`test_config_surface` pins the single normal noise law; `test_fpo_baseline` simplified.

Acceptance: `rg` for each deleted symbol returns nothing outside `outputs/`;
gates green; pins unchanged.

Status: done — 577 passed / 2 pre-existing mani_skill failures; pin_train 817 /
pin_eval 218 keys, 0 mismatches (seed-identical eval return 16.1835); ruff clean;
smoke fps 46.9 vs 51.94 baseline (single CPU sample).

## Phase 4 — One eval path (P4)

As built. The plan's premise was wrong: the duplication between `Evaluator`
(gymnasium, single-env) and `VectorEvaluator` (ManiSkill GPU) is two env-stepping
loops — unifiable only by a mode flag (forbidden) or a behavior-risky rewrite;
AST measurement showed ~18 lines of duplicated boilerplate (budget init,
`summarize` delegate), everything else is genuinely different.
`collect_chunks` reuse is infeasible (it stores training batches with autoreset).

As-built changes: `_BaseEvaluator` extracted (budget init `_check_eval_counts` +
`summarize` delegate, shared by both evaluators). No env-loop rewrite; behavior
identical — a seeded 2000-step Hopper run produced a **byte-identical**
`eval_history.csv` before and after (compare `/tmp/myfpo_smoke/p4_{pre,post}`).

Acceptance re-scoped: no `≤200 lines` claim — `evaluator.py` is 518 lines, still
~230 lines of pinned jerk/CSV/stats helpers used externally (pins, multiseed,
training). If a shared env protocol (Isaac Lab's `VecEnv` adapter pattern) is
ever wanted, that is a separate, behavior-risking decision — it changes eval seeding
for gymnasium envs and is out of scope here.

Status: done — gates green (577 pass / 2 pre-existing mani_skill failures);
pins 817/218, 0 mismatches; smoke CSV byte-identical.

## Phase 5 — Optimizer and buffer compression (P5, P7)

As built. Two planned items were rejected after checking blast radius:

- **Group merge rejected.** Merging the encoder into the actor-decay group
  (4→3 param groups) breaks resume of every pre-change checkpoint:
  `checkpointing.load_optimizer` raises `ResumeError` on param-group count
  mismatch (`checkpointing.py:97-101`). Resume contract beats a cosmetic group.
  `split_grad_params` stays; the two `clip_grad_norm_` calls per boundary were
  already one per role (independent actor/critic norms, non-negotiable).
- **Buffer refactor a no-op.** `RolloutBuffers` is already a `@dataclass` with
  instance state (`host`/`device`/`h2d_event`/`env_wrapper`);
  `tests/test_shared_env_carry.py` enforces the no-duplicate-carry invariant.
  The `layout_key`/`make_buffer` split is left alone — inlining it would add a
  second shape-list source of truth.

Actual change: total pre-clip norm via `torch.hypot` instead of
`stack`+`vector_norm`+`reshape` (logging-only; gradients untouched).

Acceptance re-scoped: gates green; pins unchanged; eval CSV byte-identical on the
seeded smoke; `fps` within noise (43.2 vs 51.94 baseline, single CPU sample;
run-to-run range across phases 43–52).

Status: done — 577 passed / 2 pre-existing mani_skill failures; pins 817/218,
0 mismatches; `/tmp/myfpo_smoke/p4_pre` and `.../p5` CSVs identical.

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

As built (scope per user confirmation: wording-only modules deleted, 8 small modules merged into 3 themed files).

Deleted modules (source-text/AST/name enforcement, no numeric behavior):
`test_time_variable_name.py`, `test_actor_signature.py`, `test_log_key_family.py`,
`test_ratio_axes.py` (user opted in), `test_config_surface.py`.

Merged into themed modules:
`test_cfm_grouping.py` + `test_cfm_shared_loss.py` + `test_fpo_baseline.py` -> `test_cfm.py`;
`test_cudagraph_safe.py` + `test_temporal_sensitivity.py` + `test_update_metrics.py` -> `test_training_logic.py`;
`test_checkpoint_roundtrip.py` + `test_random_baseline.py` -> `test_checkpoint_eval.py`.

Keep: all behavior tests (GAE, bootstrap, guard messages, metrics faithfulness,
encoder, critic, clamp saturation, action projection, etc.) and pins.

Deviation vs the ≤30-module acceptance: 50 .py files remain (48 test modules +
conftest.py + fixtures.py); collected tests 606 -> 541; 2 pre-existing mani_skill
failures unchanged. Reaching ≤30 would require merging mid-size modules with no
coverage gain — not done without explicit approval.

Status: done — pytest 539 passed / 2 pre-existing mani_skill failures; pins 817/218,
0 mismatches; seeded smoke eval CSV byte-identical.

## Phase 8 — Housekeeping and report

- Update `AGENTS.md` (structure section: actions.py/diagnostics.py, pins/gone names,
  test-module count, diagnostics lazy import note) and `README.md` (stale processor.py
  references).
- `myfpo_online.md` untouched (user research notes).

## Appendix A — Baseline (Phase 0) and outcome (Phase 8, 2026-11-13, branch `simplify`)

| metric | before | after (final) |
|---|---|---|
| `wc -l src/myfpo/*.py scripts/*.py` | 8561 | 8480 |
| `src/myfpo/*.py` only | ~7.6k | 7509 |
| test .py files (`ls tests/*.py`) | 63 | 50 (48 test modules + conftest + fixtures) |
| tests collected / pass / skip | 606 / 601 / 3 | 541 / 539 / 3 |
| pre-existing failures (mani_skill absent) | 2 (`test_eval_video.py`) | 2 (unchanged) |
| Hopper-v5 2k-step smoke `fps` (CPU) | 51.94 | 43–52 across phases (single samples; seed-identical output each phase) |
| pin_train / pin_eval | 817 / 218 keys, 0 mismatches | 817 / 218 keys, 0 mismatches |
| ruff | pass | pass |
| eval_history.csv on smoke (seed 0) | — | byte-identical pre/post phases 4, 5, 7 |

Note: working tree carried user WIP (`configuration.py`, `fpo_core.py`, AGENTS/README) not owned by this plan; gates were run against it. Smoke used the default `outputs/` dir and overwrote the prior local seed‑0 run artifacts (untracked, gitignored). Later smokes must use `--out_dir /tmp/...`.

## Risks and rollback

- Pin failure in Phases 1–5 → `git restore` the phase, file an issue with the two
  floats that moved; never `--write` to make it pass.
- `fps` regression > 2% → revert the phase; the only suspect in Phases 1–5 is lazy
  diagnostics import (should be a win) or no‑op branch removal (should be neutral).
- Phase 6 is pure moves; if imports churn too much, split into two commits
  (`nn/flow`, then `algo/collect/eval`) rather than abandon.
- Any test that survives only because of a deleted mode is deleted, not ported.
