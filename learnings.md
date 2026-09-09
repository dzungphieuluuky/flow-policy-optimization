# Learnings (consolidated — root)

Durable, cross-session lessons. Plan-scoped raw notes live in
`.omo/notepads/*/learnings.md`. Entries marked HISTORICAL describe deleted
code — kept for archaeology, do not act on them.

## Tooling (durable)

- `filesystem_edit_file` is ATOMIC: one bad `oldText` fails the whole batch.
  Split large rewrites into small batches (3–5 edits) and read the returned
  diff every time. Never truncate `oldText` to dodge validation — it caused a
  builder corruption that had to be repaired.
- Same-file parallel edits race — sequence them.
- Asserted line-range deletions beat blind ranges: anchor with exact marker
  strings, assert the first/last lines, then delete 1-indexed ranges via a
  script. Two off-by-one failures happened on markdown `# %%` cell markers
  sitting one line above the assumed boundary.
- Deleted-class residue check: after removing a class, re-read the full
  region — a second class (`PpoValueImpl`) hiding between two deleted ones
  was swallowed and had to be restored verbatim from the pre-delete read.
- `git rm --cached` is required when gitignoring already-tracked files
  (`*.ipynb`): `git check-ignore` stays silent until untracking.
- Always verify evidence-dir contents before `rm -rf`: an "empty" claim was
  wrong (`smoke_libero/` held two full smoke runs). `ls` first, delete after.
- `tail -1` on pytest output can grab atexit noise (sweep worker); grep for
  `passed|failed|error` instead.
- LSP unavailable in this env (import-resolution errors are venv-path noise):
  `py_compile` + full suite is the functional equivalent. Never treat LSP
  "unresolved import" as a code defect here.
- `chmod +x` denied by tool permissions — scripts run via interpreter
  (`bash scripts/verify_deps.sh`), never rely on exec bits.

## Process (durable)

- Parity gates make merges safe: identical 300-step `eval_history.csv` hashes
  (`a61e83d3…`) proved the GaussianPolicy merge and helper extraction were
  behavior-identical. Deterministic 200-step mfpo smoke pins
  `reward_mean_last10 = 17.34955…` — any drift is a real regression.
- CSV-hash parity generalizes: any refactor of runner/baselines needs a
  pre/post hash comparison before the old code is gone.
- Z-count discipline: track `216 = 222 − 6 (sigreg v1)` explicitly; every
  final gate recomputes it. A silent count change means a test was lost.
- Delete-gate order: universal `grep src scripts tests docs` BEFORE deleting;
  distinguish live imports from builder-embedded strings and notebook text.
- Builder-embedded code is a second codebase: deleting a module requires
  checking `build_notebook.py` MODULE_ORDER/TITLES strings, not just imports.
  Regenerate (never hand-edit) `.ipynb`, then validate cell counts (61/47).
- No commits unless explicitly requested. The tree routinely mixes
  pre-existing uncommitted work with plan work — disentangle before committing.
- Plan-subagent calls can abort silently; fallback is building the plan
  directly from retained inventories and presenting it for `proceed`.
- `yaml.safe_load` only; coerce YAML keys to dataclass fields (drop unknowns
  like `tau_range`, tuple-ize lists) so configs roundtrip
  (`MFPOConfig(**yaml) == MFPOConfig()`).
- Thresholds must be measured, not guessed: SIGReg smoke-gate bounds came
  from running `sigreg_losses` on gaussian vs degenerate inputs first.

## Codebase facts (current)

- Single-run vs matrix CLI: singular `--method/--env/--seed` vs plural
  `--methods/--envs/--seeds` + `--matrix`; `--tiny` is matrix-only and warns.
  Precedence single `CLI>YAML>dataclass`, matrix `CLI scalars>YAML>shared>dataclass`.
- `wallclock_h` CSV column is an acknowledged placeholder (analysis.py).
- `_wandb_ctx` from `initWandbRun` is never entered — dead but kept for API
  shape; future micro-cleanup drops it to a single bool.
- `Dk=64` hardcoded in `networks.py:189`; `Da` is env-dependent.
- Keep-list (do not re-propose without new evidence): MJX stub (test-locked
  `is_stub`), `assemble_update_legacy` name, `buildLegacyRunMatrix`,
  `LEGACY_MUJOCO` flag, `JointFlowTransformer`/`PerBranchFlowLoss` (live),
  `inference.py` standalone (Euler direction differs from `fpo_core`),
  `analysis.py` statistics (amendment boundary), builder LIBERO prune-note
  strings (behavior-locked file).

## HISTORICAL (deleted code — do not act on)

- `LiberoConfig`/`buildLiberoRunMatrix`/LIBERO-10 suites/`SyntheticLiberoEnv`:
  purged. Old field mapping `Z_dim↔D, K_spectral↔K, context_len↔N,
  chunk_len↔H, lambda_aux↔lambda_S` is void.
- `SampleEfficiencyBudget`/`per_suite_efficiency`/budget_pct slices: purged.
- `LeWorldSIGReg`/`sigreg_loss`/`test_sigreg_v1.py`: purged; live loss is
  VICReg `sigreg_losses` (var+cov, λ=0.1).
- `VisionEncoder`/`RobotStateEncoder`/`FusionModule` + timm guard: purged.
- `ACTPolicy`/`DiffusionPolicy`/A1-A2 ablations: purged; baselines are skrl
  PPO/SAC with shared `GaussianPolicyImpl`.
- `scripts/local_smoke.py`: replaced by `scripts/train.py`.
- `test_*_v1.py`/`test_notebook_v1.py`: deleted with the code they covered.
