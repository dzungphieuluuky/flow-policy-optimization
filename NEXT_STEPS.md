# NEXT_STEPS — from repo to paper submission

Goal: turn SJFPO (Spectral Joint Flow Policy Optimization) into a submittable
conference paper. Ordered by dependency; do not skip ahead (each phase gates
the next).

Current state (2026-09-16): method implemented, DDP path landed (`fadb333`),
Phase-0 per-bin diagnostics landed read-only (`740ae67`), 303 tests green,
tiny sweep (3 runs x 20k) in progress, full 45-run matrix empty, no manuscript.

---

## Phase 0 — Stabilize the tree (1 day, CPU-only)

- [ ] Confirm tiny sweep finishes: `tail /tmp/sweep_tiny.log` shows
      `sweep: 3 ok, 0 failed`.
- [ ] Re-scan: `PYTHONPATH=src .venv/bin/python -c` with
      `scan_runs(RUN_MATRIX, '.omo/evidence/train')` shows 3 skipped (tiny).
- [ ] `PYTHONPATH=src .venv/bin/python -m pytest -q` green.
- [ ] `git status` clean. Tag: `git tag v0-tiny && git push --tags`.

## Phase 1 — Freeze the method (1-2 days, CPU-only)

Open question that blocks all writing: Phase-1 per-bin trust region.

- [ ] Run the 2M-step Hopper diagnostic run the notebooks point at:
      `PYTHONPATH=src .venv/bin/python scripts/train.py --method sjfpo \
--env Hopper-v5 --seed 0 --total_steps 2000000 --out_dir .omo/evidence/train`
      (GPU strongly advised; on CPU run 200k and extrapolate).
- [ ] Read `flow_rho_bin_range / lowfreq / highfreq / outside_frac` from logs.
- [ ] Decision, exactly one:
      - (a) bins behave alike -> DELETE Phase-0 diagnostics, keep scalar ASPO;
      - (b) low/high bins diverge -> implement per-bin clip/penalty (Phase 1).
- [ ] Lock `SJFPOConfig` defaults. No more key renames after this point:
      every rename invalidates prior runs.
- [ ] Update `docs/architecture.md` + `docs/SHINY.md` to match the frozen code.
- [ ] Commit + tag `v0-freeze`. Record hash; every paper number must trace
      to a run dir produced by this hash.

## Phase 2 — Evidence: full benchmark matrix (needs GPU, the long pole)

Target: 45 runs = 3 methods (sjfpo, ppo, sac) x 3 envs
(HalfCheetah-v5, Walker2d-v5, Hopper-v5) x 5 seeds (0-4) x 1M steps.
ThinkPad (RTX PRO 1000 Blackwell 8GB, 16 cores): sjfpo collects with
`num_envs=8` async MuJoCo procs by default (per-update env-steps unchanged,
collection ~linear speedup); eval stays single-env. Override with
`--sjfpo.num_envs N` (8 fits 32GB; 1 reproduces the legacy loop).

- [ ] Launch: `METHODS="sjfpo ppo sac" ENVS="HalfCheetah-v5 Walker2d-v5 Hopper-v5" \
      SEEDS="0 1 2 3 4" bash scripts/sweep.sh` (out base
      `.omo/evidence/train` by default).
- [ ] Monitor via `scan_runs`: pending -> 0, diverged investigated (a
      diverged run is a result too, but three in a cell means a bug).
- [ ] Back up run dirs (manifests + `eval_history.csv` + final checkpoints).
      At minimum keep manifests + CSVs; checkpoints are large.

## Phase 3 — Evidence: ablations (GPU, Hopper-v5 only, 3-5 seeds each)

Each ablation is one flag/config change; keep everything else frozen.
These are the paper's differentiation vs. SOTA (FAFM, FreqPolicy, FAST):

- [ ] A1 — spectral placement: joint (default) vs. action-only
      (`top_p_spectral_tokens=1.0` equiv / bypass obs DCT) vs. state-only
      (time-domain action head). Tests the "joint" claim.
- [ ] A2 — pooling: `spectral_condition_mode=attention` vs. `mean`.
- [ ] A3 — bandwidth: `top_p_spectral_tokens` in {0.1, 0.3, 0.5}.
- [ ] A4 — trust region: ASPO vs. plain PPO-clip (validates SHINY claim 3).
- [ ] A5 (cheap, CPU): `n_action_steps` / `executed_vs_predicted_ratio`
      sensitivity {0.25, 0.5, 1.0}.
- [ ] A6 — evidence breadth (pick ONE): promote a prototype env from
      `KNOWN_ENVS` into the matrix. `Ant-v5` = same locomotion family, zero
      code change. `AdroitHandDoor-v1` = dexterous manipulation, bridges
      toward the manipulation SOTA (FAFM/FreqPolicy) and strengthens both
      the CoRL and NeurIPS versions. Needs `gymnasium-robotics` installed
      (`scripts/train.py` already handles the import).

## Phase 4 — Analysis + figures (CPU-only, after Phases 2-3)

`src/sjfpo/analysis.py` (IQM + bootstrap, n_boot=2000) is the only stats
path; do not hand-roll numbers.

- [ ] Main table: IQM return + 95% CI per method x env (last-10 evals).
- [ ] Learning curves: mean +- CI over seeds, all 3 envs.
- [ ] Smoothness figure: mean |action_{t+1} - action_t| and/or retained
      high-frequency energy per method. The smoothness claim needs a number.
- [ ] Ablation bar chart (A1-A4 on Hopper).
- [ ] Optional: per-bin rho diagnostic plot if Phase-1(b) was taken.
- [ ] Regenerate notebooks: `.venv/bin/python scripts/build_notebook.py`,
      verify figures reproduce from checked-in CSVs.

## Phase 5 — Manuscript (parallel with Phases 2-4)

You are entering this generic research timeline at the experiments stage
(weeks ~19-35): foundations, paper reading (FAFM/FreqPolicy/FAST surveyed),
and question-scoping are done. Remaining generic phases map to this file:
rigorous experiments = Phases 2-4, scientific writing = this phase,
submission/review = Phase 7 plus the arXiv item below.

Venue plan (sim-only, no hardware — fully acceptable at all of these):

| Venue | Why it fits | Deadline (as of 2026-09-16) |
|---|---|---|
| **CoRL 2027** (primary) | Best fit: robot-learning audience, MuJoCo-only papers routine, no hardware expected | Expect ~May/Jun 2027 (CoRL 2026 closed May 28, 2026) |
| **NeurIPS 2027** (secondary) | General-ML audience; frame as flow-matching RL method, not just robotics | Expect ~May 2027 (2027 dates not yet announced) |
| **ICML 2027** (backup) | Same framing as NeurIPS | Expect ~Jan/Feb 2027 |
| **ICLR 2027** (sprint only) | Abstract Sep 18, full paper Sep 25, 2026 — 2-9 days away; only viable if you submit current tiny-matrix results as-is | Sep 18/25, 2026 |
| **CoRL 2026 workshops** (stepping stone) | Main track closed, but workshop CFPs may still be open; fast feedback + citation | Check corl.org now |

Recommendation: target CoRL 2027 as the main goal, NeurIPS 2027 as the
co-target (timelines overlap; same evidence serves both). Skip the ICLR 2027
sprint unless you want a forced draft deadline — 9 days cannot produce the
45-run matrix. A CoRL 2026 workshop paper (4-8 pages, tiny-matrix results)
is the cheap intermediate win: it timestamps the idea and buys reviewer
feedback before the 2027 cycle.

Not recommended: ICRA/IROS — sim-only is acceptable there but reviewers
weight real-world validation more; this story is algorithmic, so the
fit is weaker than CoRL for the same effort.

Framing per venue (same evidence, different lead):

- CoRL: robotics story. Lead with spectral action generation + smoothness
  + locomotion results; ablations A1-A3 carry the contribution.
- NeurIPS/ICML: method story. Lead with flow-matching policy gradients,
  ASPO trust region for estimated likelihoods, chunk semi-MDP GAE; MuJoCo
  is the testbed. Ablations A1 (joint placement) and A4 (ASPO trust region)
  are the most important figures —
  without them "spectral smoothing" reads as incremental to an ML
  audience.

- [ ] Skeleton (maps to `docs/SHINY.md`):
      1. Intro: flow policies are time-domain; spectrum gives smoothness
         + memory for free.
      2. Method: truncated-DCT flow (exact likelihoods via orthonormality),
         chunk semi-MDP GAE, ASPO trust region, parameter-free conditioning.
      3. Related: FAFM (2606.20135), FreqPolicy-AR (2506.01583),
         FreqPolicy-Flow (NeurIPS'25), FAST (2501.09747), sub-frequency
         diffusion (2605.27919), ABPolicy, MPD/ProDMP, SFP, RFMP,
         pi0/RDT FM heads. Positioning sentence: first spectral-JOINT
         flow-matching POLICY-GRADIENT (RL), not imitation.
      4. Experiments: matrix + ablations + smoothness.
      5. Limitations: MuJoCo locomotion only, no vision, no real robot.
- [ ] Math appendix: CFM loss in DCT space, Parseval bookkeeping, ASPO
      piecewise derivation, chunk-GAE recursion with variable L_k.
- [ ] Draft in figures-first order: figures/tables -> methods -> results
      -> intro -> abstract. Do not write the intro first.
- [ ] Post to arXiv after the internal red-team pass (before or with
      submission, depending on venue anonymity rules: CoRL/NeurIPS review
      is double-blind, so arXiv-after-decision if anonymity is a concern;
      otherwise preprint early to timestamp).
- [ ] Red-team the ASPO section with one colleague before submitting
      (most likely reviewer attack surface).

## Phase 6 — Reproducibility package (1-2 days)

- [ ] `REPRODUCE.md`: fresh-env install -> smoke run -> expected output,
      with GPU model + wall-clock per run type.
- [ ] Verify `scripts/build_notebook.py` output runs end-to-end in Colab.
- [ ] Publish: code commit hash, seed list, all `eval_history.csv` files,
      analysis script invocation. Numbers in tables must be regenerable
      from checked-in CSVs via `analysis.py`.
- [ ] `tests/test_pipeline_cpu.py` + full suite green on the submission hash.

## Phase 7 — Pre-submission gate (checklist, all must be yes)

- [ ] Full `pytest -q` green on submission hash.
- [ ] Smoke run green on submission hash.
- [ ] Every table number traces to a run dir + manifest.
- [ ] No uncommitted code; no stray files (`git status` clean).
- [ ] Claims in abstract appear verbatim in results with a figure/table ref.
- [ ] Limitations section exists and is honest (sim-only, 3 envs).
- [ ] Anonymized repo / supplementary bundle builds from scratch once,
      by hand, following only `REPRODUCE.md`.

---

## Compute budget (rough, measured 23 fps sjfpo on CPU)

| Workload              | Per run (CPU) | Count | GPU estimate |
|-----------------------|---------------|-------|--------------|
| Smoke 2k              | ~2 min        | -     | < 1 min      |
| Tiny 20k              | ~15 min       | 3     | ~30 min tot  |
| Diagnostic 200k-2M    | 2.5-25 h      | 1     | 0.5-4 h      |
| Full 1M sjfpo         | ~12 h         | 15    | ~1 h each    |
| Full 1M ppo/sac       | faster        | 30    | minutes each |
| Ablations (Hopper 1M) | ~12 h CPU     | ~15   | ~1 h each    |

Do not attempt the full matrix on CPU. Everything else can proceed on CPU
while GPU time is arranged.
