# DONE — completed work log (newest first)

Companion to `NEXT_STEPS.md` (remaining work). One checkbox per landed
commit on `main`; every item was gated by `pytest -q` green + smoke/manual
verification before commit.

## Architecture refactors
- [x] Vectorized rollout collection, `num_envs=8` default (`998422b`)
      `collect_chunks_vec`: per-env chunk loops, batched proposals, one
      vector step per action; eval stays single-env.
- [x] Pure-DiT policy: single adaLN path `c = time + state + spectral`
      (`0d2421d`). Deleted concat, readout slice, position collision;
      `WindowEncoder` merged (FFN 1x->4x); `CondMLP` per DiT canon.
- [x] Bin-index encoding on policy action tokens (`c64bfe7`).
      Frequency-addressable inter-bin attention + symmetry test.
- [x] Spectral-only scope: removed `sigreg_loss` + quadrature (`fcc4007`).
- [x] Removed write-only `last_pool_weights` (`bc5568e`).

## Training infrastructure
- [x] Accelerate DDP path, single-process parity-checked (`fadb333`).
      Byte-identical `eval_history.csv` pre/post diff.
- [x] Phase-0 per-bin rho diagnostics, read-only (`740ae67`).
- [x] Pre-clip grad-norm logging; demo video + return + `video_ok`
      (`0795e0a`).

## Hygiene
- [x] Dead code: `unwrap_compiled`, `build_filtered_matrix`,
      `safe_float_value` (`566298e`); dead args: `build_notebook(output_path)`,
      `random_baseline(seed)`, skrl `outputs` (`60030a2`).
- [x] Full docs sync to live code (`60030a2`, `8b08dfd`, `bc5568e`):
      no `--matrix` flag, underscore draccus flags, 300 tests, 8-cell
      notebooks, live wandb keys; both `.ipynb` rebuilt.

## Planning
- [x] `NEXT_STEPS.md` roadmap Phases 0-7 (`fb84122`).
- [x] Venue targets: CoRL 2027 primary, NeurIPS 2027 co-target (`c68d6f9`).
- [x] Ablations A1-A6 + per-venue framing (`c68d6f9`).

## Prior work (before this session)
- [x] Five conditioning fixes: timestep scale, bin order, bin norm, decay
      groups, diagnostics (`54926b2`).
- [x] Notebook retarget to single-run CLI (`bb328ef`).
- [x] Config cleanup, CLI split, policy upgrades (`5895219`, `a4fe8b6`,
      `bf46206`, `39d02fc`).

## Standing state
- Suite: **302 passed**, 0 failed. Tree clean.
- Matrix: tiny sweep partially salvaged (sjfpo+ppo seed0 complete, sac
      partial); full 45-run matrix empty — needs ThinkPad GPU runs.
- Phase-1 per-bin trust region: undecided, gated on 2M-step diagnostics.
