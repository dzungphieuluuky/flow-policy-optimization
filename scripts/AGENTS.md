# AGENTS — scripts (ASD-STE 100)

Entry: `scripts/train.py` (CLI > `MyfpoConfig` defaults; every field is a `--myfpo.` flag). `uv sync` installs the project editable, so `import myfpo` works from any directory; no `sys.path` juggling in the scripts.

Convention: exit codes `0/1/2/4` (0 = ok, 1 = bad command line, 2 = the run failed, 4 = an artifact is missing). Gates: `pytest -q tests` (`pyproject.toml:59`) and `ruff check .` (`pyproject.toml:61`). No CI (`AGENTS.md:40`). `tests/test_config_yaml_parity.py` enforces `configs/myfpo.yaml` parity (`tests/test_config_yaml_parity.py`). Notebook pair: `myfpo_online.md` / `.ipynb` (`jupytext --sync` idempotent; `cell_metadata_filter: -all`).

Anti-patterns:
- Do not add runner scalars (`total_steps`, `eval_steps`, `save_steps`, `obs_dim`, `action_dim`) to `MyfpoConfig`. Bypass config by design (`training.py`).
- Dependency pins live in `pyproject.toml` only. `scripts/verify_deps.sh` was deleted; it had drifted three ways from the environment.
- Do not hand-edit `.ipynb`. Edit `.md`, run `jupytext --sync`.

Commands:
```bash
.venv/bin/python scripts/train.py --env Hopper-v5 --seed 0 --total_steps 2000
.venv/bin/python -m jupytext --sync myfpo_online.md
uv run ruff check .
```

Deleted artifacts (stale): `run_sweeps.sh` (passed removed flags `critic_max_grad_norm`, `critic_ema_decay`); `cuda_probe.py` (read never-emitted metric keys); `build_notebook.py` (installed deleted `requirements-linux.txt`); deleted `4bd45db`. `src/myfpo/baselines.py` + `--baseline` flag removed (`debd7e8`). Config mapping pinned by `tests/test_ratio_axes.py`.

Deleted features (`debd7e8`): `check_divergence`, `check_regression`, `interval_straddles` removed from `evaluator.py`. `lr_decays` / `max_lr_decays` fields gone from `MyfpoConfig` / generator. No early exit: run ends at `total_steps` or non-finite loss (`MyfpoAborted`). `stop_eval_sampling` stayed (selects `checkpoints/best` distribution, not stopper). Deleted tests: `test_eval_interval.py` (19), `test_tracking.py` (4), `test_metric_audit.py` (1), `test_fpo_parity.py` (1). Interval/bootstrap tests stayed.

Verified non-issues:
- `wandb` under numpy 2 fine. Default `report_to` = `"all"`, not `"wandb"`.
- `requirements-linux.txt` deleted (`747c67c`). `requirements.txt` unpinned.
- `.gitignore` does not ignore `*.ipynb`.
- `configs/shared.yaml` unreferenced; absence costs nothing.
- `requirements.txt`: `jax[cuda13]` (not cuda12); `myfpo_online.md` install cell matches.

New artifacts (added 2026-10-18):
- `README.md`: abstract, arXiv refs (`arXiv:2507.21053`, `arXiv:2602.02481`), quick start.
- `docs/ablation_reference.md`: taxonomy table, design trade-offs (`rollout_seam_jerk`, approximate ratio, ASPO, chunking).
- `literature_review.md`: full synthesis (FPO, FPO++, GoRL, GORL, GenPO, OGPO, DBPO, GPC, PSS), comparison table (`Section 11.4`), reading roadmap (`Section 11.5`), dependency graph (`11.2`), timeline (`11.1`).
- `AGENTS.md` updated (`49`).

Fixed bugs (`5d28f05`): `fpo_core.py`: restored `cfm_loss_weight` / `sigreg_weak_loss` definitions accidentally removed during docstring fix; corrected `cfm_loss_weight` docstring (`reversed vs. reference` removed); added proof-reference note (`fpo_core.py:288`) citing `arXiv:2507.21053`, `arXiv:2602.02481`, `literature_review.md` Sec 4.2. `training.py`: removed premature `eval_state[2] = 0` (`line 491`) before definition (`line 495`) — `UnboundLocalError` fixed (`5d28f05`).

Known open issues:
- `scripts/`: no `sys.path.insert` left; the package is installed editable by `uv sync`. A fresh clone needs `uv sync` before `pytest` or the scripts run.
- `rollout.py`: `rollout_seam_jerk` / `rollout_jerk` measured; no chunk-boundary mitigation implemented.
- `metrics.py`: `budget_truncated_chunk_frac` emitted; feature removed.
- `experiments/`: 10 PNG screenshots (`2026-09-26`); no quantitative tables / CSV.
- No statistical significance tests for metrics (naming/NaN only).
- No `torch.utils.checkpoint` applied (`flow.py` / `rollout.py`).
- No CI. Gates are `pytest -q tests` and `ruff check .`.
- `fpo_core.py`: approximate ratio (`ρ̂`) approximation bias unquantified in tests.
- `metrics.py`: `broadcast_condition` dense (`einx.id`); `rollout.py` chunk grouping logic sparse comments.
