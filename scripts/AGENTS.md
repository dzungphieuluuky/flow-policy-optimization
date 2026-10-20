# AGENTS — scripts (ASD-STE 100)

Entry: `scripts/train.py` (CLI > `MyfpoConfig` defaults; every field is a `--myfpo.` flag). `uv sync` installs the project editable, so `import myfpo` works from any directory; no `sys.path` juggling in the scripts.

Convention: exit codes `0/1/2/4` (0 = ok, 1 = bad command line, 2 = the run failed, 4 = an artifact is missing). Gates: `pytest -q tests` (`pyproject.toml:35`) and `ruff check .` (`pyproject.toml:37`). No CI. Notebook pair: `myfpo_online.md` / `.ipynb` (`jupytext --sync` idempotent; `cell_metadata_filter: -all`).

Anti-patterns:
- Do not add runner scalars (`total_steps`, `eval_steps`, `save_steps`, `obs_dim`, `action_dim`) to `MyfpoConfig`. Bypass config by design (`training.py`).
- Dependency pins live in `pyproject.toml` only.
- Do not hand-edit `.ipynb`. Edit `.md`, run `jupytext --sync`.

Commands:
```bash
.venv/bin/python scripts/train.py --env Hopper-v5 --seed 0 --total_steps 2000
.venv/bin/python -m jupytext --sync myfpo_online.md
uv run ruff check .
```

Deleted artifacts (stale): `run_sweeps.sh` (passed removed flags `critic_max_grad_norm`, `critic_ema_decay`); `cuda_probe.py` (read never-emitted metric keys); `build_notebook.py` (installed deleted `requirements-linux.txt`); `verify_deps.sh` (drifted three ways from the environment); `src/myfpo/baselines.py` + `--baseline` flag. Config mapping pinned by `tests/test_ratio_axes.py`.

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

New artifacts (added 2026-10-20):
- `docs/token_advantage_design_a.md`: the plan for a per-token advantage for action chunks (RLinf-VLA's action-level arm, adapted to the CFM-loss ratio). Plan only; not implemented.
- `docs/architecture.md`: the action-space section (the raw chunk, the squash on the way to the env) and the layer map with `processor`.
- `tests/test_raw_action_space.py`: the six tests that pin "the action the flow is trained on is the action the flow produces".

Fixed bugs (`5d28f05`): `fpo_core.py`: restored `cfm_loss_weight` definition accidentally removed during docstring fix; corrected `cfm_loss_weight` docstring (`reversed vs. reference` removed); added proof-reference note (`fpo_core.py:288`) citing `arXiv:2507.21053`, `arXiv:2602.02481`, `literature_review.md` Sec 4.2. `training.py`: removed premature `eval_state[2] = 0` (`line 491`) before definition (`line 495`) — `UnboundLocalError` fixed (`5d28f05`).

Action space (2026-10-20): the flow's output is the **raw** action, and that raw value is what the policy is trained on; the `tanh` squash happens on the way to the env. Before this, the buffer stored the squashed chunk, so the flow was trained on the squash of its own output, and the env received a second squash of a value that was already squashed. `processor.squash_to_env` is the one implementation: `predict_action_chunk` returns the raw chunk, `select_action` and the collector squash it, and the collector stores the raw. Consequences:
- The stored action is **not** the action the env received. `train/raw_action_abs_mean` is the stored chunk, `train/executed_action_abs_mean` the env-facing one, and the ratio between them is the squash. The old `train/action_abs_*` keys are retired.
- The action rides fp32 in the batch: it is the CFM target, and the raw values are unbounded.
- Checkpoints carry `action_space.json`. A checkpoint without it has a critic that read the squashed chunk, so `load_policy_checkpoint` refuses it with the reason named. **Every checkpoint written before 2026-10-20 is void.**
- The jerk metrics (`rollout_jerk`, `rollout_seam_jerk`) are measured on the env-facing action, since that is the trajectory the env saw.

Fixed bugs (2026-10-20): `modeling.py` `load_ema_state_dict` restored the EMA shadow on the CPU while the live parameters were on the GPU, so every resume died in `update_ema` with a device mismatch. The shadow now follows the parameters onto their device. `training.py`'s post-clip magnitude guard now checks the env-facing action against the env's bounds, not the stored chunk against 1.

Known open issues:
- `scripts/`: no `sys.path.insert` left; the package is installed editable by `uv sync`. A fresh clone needs `uv sync` before `pytest` or the scripts run.
- `rollout.py`: `rollout_seam_jerk` / `rollout_jerk` measured; no chunk-boundary mitigation implemented.
- `experiments/`: 11 PNG screenshots (`2026-09-26`); no quantitative tables / CSV.
- No statistical significance tests for metrics (naming/NaN only).
- No `torch.utils.checkpoint` applied (`flow.py` / `rollout.py`).
- No CI. Gates are `pytest -q tests` and `ruff check .`.
- `fpo_core.py`: approximate ratio (`ρ̂`) approximation bias unquantified in tests.
- `metrics.py`: `broadcast_condition` dense (`einx.id`); `rollout.py` chunk grouping logic sparse comments.
- The raw action has no clip. `train/raw_action_abs_max` is the monitor: a drifting raw magnitude would make `cfm_clamp` bind and pin the ratio at 1.
