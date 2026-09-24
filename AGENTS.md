# AGENTS.md

## Active checkout
- The active project is the outer directory `/home/vr-tts27/Downloads/spectral-joint-policy`; it owns `.venv/`, `outputs/`, and the files used by current tracebacks.
- `spectral-joint-policy/` inside it is a separate older Git checkout with conflicting files. Exclude it from searches, edits, pytest discovery, and builds; do not accidentally commit it.
- The outer tree is not a Git worktree. Do not rely on `git status` there.
- Executable source wins over README/docs and the scoped `AGENTS.md` files, which contain stale architecture, defaults, test counts, and precision claims.

## Runtime map
- `scripts/train.py` is the single-run CLI. It dispatches `sjfpo` to `sjfpo.train_sjfpo.train_sjfpo_run`, `ppo`/`sac` to `sjfpo.baselines`, and `sjfpo_jax` to `sjfpo_jax.train_sjfpo.train_sjfpo_run`.
- Torch SJFPO flow: rollout → chunk GAE → minibatch `SJFPOPolicy.forward` (flow/ASPO + critic) → backward/clip/AdamW → evaluation/checkpoint.
- `src/sjfpo/` is the Torch production package; `src/sjfpo_jax/` is a separate JAX backend. Do not port changes between them unless requested.
- `src/spectral_joint_policy/` is only the placeholder package used by the declared console script; it is not the training entrypoint.
- `scripts/sweep.sh` calls `scripts/train.py` directly for each method/env/seed. It does not use `sjfpo.sweep` or `RUN_MATRIX`.
- `scripts/play.py` evaluates checkpoints. `build_notebook.py` builds clone-based notebooks; only `build_offline_notebook.py` embeds source.
- `src/sjfpo/analysis.py` is analysis-only and is not imported by the training CLI.

## Setup and commands
- Python 3.12+ is required. Reinstall with `.venv/bin/python -m pip install -e .`; `requirements-linux.txt` is the pinned Linux alternative. `python` is not on PATH here, so use `python3` or `.venv/bin/python`.
- Do not use `scripts/verify_deps.sh` as a gate; it references a missing requirements file and obsolete version bounds.
- Single run, without wandb:
  ```bash
  PYTHONPATH=src .venv/bin/python scripts/train.py --method sjfpo --env Hopper-v5 --seed 0 --total_steps 2000 --wandb_enabled false
  ```
- Sweeps must explicitly select the interpreter in this environment:
  ```bash
  METHODS="sjfpo ppo sac" ENVS="Hopper-v5" SEEDS="0" TOTAL_STEPS=2000 bash scripts/sweep.sh --python .venv/bin/python
  ```
- Unit tests use the shared MuJoCo-free `tests/fixtures/fake_env.py` fixture:
  ```bash
  PYTHONPATH=src .venv/bin/python -m pytest -q tests
  PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_modeling_sjfpo.py::test_forward_bfloat16_storage_and_compute
  ```
- Always scope pytest to `tests`; unscoped discovery also collects the nested checkout and causes import-path conflicts.
- There is no CI, configured lint command, or configured typecheck gate. `ruff` is not installed; `basedpyright` exists but currently reports extensive pre-existing errors and has no project config.
- A fast syntax fallback for changed Python files is `PYTHONPATH=src .venv/bin/python -m py_compile <files>`.

## High-risk gotchas
- `STORAGE_DTYPE` is resolved once at import. On RTX PRO 1000 Blackwell it is BF16 even when tests or a manually constructed policy stay on CPU, while parameters remain FP32; this produces BF16/Float `Linear` errors. Start CPU test processes with `CUDA_VISIBLE_DEVICES=''` so storage resolves to FP32. Exercise real BF16 only on a CUDA model or with an explicit CPU-autocast regression.
- Any FP32 `nn.Linear` called with `STORAGE_DTYPE` tensors must be inside `autocast_context(device)`. Autocast is intentionally a no-op on CPU.
- `SJFPO_DEVICE=auto|cpu|cuda` controls device selection; `SJFPO_COMPILE=0` disables CUDA compilation. On Blackwell, `resolve_compute_dtype` returns BF16 before reading `SJFPO_DTYPE`, so that override does not disable Blackwell BF16.
- wandb is enabled through `TrainConfig.wandb_enabled` / `--wandb_enabled`; `SJFPO_DISABLE_WANDB` is documented but not read by the current code. Keep direct wandb imports confined to `src/sjfpo/wandb_logging.py`.
- `src/sjfpo/configuration_sjfpo.py::SJFPOConfig` is the live SJFPO default source. `configs/*.yaml` currently disagrees numerically with those defaults, and YAML parity tests check keys rather than values.
- `RUN_MATRIX` currently contains four core envs from `configs/envs.yaml` (60 runs) despite its “45 runs” comment, while `sweep.sh` independently hard-codes three envs (45 runs). Do not treat either as synchronized.
- Evaluation CSV is five columns: `episode,return,length,success,jerk`.
- JAX tests require working `flax`/`optax`; do not assume the current Torch-oriented virtualenv can collect or run them.

## Current test drift
- Do not report the suite as green from the docs. `pytest -q tests` currently collects 313 tests with 10 collection errors, including missing/renamed symbols (`capture_env_fingerprint`, `compute_advantages`, `is_bf16_available`, `SpectralAttentionPool`) and JAX dependency gaps.
- `tests/test_modeling_sjfpo.py` also has stale expectations around diagnostic keys, defaults, and rollout/CRN behavior. Run the exact regression first and report unrelated baseline failures instead of weakening assertions.
- This is research code: prefer the smallest direct change, keep algorithm math explicit, and do not add infrastructure, config frameworks, or dependencies unless the requested behavior requires them.
