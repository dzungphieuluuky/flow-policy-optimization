# Repository instructions

## Source of truth

- This is the active repository checkout. If a nested `spectral-joint-policy/` checkout appears, exclude it from searches, edits, pytest, builds, and commits.
- Trust executable source, `pyproject.toml`, `scripts/`, and focused tests over README/docs or scoped `AGENTS.md` files; several historical docs are stale.
- `src/sjfpo/` is the active Torch package. `src/sjfpo_jax/` is a separate JAX backend; do not port changes between them unless requested.
- `src/spectral_joint_policy/` is only the console-script placeholder, not a training entrypoint.
- `src/sjfpo/analysis.py` is analysis-only and is not imported by the training CLI.
- Automatic `RUN_MATRIX`/`TINY_MATRIX` generation was removed from `src/sjfpo/config.py`; use explicit CLI/config runs or the shell sweep helper.

## Runtime and commands

- Python 3.12+ is required. The working interpreter is `.venv/bin/python`; `python` may not be on PATH.
- Install with:
  ```bash
  .venv/bin/python -m pip install -e .
  ```
  or `.venv/bin/python -m pip install -r requirements-linux.txt`.
- Single Torch run:
  ```bash
  PYTHONPATH=src .venv/bin/python scripts/train.py \
    --method sjfpo --env Hopper-v5 --seed 0 --total_steps 2000 \
    --wandb_enabled false
  ```
- Optional shell sweep (it calls `scripts/train.py` once per axis value; select the interpreter explicitly):
  ```bash
  METHODS="sjfpo ppo sac" ENVS="Hopper-v5" SEEDS="0" TOTAL_STEPS=2000 \
    bash scripts/sweep.sh --python .venv/bin/python
  ```
- Rebuild notebooks with:
  ```bash
  PYTHONPATH=src .venv/bin/python scripts/build_notebook.py
  ```
  Notebooks clone the repository and call the scripts; do not hand-edit generated `.ipynb` files.
- Fast syntax check for changed Python files:
  ```bash
  PYTHONPATH=src .venv/bin/python -m py_compile <files>
  ```

## Training architecture

- `scripts/train.py` is the single-run CLI. `sjfpo` dispatches to `sjfpo.train_sjfpo.train_sjfpo_run`; `ppo`/`sac` dispatch to `sjfpo.baselines`; `sjfpo_jax` is separate.
- Active Torch flow is rollout → step-level GAE → `SJFPOPolicy.forward` (flow/FPO loss + critic) → accumulated backward/clip/AdamW update → evaluation/checkpoint.
- `SJFPOConfig` in `src/sjfpo/configuration_sjfpo.py` is the live SJFPO configuration source. CLI values override `--config` YAML values, which override dataclass defaults.
- `n_action_steps` must be passed consistently to collection and evaluation; the collector also receives the configured action horizon explicitly.
- `grad_accum_steps` follows the Accelerate accumulation pattern: every microbatch backwards, optimizer steps/clipping only at accumulation boundaries, including a final partial group.
- Observation normalizer updates once per completed training rollout from raw post-step frames; evaluation/inference must not update it.
- `old_actor` and `old_conditioner` are runtime-only frozen snapshots; they are reconstructed on checkpoint load and excluded from saved model tensors.

## Precision and devices

- `resolve_storage_dtype(device)` is device-aware: CPU is FP32; CUDA RTX PRO 1000 Blackwell may use BF16 storage. `resolve_compute_dtype()` honors `SJFPO_DTYPE=fp32|bf16|auto`.
- Use `CUDA_VISIBLE_DEVICES=''` for CPU tests/smokes when CUDA is present but not intended.
- `SJFPO_DEVICE=auto|cpu|cuda` selects execution device; `SJFPO_COMPILE=0` disables CUDA compilation.
- FP32 `nn.Linear` calls receiving mixed-precision tensors must run inside `autocast_context(device)`; CPU autocast is intentionally a no-op.
- The 8-bit optimizer path and bitsandbytes dependency were removed. Do not reintroduce them.
- The active Torch path has no MJX rollout. Do not add `SJFPO_MJX` or MJX imports to the Torch package; keep any JAX backend changes isolated.

## Testing

- Always scope pytest to `tests`:
  ```bash
  PYTHONPATH=src .venv/bin/python -m pytest -q tests
  ```
- Run focused regressions directly, for example:
  ```bash
  PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_fpo_core.py
  PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_policy_checkpoint.py
  PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_modeling_sjfpo.py
  ```
- There is no configured CI, lint, or typecheck gate. Do not claim the full suite is green from documentation; report unrelated baseline failures and do not weaken assertions to hide them.
- JAX tests require working `flax`/`optax` and may not collect in the Torch-oriented environment.
- The shared `tests/fixtures/fake_env.py` is the preferred MuJoCo-free fixture for unit/smoke tests.
- Evaluation CSV schema is five columns: `episode,return,length,success,jerk`.

## Change discipline

- Keep research-code changes direct and explicit; do not add config frameworks, dependencies, or infrastructure without a requested behavior change.
- Preserve tensor-shape comments and `einx.id` reshape parity in collector/FPO/trainer/DCT code.
- Do not import `wandb` outside `src/sjfpo/wandb_logging.py`.
- Do not reintroduce EMA/target networks in the active SJFPO path; only the frozen rollout snapshots are used.
- Do not touch analysis statistics/plotting without an explicit requested scope.
