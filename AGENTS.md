# AGENTS — myfpo (ASD-STE 100)

Setup, gates, commands, invariants. `scripts/AGENTS.md` holds script-local facts only.

## Setup

`uv sync` installs the package editable + dev tools. Python 3.12 (`.python-version`).

`mani_skill` is NOT a dependency: it is absent from `pyproject.toml` and `uv.lock`. `--env MS-*` needs a separate `mani_skill` 3.x + `sapien` install, or it fails at import.

## Gates

Run all four, serially.

```
.venv/bin/python -m pytest -q      # behaviour, 51 test modules, one process
uv run ruff check .                # dead code, unused names
PYTHONPATH=src .venv/bin/python scripts/pins/pin_train.py   # atol=0
PYTHONPATH=src .venv/bin/python scripts/pins/pin_eval.py    # atol=0
```

- Pins compare at `atol=0` (bit-identical floats). Missing or new keys fail. Use `--write` only when a change must move the numbers.
- The pins do not cover rollout, EMA, the LR schedule, or checkpoint round-trips.
- `addopts = "-n 0"`: one process. xdist workers race on `MUJOCO_LOG.TXT` and the global torch RNG. Override with `-n 4` only for throughput, never for diagnosing.
- One file: `.venv/bin/python -m pytest tests/test_guard_messages.py -q`
- `tests/conftest.py` deletes the `MUJOCO_LOG.TXT` the session created. Do not duplicate that cleanup.

## Entry

`scripts/train.py`, one run per invocation. Draccus builds `TrainConfig` (`scripts/train.py:106`).

- Top-level flags: `--env`, `--seed`, `--total_steps`, `--resume`, `--obs_dim`, `--action_dim`.
- `--video` is `--video bool`, not a bare flag. Pass `--video true`.
- `MyfpoConfig` is nested: `--myfpo.policy_lr`, `--myfpo.num_envs`, ...
- Runner scalars (`total_steps`, `eval_steps`, `save_steps`, `obs_dim`, `action_dim`) stay on `TrainConfig`. The runner bypasses `MyfpoConfig` for them, by design.

```
.venv/bin/python scripts/train.py --env Hopper-v5 --seed 0 --total_steps 2000
.venv/bin/python scripts/train.py --env Hopper-v5 --seed 0 --resume outputs/myfpo_Hopper-v5_seed0/checkpoint_1000 --total_steps 4000
```

Exit codes: 0 success, 1 usage error, 2 runtime/divergence abort or draccus parse error, 4 missing artifact.

`--resume` restores the checkpoint config and warns that scalar CLI overrides are ignored. A resumed run gets a new `_resume` directory, so it never overwrites the source run.

## Structure

`src/myfpo/` package, `scripts/` CLIs and pins, `tests/` 51 test modules, `configs/envs.yaml` the env registry.

`configs/envs.yaml` is the only dim source and drives `KNOWN_ENVS`. `scripts/train.py` checks the live env spaces against it at startup. Add an env there before passing its id.

`docs/` no longer exists. `tests/test_block_contract.py` previously cited `docs/architecture.md` (cleaned). No source files reference it now.

`myfpo_online.md` / `.ipynb` are one file in two formats. Edit the `.md`, then `jupytext --sync`. Never hand-edit the `.ipynb`.

## Env backends

The path is chosen by capability, not by name.

- numpy gym env: `step()`; `make_vec_env` forks workers when `num_envs > 1`.
- torch env: `step_torch()` (ManiSkill). It batches its own sims, so `make_train_env` must return it unwrapped. Forking would start one SAPIEN context per worker.
- Eval: `Evaluator` for CPU envs, `VectorEvaluator` for torch envs. `VectorEvaluator` needs lockstep rows, which is why `make_eval_env` sets `ignore_terminations=True`.
- `maniskill_gpu.NextStepShim` converts ManiSkill's SAME_STEP autoreset to NEXT_STEP. Without it the collector rejects the env, and the done-step `o_{t+1}` is a reset frame.
- `MS-*` ids are registered only by `scripts/play.py` (CPU, single env). A bare `gym.make("MS-...")` elsewhere fails.

## Known traps

- `scripts/pins/_shared.py` must not import from `src` beyond what it already builds. A pin sharing code with the code it checks cannot catch defects in the shared part.
- Actor and critic gradient clipping are independent (`training.py`). A joint clip silently shrinks whichever side has the larger gradient.
- The stored action is the raw flow output. The `tanh` squash happens only on the way to the env (`processor.py`). The env receives a different value than what is stored.
- A SAPIEN GPU context is budgeted at 1.5 GiB (`VIDEO_CONTEXT_BYTES`). Training, eval, and video are three contexts. Do not drop the free-memory check before the video env.
