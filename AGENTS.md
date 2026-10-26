# AGENTS — myfpo (ASD-STE 100)

Setup, gates, commands, invariants. `scripts/AGENTS.md` holds only script-local facts.

## Setup

`uv sync` installs the package editable + dev tools. A fresh clone needs this before anything runs.
Python 3.12 (see `.python-version`). The package imports as `src/myfpo`; nothing adds `sys.path`.

```
uv sync
```

## Gates

Run all three. Serial by default — do not add `-n` unless you are not diagnosing.

```
.venv/bin/python -m pytest -q      # behaviour, 51 test files, one process
uv run ruff check .                # dead code, unused names
PYTHONPATH=src .venv/bin/python scripts/pins/pin_train.py   # behaviour pin, atol=0
PYTHONPATH=src .venv/bin/python scripts/pins/pin_eval.py    # behaviour pin, atol=0
```

The pins compare at `atol=0` (bit-identical floats). Missing or new keys fail. Regenerate baselines only with `--write`, and only when a change is meant to move the numbers. The pins do not cover rollout, EMA, the LR schedule, or checkpoint round-trips.

Tests run in one process (`addopts = "-n 0"`). xdist workers fight over `MUJOCO_LOG.TXT` and the global torch RNG. Override with `-n 4` only for throughput, never for diagnosing.

Run one test file:

```
.venv/bin/python -m pytest tests/test_guard_messages.py -q
```

## Entry

`scripts/train.py` — one run per invocation. Draccus builds `MyfpoConfig`; every field is a `--myfpo.` flag.

```
.venv/bin/python scripts/train.py --env Hopper-v5 --seed 0 --total_steps 2000
.venv/bin/python scripts/train.py --env Hopper-v5 --seed 0 --resume outputs/myfpo_Hopper-v5_seed0/checkpoint_1000 --total_steps 4000
```

Exit codes: 0 success, 1 usage error, 2 runtime/divergence abort, 4 missing artifact, 2 draccus parse error.

Do not add runner scalars (`total_steps`, `eval_steps`, `save_steps`, `obs_dim`, `action_dim`) to `MyfpoConfig`. The runner bypasses the config for them, by design.

`--resume` makes the checkpoint config win; CLI overrides of scalar fields are then ignored (the script warns).

## Structure

`src/myfpo/` — the package. `scripts/` — CLIs and pins. `tests/` — 51 test files, no `conftest` magic beyond `fixtures.py`. `docs/` — architecture record, ablation reference, token-advantage design. `myfpo_online.md` / `.ipynb` are one file in two formats: edit the `.md`, then `jupytext --sync`. Never hand-edit the `.ipynb`.

## Known traps

- `scripts/pins/_shared.py` must not import anything from `src` beyond what it already builds. A pin sharing code with the code it checks cannot catch defects in the shared part.
- `critic` and `actor` gradient clipping are independent (`training.py`). A joint clip silently shrinks whichever side has the larger gradient.
- The stored action is the raw flow output; `tanh` squash to env bounds happens only on the way to the env (`processor.py`). The env receives a different value than what is stored.
- `AGENTS.md` was deleted from the repo root and restored here. `scripts/AGENTS.md` still points at it; keep them consistent.
