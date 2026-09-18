# Troubleshooting

## Tests

**`ModuleNotFoundError: No module named 'sjfpo'`**

Run with `PYTHONPATH=src`:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

**Spectral test flake**

Keep fp32 storage; bf16 autocast applies to matmuls on CUDA only (CPU paths
stay fp32). Check the state→conditioner→flow order in
`networks.py::StateConditioner` first.

## Smoke / Runner

**`No eval histories` after smoke**

Smokes write under `--out_dir` (default `.omo/evidence/train`). If you changed
`--out_dir` or sweep axes mid-run, clear the old dir or re-run with the
intended flags. `eval_history.csv` appears only after the first eval steps.

**SJFPO runs** exit non-zero

`scripts/train.py` exits: 0 ok, 1 usage error, 2 runtime/divergence abort,
4 missing artifact (draccus `--help` shows the flags; flags use underscores).

**Config parsed at import time**

`config.py` reads env flags at import — restart the process after flipping an
env var (don't just set `os.environ` in a REPL).

**`Sync batchnorm` / `Distributed` errors**

Single-process by default (CUDA with CPU fallback); multi-GPU goes through
`accelerate launch`, not manual `DistributedDataParallel`. If you pulled in
`skrl` distributed helpers, drop them; the supported paths are the plain
loop and `baseline_train_loop`.

## Notebook Build

**Notebook content looks stale**

Notebooks carry no source — rebuild after changing the command set:
`python scripts/build_notebook.py`, then the jupytext structure check passes
and both files stay at 8 cells each. Never hand-edit ipynbs.

**`jupyter nbconvert --execute` hangs**

Full runs (1M steps, let alone the 45×1M matrix) exceed Colab/laptop time
budgets — execute only the smoke cell remotely, keep full runs for machines
with accelerators.

## Git

**`.omo/evidence` ignored**

Correct — `.gitignore` ignores it. Force-add (`git add -f`) only evidence you
intend to persist.
