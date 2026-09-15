# Troubleshooting

## Tests

**`ModuleNotFoundError: No module named 'sjfpo'`**

Run with `PYTHONPATH=src`:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

**Spectral test flake**

Keep fp32, no `autocast`/`half`/`bfloat16` — the fp32-only gate scans for
these. Check the DCT→low-K→Linear order in
`networks.py::SpectralEncoder` first.

## Smoke / Runner

**`No eval histories` after smoke**

Smokes write under `--out-dir` (default `.omo/evidence/train`). If you changed
`--out-dir` or matrix filters mid-run, clear the old dir or re-run with the
intended flags. `eval_history.csv` appears only after the first eval steps.

**SJFPO runs** exit non-zero

`scripts/train.py` exits: 0 ok, 1 usage conflict (e.g. `--tiny` without
`--matrix`, singular + plural flag mix), 2 runtime exception, 4 missing
artifact. `--help` lists the conflict rules.

**Config parsed at import time**

`config.py` reads env flags at import — restart the process after flipping an
env var (don't just set `os.environ` in a REPL).

**`Sync batchnorm` / `Distributed` errors**

Not used — everything is single-process, CUDA-default with CPU fallback. If you pulled in `skrl`
distributed helpers, drop them; the supported path is `baseline_train_loop`.

## Notebook Build

**Notebook content looks stale**

Notebooks carry no source — rebuild after changing the command set:
`python scripts/build_notebook.py`, then `nbformat.validate` passes and both
files stay at 6 cells each. Never hand-edit ipynbs.

**`jupyter nbconvert --execute` hangs**

Full runs (1M steps, let alone the 45×1M matrix) exceed Colab/laptop time
budgets — execute only the smoke cell remotely, keep full runs for machines
with accelerators.

## Git

**`.omo/evidence` ignored**

Correct — `.gitignore` ignores it. Force-add (`git add -f`) only evidence you
intend to persist.
