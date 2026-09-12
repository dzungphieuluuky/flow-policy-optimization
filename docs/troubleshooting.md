# Troubleshooting

## Tests

**`ModuleNotFoundError: No module named 'sjfpo'`**

Run with `PYTHONPATH=src`:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

**Spectral test flake**

Keep fp32, no `autocast`/`half`/`bfloat16` — the fp32-only gate scans for
these. Check the DCT→low-K→frozen→Linear order in
`networks.py::SpectralEncoder` first.

## Smoke / Runner

**`CalibrationError` or `No eval histories` after smoke**

Smokes write under `--out-dir` (default `.omo/evidence/...`). If you changed
`--out-dir` or matrix filters mid-run, clear the old dir or re-run with the
intended flags. The analysis cell's empty-history guard skips gracefully.

**`train_run` exits non-zero**

`scripts/train.py` exits: 0 ok, 1 usage conflict (e.g. `--tiny` without
`--matrix`, singular + plural flag mix), 2 runtime exception, 4 missing
artifact. `--help` lists the conflict rules.

**Config parsed at import time**

`config.py` reads env flags at import — restart the process after flipping an
env var (don't just set `os.environ` in a REPL).

**`Sync batchnorm` / `Distributed` errors**

Not used — everything is single-process CPU-first. If you pulled in `skrl`
distributed helpers, drop them; the supported path is `baseline_train_loop`.

## Notebook Build

**`build_notebook.py` output drifts from `src/`**

Rebuild after editing `src/` — never hand-edit ipynbs. Verify:
`grep -n "from src\." spectral_joint_flow.ipynb` must be 0, cell counts
61 / 47, `nbformat.validate` passes.

**`jupyter nbconvert --execute` hangs**

Tiny preset is required locally: `SJFPO_TINY_MATRIX=1 … --ExecutePreprocessor.timeout=1800`.
Full matrix (45×1M) will time out on a laptop.

**Duplicate class / global collision in concatenated ipynb**

The builder's register-snippet filters `__`-prefixed globals before pumping
into `sys.modules`. New modules must `__`-prefix internal globals or use a
unique module name.

## Git

**`.omo/evidence` ignored**

Correct — `.gitignore` ignores it. Force-add (`git add -f`) only evidence you
intend to persist.
