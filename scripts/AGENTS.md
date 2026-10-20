# AGENTS — scripts (ASD-STE 100)

Read the root `AGENTS.md` first. It holds the setup, the gates, the commands, and the invariants. This file holds only the facts that are specific to `scripts/`.

Entry: `scripts/train.py`. The CLI takes the `MyfpoConfig` dataclass. Every field is a `--myfpo.` flag. The runner reads the dataclass, it does not copy the fields.

Do not add runner scalars (`total_steps`, `eval_steps`, `save_steps`, `obs_dim`, `action_dim`) to `MyfpoConfig`. The runner bypasses the config for them, by design.

`--resume` makes the checkpoint config win. CLI overrides of scalar fields are then ignored. `scripts/train.py` warns about them.

`scripts/pins/` holds the behaviour gates. Read `scripts/pins/README.md` before you change anything under `scripts/pins/`.

`myfpo_online.md` and `myfpo_online.ipynb` are one file in two formats. Edit the `.md`, then run `jupytext --sync`. Do not hand-edit the `.ipynb`.

`uv sync` installs the package editable. The scripts have no `sys.path` code. A fresh clone needs `uv sync` before any script runs.
