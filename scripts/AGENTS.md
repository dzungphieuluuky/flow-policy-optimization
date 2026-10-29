# AGENTS — scripts (ASD-STE 100)

Read the root `AGENTS.md` first. This file holds only script-local facts.

Entry: `scripts/train.py`. Draccus builds `TrainConfig`; `MyfpoConfig` is the nested `--myfpo.` block.

Do not add runner scalars (`total_steps`, `eval_steps`, `save_steps`, `obs_dim`, `action_dim`) to `MyfpoConfig`. They stay on `TrainConfig`, and the runner bypasses the config for them, by design.

`--resume` makes the checkpoint config win. Scalar CLI overrides are then ignored, and `scripts/train.py` warns.

`scripts/pins/` holds the behaviour gates. Read `scripts/pins/README.md` before you change anything under `scripts/pins/`.

`pin_train.py` and `pin_eval.py` add their own directory to `sys.path` to import `_shared`. That is the only `sys.path` code in the repo.

`scripts/play.py` registers the `MS-*` ids against a single-env CPU sim before `gym.make`.
