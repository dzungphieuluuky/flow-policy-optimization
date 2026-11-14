# AGENTS — scripts (ASD-STE 100)

Read the root `AGENTS.md` first. This file holds only script-local facts.

Entry: `scripts/train.py`. Draccus builds `TrainConfig`; `MyfpoConfig` is the nested `--myfpo.` block.

Do not add runner scalars (`total_steps`, `eval_steps`, `save_steps`, `obs_dim`, `action_dim`) to `MyfpoConfig`. They stay on `TrainConfig`, and the runner bypasses the config for them, by design.

`--resume` restores model, optimizer and step only. The config is not restored, so scalar CLI overrides apply to the resumed run. It is myfpo-only.

`scripts/pins/` held the `atol=0` behaviour gates. It was deleted on 2026-11-15
when the trust region and the ratio cap changed by design. See
`SIMPLIFICATION_PLAN.md`, Amendment 1.

`scripts/play.py` registers the `MS-*` ids against a single-env CPU sim before `gym.make`.
