# AGENTS — scripts (ASD-STE 100)

Read the root `AGENTS.md` first. This file holds only script-local facts.

Entry: `scripts/train.py`. Draccus builds `TrainConfig`; `FlowPolicyOptimizationConfig` is the nested `--flow_policy_optimization.` block.

Do not add runner scalars (`total_steps`, `eval_steps`, `save_steps`, `obs_dim`, `action_dim`) to `FlowPolicyOptimizationConfig`. They stay on `TrainConfig`, and the runner bypasses the config for them, by design.

`--resume` restores model, optimizer and step only. The config is not restored, so scalar CLI overrides apply to the resumed run. It is flow_policy_optimization-only.

`scripts/play.py` registers the `MS-*` ids against a single-env CPU sim before `gym.make`.
