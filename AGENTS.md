# AGENTS — myfpo (ASD-STE 100)

Research repo for FPO++ on MuJoCo robots. Flow-matching policy, open-loop action chunks, online RL with a CFM-loss surrogate ratio.

Write in Simplified Technical English: short sentences, active voice, no semicolons, one instruction per sentence. Name the file and the line.

## Setup

```bash
uv sync                 # installs myfpo editable plus the dev group
```

A fresh clone has no `.venv`. Run `uv sync` before `pytest` or any script. `uv.lock` is the pin source. `requirements.txt` does not exist.

## Gates

```bash
.venv/bin/python -m pytest -q    # 518 tests, about 10 s
uv run ruff check .              # dead code and unused names
```

No CI. Run both gates before you commit.

`pyproject.toml` sets `addopts = "-n 0"`. The suite runs in one process on purpose. Do not add `-n`. Several workers contend for the same cores, the same `MUJOCO_LOG.TXT`, and the same torch RNG.

## The behaviour pins

The third gate. It compares every number a fixed computation produces, at `atol=0`.

```bash
PYTHONPATH=src .venv/bin/python scripts/pins/pin_train.py
PYTHONPATH=src .venv/bin/python scripts/pins/pin_eval.py
```

Exit code 0 means no difference. Any other value means at least one.

The baselines are binary and are not tracked. A fresh clone has the scripts and no baseline. The first run fails with "no baseline. Run with --write first." Use `--write` only when the change is meant to move the numbers. Record the key count in the commit message.

`pytest` is not this gate. On 2026-10-17 the suite passed 588 tests while a change to the parameter construction order silently changed every actor weight. The pins caught it.

A pin that fails at HEAD, before any change, has a stale baseline. Regenerate it with `--write` and say so in the commit.

Do not import from `src` into `scripts/pins/_shared.py` beyond what it already builds. A pin that shares code with the code it checks cannot catch a defect in the shared part.

## Commands

```bash
.venv/bin/python scripts/train.py --env Hopper-v5 --seed 3407 --total_steps 2000
.venv/bin/python scripts/multiseed.py run --env Hopper-v5 --seeds 3407,1,2 --total_steps 20000 --out outputs/sweep
.venv/bin/python scripts/multiseed.py compare outputs/sweep_a/sweep_summary.csv outputs/sweep_b/sweep_summary.csv
.venv/bin/python scripts/play.py --env Hopper-v5 --checkpoint outputs/myfpo_Hopper-v5_seed3407/checkpoints/best
.venv/bin/python scripts/determinism_probe.py --env Hopper-v5 --checkpoint outputs/myfpo_Hopper-v5_seed3407/checkpoints/best --repeats 3
.venv/bin/python -m jupytext --sync myfpo_online.md
```

Exit codes: 0 ok, 1 bad command line, 2 the run failed, 4 an artifact is missing.

`determinism_probe.py` measures the noise floor of the evaluator. Run it before you trust any small delta.

One run writes to `outputs/myfpo_<env>_seed<seed>/`. The checkpoints sit in `checkpoints/` inside it. `--out_dir` changes the parent.

## Config

`src/myfpo/configuration.py` holds every hyperparameter. It is the only source. Each field is a `--myfpo.` flag. `scripts/train.py` reads the dataclass, it does not copy the fields.

Do not add runner scalars (`total_steps`, `eval_steps`, `save_steps`, `obs_dim`, `action_dim`) to `MyfpoConfig`. The runner bypasses the config for them, by design.

One file holds both. The top of `src/myfpo/configuration.py` is the dataclass; the environment registry read from `configs/envs.yaml` sits below it.

`configs/envs.yaml` is the source of truth for `obs_dim` and `action_dim`. The runner checks the live env space against it at startup. A mismatch is a loud error. It usually means a gym version change. Re-measure and update the file. Do not pass dims for a normal run.

`--resume` makes the checkpoint config win. CLI overrides of scalar fields are then ignored, and `scripts/train.py` warns about them.

## Layout

```
src/myfpo/          the package, 13 modules. The leaves are layers,
                     configuration, critic, evaluator and tracking.
                     processor builds on layers (the obs encoder and the
                     running normaliser live there). flow builds on layers.
                     fpo_core builds on configuration (the training metric
                     functions live there). rollout builds on processor,
                     fpo_core and layers (the collector, the chunk GAE and
                     the rollout procedure live there). modeling builds on
                     flow, critic, processor, rollout, fpo_core,
                     configuration and layers. checkpointing builds on
                     modeling. training builds on checkpointing, rollout,
                     configuration, evaluator, fpo_core, modeling,
                     processor, runner, tracking and layers.
scripts/            train.py is the entry. pins/ holds the behaviour gates.
tests/              49 files. test_guard_messages.py pins 77 guard
                     messages. test_ratio_axes.py pins the paper configs.
docs/               architecture.md explains the networks and why.
                     ablation_reference.md lists ready-made ablations.
                     token_advantage_design_a.md is a deferred plan. It is
                     not implemented. Read it before you change the advantage
                     granularity.
configs/envs.yaml   the environment registry.
```

No cycles, including the deferred imports. `docs/architecture.md` has the layer map.

## Invariants that break silently

**The action space.** The flow outputs the raw action, and the raw value is what the policy trains on. The `tanh` squash happens once, on the way to the env. The collector stores the raw chunk. `processor.squash_to_env` is the one implementation. `processor.env_squash_terms` owns the affine terms. Use both, and do not write the formula again.

**Checkpoints carry `action_space.json`.** A checkpoint without it has a critic that read the squashed chunk, so `load_policy_checkpoint` refuses it. Every checkpoint written before 2026-10-20 is void.

**The ratio is per token, the advantage is per chunk.** `ratio_token` has shape `[n, M, H]`. `normalised_advantages` has shape `[n]`. One scalar covers all 8 tokens of a chunk. This mismatch is the open problem in `docs/token_advantage_design_a.md`. Do not change the granularity in either direction without reading that file.

**The chunk reward is discounted per environment step.** `rollout.py` (the absorbed `gae.py` section) sums `gamma**offsets * r` over the 8 steps and bootstraps with `gamma**length`. `length` is 8 for a full chunk and smaller when the episode ended first. The discount clock is per step, not per chunk. Do not re-discount on top of that.

**Causal attention.** `layers.py` (the absorbed `attention.py` section) uses `is_causal=True`. Token `i` depends on the prefix only. The actor is factorised over the chunk. This is what makes a per-token advantage well posed.

**The actor outputs zero at init.** `out` and `adaLN_out` are zero-initialised. A fresh actor predicts zero velocity.

**The critic reads the action chunk without a norm.** `critic.ActionAffine` is a learned affine. It computes no statistics from the chunk. A `LayerNorm` there would subtract the chunk's mean, so a chunk of all +5 and a chunk of all 0 would reach the critic as the same input. Do not put a norm back on the action.

**The critic ensemble default is 2, not 5.** TD-MPC2 ships 5. `--myfpo.num_critics 5` restores the paper width. The docs follow the code.

**Storage dtype follows the GPU model.** `utils.resolve_storage_dtype` returns bf16 on an RTX PRO 1000 Blackwell and fp32 elsewhere. The same code then gives different numbers on different machines. This is a known hazard. It is not fixed.

## Names and functions

Every name in `src/myfpo` is public. Do not write a leading underscore on a function, a class, a method, an attribute, or a constant.

Two kinds of name keep the underscore. Python and the frameworks force them:

- the dunders, `__init__` and `__call__`.
- `RunningMeanStd._load_from_state_dict`, the `nn.Module` hook. Rename it and the normaliser latch never resets on load.

Check the package with one command. It must print the `nn.Module` hook and nothing else.

```bash
grep -rn "def _[^_]\|class _\|^_[A-Z]" src/myfpo/
```

A function does one job. Keep it under 100 lines. Split a longer one into named steps. The next command must print nothing.

```bash
.venv/bin/python -c "import ast,pathlib;[print(p,n.name) for p in pathlib.Path('src/myfpo').rglob('*.py') for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef) and n.end_lineno-n.lineno+1>100]"
```

A helper with one caller and a short body is inlined. The name goes.

A helper that is a real step keeps its name. The name is plain.

A public name is part of the package API. Add one only when it names a step of the algorithm or when it replaces two or more old names.

The test helpers and the script helpers keep their underscore. They are local to their file and no other file imports them.

## Adding a guard

`tests/test_guard_messages.py` has a table of 73 cases and a count. Add the case, then update `EXPECTED_CASES`. The test fails if the count moves without a deliberate change.

## The notebook pair

`myfpo_online.md` and `myfpo_online.ipynb` are one file in two formats. Edit the `.md`, then run `jupytext --sync`. Do not hand-edit the `.ipynb`. Ruff skips `.ipynb` files.

## The guard test at `tests/test_guard_messages.py` and the paper

`tests/test_ratio_axes.py` pins the FPO and FPO++ paper configurations. The comment there names the equations. Keep those mappings true when you touch the ratio or the trust region.

## Known open issues

- No statistical significance test for the metrics. The naming and the NaN checks only.
- No chunk-boundary mitigation. `rollout_seam_jerk` and `rollout_jerk` are measured, not fixed.
- The approximate ratio `ρ̂ = exp(L_old - L_new)` is a biased proxy. The bias is not quantified in a test.
- The raw action has no clip. `train/raw_action_abs_max` is the monitor. A drifting raw magnitude makes `cfm_clamp` bind and pins the ratio at 1.
- No `torch.utils.checkpoint` in `flow.py` or `rollout.py`.
- No experiment results in git. A run's numbers live in its wandb run.

## Style

- Result first, and the bad news first when it changes what to do.
- Be concise. Bullets, not paragraphs. One idea per line.
- Say whether a claim was verified by running the code or inferred.
- A behaviour-preserving change must show the exact invariant, `atol=0`.
- A plan that conflicts with an existing decision: stop and surface it.
- Correct an error in one line.
- Ask before a large or hard-to-reverse change.
- Commit in focused chunks. A commit message is under 30 words and says why.
