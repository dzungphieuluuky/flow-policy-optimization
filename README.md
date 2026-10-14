# myfpo

Online reinforcement learning for **action chunks** with a flow-matching actor,
built on [LeRobot](https://github.com/huggingface/lerobot) and PyTorch.

The estimator is FPO++ ([arXiv:2602.02481](https://arxiv.org/abs/2602.02481),
Eq. 10 — one ASPO ratio per Monte-Carlo `(tau, eps)` pair), run over action
chunks: the policy plans `action_horizon` steps and executes the first
`n_action_steps` open-loop, then replans from the observation it lands on.

---

## Status

Working, with two things worth knowing before you trust a run.

**Default runs are FPO++.** Ported from their config defaults: `advantage_clamp`,
Huber CFM loss, LR warmup, joint gradient clipping, `2**arange` Fourier
frequencies, `cfm_clamp=20`. The core math (STE log-ratio cap, inverse-CDF Beta
timesteps, ASPO, the `mean`/`sum`/`sqrt` CFM reductions) is identical
expression-for-expression.

**`--baseline dppo` is not DPPO.** It only sets `num_euler_steps=10` and runs the
same FPO++ surrogate, so it is FPO++ at 10 NFE. `denoising_mdp.py` holds the
per-step sampler and is reachable only from its own test. Treat the arm as
mislabeled.

**Open performance issue.** A `Humanoid-v5` run peaked at 131,072 steps and
decayed 46% by 614,400. The leading suspect is the joint gradient clip, which
saturated at 2291x against a budget of 1 — and the actor/critic split is not
recorded, so it cannot yet be attributed.

---

## Install

Requires Python >= 3.12.

```bash
uv sync
```

`pyproject.toml` references this file, so `uv sync` and `uv run` fail without it.

## Quickstart

```bash
# cheapest env first: Pendulum-v1 is 3-dim obs and converges fast
uv run python scripts/train.py \
  --method sjfpo --env Pendulum-v1 --seed 3407 \
  --total_steps 20000 --eval_steps 2000 --save_steps 10000 \
  --sjfpo.report_to tensorboard
```

Runs land in `outputs/sjfpo_<env>_seed<seed>/`, with `eval_history.csv`,
`run_metadata.json` and `checkpoints/{best,final,step_N}/`.

### Environments

`configs/envs.yaml` is the registry; `scripts/train.py` cross-checks the declared
dims against the live spaces and fails loudly on a mismatch.

| tier | envs |
|---|---|
| smoke | `Pendulum-v1`, `InvertedPendulum-v5`, `InvertedDoublePendulum-v5`, `Reacher-v5` |
| core | `Hopper-v5`, `HalfCheetah-v5`, `Walker2d-v5`, `Swimmer-v5` |
| hard | `Pusher-v5`, `Ant-v5`, `Humanoid-v5` |
| dexterous | `AdroitHandDoor-v1` (needs `gymnasium-robotics`) |

## Baselines

A baseline is a **config diff, not a branch** — it prints every field it changes
to stderr and records them in `run_metadata.json`.

```bash
uv run python scripts/train.py ... --baseline fpo     # FPO, McAllister et al. arXiv:2507.21053
uv run python scripts/train.py ... --baseline fpo++   # FPO++, Yi et al. arXiv:2602.02481
uv run python scripts/train.py ... --baseline dppo    # NOT DPPO — see Status
```

## How it works

One chunk is one macro-action, and one macro-action is one macro-transition.

- **Actor.** A DiT (`dit_depth=3`, `dit_heads=2`, `hidden_dim=128`) predicts a
  velocity field over a chunk of `action_horizon=8` actions conditioned on a
  `n_obs_steps=16` observation window. Only the first `n_action_steps=4` execute,
  then the policy replans — receding horizon, open loop within a chunk.
- **Target.** Flow matching with `v* = a - eps`. The loss is the squared velocity
  error, Huber by default (`delta=0.5`), masked by `action_is_pad` and
  renormalised by executed count so a short chunk keeps a full chunk's magnitude.
- **Ratio.** `exp(l_old - l_new)`, standing in for a log-probability ratio. The
  same `(tau, eps)` draws are reused for both, and `l_old` comes from a frozen
  snapshot of the behaviour policy. One log-ratio per action step, capped by a
  straight-through estimator at `log_ratio_cap=10`.
- **Trust region.** ASPO: PPO where `A > 0`, SPO where `A < 0`, elementwise,
  `clip_param=0.05`.
- **Critic.** A 2-member ensemble conditioned on the chunk's executed fraction.
  `amin` for the bootstrap, mean for the loss. No target network.
- **Advantages.** Chunk-level GAE. `A_i = delta_i + gamma^h * lambda * (1-term)
  * (1-trunc) * A_j` over the next chunk of the same env; `gae_lambda=0.95` is
  the default and `0.0` is QC's one-backup opt-out. Timeouts enter through the
  reward the way rsl-rl does it, with `done` masking the bootstrap so the
  correction lands once.

Full metric list in `train_sjfpo.py`; the mapping from GAE diagnostics to log keys
is the `_GAE_DIAG_LOG_KEYS` table there.

## Configuration

`configs/sjfpo.yaml` is generated, not hand-edited:

```bash
uv run python scripts/gen_sjfpo_config.py
```

It is parity-tested against the dataclass, so regenerating after a field change
is required, not optional.

Any field can be overridden per run:

```bash
--sjfpo.action_horizon 16 --sjfpo.gae_lambda 0.0 --sjfpo.num_envs 8
```

Runner scalars (`total_steps`, `eval_steps`, `save_steps`, `obs_dim`,
`action_dim`) are **top-level flags, not config fields** — putting them in the
dataclass would bypass it by design.

## Testing

```bash
uv run pytest -q tests            # 376 tests, distributed by default (-n 4)
uv run pytest -q tests -n 0       # serialise when a failure needs a readable trace
```

There is **no CI**. `pytest -q` is the only gate.

The guard-message snapshot is the one thing to know about:

```bash
uv run pytest -q tests/test_guard_messages.py --snapshot-regen -n 0
```

Every `ValueError` guard in the config and the model modules has its message
snapshotted, so a reworded guard fails a test instead of silently changing what a
user sees. `EXPECTED_CASES` in `tests/test_guard_messages.py` trips if a guard is
added or dropped without the snapshot being regenerated.

## Repo layout

```
src/myfpo/
  configuration_sjfpo.py   every hyperparameter, all validation inline
  modeling_sjfpo.py        SJFPOPolicy: rollout, forward, the loss
  fpo_core.py              CFM loss, ratio, ASPO, Euler sampler
  gae.py                   chunk-level advantages
  collector.py             open-loop chunk collection, RolloutBuffer
  critic.py                the ensemble
  flow.py, layers.py, attention.py, encoder.py   actor and building blocks
  baselines.py             fpo / fpo++ / dppo as auditable config diffs
  denoising_mdp.py         test-only, see Status
  evaluator.py, tracking/, checkpointing.py
scripts/
  train.py                 the CLI, the real entry point
  play.py, cuda_probe.py, gen_sjfpo_config.py, build_notebook.py
  verify_deps.sh           stale, see scripts/AGENTS.md
configs/
  sjfpo.yaml               generated
  envs.yaml                env registry with dims and tiers
  ablations/
tests/                     380 tests
sjfpo_online.ipynb / .md   Colab + Kaggle notebook, paired via jupytext
```

## Online notebook

`sjfpo_online.ipynb` (paired with `.md` via `jupytext`) clones the repo, installs
dependencies and runs training on Colab or Kaggle. It ends with sweep cells that
reproduce the degradation diagnosis. It needs `GITHUB_TOKEN` and optionally
`WANDB_API_KEY` from the runtime secret store.

Keep the two files in sync after editing either:

```bash
uv run --no-project --with jupytext jupytext --sync sjfpo_online.md
```

`--set-formats ipynb,md` is required once to establish the pairing; without a
`jupytext:` front matter `--sync` exits 0 and writes nothing.

## Known rough edges

- `scripts/verify_deps.sh` asserts `torch>=2.2,<2.6` and `numpy<2.1`, and installs
  a `requirements-local.txt` that does not exist. It fails at that install step
  before reaching the version checks. See `scripts/AGENTS.md`.
- `--baseline dppo` is mislabeled, and `denoising_mdp.py` is dead weight.
- No CI, and `trajectory_straightness` in `fpo_core.py` is never called from
  training.

## License

See [`LICENSE`](LICENSE).