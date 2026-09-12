# spectral-joint-policy

SJFPO (Spectral Joint Flow Policy Optimization) — a flow-matching policy-gradient method with spectral state conditioning — benchmarked against skrl PPO/SAC on MuJoCo v5 continuous control.

Research code: correctness, clarity, reproducibility over infrastructure. See `AGENTS.md` and `src/sjfpo/AGENTS.md` before modifying code.

## Method

**SJFPO (live train path, actions-only):**

- Stack `T=16` obs frames → `StateEncoder` (per-frame MLP + self-attn) → `Z [N,T,D=128]`.
- DCT-II over the raw obs window (time axis) → keep `K` lowest low-frequency bins (`K = round(top_p*T)`, default 5) → raw bins = `SpectralEncoder` conditioning `[N,K,obs]`, no projector, no calibration, no encoder latent. (rFFT is reserved for `observation.image`; state and action use DCT.)
- Action chunk `[N,H=16,Da]` stored as orthonormal DCT-II coefficients. Flow (DiT-8, adaLN-Zero, `SJFOPolicyModel`) attends over `[state_embed(raw obs); noisy actions]` prefix tokens and predicts action velocities in DCT space.
- Sample with `fpo_core.euler_sample` (descending `t: 1→0`, `x ← x − δ·v̂`); decode with `runner.decode_chunk_actions` (IDCT + single `clamp(-1,1)`). Training executes the full predicted chunk open-loop (`H=16`); inference/eval execute the `executed_vs_predicted_ratio=0.5` prefix (8 steps), then replan.
- Update (`SJFPOTrainer.assemble_update`): per-sample MC log-liks `[N,m=16]` → STE-capped ratios → ASPO surrogate (PPO-clip for `A≥0`, SPO penalty for `A<0`, mean-of-ψ) + `V(s,a)` TD critic (`R_k + γ^L·V(s',a')`, terminated never bootstraps) + Epps-Pulley `sigreg_loss` on the critic latent. One AdamW optimizer over policy + critic + critic trunk (spectral is parameter-free); per-module grad-clip; NaN aborts via `SJFPOAborted`.
- Chunk rollout uses LeRobot-keyed `Transition` dicts; chunk-level semi-MDP GAE (`gae.chunk_gae`, numpy fp64, advantages centered).

**Baselines:** skrl 2.1.0 PPO / SAC via shared `baseline_train_loop` (`src/sjfpo/baselines.py`). No spectral/flow code.

Single source of truth: every module under `src/sjfpo/` is on the live path. The notebooks carry no source — they clone this repo and shell out to `scripts/train.py`.

| | |
|---|---|
| Tasks | `HalfCheetah-v5`, `Walker2d-v5`, `Hopper-v5` (flat Box, MuJoCo-only) |
| Methods | `sjfpo`, `ppo`, `sac` |
| Matrix | 45 runs: 3 methods × 3 envs × 5 seeds × 1M steps; tiny = 3 runs (Hopper-v5, seed 0, 20k steps) |
| Precision/device | fp32 only (no autocast/half/compile); CUDA-default with CPU fallback (`SJFPO_DEVICE=cuda\|cpu\|auto`) |

## Layout

```
src/sjfpo/          # all production code (MuJoCo-only, fp32)
  config.py         # frozen SJFPO/PPO/SAC dataclasses, RUN_MATRIX, build_filtered_matrix
  networks.py       # policy/critic StateEncoders, SpectralEncoder, SJFOPolicyModel (DiT), Critic, sigreg_loss
  fpo_core.py       # CFM losses, flow_ratio (STE cap), aspo_surrogate, policy_ratio, euler_sample, total_update_loss
  collector.py      # Transition dict (9 LeRobot keys), open-loop full-H chunks in training, next.action chaining
  gae.py            # chunk semi-MDP GAE (R_k intradiscounted, γ^L bootstrap, (γλ)^L recursion)
  trainer.py        # SJFPOTrainer.assemble_update (ASPO + V(s,a) TD + SigReg)
  runner.py         # train_run: collect → GAE → update → eval → ckpt; decode_chunk_actions
  baselines.py      # skrl PPO/SAC (FrameStackWrapper, baseline_train_loop)
  evaluator.py      # Evaluator → 3-col CSV [episode,return,length] + divergence gate
  checkpointing.py  # atomic checkpoints (models + optimizer + counters)
  sweep.py          # scan_runs / run_pending, format_run_directory_name
  analysis.py       # IQM + bootstrap (n_boot=2000), invoked from notebooks
  wandb_logging.py  # sole wandb importer (REQUIRED); 6 fns, SJFPO_DISABLE_WANDB=1 off switch
  utils.py          # resolve_device, safe_float_value (no sjfpo imports)
configs/            # sjfpo.yaml, ppo.yaml, sac.yaml, shared.yaml (mirror config.py)
scripts/train.py    # unified CLI (single-run + --matrix)
scripts/build_notebook.py  # builds both minimal Colab .ipynb (never hand-edit notebooks)
tests/              # 290 tests; test_pipeline_cpu.py invariants never touched
docs/               # architecture, modules, config-flags, testing, wandb, troubleshooting
```

## Install

Requires Python ≥ 3.12.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .                    # or:
.venv/bin/python -m pip install -r requirements-linux.txt   # linux / -amd variant on AMD
```

Hard pins: `gymnasium==1.3.0` (range `>=1.0,<1.3`), `skrl==2.1.0`, `einx==0.4.3`, `torch==2.14.0`, `mujoco==3.13.0`. `sjfpo.config.assert_runtime` enforces the gymnasium/skrl pins; `scripts/verify_deps.sh` is stale (references deleted `requirements-local.txt` and an old torch range — use the pins above).

## Usage

```bash
# single run (defaults: total-steps 2000, eval/ckpt min(1000, total))
PYTHONPATH=src .venv/bin/python scripts/train.py \
  --method sjfpo --env Hopper-v5 --seed 0 --total-steps 2000

# matrix sweep (RUN_MATRIX order: method × env × seed)
PYTHONPATH=src .venv/bin/python scripts/train.py \
  --matrix --methods sjfpo ppo --envs Hopper-v5 --seeds 0 1

# tiny smoke: 3 runs, Hopper-v5 seed 0, 20k steps (overrides filters)
PYTHONPATH=src .venv/bin/python scripts/train.py --matrix --tiny
```

Precedence: **CLI flags > `--config <file>.yaml` > dataclass defaults**. Singular `--method/--env/--seed` for single runs; plural `--methods/--envs/--seeds` only with `--matrix` (`--tiny` requires `--matrix`).

Exit codes: `0` success · `1` usage error · `2` runtime/divergence abort · `4` missing artifact.

Env switches: `SJFPO_DEVICE=cuda|cpu|auto` · `SJFPO_DISABLE_WANDB=1` (only wandb off switch; wandb is a required import) · `SJFPO_TINY_MATRIX=1` (same as `--tiny`) · `SJFPO_LEGACY_MUJOCO=1` (default, MuJoCo-only).

Key `SJFPOConfig` defaults: `num_mc_samples=16`, `euler_train=64`, `euler_eval=16`, `chunks_per_update=512`, `n_obs_steps=16`, `chunk_size=16`, `executed_vs_predicted_ratio=0.5`, `top_p_spectral_tokens=0.3`, `hidden_size=128`, `dit_depth=8`, `dit_heads=4`, `dit_mlp_ratio=4.0`, `dit_time_dim=256`, `lr=1e-4`, `wd=1e-4`, `clip_range=0.05`, `cfm_clamp=20`, `log_ratio_cap=10`, `sigreg_coef=0.1`, `vf_coef=1.0`, `grad_clip=1.0`, `gamma=0.99`, `gae_lambda=0.95`.

## Outputs

`--out-dir` defaults to `.omo/evidence/train` (git-ignored). Single mode writes under `<out-dir>/<method>/`; matrix mode under `<out-dir>/<method>_<env>_seed<seed>/` (`sweep.format_run_directory_name`):

- `eval_history.csv` — `episode,return,length`
- `checkpoint_<step>/` + `checkpoint_final/` — `checkpoint.pt` + `manifest.json`
- wandb `train/*`, `eval/*`, `gae/*`, `timing/*` when enabled (one `log_metrics(data, step=global_step)` per update)

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q                  # 290 tests
PYTHONPATH=src .venv/bin/python -m pytest tests/test_pipeline_cpu.py -v
```

Numeric gates: DCT roundtrip `<1e-5`, spectral order tol `2e-3`, `einx.id` matches torch at `1e-6`.

## Notebooks

```bash
.venv/bin/python scripts/build_notebook.py
```

Regenerates both minimal Colab notebooks (clone + install + `!train.py`, no source inlined). `*.ipynb` are git-ignored build artifacts.

## Conventions (read `src/sjfpo/AGENTS.md` first)

- Tensor-order comments on every transform (`[N,T,D]`, `[N,H,Da]`, `[N,K,obs]`, `[N,Ka,Da]`, `[N,m]`); `einx.id` reshapes must match torch.
- Frozen dataclass configs; YAML files mirror them field-for-field (`tests/test_config_yaml_parity.py` locks this).
- `wandb` imported only in `wandb_logging.py`.
- `retained:`-marked helpers (`cfm_loss`, `flow_loglik`, `policy_loss`, `assert_runtime`, …) are test/notebook seams — don't delete.
