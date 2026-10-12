# Sweeps — spectral prior on the noise base

Generated after `1008064` (the prior) and `d0f97c2` (DCT removed).

**Run these yourself.** Nothing here has been run. The numbers in
"prior evidence" come from tensorboard logs already on disk, not from these
sweeps.

---

## Before you run anything

```
.venv/bin/python -m pytest tests/ -q
```

Expected: **231 passed, 11 failed**. The 11 are pre-existing
(`chunk_budget_truncated` assertions against a field the collector no longer
produces). If you see anything else, stop — something changed.

Two things that will bite you otherwise:

- **`max_grad_norm` is now `1.0`, your old runs used `48.0`.** New results do
  not compare to runs in `tensorboard/`.
- **Run names changed.** `k<num_kept_tokens>` is now `h<action_horizon>`, and
  spectral arms get a `b<beta>` tag. Old log dirs are not comparable by name.
- **Smoke runs are nondeterministic** — the same config returns anywhere in a
  ±1.5 band on Hopper. Never compare single 12-step runs.

All flags below use underscores (`--sjfpo.n_action_steps`, not dashes).
Runner scalars are `--total_steps`, not `--sjfpo.total_steps`.

---

## T1 — The headline question

**Does the 1/f^β base prior help?**

```bash
for BETA in 0.0 0.5 1.0; do
  for SEED in 0 1 2 3 4; do
    .venv/bin/python scripts/train.py \
      --env Hopper-v5 --total_steps 200000 --seed $SEED \
      --eval_steps 10000 --save_steps 50000 --out_dir outputs/t1_beta$ \
      --sjfpo.num_envs 64 --sjfpo.chunks_per_update 1024 --sjfpo.batch_size 128 \
      --sjfpo.report_to wandb --sjfpo.stop_patience 999 \
      --sjfpo.noise_scaling_mode freq_weighted_time --sjfpo.noise_scaling_beta $BETA
  done
done
```

`BETA=0.0` is bit-identical to `noise_scaling_mode white` — it is the
ablation baseline and it short-circuits before the filter.

**Also run the white control**, to confirm β=0 reproduces it:

```bash
.venv/bin/python scripts/train.py --env Hopper-v5 --total_steps 200000 --seed 0 \
  --sjfpo.report_to wandb --sjfpo.stop_patience 999 \
  --sjfpo.noise_scaling_mode white
```

### What to read

| Metric | Where | Why |
|---|---|---|
| `noise_band_energy_hf_lf_ratio` | train | **Check this first.** ~1.0 = prior didn't run. Must fall as β rises. |
| episodic return | eval | The outcome |
| `clipfrac`, `mean_log_ratio` | train | Trust-region behaviour |
| `cfm_above_clamp_frac` | train | Should stay near 0 |

### Kill criterion — set this down now

**If β=1.0 does not beat β=0.0 on mean return across 5 seeds, that is the
result.** Publish the negative; do not extend to β=2 or to more envs.

Note the expected effect size is small. At `H=8` the lag-1 autocorrelation of a
1/f prior is **0.26, not ~1** — it is a gentle low-pass. Anything promising less
than ~1–2% return is inside seed noise at n=5.

---

## T2 — Test LP-FM's claim (the risk to the whole idea)

LP-FM (arXiv:2606.02177) reports spectral bias helps **only with adaptive
solvers**: *"we do not observe benefits from introducing a spectral bias under
Euler."* You use Euler. This sweep checks whether that transfers to RL.

```bash
for BETA in 0.0 1.0; do
  for SEED in 0 1 2; do
    .venv/bin/python scripts/train.py \
      --env Hopper-v5 --total_steps 200000 --seed $SEED \
      --sjfpo.report_to wandb --sjfpo.stop_patience 999 \
      --sjfpo.num_euler_steps 10 \
      --sjfpo.noise_scaling_mode freq_weighted_time --sjfpo.noise_scaling_beta $BETA
  done
done
```

Then rerun at `--sjfpo.num_euler_steps 32` (closer to an adaptive solver's step
count without needing a new sampler).

**If the prior only helps at 32 steps, you have a real, publishable result** —
and a cheap one. If it helps at neither, LP-FM generalises and the prior is
dead here.

---

## T3 — Does it survive the horizon?

`n_action_steps` sets the executed chunk length, which sets `γ^h` in the
backup and how much of the filter is exercised.

```bash
for NAS in 2 4 8; do
  for BETA in 0.0 1.0; do
    for SEED in 0 1 2; do
      .venv/bin/python scripts/train.py \
        --env Hopper-v5 --total_steps 200000 --seed $SEED \
        --sjfpo.report_to wandb --sjfpo.stop_patience 999 \
        --sjfpo.n_action_steps $NAS \
        --sjfpo.noise_scaling_mode freq_weighted_time --sjfpo.noise_scaling_beta $BETA
    done
  done
done
```

Shorter chunks discount less per macro-step (`gae.py`, `gamma ** lengths`), so
retune `gamma` before concluding anything from a horizon arm.

---

## T4 — Wider states, where the critic matters

Run this on a **high-`obs_dim` env**. `critic_hidden_dim` exists because
`obs_dim` runs 3 (Pendulum) to 348 (Humanoid-v5), and the critic used to be
welded to the actor's width. On Hopper (obs 11) this sweep will show nothing.

```bash
for HID in null 256 512; do
  for SEED in 0 1 2; do
    .venv/bin/python scripts/train.py \
      --env Humanoid-v5 --total_steps 200000 --seed $SEED \
      --sjfpo.report_to wandb --sjfpo.stop_patience 999 \
      --sjfpo.critic_hidden_dim $HID
  done
done
```

Optional, same seeds: `--sjfpo.num_critics 4`, `--sjfpo.critic_num_layers 3`.

---

## T5 — Re-baseline everything

Defaults moved (critic exec-length, `n_action_steps 8→4`, `max_grad_norm
48→1.0`, `cfm_loss_reduction sum→mean`). **Old results are not a baseline.**

```bash
for BL in fpo fpo++ dppo; do
  for SEED in 0 1 2 3 4; do
    .venv/bin/python scripts/train.py \
      --env Hopper-v5 --total_steps 200000 --seed $SEED \
      --baseline $BL --sjfpo.report_to wandb --sjfpo.stop_patience 999
  done
done
```

Run this **first** if you intend to compare anything.

---

## T6 — Open decisions, cheap

```bash
# gae_lambda: 0.0 is QC's one-chunk backup, which is the default.
# 0.95 is what FPO++ reports. See the note below.
--sjfpo.gae_lambda {0.0, 0.95}

# critic capacity
--sjfpo.critic_max_grad_norm {1.0, 48.0, null}
```

**Read `pre_clip_actor` / `pre_clip_critic` from T5 before touching
`critic_max_grad_norm`.** Both are already logged; the answer may be "leave it."

### On `gae_lambda`

Your repo's reference is QC (one chunk). FPO++ reports λ=0.95. They are
genuinely different papers and cannot both be the reference. `--baseline fpo++`
currently inherits `gae_lambda=0.0`, so the FPO++ baseline runs **without** the
λ that paper specifies. That is defensible only if you say so.

---

## Not wired up

`trajectory_straightness` exists and is tested
(`sample_action(return_velocities=True)`) but is **not** logged. Without it you
measure return, not the S3 mechanism (does the prior lower flow curvature?).
Wiring it is ~20 lines of rollout plumbing if you want the mechanism rather than
just the outcome.

---

## Order

1. **T5** — re-baseline. Everything is relative to it.
2. **T1** — the headline. Check `noise_band_energy_hf_lf_ratio` before reading returns.
3. **T2** — only if T1 is not flat.
4. **T3**, **T4**, **T6** — independent, can run in parallel.

Full grid is ~100 runs at 200k steps. For a first pass use
`SEEDS="0 1 2"` and widen only what looks promising.