# Diagnosing late-stage degradation

## The symptom

`Humanoid-v5`, seed 3407, 614,400 env steps, 2700 s at 227 fps.

| | |
|---|---|
| best return | 367.94 at step **131,072** (21% of the run) |
| final return | 198.67 |
| `regressed` | 1, at step 614,400 (the last step) |
| `lr_decays` | 3 of 3 — exhausted |
| vs random baseline 113.97 | 1.74x at the end, 3.23x at the peak |

The policy peaked early and gave back 46%. `eval/mean_length` is 42.16 ± 0.42
across 50 episodes, so every eval episode ends almost immediately.

## What the log already proves

| Metric | Value | Reading |
|---|---|---|
| `train/grad_norm_preclip` | 2291.5 vs `max_grad_norm` 1 | clip saturated, **2291x** |
| `train/value_loss` | 611.44 | 5 orders above the policy loss |
| `train/policy_loss` | −0.00247 | contributes nothing to the gradient |
| `train/explained_var` | 0.186 | critic never fit |
| `train/value_pred_std` | 51.71 vs `value_target_std` 38.72 | over-dispersed |
| `train/log_ratio_max` | 0.0717 vs cap 10 | never binds |
| `train/log_ratio_clip_frac` | 0 | never binds |
| `train/flow_ratio_std` | 0.0098 vs `clip_param` 0.05 | 0% outside the clip band |
| `train/cfm_above_clamp_frac` | 0 | `cfm_clamp=20` is safe |

The trust region is inert and the gradient budget is saturated by something.

## What it does not tell us

**How the 2291x splits between actor and critic.** A synthetic probe gives 6.3x,
not 2291x, so the split is data-dependent. `train/actor_grad_norm_preclip` and
`train/critic_grad_norm_preclip` were collapsed into `train/grad_norm_preclip` in
commit `fc8724b`, which is the one metric that could answer it.

**Whether the spectral prior helped or hurt.** That run used
`noise_scaling_mode=freq_weighted_time`. There is no `white` control, so the
clip and the prior are confounded.

---

## Step 0 — instrument before you run

Do this first. Without it, arm B below cannot be interpreted.

Restore the per-role norms in `train_sjfpo.py`. Replace:

```python
pre_clip_total = torch.nn.utils.clip_grad_norm_(
    clip_params, cfg.max_grad_norm
)
```

with:

```python
pre_clip_total = torch.nn.utils.clip_grad_norm_(
    clip_params, cfg.max_grad_norm
)
# Diagnostics only -- the clip itself stays joint.
actor_preclip = torch.linalg.vector_norm(
    torch.stack(torch._foreach_norm([p.grad for p in actor_params if p.grad is not None]))
)
critic_preclip = torch.linalg.vector_norm(
    torch.stack(torch._foreach_norm([p.grad for p in critic_params if p.grad is not None]))
)
```

Rebuild `actor_params` / `critic_params` from
`policy.get_actor_optim_params()` and `policy.get_critic_optim_params()`, log
both as `train/actor_grad_norm_preclip` and `train/critic_grad_norm_preclip`,
and assert `atol=0` against a norm over the concatenation in a test.

Also log `train/value_huber_knee = float(policy._huber_resid_rms *
cfg.value_huber_delta_scale)`. It is now meaningful (the knee tracks the
residual), and watching it move tells you whether the critic is drifting in
scale.

---

## Step 1 — find the onset (cheap)

The peak was at 131k, so 200k steps captures peak plus onset. Eval every 16k
gives 12 points instead of 4. About 15 minutes per arm at 227 fps.

```bash
.venv/bin/python scripts/train.py \
  --method sjfpo --env Humanoid-v5 --seed 3407 \
  --total_steps 200000 --eval_steps 16384 --save_steps 50000 \
  --out_dir outputs/diag_onset_white \
  --sjfpo.report_to tensorboard \
  --sjfpo.noise_scaling_mode white
```

Repeat with `--out_dir outputs/diag_onset_spectral` and
`--sjfpo.noise_scaling_mode freq_weighted_time`.

Everything else stays at defaults so the arms differ in one variable.

### Read from `outputs/diag_onset_*/`

- `eval/mean_return` and `eval/best_mean_return` vs `eval/steps_to_best`
- `eval/mean_length` — falling return with falling length is a balance failure,
  not an optimisation failure
- `train/explained_var` per eval — if it never climbs, the critic is the ceiling
- `train/actor_grad_norm_preclip` / `train/critic_grad_norm_preclip` — the ratio
  is the question
- `train/value_huber_knee` — should be flat if the fix is holding
- `train/rollout_seam_jerk` vs `train/rollout_jerk`

### Decision table

| Observation | Implication |
|---|---|
| critic norm ≫ actor norm (≫10x) | joint clip is starving the actor — go to arm B |
| both norms ≫ 1 and roughly equal | genuine scale blow-up — the loss needs rebalancing, not the clip |
| `explained_var` flat below ~0.3 | the critic is the bottleneck; advantages are critic error |
| `log_ratio_max` stays under ~0.2 | trust region inert — raise `clip_param` or the LR, not the clamp |
| spectral ≈ white on return | the 1/f prior is not earning its place on this env |
| spectral clearly worse | it interacts badly with this estimator; stop spending runs on it |

---

## Step 2 — confirm the clip hypothesis

Only if Step 1 shows the critic owning the budget.

```bash
.venv/bin/python scripts/train.py \
  --method sjfpo --env Humanoid-v5 --seed 3407 \
  --total_steps 614400 --eval_steps 8192 --save_steps 50000 \
  --out_dir outputs/diag_sepclip_white \
  --sjfpo.report_to tensorboard \
  --sjfpo.noise_scaling_mode white
```

with per-role clipping restored:

```python
torch.nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm)
torch.nn.utils.clip_grad_norm_(critic_params, cfg.max_grad_norm)
```

Two seeds minimum. One seed is not a result on Humanoid.

Run it against `white`, not the spectral arm, so the only variable against the
original run is the clip.

### Expected signature if the hypothesis holds

- `actor_grad_norm_preclip` stops scaling with `critic_grad_norm_preclip`
- `policy_loss` stops being ~1e-3 against a value loss of ~1e2
- `log_ratio_max` starts moving toward the clip band
- `steps_to_best` moves later than 131k

If none of those move, the clip was not the cause and you have spent the run
learning that — which is why the instrumentation comes first.

---

## Step 3 — only then

Re-run the spectral arm with whatever won, and add a second seed. Do not
reintroduce `cfm_clamp` or `log_ratio_cap` changes; both were verified inert on
this data (`cfm_above_clamp_frac = 0`, `log_ratio_clip_frac = 0`) and neither is
a lever.

The target-critic removal (`c157a42`) is also still unattributed. Critic
`amin`-for-target with mean-for-loss and no teacher is a live overestimation
channel, and `value_pred_std > value_target_std` is consistent with it. If Step 1
shows `explained_var` flat, restoring the EMA teacher is the next thing to try —
but that is a separate experiment, not part of this one.

---

## Reproducing the original run exactly

```bash
.venv/bin/python scripts/train.py \
  --method sjfpo --env Humanoid-v5 --seed 3407 \
  --total_steps 614400 --eval_steps 8192 --save_steps 50000 \
  --out_dir outputs/diag_repro \
  --sjfpo.report_to tensorboard \
  --sjfpo.noise_scaling_mode freq_weighted_time \
  --sjfpo.compile false
```

Run this **first**, before anything else. Two fixes landed since the original
run (encoder EMA, Huber knee), so this is no longer a reproduction — it tells
you what those two bought you. If it now peaks later than 131k, the fixes moved
the needle and Step 1 is cheaper to interpret.

## Notes

- One process at a time. `num_envs=64` forks a pool; do not run two arms
  concurrently on one box.
- `Humanoid-v5` is 348-dim obs, 17-dim action, tier `hard` (`configs/envs.yaml`).
- Run dirs are `outputs/sjfpo_Humanoid-v5_seed3407`; `--resume` writes to a
  `_resume` suffix and never overwrites.
- `run_metadata.json` is written before the first rollout. It records the device
  and dtype settings (`compute_dtype`, `storage_dtype`, `tf32_*`, `gpu`), plus
  `vector_env` when `num_envs > 1` and a `baseline` block when `--baseline` was
  passed. It does **not** record the hyperparameter diff — read the `--baseline`
  report on stderr for that.
- Exit codes: `0` ok, `1` usage, `2` runtime or divergence, `4` missing artifact.