# Experiment plan

State as of `da799b7`. Every number below is read out of this repo, not recalled.

## What the existing runs say

| Run | evals | best | final 10 | decile means |
|---|---|---|---|---|
`outputs/sjfpo_Hopper-v5_seed3407` | 384 | **997** | **74** | 165 122 132 235 122 86 68 84 **405** 60 74 |
`outputs/sjfpo_InvertedPendulum-v5_seed3407` | 864 | 53 | 15 | 18 17 15 24 26 25 20 17 7 17 14 |
`outputs/sjfpo_InvertedDoublePendulum-v5_seed3407` | 120 | 148 | 83 | 79 77 78 78 78 80 80 85 86 87 |

**Hopper peaks at 997 on eval 334 of 384 and ends at 74.** InvertedPendulum peaks at
decile 4 and decays. This is a policy that works and is then destroyed, not one that
never learns. `jerk` is not the cause: it rises 1.8e-6 → 4e-4, real but off a floor.

K is not currently binding: `tmp/es4` 307.8 vs `tmp/es16` 309.6 at 50 evals, 1 seed —
a difference that size cannot resolve. Do E0–E3 before spending on K.

## Preconditions

```bash
# After EVERY run and every pytest. wandb/ and tensorboard/ are tracked.
git checkout -- wandb/ tensorboard/
```

`--sjfpo.report_to none` skips the wandb import entirely and is what you want for
measurement runs. Use `--sjfpo.report_to tensorboard` when you need the `train/cond_*`
collapse probes, which only exist when a tracker is enabled.

---

## E0 — Does the critic gradient live in the retained subspace?

**Run this first.** 20 lines, no training, and it either opens the contribution or
closes it. Everything downstream of the value-gradient idea depends on it.

Save as `tmp/rho_probe.py`:

```python
import sys; sys.path.insert(0, "src")
import torch
from spectral_joint_policy.dct_utils import extract_dct_coeffs

def spectral_energy_fraction(critic, states, actions, ks):
    """rho_K = ||Pi_K grad_a Q||^2 / ||grad_a Q||^2. One backward pass for all K."""
    actions = actions.clone().requires_grad_(True)
    critic(states, actions).sum().backward()      # [num_critics, B]
    g = actions.grad                              # [B, H, action_dim]
    total = g.float().square().sum().item()
    return {k: extract_dct_coeffs(g, k).float().square().sum().item()
               / max(total, 1e-12) for k in ks}
```

Verified sanity: ρ = 1.0 at `k == action_horizon` (Parseval), monotone in k.

Two things to know before you read a number:

- **The FiLM head is zero-initialised, so an untrained critic is exactly
  action-blind and ρ = 0.0 at every k.** That is adaLN-Zero working, not a bug. Only
  a critic that has trained past the first update tells you anything.
- Run it on `outputs/sjfpo_Hopper-v5_seed3407/checkpoints/*/model.safetensors` at
  several points along the run, not just the final one.

**Kill criterion:** median ρ < 0.3 at K=4 → the projection discards the signal, the
value-gradient contribution dies, and the paper becomes a representation paper. Stop
and re-scope. Above 0.3 it survives.

---

## E1 — What kills the policy after its peak?

Distinguish the two collapse signatures, which are indistinguishable from the eval CSV
and need opposite fixes.

Already logged, no code change needed — `info` keys all become `train/<key>`:

| Signature | Metric | Dohare collapse | Trust region |
|---|---|---|---|
rank collapse | `train/cond_effective_rank` | falls | flat |
weight spike | `train/cond_weight_magnitude` | rises sharply | flat |
trust region | `train/clipfrac`, `train/flow_ratio_std` | flat | `clipfrac` → 1 |

```bash
WANDB_MODE=offline .venv/bin/python scripts/train.py \
  --env Hopper-v5 --seed 3407 --total_steps 400000 --eval_steps 1000 \
  --out_dir ./tmp/e1 --sjfpo.report_to tensorboard
```

Caveat on the rank probe: `forward` sizes it at
`hidden_dim // num_flow_tokens` = 64 rows, so it is an SVD of a 64×256 matrix.
Directionally useful, not a precise rank. Raise the probe width before quoting it.

---

## E2 — The eval/stochastic gap

A 12.5× gap was measured on Hopper: eval 212, noise-start rollout 16.9, random 18.9.
The training-distribution policy is at random. Until that closes, no representation
result means anything, because training and deployment are different policies.

The trainer already runs both evals each time (`train_sjfpo.py:576-586`) — same seeds,
same evaluator. Read `eval/return` against `train/stochastic_mean_return` from one run.
Then isolate the sampler:

```bash
# Noise start, 4 steps vs zero start, 4 steps, same checkpoint.
--sjfpo.num_euler_steps 4     # stochastic path
# predict_action_chunk(zero_sampling=True) is the eval path
```

**If stochastic stays at random, the flow is not the policy during training** and the
DCT is being optimised in a regime you never deploy.

---

## E3 — Is the collapse the EMA?

Eval already swaps EMA weights in and out (`swap_in_ema` / `swap_out_ema`), so the
collapse is visible in the EMA too. That is evidence the live weights collapsed, not
that averaging smeared a healthy policy — but confirm:

```bash
--sjfpo.ema_decay 0.0    # live weights at eval
```

Same seed, same length as E1. If the peak-then-collapse survives, the collapse is in
training (E1). If it flattens, `ema_decay=0.95` is too slow for a bimodal weight
distribution and the decay wants to be much smaller, not larger.

---

## E4 — K sweep (needs 3 seeds and full-length runs)

```bash
for k in 2 4 8 16; do for s in 0 1 2; do
  WANDB_MODE=offline .venv/bin/python scripts/train.py \
    --env Hopper-v5 --seed $s --total_steps 32768 \
    --sjfpo.num_kept_tokens $k --out_dir ./tmp/e4_k${k}_s${s} \
    --sjfpo.report_to none
done; done
```

The existing `es4`/`es16` pair is 1 seed and 50 evals and cannot resolve anything.
Three seeds is the minimum for a table that goes in a paper.

Expect K=4 ≈ K=16 on Hopper — its actions are near constant-velocity. If you want K to
matter, test a task with genuine high-frequency action content, not locomotion. Report
the null result if that is what you get; it is the honest finding.

---

## E5 — Noise colouring β (the DCT-specific interaction)

`white` puts equal variance in all 4 retained modes. `compute_freq_weighted_std`
already implements 1/f^β and the config notes coloured-noise PPO puts the on-policy
optimum at β = 0.5.

```bash
for b in 0.0 0.5 1.0; do
  WANDB_MODE=offline .venv/bin/python scripts/train.py \
    --env Hopper-v5 --seed 0 --total_steps 32768 \
    --sjfpo.noise_scaling_mode freq_weighted --sjfpo.noise_scaling_beta $b \
    --out_dir ./tmp/e5_b${b} --sjfpo.report_to none
done
```

This is the one interaction that is genuinely DCT-specific and unpublished: a
frequency basis changes what the *noise* should look like, not just what the policy
represents. Could be noise. It is cheap.

---

## Expected outcome, ranked

1. **The collapse.** Large, and not a DCT result. Fix likely in the trust region or
   EMA cadence.
2. **The stochastic gap.** Large. Closing it is worth more than any K sweep.
3. **K.** Small, probably a null result on Hopper. Publishable either way.
4. **β.** Uncertain. Best case is the only DCT-specific win available.

## Do not spend time on

- **Capacity.** 20.7M params on a 3-dim action space is not the constraint.
- **`num_euler_steps`.** 4 ≈ 16 already measured (211.7 vs 212.0). Saturated.
- **Longer chunks.** `n_action_steps == action_horizon == 16` already runs the whole
  chunk open-loop.

## Framing caveat

The paper claim currently on the table — an orthonormal basis makes a value-gradient
distillation step exact in action units and the step size meaningful — does **not**
require K to matter for return. It requires ρ, which is E0. Run E0 before writing any
of it. Critic-gradient distillation itself is prior art (QSM ICML 2024, QAM ICLR
2026, FlowDPG, RLDT); the contribution has to be the spectral basis, not the
distillation.