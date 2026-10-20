# Architecture Record

Why each network is shaped the way it is, and which reference each choice follows.
Written so the next reader does not have to reverse it out of `git log`.

`README.md` describes what the repo does. This describes how it is built.

---

## Actor — `src/myfpo/flow.py`, `src/myfpo/attention.py`

Conditional flow-matching velocity model over `action_horizon` action tokens.
Two variants behind `config.actor_arch`; `dit` is the default.

| Element | Choice | Reference |
|---|---|---|
| Block | adaLN-Zero DiT block, shared by both actors | Peebles & Xie, DiT (`arXiv:2212.09748`) |
| Norm | RMSNorm, pre-norm | |
| FFN | SwiGLU, 4x expansion | Shazeer, GLU Variants (`arXiv:2002.05202`) |
| Output init | zero on `adaLN_out` and `out` | DiT |
| Position | RoPE on q/k | Su et al. (`arXiv:2104.09864`) |
| Timestep | per-token, not per-sample | SD-DiT / rectified-flow training |
| Time embedding | log-spaced DDPM table | Ho et al. (`arXiv:2006.11239`) |
| Depth / width | `dit_depth=4`, `hidden_dim=128` | see Depth and width |

Defaults: **1.24M parameters**, 4 blocks of 281k each.

The block is one class (`attention.AdaLNBlock`) with the MLP as the only
difference: `DiTBlock` supplies a SwiGLU, `ResMLPBlock` the paper's residual
MLP. The counts and the checkpoint keys are pinned in
`tests/test_block_contract.py`.

### Verified, not assumed
- **adaLN-Zero is real.** All-zero `adaLN.fc2` makes the block an exact identity at
  init; perturbing any of its six outputs changes the velocity by exactly 0.
- **All four conditioning paths are live.** With the gates held open, perturbing
  `shift_msa` / `scale_msa` / `shift_mlp` / `scale_mlp` changes the output by
  1.08 / 1.15 / 1.21 / 1.20 relative.
- **RoPE is correct** — translation invariance holds to 3.6e-7.

### Causal attention — a deliberate deviation
`SelfAttention` passes `is_causal=True`. ACT and Diffusion Policy attend
bidirectionally over the chunk; this does not.

The reason is prefix consistency: the first `n_action_steps` tokens of a chunk
are the ones actually executed, and under a causal mask they cannot depend on the
tail that gets discarded. That is the same seam the repo already measures with
`rollout_seam_jerk`. The cost is a halved receptive field per token.

This is a choice, not an oversight. If it is ever revisited, the ablation is to
flip one flag and compare `rollout_seam_jerk` against return.

### Time embedding
`SinusoidalEncoding` uses `exp(-log(10000) * i / half)`. An earlier
`2**i` table ran to 32768 rad over `t in [0, 1]`, which made neighbouring
timesteps near-orthogonal (cosine similarity 0.35 between `t=0` and `t=0.05`)
rather than smooth (0.9999 with the current table). A continuous velocity
field is regressed against these codes, so smoothness is not cosmetic.

### Depth and width
`dit_depth` 3 → 4 and `hidden_dim` 128 are capacity ceilings, not literature
defaults. DiT-1/2 for images uses 12 blocks at width 768. Depth 3 is defensible
for 8-token low-dimensional actions and probably was not the bottleneck; the
8-block variant cost ~2x the actor FLOPs and the full test suite went 57s →
126s.

**These two numbers went in unmeasured.** There is no shallow-actor
config, so going back means passing `dit_depth` on the CLI. Run names encode the
value (`tracking.py` emits `d128x4`), so a depth-3 run opens a separate
series rather than continuing this one.

### Known, not fixed
`adaLN` is 115k of each 280k block — 41%, more than attention. The
"modulate once, share across branches" variant roughly halves it. It was not
adopted: at width 128 it saves ~9% of total parameters, collapses four
diagnostics (`gate_msa`/`gate_mlp`/`scale_msa`/`scale_mlp`) into two, and
requires a retrain. Revisit if depth and width both grow.

---

## State encoder — `src/myfpo/encoder.py`

`CondMLP(n_obs_steps * obs_dim → hidden → hidden)`, then **RMSNorm**.

The norm is the point. This was the only unnormalised path in the whole stack:
every observation reaches the actor through it and then through an unnormalised
adaLN projection, so nothing between the raw state and the modulation was
normalised. RMSNorm over LayerNorm because it does not subtract the mean, and
because it matches the actor's `norm1`/`norm2`/`norm_out`.

Normalising the output also means the actor sees a `cond` of consistent
magnitude regardless of observation scale, which matters because
`cond` is *added* to `time_mlp(temb)`.

---

## Critic — `src/myfpo/critic.py`

Ensemble of 2 `Q(s_t, a_{t:t+h})` heads over the full window. Trunk follows
**TD-MPC2** (`arXiv:2310.16828`) throughout.

| Element | Choice | Why |
|---|---|---|
| Hidden layers | `Linear → LayerNorm → Mish` | TD-MPC2. Its stated result: scaling an *unnormalised* critic does not improve performance, while adding LayerNorm does. The norm is the load-bearing part. |
| Dropout | 1% after the first `Linear` | TD-MPC2. Every member regresses onto the same scalar target, so initialisation is the only thing separating them. |
| Ensemble | 2 | the repo's default; TD-MPC2 ships 5 (`--myfpo.num_critics 5` restores it) |
| Output layer | bare `Linear` | no activation or norm on the scalar head |
| Width | `critic_hidden_dim=512` | matches TD-MPC2 |
| Input norms | `LayerNorm` on state and action **separately** | one norm over the concat would couple state dims to action dims, which are unrelated quantities |

Defaults: **631k parameters**, 2 members of 316k.

`norm_state` is skipped when `critic_share_encoder=True`, so that flag keeps
meaning "raw shared-encoder vector" rather than double-normalising.

---

## Action space — `src/myfpo/processor.py`, `src/myfpo/collector.py`

The flow's output is the **raw** action, unbounded, and that raw value is what
the policy is trained on. The squash happens on the way to the env.

```
flow output r  ──►  stored chunk (raw)  ──►  CFM target, critic input
                └─►  squash_to_env(r) ──►  the action the env steps with
```

`bias + scale · tanh(r)`, with `scale = (high - low) / 2` and
`bias = (high + low) / 2` from the env's bounds. One implementation
(`processor.squash_to_env`) serves the postprocessor and the collector, so the
two cannot drift.

**Why the raw value is what is stored.** The CFM regression target is built
from the stored chunk. If the buffer kept the squashed value, the flow would be
trained on the squash of its own output: the action the policy is trained on
would not be the action the policy produces, and the env would receive a
second squash of a value that was already squashed. The invariant is pinned by
`tests/test_raw_action_space.py`.

The consequence for the reader: **the stored action is not the action the env
received.** `train/raw_action_abs_mean` is the stored chunk, and
`train/executed_action_abs_mean` is the env-facing one; the ratio between them
is the squash, and it is a property of the distribution, not a constant.

The action stays in fp32 in the batch: it is the regression target, the raw
values are unbounded, and a bf16 rounding would put a 0.4% error on it.

Checkpoints record the action space (`action_space.json`). A checkpoint without
it was trained with the critic reading the squashed chunk, which is a different
quantity, and the loader refuses it.

---

## Value targets and advantages — `src/myfpo/metrics.py`, `src/myfpo/rollout.py`

`subsampled_min(q, size=2)` — the min of **2 randomly drawn members, per column**.
REDQ (Chen et al. 2021) and TD-MPC2.

A full `amin` over all members is strictly more pessimistic than either, and
`measure_value_drift` uses the same reduction so both sides of its subtraction
are comparable.

This is a PPO loop, not a TD loop: `raw_returns` is computed once per rollout
(`gae.py`) and reused frozen across all 10 epochs. That is already the property a
target critic network would provide, so there is none.

### Ensemble collapse is logged, not gated
`train/value_pred_member_gap_p95` is the tail of the per-chunk spread between
the highest and lowest member. A trend to zero means the members have converged
onto one function. There is no abort on it: an unsupervised statistic with no
baseline cannot be thresholded honestly. Watch it against
`--myfpo.critic_dropout 0`.

---

## Gradient clipping — `src/myfpo/training.py`

Actor and critic are clipped **independently**, each to `max_grad_norm=1.0`,
sharing one `optimizer.step()`. One joint clip lets whichever side has the larger
gradient silently shrink the other.

`split_grad_params(policy)` derives the split from `get_actor_optim_params` /
`get_critic_optim_params`, **not** from `optimizer.param_groups`. That is
deliberate: `checkpointing.py` replaces `param_groups` wholesale on
`load_state_dict`, so tagging them with a role key would make any pre-existing
checkpoint load with the key missing.

`train/grad_norm_before_clip` is `sqrt(sum of squared per-role norms)`, which over
a partition is exactly the old joint norm. The series stays comparable with
history; `..._actor_before_clip` and `..._critic_before_clip` are additive.

---

## Observation normalisation — `src/myfpo/encoder.py`

`RunningMeanStd` uses a parallel (Chan) batch update with a float64 counter,
registered as buffers, and the **same object** is shared between training and
inference. Better than the frozen dataset normaliser most policy code uses,
because it tracks the online distribution.

---

## Layer map

```
utils ──────────────► (leaf: device, dtype, tensor helpers)
layers ─────────────► (leaf: RMSNorm, SwiGLU, CondMLP, sinusoidal)
attention, encoder, critic, gae, evaluator ──► (leaf or near-leaf networks)
processor ───────────► encoder (the squash, shared by the postprocessor and
                       the collector)

flow ───────────────► attention, layers
metrics ────────────► fpo_core (sampling + broadcasting)
collector ──────────► configuration, processor
rollout ────────────► collector, gae, metrics
modeling ────────────► flow, critic, encoder, fpo_core, metrics, rollout
training ────────────► everything above, plus tracking, runner, checkpointing
```

No cycles, including deferred imports. `rollout` depends on nine members of the
policy object without a declared interface — a known inversion, not yet fixed.