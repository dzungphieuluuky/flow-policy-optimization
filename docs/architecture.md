# Architecture Record

Why each network is shaped the way it is, and which reference each choice follows.
Written so the next reader does not have to reverse it out of `git log`.

`README.md` describes what the repo does. This describes how it is built.

---

## Actor — `src/myfpo/flow.py`, `src/myfpo/layers.py` (the attention section)

Conditional flow-matching velocity model over `action_horizon` action tokens.
Two variants behind `config.actor_arch`; `dit` is the default.

| Element | Choice | Reference |
|---|---|---|
| Block | adaLN-Zero DiT block, shared by both actors | Peebles & Xie, DiT (`arXiv:2212.09748`) |
| Norm | RMSNorm, pre-norm | |
| FFN | MishFFN, `dit_mlp_ratio=2` | Shazeer, GLU Variants (`arXiv:2002.05202`) |
| Output init | zero on `adaLN_out` and `out` | DiT |
| Position | RoPE on q/k | Su et al. (`arXiv:2104.09864`) |
| Timestep | per-token, not per-sample | SD-DiT / rectified-flow training |
| Time embedding | log-spaced DDPM table | Ho et al. (`arXiv:2006.11239`) |
| Depth / width | `dit_depth=4`, `hidden_dim=128` | see Depth and width |

`dit_mlp_ratio=2` means the MishFFN hidden width is twice the model width. Its
parameter count equals that of a 4x GELU MLP, because MishFFN keeps two
matrices of that width.

Defaults: **1.24M parameters**, 4 blocks of 281k each.

The block is one class (`layers.AdaLNBlock`) with the MLP as the only
difference: `DiTBlock` supplies a MishFFN, `ResMLPBlock` the paper's residual
MLP. The counts and the checkpoint keys are pinned in
`tests/test_block_contract.py`.

### Verified, not assumed
- **adaLN-Zero is real.** All-zero `adaLN.fc2` makes the block an exact identity at
  init: `max(abs(out - x)) = 0.0`.
- **All six conditioning paths are live.** Add `+1.0` to one of the six adaLN
  outputs and read `norm(out - base) / norm(base)`. The recipe below is
  deterministic, so the numbers are reproducible.

  ```python
  torch.manual_seed(0)
  m = MyfpoFlowModel(3, 8, 128, 2, 2, 2.0, 32)
  for p in (m.out.weight, m.out.bias, m.adaLN_out.weight, m.adaLN_out.bias):
      torch.nn.init.normal_(p, std=0.2)
  b = m.blocks[0]
  torch.nn.init.normal_(b.adaLN.fc2.weight, std=0.1)
  torch.nn.init.normal_(b.adaLN.fc2.bias, std=0.1)
  x = torch.randn(2, 8, 128)
  c = torch.randn(2, 8, 128)
  ```

  The four conditioning paths give `0.128 / 0.114 / 0.079 / 0.153` for
  `shift_msa` / `scale_msa` / `shift_mlp` / `scale_mlp`. The two gates give
  `0.236` and `0.150`.
- **RoPE is correct.** The attention score depends on the relative offset only.
  Shifting q and k by the same `+5` positions leaves the score matrix unchanged
  to `3.3e-06` at the default width (`hidden_dim=128`, `dit_heads=2`,
  `action_horizon=8`). Over five seeds the figure stays below `3.9e-06`. This
  is fp32 roundoff, not a drift.
- **The attention matches an independent reference.** A naive implementation with
  explicit per-head loops, an explicit causal mask and an explicit RoPE gives the
  same output to `1.2e-07`.

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
`SinusoidalEncoding` uses `exp(-log(10000) * i / half)` and multiplies `t` by
`TIME_SCALE = 1000` before the sinusoid. The table alone was not enough: with
`t` on `[0, 1]` the top channel turned through 1 rad over the whole path, so
`t=0.1` and `t=0.9` had cosine similarity 0.97. The scale is the DiT
convention, whose timesteps run to 1000. It puts 127 cycles on the top channel
over the trained band `[0.1, 0.9]`, so a 0.01 step moves the code (similarity
0.63) and a 0.001 step stays close (0.96). A `2**i` table is still rejected: it
runs to 32768 rad.

### Depth and width
`dit_depth` 3 → 4 and `hidden_dim` 128 are capacity ceilings, not literature
defaults. DiT-1/2 for images uses 12 blocks at width 768. Depth 3 is defensible
for 8-token low-dimensional actions and probably was not the bottleneck. The
8-block variant cost about twice the actor FLOPs.

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

## State encoder — `src/myfpo/processor.py` (the absorbed `encoder.py` section)

One shared `CondMLP(obs_dim → hidden → hidden)` runs on every frame, one causal
`SelfAttention` layer mixes the window, and the last frame is read out through
**RMSNorm**.

The flat `CondMLP(n_obs_steps * obs_dim → hidden → hidden)` was the capacity
sink. On Humanoid (348 obs, 16 frames) it held 713k of the encoder's 729k, 59%
of the whole actor. The per-frame projection makes the width follow `obs_dim`:
Humanoid is now 127k. Hopper pays more, 39k → 84k, because the attention layer
is 66k at any window.

Attention keeps the frame order, which a pool would drop. It is causal, so
frame `i` reads the prefix only, and the readout is the last frame.

The norm is the point. This was the only unnormalised path in the whole stack:
every observation reaches the actor through it and then through an unnormalised
adaLN projection, so nothing between the raw state and the modulation was
normalised. RMSNorm over LayerNorm because it does not subtract the mean, and
because it matches the actor's `norm1`/`norm2`/`norm_out`.

Normalising the output also means the actor sees a `cond` of consistent
magnitude regardless of observation scale, which matters because
`cond` is *added* to `time_mlp(temb)`.

---

Ensemble of 2 **TD-MPC2** (`arXiv:2310.16828`) heads over the full window.

| Element | Choice | Why |
|---|---|---|
| State input | `StateAffine` on the raw window, then the actor's window trunk (`StateEncoder`), one token per frame | one norm over the state would couple state dims to action dims, which are unrelated quantities. A `LayerNorm` also subtracts the window's mean, so a raised state and the original state reached the critic as one input. The trunk replaces the flat `T * obs_dim` first layer. |
| Layout | two token streams, a causal `SelfAttention` over the chunk steps, then `CrossAttention` both ways, then the MLP | the window and the chunk condition each other before the head. Each cross direction reads the raw stream, so neither pass sees a half-mixed one. The pool is the last state token plus the mean action token. |
| Chunk attention | causal `SelfAttention`, RoPE step codes | the pool is a mean, which is order-blind. Without an intra-chunk pass a reversed chunk reads as the same chunk. RoPE gives each step its position, and the mask makes token `i` read the prefix only. There is no convolution in the critic. |
| Hidden layers | `Linear → LayerNorm → Mish` | TD-MPC2. Its stated result: scaling an *unnormalised* critic does not improve performance, while adding LayerNorm does. The norm is the load-bearing part. |
| Dropout | 1% after the first `Linear` | TD-MPC2. Every member regresses onto the same scalar target, so initialisation is the only thing separating them. |
| Activation | `Mish` | TD-MPC2 uses GELU. `Mish` is the swap. |
| Action input | learned affine, **not** a norm | a `LayerNorm` over the chunk subtracts the chunk's mean, so a chunk of all +5 and a chunk of all 0 reach the critic as the same input. The critic was blind to a sustained push, and blind to the drifting raw magnitude that `train/raw_action_abs_max` monitors. |

Defaults for Hopper (`obs_dim=11`, `action_dim=3`, 16 frames, 8-step chunk):
**668k parameters**, 2 members of 334k. Humanoid (`obs_dim=348`) is **780k**.

Two places depart from the reference implementation, not from the paper.
The official tdmpc2 `mlp.py` puts dropout **after** the LayerNorm and applies
it after **every** hidden layer. The repo follows the paper's wording, which
says "1% Dropout after the first linear layer in each Q-function".

The chunk self-attention is causal, and so is the actor's. The two cross blocks
stay bidirectional: the critic reads a chunk it has already executed.

`norm_state` is skipped when `critic_share_encoder=True`, so that flag keeps
meaning "raw shared-encoder vector" rather than double-normalising. The two
paths are not comparable: the shared path ends in an RMSNorm, which does not
subtract the mean, so the shared critic is shift-sensitive on the state and the
default critic is not.

---

## Action space — `src/myfpo/processor.py` and the collector in `src/myfpo/rollout.py`

The flow's output is the **raw** action, unbounded, and that raw value is what
the policy is trained on. The squash happens on the way to the env.

```
flow output r  ──►  stored chunk (raw)  ──►  CFM target, critic input
                └─►  squash_to_env(r) ──►  the action the env steps with
```

`bias + scale · tanh(r)`, with `scale = (high - low) / 2` and
`bias = low + scale` from the env's bounds. The bias is the low end plus the
half width, so both terms come from one difference.
`processor.env_squash_terms` owns the terms and
`processor.squash_to_env` owns the map. The postprocessor, the collector and
the training diagnostics all call them, so the three cannot drift.

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

## Value targets and advantages — `src/myfpo/fpo_core.py` (the metrics section), `src/myfpo/rollout.py`

`subsampled_min(q, size=2)` — the min of **2 randomly drawn members, per column**.
REDQ (Chen et al. 2021) and TD-MPC2.

A full `amin` over all members is strictly more pessimistic than either, and
`measure_value_drift` uses the same reduction so both sides of its subtraction
are comparable.

This is a PPO loop, not a TD loop: `raw_returns` is computed once per rollout
(`rollout.py`, the absorbed `gae.py` section) and reused frozen across all 10 epochs. That is already the property a
target critic network would provide, so there is none.

### Ensemble collapse is logged, not gated
`train/value_pred_member_gap_p95` is the tail of the per-chunk spread between
the highest and lowest member. A trend to zero means the members have converged
onto one function. There is no abort on it: an unsupervised statistic with no
baseline cannot be thresholded honestly. Watch it against
`--myfpo.critic_dropout 0`.

`train/value_pred_member_corr` is the mean Pearson correlation between the
members, over the batch axis. It compares the members' value functions, not
their values on one state. The correlation is invariant to a shift and to a
scale, so a member that is the same function with a different offset still
reads one. A collapse shows as a correlation near one **together with** a gap
near zero. A single member, or a member that is constant over the batch, has
no correlation to measure, so the value is NaN. NaN is how the repo says
"unmeasured".

The metric reads the training forward, so the critic dropout is active. The
confound was measured: on a critic fit to one shared target, the train-mode
and eval-mode correlations differ by about 0.01. The dropout does not hide a
collapse.

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

## Observation normalisation — `src/myfpo/processor.py` (the absorbed `encoder.py` section)

`RunningMeanStd` uses a parallel (Chan) batch update with a float64 counter,
registered as buffers, and the **same object** is shared between training and
inference. Better than the frozen dataset normaliser most policy code uses,
because it tracks the online distribution.

---

## Layer map

Every import edge in the package. Read it as "X builds on Y".

Every import edge in the package. Read it as "X builds on Y".

```
layers, configuration, critic, evaluator, tracking ──► (leaves; layers holds
                         utils and attention)

processor ─────────────► layers (encoder + running obs norm + the squash)
fpo_core ──────────────► configuration, layers (the metrics live inside)
flow ──────────────────► layers
rollout ───────────────► processor, fpo_core, layers (collector and gae
                         live inside)
modeling ──────────────► flow, critic, processor, rollout, fpo_core,
                         configuration, layers
checkpointing ─────────► modeling
training ──────────────► checkpointing, rollout, configuration, evaluator,
                         fpo_core, modeling, processor, runner, tracking,
                         layers
```

No cycles, including the deferred imports. All absorbed-section notes name
the former single-job module.

No cycles, including deferred imports. `rollout` depends on nine members of the
policy object without a declared interface — a known inversion, not yet fixed.