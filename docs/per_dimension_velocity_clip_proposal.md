# Research Proposal: Per-Dimension Velocity Trust Regions for Flow-Policy RL

Working title: **Per-Dimension Velocity Trust Regions**
Author(s): myfpo team
Status: research proposal, 2026-11-14
Estimated effort: 10–12 weeks to submission (target CoRL / ICRA; fallback RSS workshop)

---

## 0. One-paragraph thesis

QF3 (*Fast Flow RL with Filtered Q-Gradients*, arXiv:2610.08789) trains a flow actor by
backpropagating the critic's `Q` action-gradient through a single-step Euler prediction, with a
per-dimension velocity clip around the replay action that keeps the one-step prediction inside
a trust region. It is the current off-policy SOTA for flow-RL from scratch (roughly 10×
wall-clock over FPO++ on humanoid locomotion and motion tracking). Its alpha is a single scalar
shared by every action dimension — a structural constraint the QF3 authors themselves concede
costs task-specific tuning and limits how far the method generalizes. This proposal claims:
**the velocity trust region should be a per-dimension vector, not a scalar, computed from the
critic's per-dimension sensitivity, and that this alone closes most of the alpha-tuning gap
(Hopper/Walker2d/Ant/Humanoid) without touching any other part of QF3.** The win is measured on
both return-per-step and return-per-second, with paired-seed statistics and pinned behavior as
this repo's advantage demands.

---

## 1. Problem statement

### 1.1 What QF3 is, precisely

QF3 implements the following update (Alpha = a single hyperparameter):

1. Replay buffer stores transitions `(s, a_buf, r, s')`, where `a_buf` is a **stored noised
   action** (the buffer action), not a noise-free target.
2. Actor is a flow policy `v_θ(s, t, ε)`. The actor's one-step prediction from a replay action is
   controlled by a velocity clip around the **replay action's** velocity:
   `Δ_θ = v_θ − u_buf` is clipped per-dimension to `[-α, +α]`, defining
   `x̂_{1,clip}`, a one-step Euler prediction that stays in the trust region.
3. Actor loss combines:
   - **Q-maximization** on the clipped prediction: `Q(s, x̂_{1,clip})`, and
   - **flow-matching anchor** to the replay distribution (`‖v_θ − v_buf‖²`), weighted by `λ_cfm`.
4. The clip is what prevents the critic from dominating: without it, the one-step approximation
   drifts, the critic is queried far from replay data, and learning collapses on Humanoid even at
   lower learning rates (their ablation).
5. Alpha is **one scalar** for every dimension of the action.

### 1.2 Why a scalar is structurally wrong here

- **Action dimensions are not interchangeable in locomotion.** Hip pitch and ankle torque are in
  very different dynamical roles; a fixed `α` that is safe for the former permits unbounded
  effective step-size for the latter early in training, and vice versa late in training.
- **Sensitivity is state-dependent.** The same joint wants a tiny α near contact (precise balance)
  and a large α in free flight (fast reconfiguration). A scalar cannot express either.
- **The alpha tuning burden is the most-reported failure mode.** The QF3 release notes name
  `(α, λ_cfm)` as task-sensitive and give data: Hopper, Walker2d, Ant, Humanoid each need their
  own settings, and Robomimic needs smaller than all of them. The author's own analysis
  (source notes) calls this "the constraint that limits how far the method generalizes".

### 1.3 The gap

Existing flow-RL methods either clip uniformly (QF3), clip nothing and rely on a trust-region
surrogate for positive advantages (FPO++/ASPO), or impose a global KL/velocity bound with no
dimension information (Flow-DPPO's divergence mask). **No flow-RL method computes a
per-dimension, state-dependent trust region from the critic.** This proposal closes that gap with
one algorithmic delta and one engineering contribution (a clean, auditable implementation inside
this repo's pinned-gate framework).

---

## 2. Claim, hypothesis, and the experiment that falsifies it

**Primary hypothesis (H1).** A per-dimension velocity trust region `α(s) ∈ R^{d_a}_+`
computed from the critic's diagonal sensitivity yields better return-per-step **and**
return-per-second than QF3's homogeneous `α` on ≥ 3 of 4 MuJoCo tasks, at matched interaction
budget, with no increase in seed variance.

**Secondary hypotheses (H2–H4).**

- **H2 (tuning lift):** The heterogeneous clip is far less sensitive to its own hyperparameters —
  the same `(λ_cfm, α_max)` works reasonably on all four tasks, where homogeneous-α needs
  per-task retuning.
- **H3 (failure-mode removal):** The heterogeneous clip suppresses the early-success / late-collapse
  pattern observed in this repo's own FPO runs and in QF3's reported ablations, measured by
  episode-length streak histograms and drift-norm trajectories.
- **H4 (cost):** The added computation (one extra critic head or a diagonal EMA) is ≤ 5% of
  wall-clock per update; total speedup vs homogeneous QF3 ≥ 1.0× (i.e., not worse).

**Falsification conditions (pre-registered, so the paper is honest out of the gate).**

1. If homogeneous α matches heterogeneous α on every task and every budget, H1 fails — report it
   as a negative result, still publishable with the diagnostics.
2. If the heterogeneous α only helps by consuming more tunings of its own, H2 fails.
3. If the wall-clock cost exceeds 5%, H4 fails; the method must be re-engineered, not hidden.

---

## 3. Method

### 3.1 Core delta

Replace the scalar clip with a **per-dimension trust region vector** `α(s) ∈ R^{d_a}_+`:

```
Δ_θ = v_θ − u_buf
Δ̂_θ = clip(Δ_θ / α(s), −1, +1) * α(s)       # per-dim, in units of each dim's own allowance
x̂_{1,clip} = one_step_prediction(v_θ clipped via α(s), a_buf, t)
L_actor = −Q(s, x̂_{1,clip}) + λ_cfm * flow_match(v_θ, v_buf)
```

Everything else from QF3 stays: the Q-gradient path, the one-step Euler prediction, the flow-match
anchor to replay data, the clip as the anti-collapse stabilizer, and the order of operations.

### 3.2 Where `α(s)` comes from — three realizations, ablated

**(A) Intrinsic-velocity scale (cheapest; no extra critic math).**
`α_i(s) = α_max · σ_i(s) / σ̄(s)` where `σ_i(s)` is an EMA of `|Δ_θ,i|` at states visited (this is
RMSProp-style normalization on the velocity discrepancy). Pros: zero extra inference cost;
principled (dims that move need more allowed drift). Cons: uses dynamics info, not
quality-of-criticality info.

**(B) Critic-curvature (the actual claim).**
`α_i(s) = α_max / sqrt(ε + H_ii)` where `H_ii = ∂²Q/∂a_i²` — second derivative of Q in each action
dimension, i.e., per-dimension sensitivity of the return landscape. Dimensions where Q is flat
(can move far without harming return) get large α; dimensions where Q peaks sharply get small α.
`H_ii` is computed by a diagonal Gauss-Newton pass on one critic layer per update — cheap because
it reuses the same forward/backward infrastructure the actor already has.

**(C) Hybrid (EMA of clip-fraction, IIRC PPO-style clip adaptation).**
Maintain per-dimension clip-fraction `c_i` (fraction of times dim `i` was clipped in the last EMA
window), and adapt `α_i` to push `c_i` toward a target (e.g., 0.15). Dims clipped too often get
more allowance; dims untouched get less. This is the “controllable trust region” and is the most
robust to Q being wrong early, at the cost of one extra controller per dim.

The paper reports all three, plus homogeneous scalar α as the re-implemented baseline (QF3
faithful re-implementation), with the same codebase, same seeds, same tasks.

### 3.3 Implementation plan (this repo)

- **Stage 0 (1 week).** Port a minimal QF3-style off-policy flow actor-critic into the sibling of
  `algo.py` that already carries the flat-MLP trunk (post-swap architecture). Reuse:
  - `collect.py` storage and `raw_returns`/`normalized_advantages` for the anchor,
  - the flow head for `v_θ` unchanged,
  - `maybe` the flat-MLP actor as the base policy network (the MLP trunk is the better default
    for an MLP-ish head anyway, and matches QF3's architecture).
  Success criterion: reproduces QF3's published ordering on Hopper-v5 (offline smoke + one full seed
  at ~500k steps, return within 10% of reported QF3 behavior at matched alpha).
- **Stage 1 (1 week).** Implement `α(s)` realizations (A), (B), (C) behind one config flag
  (`velocity_clip_scale ∈ {"uniform", "rms_ema", "critic_curvature", "clip_adapt"}`), defaulting to
  `uniform`. Pin the homogeneous path with new pins (`pin_qclip_variant.py`) so the heterogeneous
  code paths cannot silently change the homogeneous result.
- **Stage 2 (2–3 weeks).** Experiments — see Section 4.
- **Stage 3 (2 weeks).** Paper writing + artifacts.

### 3.4 What changes in the architecture (nothing, deliberately)

No trunk, no encoder changes. `FlowMLP` stays; critic untouched. This keeps the implementation
surface small and the attribution clean: **one algorithmic delta, one family of hyperparameters.**

---

## 4. Experiments

### 4.1 Tasks

| Tier | Tasks | Purpose |
|---|---|---|
| Core | Hopper-v5, Walker2d-v5 | Locomotion standard; where homogeneous α is known to work |
| Hard | Ant-v5, Humanoid-v5 | Where homogeneous α demonstrably fails / needs different α |
| Sparse (stretch) | Pusher-v5 | Stress test for Q-quality in the sensitivity signal |

### 4.2 Baselines (all re-trained and evaluated in this repo, matched seeds `eval_seed = 10000 + seed`)

1. **QF3 (homogeneous α)** — re-implemented faithfully per arXiv:2610.08789; `α` swept per task on
   the reported grid, best-of reported as "tuned homogeneous α". This is the honest comparator,
   not a straw man.
2. **FPO++ (this repo's default, ASPO, per-token ratio)** — current head, matched at budget.
3. **SAC-Gaussian (rsl‑rl-quality Gaussian baseline)** — context for absolute numbers.
4. **QF3 with pooled α per task** — incumbents' published settings, as control for tuning luck.

### 4.3 Metrics

- Return vs environment steps, mean last-10 episodes, 3 seeds minimum per arm, 95% CI from paired
  seeds; **bootstrap CI on mean difference** (paired, same seeds across arms).
- Wall-clock per environment step and per update (to test H4).
- Episode-length streaks and their bimodality (to test H3 via streak histograms).
- Drift norm (`‖V‖`) per update; clip fraction per dimension (sanity that the heterogeneous clip
  is doing what it claims to).
- **Paired-seed gates**, exactly as this repo's `abl_base` harness: every comparison reports
  `median_last10`, `pooled_se`, and the gap must exceed 2× pooled SE to claim a difference.
- Determinism probe on the final checkpoint (eval noise floor = 0.0 required).

### 4.4 Ablation matrix

| α(s) | λ_cfm | tasks | reading |
|---|---|---|---|
| uniform (tuned) | per task (best) | all 4 | honest QF3 re-implementation, ceiling of homogeneous |
| rms_ema | 0.1 / 0.3 / 1.0 | all 4 | does dynamics-scale info carry the day? |
| critic_curvature | 0.1 / 0.3 / 1.0 | all 4 | does return-sensitivity info carry the day? |
| clip_adapt | 0.1 / 0.3 / 1.0 | all 4 | robustness under Q miscalibration |
| **heterogeneous** | **one setting** | all 4 | the claim; must work out-of-the-box-ish |

### 4.5 Expected results (revised down to keep the paper honest)

Honestly, all of: (i) a clean win on Humanoid-v5 where scalar α is known hard to tune and where
the Q is least flat; (ii) a wash or small win on Hopper-v5; (iii) tuning gap closure on Robomimic-
style or Pusher-like regimes; (iv) QF3 homogeneous re-implementation beating FPO++ at wall-clock,
reproducing the QF3 motivation inside your codebase. Any of (ii) or (iii) plus (i) makes the paper;
(i) alone is the headline.

---

## 5. Theory targets (heuristic, not a proof)

**T1 (bounded drift).** Show the per-dimension update keeps `x̂_{1,clip}` within a bounded ellipsoid
of the replay action, with the ellipsoid axes exactly `α(s)`; relate to the QF3 one-step error bound
`‖x̂₁ − x₁‖ ≤ ...` from their analysis. The heterogeneous bound should strictly dominate the scalar
one when `σ_i` differ by > 2×.

**T2 (improvement direction, informal).** Decompose the actor gradient contribution from the clip vs
the Q term; show the heterogeneous α aligns the update with the steepest-descent direction of a
local approximation of Q (Gauss-Newton), with alignment error bounded by the curvature estimate's
diagonal approximation error. A 2D tabular case (1 action dim, toy Q) makes this concrete.

**T3 (stability under noise).** Drift-norm dynamics under advantage/Q noise, with the safety floor
argument from the drifting proposal (internal prior art) as motivation. Deliverable is a
**stability diagnostic**, not a theorem: `‖V‖` trajectory and clip-fraction phase plots over training,
on all four tasks, three seeds.

No convergence proof is claimed — this is an empirical RL paper with a principle-backed heuristic,
in the same genre as QF3 itself.

---

## 6. Related work (positioning)

- **FPO / FPO++** (arXiv:2507.21053, arXiv:2602.02481): on-policy CFM-ratio methods; per-sample
  ratios, ASPO trust region, humanoid/locomotion benchmarks. Orthogonal to this work: we change
  the *clip geometry*, not the objective family.
- **QF3** (arXiv:2610.08789): the method being extended; its homogeneity is the contribution gap.
- **AWR / advantage-weighted regression family**: different improvement operators; both share the
  "improvement as distribution matching" umbrella (see local survey `docs/drifting_robot_policy_proposal.md`, which positions the same axes).
- **Diffusion/Flow RL** (Diffusion-QL, RFO, Flow-DPPO): velocity-space clipping is QF3's innovation;
  this work does not compete with them on the same axis, and we cite them for context.
- **Advantage-filtered buffers / OGPO-style**: QF3 wins where replay is quality-filtered; we keep
  QF3's recipe and only re-geometrize the trust region — minimal moving parts.
- **Trust-region/PPO literature** (clip, adaptive clip, per-token ratio granularity): homogeneous
  clipping is standard; per-dimension clipping guided by return geometry is the delta.

Venue reading: this is a **method + evidence paper** (valuable for CoRL/ICRA/RSS), not a
theory paper (too thin for a full NeurIPS/ICLR submission), and not a benchmark paper (comparison
is deliberately minimal, not a survey).

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| QF3 re-implementation not faithful (numbers off) | Stage 0 gate: reproduce the paper's Hopper ordering at matched budget before starting Stage 1 |
| Heterogeneous α helps little (diagnosis: Q is already flat in Humanoid with clip active) | Fall back to (C), the controller-based α — robust regardless of Q quality; H2 still testable |
| Tuning burden just moves (λ_cfm × α matrix) | Pre-registered single-setting-for-all-tasks protocol is the headline claim; if that fails H2, say so |
| Wall-clock regressions from curvature pass | Realization A (pure EMA) is a strict super-set of the claim with zero extra cost; curvature pass only run every `K` steps |
| Compute budget too small to show learning | 500k steps minimum on 2 tasks (Ant, Walker2d is cheapest at ~2k steps/s; budget accordingly); report last-10 with CI |
| The repo is not a QF3 codebase (it's PPO/FPO) | Stage 0 is a port, not a refactor; FPO path preserved behind `update_impl ∈ {fpo, qf3, qf3_het}` flag |

---

## 8. Timeline

| Week | Milestone |
|---|---|
| W1 | Stage 0 done: homogeneous QF3 runs on Hopper-v5; return within 10% of reference at matched α; `pin_qclip_variant.py` green |
| W2 | Stage 1 done: all three α(s) behind one config flag; homogeneous control confirmed stable across seeds |
| W3–4 | Core experiments: 2 core tasks × 3 seeds, heterogeneous arms; H1/H2 readout |
| W5 | Hard tasks (Ant, Humanoid-v5) on same protocol; H3 diagnostics |
| W6 | Ablations (β curvature vs EMA vs controller; λ_cfm grid) + H4 timing |
| W7 | Determinism floor, artifacts (pinned code, seeds, configs), Selfcheck of reproducibility |
| W8 | Paper draft v1 |
| W9 | Internal review pass; revise |
| W10 | Submit |
| W11–12 | Buffer for rebuttals / workshop fallback |

---

## 9. What this paper is NOT

- Not a new RL algorithm family; it is a trust-region geometry revision on QF3's foundation.
- Not an architectural change; trunk, encoder, and critic are untouched.
- Not a theory paper; stability is monitored, not proven.
- Not a demo video; all claims are MuJoCo with deterministic eval and paired seeds, in this repo.

---

## Appendix A — Concrete config surface (to be implemented in `config.py`)

```python
# QF3-style clip geometry (Stage 1)
velocity_clip_scale: str = "uniform"        # {"uniform", "rms_ema", "critic_curvature", "clip_adapt"}
velocity_clip_alpha_max: float = 3.0        # absolute cap for (A)/(C)
velocity_clip_rms_ema_decay: float = 0.95   # (A) EMA for per-dim |Δθ|
velocity_clip_curvature_eps: float = 1e-6   # (B) diagonal Hessian floor
velocity_clip_curvature_every: int = 4      # (B) update curvature only every K steps
velocity_clip_adapt_target: float = 0.15    # (C) per-dim clip-fraction setpoint
velocity_clip_adapt_lr: float = 0.01        # (C) controller gain
```

And a new pin pair, `pin_qclip_uniform.pt` / `pin_qclip_het.pt`, one per realizability path, so
the heterogeneous math is pinned independently of homogeneous QF3.

## Appendix B — Claims to be made in the paper (pre-written, to be verified)

1. Per-dimension velocity trust region raises return/interaction-step on Humanoid-v5 and Ant-v5
   at matched budget (paired seeds, pooled-SE gate, 3 seeds).
2. The tuning burden of QF3 is substantially reduced (same `α_max`/`λ_cfm` across all four tasks
   in the heterogeneous arm).
3. The homogeneous re-implementation in this repo reproduces the QF3-vs-FPO++ gap, confirming
   the method's relevance on tasks QF3 is known to struggle with.
4. All code, configs, seeds, eval scripts, and thresholds are released; behavior is pinned at
   `atol=0` for the homogeneous path.

---

## Appendix C — Stage 0 implementation sketch (what “port QF3” actually means here)

This is the part the viability of the whole proposal hinges on, so it needs to be honest
before anyone writes code: **myfpo today is an on-policy PPO/FPO trainer, not an off-policy
actor-critic.** There is no replay buffer, no Bellman vacuum, no TD target that lives for more
than one rollout. “Porting QF3” therefore means *adding an off-policy update path*, not
remapping function calls. That is the single hardest and longest part of Stage 0, and the
timeline only holds if it is treated as its own unit of work with its own tests.

### C.1 Module map after Stage 0

```
config.py        + update_impl, qf3_* scalars, replay buffer knobs
qf3.py           NEW ~400 lines — QF3 update (actor clip path, critic backup, replay)
train.py         dispatch on cfg.update_impl; FPO path untouched and still default
critic.py        unchanged (already an ensemble with target critic + polyak)
collect.py       unchanged (FPO path keeps using it)
replay.py        NEW ~150 lines — flat replay deque + sampling; does NOT reuse RolloutBuffer
checkpointing.py + replay buffer state (count, cursor, tensors) for resume
scripts/pins/pin_qclip_uniform.py   NEW — pins the homogeneous path
tests/test_qf3.py                   NEW — clip path unit tests + toy-buffer integration
```

### C.2 What carries over vs. what does not

| Piece                  | From where        | Notes                                                    |
|------------------------|-------------------|----------------------------------------------------------|
| Flow head `v_θ`        | existing          | reused as the actor; same signature `v(s, t, ε)`          |
| Flat-MLP trunk         | existing          | QF3 is MLP-ish anyway; fine for a first implementation   |
| Ensemble critic        | existing          | double-Q via the 2-member ensemble (TD3/SAC trick)       |
| Target critic          | existing          | soft-updated at rate τ = 0.005 as today                  |
| obs / reward normalizer| existing          | reused as-is; no new normalizer                          |
| `RolloutBuffer`        | **not reusable**  | on-policy chunks, no TD; do not bend it into a replay    |
| `RolloutBuffers`       | **not reusable**  | same reason                                              |
| `raw_returns` / GAE    | **not reusable**  | QF3 targets come from `Q_target(s', a')`, not from GAE   |
| `normalized_advantages`| **not reusable**  | actor signal comes from `Q(s, x̂₁_clip)` directly          |
| `collect.py` pipeline  | **not reusable**  | QF3 never needs a rollout pass to build a batch          |
| eval pipeline          | reused            | unchanged; paired seeds, same CSVs                        |

### C.3 The new replay buffer — minimal viable shape

```python
@dataclass
class ReplayBuffer:
    obs_dim: int; action_dim: int; horizon: int
    obs: torch.Tensor        # [cap, obs_dim]
    action: torch.Tensor     # [cap, horizon, action_dim]
    reward: torch.Tensor     # [cap]
    next_obs: torch.Tensor   # [cap, obs_dim]
    done: torch.Tensor       # [cap, bool]
    pos: int = 0; size: int = 0
    def __init__(..., capacity: int): ...
    def add(...): ...
    def sample(self, batch_size: int, device): ...
```

Observations are **raw** (normalizer applied at read time) so `config.obs_normalizer` can keep
a single writer and remain the same object the policy is built on. Reward stays unnormalized
for critic targets (`gamma` is what it is); advantage scaling is not needed in QF3.

### C.4 The new update — actor side (the actual QF3 clip)

```python
def qf3_actor_update(batch, policy, critic, critic_target, cfg):
    s   = normalizer(batch.obs); s_ = normalizer(batch.next_obs)
    eps = torch.randn_like(batch.action)                       # same noise distribution as training
    t   = torch.rand(batch.action.size(0)).expand_as(batch.action[..., :1]).squeeze(-1)
    a   = batch.action
    # One-step Euler predictor from the replay action (NOT a full ODE solve):
    # x_1 ≈ a + v(s, 1, eps) * 1.0  -- from the noise endpoint, not from a_buf;
    # here the *clip* is applied around the replay action, not around x_1.
    # (Exact Euler form to confirm against the paper/repo's released code in Stage 0,
    #  this is the one open implementation detail flagged to resolve before Stage 1.)
    v    = policy.flow.velocity(s, t, eps)
    delta = (v - a)  # replay-anchored delta; the clip region centers on a
    # homogeneous baseline
    alpha = cfg.velocity_clip_alpha             # scalar, QF3-faithful
    delta_clip = delta.clamp(-alpha, +alpha)
    x1_clip = a + delta_clip                    # [B, H, D], stays inside the trust region
    # actor objective
    q = critic(s, x1_clip)                      # [num_critics, B] or [B]
    q = q.mean(0) if q.dim() == 2 else q
    flow_anchor = (v - flow_target(s, eps, a)) ** 2     # `flow_match` to replay data
    actor_loss  = -q.mean() + cfg.qf3_cfm_weight * flow_anchor.mean()
    actor_loss.backward()
```

Notes on the clip mechanics:

- The clip acts on **velocity space** (`v − u_buf`), matching the QF3 paper rather than clipping
  action space directly.
- Exact form of `x̂_{1,clip}` (whether one Euler step from `a_buf` at `t=0` velocity, or from
  the noise side) is an **implementation detail to verify against the QF3 release** before Stage
  1 — not a research degree of freedom to tune away. It is deliberately flagged, not silently
  resolved, because getting it wrong invalidates the Stage 0 fidelity gate.

### C.5 The new update — critic side (and what is genuinely borrowed)

The critic needs SARSA-style targets from replay, not GAE returns. The safest starting point,
picked deliberately over inventing something:

```python
with torch.no_grad():
    eps_  = torch.randn_like(batch.action)
    t_    = torch.rand(batch.action.size(0))
    a_    = policy.flow.sample_action(s_, t_, eps_, num_steps=1)  # one-step proxy, SAC-ish
    q_next = critic_target(s_, a_)               # target critic, polyak-updated as today
    target  = batch.reward + cfg.gamma * (1 - batch.done.float()) * q_next
pred   = critic(s, batch.action)
critic_loss = huber(pred - target, beta=cfg.value_loss_beta)
critic_loss.mean().backward()
```

This deliberately borrows: ensemble double-Q, target critic with polyak, Huber loss. This is the
*smallest* actor-critic that is recognizable as "QF3-like" without inventing new value machinery
on top of an unproven clip. If Stage 0 reveals the paper wants a plain SAC single-critic instead,
the delta is one line (drop the ensemble, share state dict), not a design change.

### C.6 Config surface added in Stage 0

```python
update_impl: str = "fpo"                 # {"fpo", "qf3", "qf3_het"} — flag gates everything
qf3_update_every: int = 1                # gradient steps per env step
qf3_cfm_weight: float = 1.0              # λ_cfm; homogenous reference suggests ~1.0
qf3_batch_size: int = 256                # replay minibatch
qf3_replay_capacity: int = 100_000
qf3_min_buffer: int = 10_000             # don't start updates until this filled
qf3_alpha: float = 1.0                   # homogeneous clip-magnitude (tuned per task)
velocity_clip_scale: str = "uniform"     # becomes "rms_ema" | "critic_curvature" | "clip_adapt" in Stage 1
velocity_clip_alpha_max: float = 3.0
qf3_cfm_weight: float = 1.0
```

(Config field names may shift to match house style during implementation; the semantic set is
fixed. `qf3_cfm_weight` doubles as the λ_cfm knob for the homogeneous re-implementation.)

### C.7 The order of work inside Stage 0

1. **ReplayBuffer + sampling tests** — knock on the simplest door first: buffer add/sample,
   wrap-around, `min_buffer` gate, done-masking correctness. 4 tests, pure CPU, fast.
2. **One-step Euler predictor** — test that `policy.flow.predict(s, t, ε)` returns the right
   shape and that a *full* ODE solve at `t=0 → 1` produces the same vector as the one-step call
   up to the clip (algebraic consistency check, not a bitwise assert). 2 tests.
3. **Homogeneous QF3 update** — one update, tiny buffer, tiny net: verify no NaNs, actor loss
   finite, critic loss decreases on a toy fixed buffer over 100 steps. 2 tests.
4. **Dispatch wiring** — training loop's FPO path unchanged; smoke run with
   `--myfpo.update_impl qf3` for 200 steps end-to-end on Hopper; pins re-baked if off by intent.
5. **Fidelity gate** — the real test: one full seed on Hopper-v5, `α` swept on the reported grid,
   result must land within ~10% of the QF3 paper's behaviour at matched `α`. Treat this as
   pass/fail, not "about as good as we can do." If it fails, Stage 1 does not start.

### C.8 Cost estimate, itemized

| Item                                     | Est. lines | Est. time |
|------------------------------------------|-----------|-----------|
| `replay.py`                              | ~150      | 1 day     |
| `qf3.py` actor+critic updates            | ~350–450  | 3–4 days  |
| `config.py` + validation                 | ~30       | ½ day     |
| `train.py` dispatch                      | ~40       | 1 day     |
| `checkpointing.py` replay state          | ~30       | ½ day     |
| `tests/test_qf3.py`                      | ~250      | 1–2 days  |
| smoke + tune `α` for Hopper              | —         | 1 day     |
| **Total**                                |          | **8–10 days** |

Buffer and update paths each get their own test file from day one — this is where the
Stage 0 risk lives, and the pins are the floor, not the ceiling.
