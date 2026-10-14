# Actionable Contributions for Paper Submission
Based on repo audit: README, docs/DIAGNOSE_DEGRADATION.md, configuration_sjfpo.py, fpo_core.py

## A. Spectral Prior Validation (Highest Priority — Unvalidated Claim)

### Gap
README: "The spectral idea... Shipped as noise_scaling_mode=freq_weighted_time. The companion idea — a per-band trust region sharing that profile — is NOT implemented."
Config comment: "A per-band trust region sharing that profile — is NOT implemented."
DIAGNOSE_DEGRADATION.md: "Whether the spectral prior helped or hurt. That run used `noise_scaling_mode=freq_weighted_time`. There is no `white` control, so the clip and the prior are confounded."

### Action
Run controlled ablation on same env/seed/config:
- Arm A: `noise_scaling_mode='white', noise_scaling_beta=1.0` (plain FPO++)
- Arm B: `noise_scaling_mode='freq_weighted_time', noise_scaling_beta=1.0` (spectral)
- Arm C: `freq_weighted_time` with `beta=0.5, 2.0` (sensitivity sweep)

Log all metrics from DIAGNOSE_DEGRADATION.md (grad_norm split, value_loss, explained_var, log_ratio_max, cfm_above_clamp_frac) plus new per-role norms (actor/critic split — requires restoring instrumentation per doc's Step 0).

### Contribution Claim
"We show that a `1/f^beta` spectral base measure (via rfft gain on action-step axis) improves/stabilizes flow RL training compared to white noise, or identify conditions where it degrades performance."

### Deliverable
Ablation table + training curves + per-role gradient diagnostics. Without this, the spectral claim is unsupported.

---

## B. Per-Band Trust Region (Explicitly Unimplemented — Clear Extension)

### Gap
Config line 6-7: "The companion idea — a per-band trust region sharing that profile — is NOT implemented."
This means the ASPO clip (`clip_param=0.05`) applies uniformly across frequency bands, ignoring that low-frequency modes (large-scale action structure) may need different clipping from high-frequency modes (fine correction).

### Action
Modify `fpo_core.py`: compute `mag` (the spectral std) for each frequency bin, then apply frequency-dependent clipping:
```
clip_param_k = clip_param * (1 + (kπ/H)^2)^(β/2)  # larger clip for high-freq modes
```
Or inverse: tighter clipping on high-frequency noise to prevent overfitting fine details.
Compare against uniform ASPO.

### Contribution Claim
"We extend ASPO to frequency-aware clipping, aligning the trust region with the spectral base measure."

### Deliverable
Modified `compute_cfm_loss` / trust region function; comparison table; brief mathematical justification (line 6-8 of config file becomes implemented, not just commented).

---

## C. Degradation Diagnosis & Fix (Real Issue — Open in Repo)

### Gap
DIAGNOSE_DEGRADATION.md: 46% decay at 614,400 steps; grad_norm_preclip 2291x budget; actor/critic split unrecorded (collapsed in commit fc8724b); value_loss 611.44 (5 orders above policy loss); explained_var 0.186.

### Action (Step 0 from doc — Required Before Any Interpretation)
1. Restore per-role gradient norms in `train_sjfpo.py` (rebuild `actor_params` / `critic_params` from `get_actor_optim_params()` / `get_critic_optim_params()`; log `actor_grad_norm_preclip` and `critic_grad_norm_preclip`).
2. Assert `atol=0` against concatenated norm in a test.
3. Log `value_huber_delta` as `float(policy._huber_resid_rms * cfg.value_huber_delta)`.

Then run with white vs. spectral control to isolate cause:
- Is it spectral-related? (Compare with white control)
- Is it gradient saturation? (Split actor vs. critic norms)
- Is it critic misfit? (`explained_var`, `value_loss`, `huber_delta` tracking)

### Contribution Claim
"We diagnose and mitigate late-stage degradation in flow RL for humanoid control by restoring per-role gradient diagnostics and identifying [cause from above]."

---

## D. Spectral Mode for Exploration / Entropy Control

### Gap
The spectral profile (`std = (1 + (kπ/H)²)^(-β/2)`) controls the magnitude of noise per frequency. Currently `beta=1.0` (pink noise). No analysis connects this to exploration or entropy.

### Action
Analyze entropy of the flow model under spectral vs. white noise:
- Measure `train/entropy` or approximate via flow-ratio std (`train/flow_ratio_std` is 0.0098 vs clip 0.05 — very low variance).
- Test hypothesis: spectral noise increases low-frequency exploration (larger action structure variation) while suppressing high-frequency jitter, improving temporal consistency.

### Contribution Claim
"We show that spectral noise profiles provide structured exploration in flow RL, with frequency-dependent exploration that improves temporal consistency compared to isotropic noise."

---

## E. Minimal Paper Package (What You Need to Submit)

| Component | Status | Action Needed |
|---|---|---|
| Spectral validation (white vs. freq) | Not done | Run A/B/C (Section A) |
| Per-band trust region | Not implemented | Implement + compare (Section B) |
| Degradation fix / diagnosis | Partially documented | Restore instrumentation + isolate (Section C) |
| Mathematical justification | Partial (fpo_core.py lines 63-146) | Formalize spectral-consistency proof |
| Reproducibility note | Exists (README) | Explicit control experiments |

### Minimal Viable Paper (MVP)
1. Run A vs. B on 2-3 envs (Pendulum-v1 smoke, Humanoid-v5 core, one manipulation).
2. Implement per-band ASPO (even if results are neutral — negative results are publishable with analysis).
3. Restore gradient instrumentation; report split norms.
4. Include `docs/DIAGNOSE_DEGRADATION.md` resolution (even if partial).
5. Write 4-6 pages: intro (flow RL + spectral prior), method (fpo_core modifications), experiments (controlled ablation), analysis (degradation diagnosis), conclusion.

### What Makes It Publishable
- **Novel mechanism**: Spectral base measure for flow RL is a concrete algorithmic change, not just a hyperparameter sweep.
- **Controlled evaluation**: The repo explicitly lacks white controls — providing them is a scientific contribution.
- **Implementation transparency**: The repo documents exactly what's missing (`NOT implemented`, no control, collapsed metrics) — addressing these demonstrates rigor.
- **Connection to known issue**: Degradation diagnosis is a real, unsolved problem in the FPO family.
