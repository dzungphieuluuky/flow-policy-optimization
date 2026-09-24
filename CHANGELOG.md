# Changelog

## 2026-09-24

### Fixed

- **Torch BF16 policy updates:** Wrapped the live state conditioner and diagnostic actor pass in CUDA autocast so BF16 storage tensors no longer reach FP32 `Linear` layers.
- **Torch compile stability:** Normalized scalar flow timesteps to shape `[1]`, preventing Dynamo recompilation-limit warnings caused by scalar-versus-batched timestep inputs.
- **JAX vector rollout seeds:** Bounded random environment seeds to the int32 range and reserved space for per-environment seed offsets.
- **JAX JIT non-finite guard:** Replaced traced Python `bool` conversion with a JAX debug callback that still aborts on non-finite losses.
- **JAX DCT tracing:** Removed global DCT basis caching that could retain JAX tracers inside compiled training functions.
- **JAX vector environments:** Switched multi-environment Gymnasium workers from `fork` to `spawn` to avoid the JAX/fork deadlock warning.
- **JAX result metadata:** Corrected successful JAX run results to report `method: "sjfpo_jax"`.
- **Low-frequency DCT training:** Training now retains exactly `Ka = round(top_p_spectral_tokens * action_horizon)` coefficients in both Torch and JAX paths, matching the partial-IDCT inference path.
- **Single-chunk critic updates:** Removed a duplicate squeeze that reduced a one-element critic prediction to a scalar and caused value/return shape errors.
- **Random baseline evaluation:** Increased random-policy baseline evaluation from 12 to 30 episodes while keeping regular policy evaluation at 12 episodes.
- **Metric organization:** Grouped related return, length, success, jerk, and baseline metrics consistently in Torch and JAX W&B logging.
- **Shared MC flow updates:** Rollouts now retain the exact shared MC timesteps/noise and initial behavior CFM losses; minibatch updates reuse those losses instead of recomputing the frozen actor loss.
- **Partial chunk policy loss:** Chunks with an unexecuted action suffix no longer contribute to the policy loss; their valid initial-state value target still trains the critic.
- **Explicit shared MC keys:** Torch rollout batches now use `shared_mc_tau` and `shared_mc_eps` instead of ambiguous `tau`/`eps` batch keys.
- **Split optimizers:** Actor/conditioner and critic now use separate AdamW instances, with checkpoints storing both optimizer states in a versioned payload.
- **Action diagnostics:** Added `train/action_magnitude` logging to Torch and JAX training paths.

### Updated

- Added a Pi-style primary OpenCode agent at `.opencode/agent/pi.md`.
- Replaced the generic root agent guidance with project-specific `AGENTS.md` instructions.
- Added `temp.md` documenting action scaling, transition storage, DCT, and flow-matching loss behavior.

### Validation

- Torch BF16 CUDA forward/backward passed.
- Torch low-frequency DCT smoke passed with `H=16`, `Ka=8`, `top_p_spectral_tokens=0.5`, and wandb disabled.
- JAX CPU smokes passed for 32 and 128 training steps, including evaluation and checkpoint writing.
- JAX two-environment spawn smoke passed without the `os.fork()` warning.
- Shared MC initial-loss, split-optimizer, low-frequency DCT, and split-checkpoint roundtrip tests passed.
- Focused network, modeling, evaluator, and W&B checks were run; the repository still has pre-existing test drift, including stale diagnostic-key expectations and JAX test imports.
