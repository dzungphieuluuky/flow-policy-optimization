#!/usr/bin/env bash
# =============================================================================
# Sweep driver for the critic fix + new defaults (commit 10de902).
#
# Usage:
#   bash scripts/run_sweeps.sh              # runs everything, in order
#   bash scripts/run_sweeps.sh s1 s2        # only those sweeps
#   bash scripts/run_sweeps.sh list         # list sweep names, run nothing
#
# Override any knob from the environment, e.g.
#   ENV_ID=Humanoid-v5 TOTAL_STEPS=100000 SEEDS="0 1" bash scripts/run_sweeps.sh s4
#
# IMPORTANT: every run below starts from the NEW defaults. Runs from before
# commit 10de902 are not comparable (cfm_loss_reduction, n_action_steps and
# max_grad_norm all changed), and old checkpoints will NOT resume -- the critic's
# first layer grew by one input. Re-run S0 before trusting any comparison.
#
# Stop early and skip if not on the PATH as given.
set -uo pipefail

# ---------------------------------------------------------------- knobs ------
PY="${PY:-.venv/bin/python}"
TRAIN="${TRAIN:-scripts/train.py}"
ENV_ID="${ENV_ID:-Hopper-v5}"
TOTAL_STEPS="${TOTAL_STEPS:-200000}"
SEEDS="${SEEDS:-0 1 2 3 4}"
OUT_ROOT="${OUT_ROOT:-outputs/sweeps}"
NUM_ENVS="${NUM_ENVS:-64}"
CHUNKS="${CHUNKS:-1024}"
BATCH="${BATCH:-128}"
REPORT_TO="${REPORT_TO:-wandb}"
# Early stopping compares against stop_return_frac with this patience. Left at
# its default, a good seed can finish early while a bad one runs to budget,
# which makes a mean over seeds incomparable. Effectively disabled here.
STOP_PATIENCE="${STOP_PATIENCE:-999}"
EVAL_STEPS="${EVAL_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-50000}"

die() { echo "ERROR: $*" >&2; exit 1; }

# run <sweep-tag> <extra args...>
run() {
  local tag="$1"; shift
  local seed
  for seed in $SEEDS; do
    local out="${OUT_ROOT}/${tag}/seed${seed}"
    echo "=============================================================="
    echo "[$(date +%H:%M:%S)] ${tag}  env=${ENV_ID}  seed=${seed}  steps=${TOTAL_STEPS}"
    echo "  $PY $TRAIN --env $ENV_ID --seed ${seed} --total_steps ${TOTAL_STEPS} \\"
    echo "    --eval_steps ${EVAL_STEPS} --save_steps ${SAVE_STEPS} --out_dir ${out} \\"
    echo "    --sjfpo.num_envs ${NUM_ENVS} --sjfpo.chunks_per_update ${CHUNKS} \\"
    echo "    --sjfpo.batch_size ${BATCH} --sjfpo.report_to ${REPORT_TO} \\"
    echo "    --sjfpo.stop_patience ${STOP_PATIENCE} $*"
    echo "=============================================================="
    $PY "$TRAIN" \
      --env "$ENV_ID" \
      --seed "$seed" \
      --total_steps "$TOTAL_STEPS" \
      --eval_steps "$EVAL_STEPS" \
      --save_steps "$SAVE_STEPS" \
      --out_dir "$out" \
      --sjfpo.num_envs "$NUM_ENVS" \
      --sjfpo.chunks_per_update "$CHUNKS" \
      --sjfpo.batch_size "$BATCH" \
      --sjfpo.report_to "$REPORT_TO" \
      --sjfpo.stop_patience "$STOP_PATIENCE" \
      "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
      echo "!! ${tag} seed=${seed} exited ${rc} -- continuing" >&2
      FAILED="${FAILED:-}${tag}/seed${seed}(rc=${rc}) "
    fi
  done
}

# =========================================================== the sweeps =====

# S0. Re-baseline. Everything below is relative to this. --baseline pins its own
# cfm_loss_reduction/ratio/trust-region, so these numbers are the reference the
# default-config sweeps get compared against.
sweep_s0_baselines() {
  local b
  for b in fpo fpo++ dppo; do
    run "s0_${b//+/p}" --baseline "$b"
  done
}

# S1. The critic correctness ablation. Do this BEFORE any other critic sweep --
# if the length feature changes results, every later critic number is read
# against a moving reference.
sweep_s1_exec_length() {
  run s1_exec_len_on  --sjfpo.critic_condition_on_exec_length true
  run s1_exec_len_off --sjfpo.critic_condition_on_exec_length false
}

# S2. cfm_loss_reduction. Tests the claim that "sum" inflates the CFM loss ~32x
# (62.5 vs 1.95), saturates log_ratio_cap, and collapses ASPO so that
# per_chunk and per_action become numerically identical.
# READ: clipfrac, cfm_above_clamp_frac, mean_log_ratio.
# PREDICTION under "sum": high clipfrac, and both granularities coincide.
# If clipfrac comes back near zero under "sum", that explanation is WRONG.
sweep_s2_loss_reduction() {
  run s2_reduction_mean --sjfpo.ratio_granularity per_action --sjfpo.cfm_loss_reduction mean
  run s2_reduction_sum  --sjfpo.ratio_granularity per_action --sjfpo.cfm_loss_reduction sum
  # Does the granularity still matter at all? Only meaningful under a
  # non-saturating ratio.
  run s2_gran_chunk_mean --sjfpo.ratio_granularity per_chunk --sjfpo.cfm_loss_reduction mean
  run s2_gran_sample_mean --sjfpo.ratio_granularity per_sample --sjfpo.cfm_loss_reduction mean
}

# S3. A0, receding horizon. n_action_steps=8 meant the chunk was planned AND
# executed fully open-loop, with no closed-loop correction for 8 env steps.
# READ: clipfrac and the gamma interaction -- shorter chunks discount less per
# macro-step (gae.py gamma**lengths), so retune gamma before concluding.
sweep_s3_horizon() {
  run s3_exec2 --sjfpo.n_action_steps 2
  run s3_exec4 --sjfpo.n_action_steps 4
  run s3_exec8 --sjfpo.n_action_steps 8
}

# S4. Critic capacity. Decoupled from the actor for the first time.
# Run this on a WIDE-STATE env too -- the whole point of critic_hidden_dim is
# that obs_dim runs 3 (Pendulum) to 348 (Humanoid-v5).
#   ENV_ID=Humanoid-v5 bash scripts/run_sweeps.sh s4
sweep_s4_critic_capacity() {
  run s4_hidden_actor_default --sjfpo.critic_hidden_dim null
  run s4_hidden_256           --sjfpo.critic_hidden_dim 256
  run s4_hidden_512           --sjfpo.critic_hidden_dim 512
  run s4_critics_4            --sjfpo.num_critics 4
  run s4_layers_3             --sjfpo.critic_num_layers 3
}

# S5. Grad-norm asymmetry. max_grad_norm went 48.0 -> 1.0; critic stayed 1.0.
# BUT: read pre_clip_actor / pre_clip_critic from S0 FIRST. Both are already
# logged. If the critic's norm was never near 1.0, 48.0 is free performance
# and the answer here is "leave it at 1.0".
sweep_s5_grad_clip() {
  run s5_critic_clip_1    --sjfpo.critic_max_grad_norm 1.0
  run s5_critic_clip_48   --sjfpo.critic_max_grad_norm 48.0
  run s5_critic_clip_none --sjfpo.critic_max_grad_norm null
  run s5_actor_clip_48    --sjfpo.max_grad_norm 48.0
}

# S6. Low priority, independent. critic_ema_decay=0.05 is a ~20-step lag
# (~1.25 rollouts); qc and TD-JEPA use 0.005. Justified on-policy, but untested.
sweep_s6_misc() {
  run s6_ema_0005 --sjfpo.critic_ema_decay 0.005
  run s6_epochs5  --sjfpo.num_learning_epochs 5
}

# S7. The spectral base prior: does a 1/f^beta base measure help? beta=0 is
# bit-identical to white and is the ablation baseline.
# READ noise_band_energy_high_over_low first: ~1.0 means the prior did not run.
# Kill criterion, set in advance: if beta=1.0 does not beat beta=0.0 on mean
# return across the seeds, that is the result. Do not extend to larger beta.
# Expected effect is small: at action_horizon=8 the lag-1 autocorrelation of a
# 1/f prior is ~0.26, so anything under ~1-2% return is seed noise at n=5.
sweep_s7_noise_prior() {
  local beta
  for beta in 0.0 0.5 1.0; do
    run "s7_beta${beta}" \
      --sjfpo.noise_scaling_mode freq_weighted_time \
      --sjfpo.noise_scaling_beta "$beta"
  done
  run s7_white_control --sjfpo.noise_scaling_mode white
}

ALL_SWEEPS="s0_baselines s1_exec_length s2_loss_reduction s3_horizon s4_critic_capacity s5_grad_clip s6_misc s7_noise_prior"

dispatch() {
  case "$1" in
    s0|s0_baselines)          sweep_s0_baselines ;;
    s1|s1_exec_length)        sweep_s1_exec_length ;;
    s2|s2_loss_reduction)     sweep_s2_loss_reduction ;;
    s3|s3_horizon)            sweep_s3_horizon ;;
    s4|s4_critic_capacity)    sweep_s4_critic_capacity ;;
    s5|s5_grad_clip)          sweep_s5_grad_clip ;;
    s6|s6_misc)               sweep_s6_misc ;;
    s7|s7_noise_prior)        sweep_s7_noise_prior ;;
    list) echo "$ALL_SWEEPS" | tr ' ' '\n' | nl ;;
    all) for s in $ALL_SWEEPS; do "sweep_$s"; done ;;
    *) die "unknown sweep '$1'. Try: list | all | s0 s1 s2 s3 s4 s5 s6 s7" ;;
  esac
}

# ------------------------------------------------------------------ main -----
[ -f "$TRAIN" ] || die "not found: $TRAIN (run from the repo root)"
[ -x "$PY" ]   || die "not executable: $PY (set PY=/path/to/python)"

if [ $# -eq 0 ]; then set -- all; fi
for arg in "$@"; do dispatch "$arg"; done

echo
echo "done. outputs under ${OUT_ROOT}/"
[ -n "${FAILED:-}" ] && echo "non-zero exits: ${FAILED}"
exit 0