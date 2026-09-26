#!/usr/bin/env bash
# Cartesian sweep: envs x seeds, one scripts/train.py call per run.
# Run dirs (<out-base>/<method>_<env>_seed<seed>/) are composed by train.py.
#
#   scripts/sweep.sh [--tiny] [--out-base DIR] [--python BIN] [extra train.py args...]
#
# Axes from the environment (space-separated); defaults = full 15-run matrix:
#   ENVS="HalfCheetah-v5 Walker2d-v5 Hopper-v5" SEEDS="0 1 2 3 4"
# Budgets default to train.py's own resolution (CLI > YAML > defaults); pin them:
#   TOTAL_STEPS / EVAL_STEPS / SAVE_STEPS
# --tiny: Hopper-v5 x seed 0 at 20k/10k/10k.
# Remaining args pass through to every train.py call (e.g. --config, --sjfpo.*).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
OUT_BASE="outputs/"
TINY=0
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tiny) TINY=1; shift ;;
    --out-base) OUT_BASE="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,16p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

METHODS="${METHODS:-sjfpo}"
ENVS="${ENVS:-HalfCheetah-v5 Walker2d-v5 Hopper-v5}"
SEEDS="${SEEDS:-0 1 2 3 4}"

BUDGET_FLAGS=()
[[ -n "${TOTAL_STEPS:-}" ]] && BUDGET_FLAGS+=(--total_steps "$TOTAL_STEPS")
[[ -n "${EVAL_STEPS:-}" ]] && BUDGET_FLAGS+=(--eval_steps "$EVAL_STEPS")
[[ -n "${SAVE_STEPS:-}" ]] && BUDGET_FLAGS+=(--save_steps "$SAVE_STEPS")

if [[ "$TINY" == 1 ]]; then
  METHODS="sjfpo"
  ENVS="Hopper-v5"
  SEEDS="0"
  BUDGET_FLAGS=(--total_steps 20000 --eval_steps 10000 --save_steps 10000)
fi

ok=0
failed=0
# Word-splitting the axes is intentional.
# shellcheck disable=SC2086
for method in $METHODS; do
  # shellcheck disable=SC2086
  for env in $ENVS; do
    # shellcheck disable=SC2086
    for seed in $SEEDS; do
      if "$PYTHON" "$ROOT/scripts/train.py" \
          --method "$method" --env "$env" --seed "$seed" \
          --out_dir "$OUT_BASE" "${BUDGET_FLAGS[@]}" "${EXTRA[@]}"; then
        ok=$((ok + 1))
      else
        failed=$((failed + 1))
        echo "sweep: FAILED method=$method env=$env seed=$seed" >&2
      fi
    done
  done
done

echo "sweep: $ok ok, $failed failed"
if [[ "$failed" != 0 ]]; then
  exit 2
fi
