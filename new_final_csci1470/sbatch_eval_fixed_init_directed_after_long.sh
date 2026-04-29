#!/usr/bin/env bash
#SBATCH --job-name=csci1470_fixedinit_directed_eval
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
fi
mkdir -p logs

PIPE_TAG="${PIPE_TAG:?PIPE_TAG is required (set by queue script)}"
PIPE_DIR="$PWD/artifacts/pipelines/$PIPE_TAG"

if [[ ! -d "$PIPE_DIR" ]]; then
  echo "ERROR: missing pipeline directory: $PIPE_DIR"
  exit 2
fi

VENV_DIR="${VENV_DIR:-$PWD/.venv_csci1470_smoke}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PWD/.pip_cache}"
export PIP_CACHE_DIR

bash ./bootstrap_venv.sh "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --cache-dir "$PIP_CACHE_DIR" -r requirements-train.txt

which python
python --version
nvidia-smi || true

if [[ -f "$PIPE_DIR/checkpoint_best_path.txt" ]]; then
  CKPT_PATH=$(cat "$PIPE_DIR/checkpoint_best_path.txt")
else
  echo "ERROR: checkpoint path metadata missing in $PIPE_DIR"
  exit 3
fi

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "ERROR: checkpoint not found: $CKPT_PATH"
  exit 4
fi

if [[ -f "$PIPE_DIR/train_run_dir.txt" ]]; then
  RUN_DIR=$(cat "$PIPE_DIR/train_run_dir.txt")
else
  RUN_DIR="$(dirname "$CKPT_PATH")"
fi

OUTDIR="$RUN_DIR/inference_fixed_init_10examples"
mkdir -p "$OUTDIR"

echo "=== EVAL FIXED INIT (DIRECTED LONG) ==="
echo "PIPE_TAG=$PIPE_TAG"
echo "checkpoint=$CKPT_PATH"
echo "outdir=$OUTDIR"

python evaluate_fixed_init_quick.py \
  --checkpoint "$CKPT_PATH" \
  --episodes "${EVAL_EPISODES:-24}" \
  --num-setups "${NUM_SETUPS:-10}" \
  --max-steps "${MAX_STEPS:-420}" \
  --pos-threshold "${POS_THRESHOLD:-0.07}" \
  --vel-threshold "${VEL_THRESHOLD:-0.10}" \
  --consecutive-converged "${CONSECUTIVE_CONVERGED:-300}" \
  --min-total-steps-for-converged "${MIN_TOTAL_STEPS_FOR_CONVERGED:-360}" \
  --trail-len "${TRAIL_LEN:-90}" \
  --frame-stride "${FRAME_STRIDE:-1}" \
  --axis-pad "${AXIS_PAD:-0.25}" \
  --seed "${SEED:-1234}" \
  --device auto \
  --outdir "$OUTDIR"

echo "$OUTDIR" > "$PIPE_DIR/inference_outdir.txt"
echo "end_time=$(date)"
