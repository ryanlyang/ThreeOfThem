#!/usr/bin/env bash
#SBATCH --job-name=csci1470_infer_best_gif
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
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
  RUN_NAME="ppo_long_${PIPE_TAG}"
  RUN_DIR=$(ls -dt "$PWD"/artifacts/${RUN_NAME}_* | head -n1)
  CKPT_PATH="$RUN_DIR/checkpoint_best.pt"
fi

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "ERROR: checkpoint not found: $CKPT_PATH"
  exit 3
fi

if [[ -f "$PIPE_DIR/train_run_dir.txt" ]]; then
  RUN_DIR=$(cat "$PIPE_DIR/train_run_dir.txt")
else
  RUN_DIR="$(dirname "$CKPT_PATH")"
fi

OUTDIR="$RUN_DIR/inference_best_gif"
mkdir -p "$OUTDIR"

echo "=== RUN INFERENCE + SETUP GIFS + BEST GIF ==="
echo "PIPE_TAG=$PIPE_TAG"
echo "checkpoint=$CKPT_PATH"
echo "outdir=$OUTDIR"

python run_inference_best_gif.py \
  --checkpoint "$CKPT_PATH" \
  --num-setups "${NUM_SETUPS:-10}" \
  --max-steps "${MAX_STEPS:-320}" \
  --policy deterministic \
  --pos-threshold "${POS_THRESHOLD:-0.35}" \
  --vel-threshold "${VEL_THRESHOLD:-0.45}" \
  --seed "${SEED:-1234}" \
  --device auto \
  --outdir "$OUTDIR"

echo "$OUTDIR" > "$PIPE_DIR/inference_outdir.txt"

echo "end_time=$(date)"
