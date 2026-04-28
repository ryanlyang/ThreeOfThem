#!/usr/bin/env bash
#SBATCH --job-name=csci1470_fixedinit_train
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
fi
mkdir -p logs

PIPE_TAG="${PIPE_TAG:-fixedinit_$(date +%Y%m%d_%H%M%S)}"
PIPE_DIR="$PWD/artifacts/pipelines/$PIPE_TAG"
mkdir -p "$PIPE_DIR"

echo "=== TRAIN FIXED INIT (QUICK) ==="
echo "PIPE_TAG=$PIPE_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "start_time=$(date)"

echo "$SLURM_JOB_ID" > "$PIPE_DIR/train_job_id.txt"
echo "$PIPE_TAG" > "$PIPE_DIR/pipe_tag.txt"

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

RUN_NAME="${RUN_NAME:-fixedinit_quick_${PIPE_TAG}}"
UPDATES="${UPDATES:-300}"
NUM_ENVS="${NUM_ENVS:-4}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-64}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-128}"
EVAL_EVERY="${EVAL_EVERY:-10}"
EVAL_EPISODES="${EVAL_EPISODES:-8}"
SEED="${SEED:-4301}"

python train_fixed_init_quick.py \
  --updates "$UPDATES" \
  --num-envs "$NUM_ENVS" \
  --rollout-steps "$ROLLOUT_STEPS" \
  --ppo-epochs "$PPO_EPOCHS" \
  --minibatch-size "$MINIBATCH_SIZE" \
  --eval-every "$EVAL_EVERY" \
  --eval-episodes "$EVAL_EPISODES" \
  --seed "$SEED" \
  --run-name "$RUN_NAME" \
  --save-dir artifacts \
  --device auto

RUN_DIR=$(ls -dt "$PWD"/artifacts/${RUN_NAME}_* | head -n1)
echo "$RUN_DIR" > "$PIPE_DIR/train_run_dir.txt"
echo "$RUN_DIR/checkpoint_best.pt" > "$PIPE_DIR/checkpoint_best_path.txt"
echo "$RUN_DIR/checkpoint_latest.pt" > "$PIPE_DIR/checkpoint_latest_path.txt"

echo "RUN_DIR=$RUN_DIR"
echo "end_time=$(date)"
