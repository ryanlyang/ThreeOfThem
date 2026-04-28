#!/usr/bin/env bash
#SBATCH --job-name=csci1470_train_ppo_long
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=23:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
fi
mkdir -p logs

PIPE_TAG="${PIPE_TAG:-pipe_$(date +%Y%m%d_%H%M%S)}"
PIPE_DIR="$PWD/artifacts/pipelines/$PIPE_TAG"
mkdir -p "$PIPE_DIR"

echo "=== TRAIN LONG PPO (23h) ==="
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

RUN_NAME="ppo_long_${PIPE_TAG}"

# Tuned long-run defaults; override via environment variables if needed.
UPDATES="${UPDATES:-2200}"
NUM_ENVS="${NUM_ENVS:-8}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-128}"
PPO_EPOCHS="${PPO_EPOCHS:-8}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-256}"
EVAL_EVERY="${EVAL_EVERY:-10}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-7}"

# PPO hyperparameters.
LR="${LR:-3e-4}"
GAMMA="${GAMMA:-0.995}"
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
CLIP_COEF="${CLIP_COEF:-0.2}"
VF_COEF="${VF_COEF:-0.5}"
ENT_COEF="${ENT_COEF:-0.005}"

# Reward weights.
W_POS="${W_POS:-1.0}"
W_VEL="${W_VEL:-0.35}"
W_FUEL="${W_FUEL:-0.03}"
W_COLLISION="${W_COLLISION:-60.0}"
W_SWITCH="${W_SWITCH:-0.15}"
W_PHASE="${W_PHASE:-0.01}"

python train_ppo_figure8.py \
  --updates "$UPDATES" \
  --num-envs "$NUM_ENVS" \
  --rollout-steps "$ROLLOUT_STEPS" \
  --ppo-epochs "$PPO_EPOCHS" \
  --minibatch-size "$MINIBATCH_SIZE" \
  --eval-every "$EVAL_EVERY" \
  --eval-episodes "$EVAL_EPISODES" \
  --seed "$SEED" \
  --lr "$LR" \
  --gamma "$GAMMA" \
  --gae-lambda "$GAE_LAMBDA" \
  --clip-coef "$CLIP_COEF" \
  --vf-coef "$VF_COEF" \
  --ent-coef "$ENT_COEF" \
  --w-pos "$W_POS" \
  --w-vel "$W_VEL" \
  --w-fuel "$W_FUEL" \
  --w-collision "$W_COLLISION" \
  --w-switch "$W_SWITCH" \
  --w-phase "$W_PHASE" \
  --device auto \
  --run-name "$RUN_NAME"

RUN_DIR=$(ls -dt "$PWD"/artifacts/${RUN_NAME}_* | head -n1)

echo "$RUN_DIR" > "$PIPE_DIR/train_run_dir.txt"
echo "$RUN_DIR/checkpoint_best.pt" > "$PIPE_DIR/checkpoint_best_path.txt"
echo "$RUN_DIR/checkpoint_latest.pt" > "$PIPE_DIR/checkpoint_latest_path.txt"

echo "RUN_DIR=$RUN_DIR"
echo "end_time=$(date)"
