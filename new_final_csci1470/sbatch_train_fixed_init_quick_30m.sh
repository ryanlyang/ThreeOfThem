#!/usr/bin/env bash
#SBATCH --job-name=csci1470_fixedinit_train
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
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

# Keep BLAS/OpenMP thread pools from oversubscribing the env worker processes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mpl_cache}"
mkdir -p "$MPLCONFIGDIR"

RUN_NAME="${RUN_NAME:-fixedinit_quick_${PIPE_TAG}}"
UPDATES="${UPDATES:-900}"
NUM_ENVS="${NUM_ENVS:-8}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-128}"
PPO_EPOCHS="${PPO_EPOCHS:-6}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-256}"
EVAL_EVERY="${EVAL_EVERY:-15}"
EVAL_EPISODES="${EVAL_EPISODES:-12}"
SEED="${SEED:-4301}"
VEC_ENV="${VEC_ENV:-subproc}"
MP_START_METHOD="${MP_START_METHOD:-spawn}"
FIXED_INIT_PROFILE="${FIXED_INIT_PROFILE:-near_ref}"
FIXED_INIT_POSITIONS="${FIXED_INIT_POSITIONS:-}"
FIXED_INIT_VELOCITIES="${FIXED_INIT_VELOCITIES:-}"
FIXED_INIT_POS_JITTER_STD="${FIXED_INIT_POS_JITTER_STD:-0.008}"
FIXED_INIT_VEL_JITTER_STD="${FIXED_INIT_VEL_JITTER_STD:-0.006}"
FIXED_INIT_JITTER_TRIES="${FIXED_INIT_JITTER_TRIES:-64}"
HORIZON_STEPS="${HORIZON_STEPS:-360}"
ENT_COEF="${ENT_COEF:-0.0005}"
INITIAL_LOG_STD="${INITIAL_LOG_STD:--1.2}"
REWARD_SCALE="${REWARD_SCALE:-600.0}"
MAX_ACTION_NORM="${MAX_ACTION_NORM:-0.16}"
NEAR_COLLISION_DISTANCE="${NEAR_COLLISION_DISTANCE:-0.35}"
ESCAPE_RADIUS="${ESCAPE_RADIUS:-4.0}"
W_POS="${W_POS:-1.20}"
W_VEL="${W_VEL:-0.90}"
W_FUEL="${W_FUEL:-0.05}"
W_NEAR_COLLISION="${W_NEAR_COLLISION:-8.0}"
W_COLLISION="${W_COLLISION:-150.0}"
W_ESCAPE="${W_ESCAPE:-8.0}"
W_SWITCH="${W_SWITCH:-0.35}"
W_PHASE="${W_PHASE:-0.08}"
EVAL_POS_THRESHOLD_TRAIN="${EVAL_POS_THRESHOLD_TRAIN:-0.08}"
EVAL_VEL_THRESHOLD_TRAIN="${EVAL_VEL_THRESHOLD_TRAIN:-0.12}"
EVAL_CONSECUTIVE_CONVERGED_TRAIN="${EVAL_CONSECUTIVE_CONVERGED_TRAIN:-180}"
EVAL_MIN_TOTAL_STEPS_TRAIN="${EVAL_MIN_TOTAL_STEPS_TRAIN:-220}"
SAVE_TOPK="${SAVE_TOPK:-3}"
EARLY_STOP_ON_STRICT_SUCCESS="${EARLY_STOP_ON_STRICT_SUCCESS:-0}"
EARLY_STOP_SUCCESS_RATE="${EARLY_STOP_SUCCESS_RATE:-1.0}"
EARLY_STOP_MAX_FAILURE_RATE="${EARLY_STOP_MAX_FAILURE_RATE:-0.0}"
EARLY_STOP_PATIENCE_EVALS="${EARLY_STOP_PATIENCE_EVALS:-1}"
EARLY_STOP_MIN_EVALS="${EARLY_STOP_MIN_EVALS:-1}"

EARLY_STOP_FLAG=""
if [[ "$EARLY_STOP_ON_STRICT_SUCCESS" == "1" ]]; then
  EARLY_STOP_FLAG="--early-stop-on-strict-success"
fi

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
  --device auto \
  --vec-env "$VEC_ENV" \
  --mp-start-method "$MP_START_METHOD" \
  --fixed-init-profile "$FIXED_INIT_PROFILE" \
  --fixed-init-positions "$FIXED_INIT_POSITIONS" \
  --fixed-init-velocities "$FIXED_INIT_VELOCITIES" \
  --fixed-init-pos-jitter-std "$FIXED_INIT_POS_JITTER_STD" \
  --fixed-init-vel-jitter-std "$FIXED_INIT_VEL_JITTER_STD" \
  --fixed-init-jitter-tries "$FIXED_INIT_JITTER_TRIES" \
  --horizon-steps "$HORIZON_STEPS" \
  --ent-coef "$ENT_COEF" \
  --initial-log-std "$INITIAL_LOG_STD" \
  --reward-scale "$REWARD_SCALE" \
  --max-action-norm "$MAX_ACTION_NORM" \
  --near-collision-distance "$NEAR_COLLISION_DISTANCE" \
  --escape-radius "$ESCAPE_RADIUS" \
  --w-pos "$W_POS" \
  --w-vel "$W_VEL" \
  --w-fuel "$W_FUEL" \
  --w-near-collision "$W_NEAR_COLLISION" \
  --w-collision "$W_COLLISION" \
  --w-escape "$W_ESCAPE" \
  --w-switch "$W_SWITCH" \
  --w-phase "$W_PHASE" \
  --eval-pos-threshold "$EVAL_POS_THRESHOLD_TRAIN" \
  --eval-vel-threshold "$EVAL_VEL_THRESHOLD_TRAIN" \
  --eval-consecutive-converged "$EVAL_CONSECUTIVE_CONVERGED_TRAIN" \
  --eval-min-total-steps "$EVAL_MIN_TOTAL_STEPS_TRAIN" \
  --save-topk "$SAVE_TOPK" \
  --early-stop-success-rate "$EARLY_STOP_SUCCESS_RATE" \
  --early-stop-max-failure-rate "$EARLY_STOP_MAX_FAILURE_RATE" \
  --early-stop-patience-evals "$EARLY_STOP_PATIENCE_EVALS" \
  --early-stop-min-evals "$EARLY_STOP_MIN_EVALS" \
  $EARLY_STOP_FLAG

RUN_DIR=$(ls -dt "$PWD"/artifacts/${RUN_NAME}_* | head -n1)
echo "$RUN_DIR" > "$PIPE_DIR/train_run_dir.txt"
echo "$RUN_DIR/checkpoint_best.pt" > "$PIPE_DIR/checkpoint_best_path.txt"
echo "$RUN_DIR/checkpoint_latest.pt" > "$PIPE_DIR/checkpoint_latest_path.txt"

echo "RUN_DIR=$RUN_DIR"
echo "end_time=$(date)"
