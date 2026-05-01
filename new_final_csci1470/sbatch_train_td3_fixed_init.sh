#!/usr/bin/env bash
#SBATCH --job-name=csci1470_td3_train
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
fi
mkdir -p logs

PIPE_TAG="${PIPE_TAG:-td3_fixedinit_$(date +%Y%m%d_%H%M%S)}"
PIPE_DIR="$PWD/artifacts/pipelines/$PIPE_TAG"
mkdir -p "$PIPE_DIR"

echo "=== TRAIN TD3 FIXED INIT ==="
echo "PIPE_TAG=$PIPE_TAG"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "start_time=$(date)"

echo "${SLURM_JOB_ID:-none}" > "$PIPE_DIR/train_job_id.txt"
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

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mpl_cache}"
mkdir -p "$MPLCONFIGDIR"

RUN_NAME="${RUN_NAME:-td3_fixedinit_${PIPE_TAG}}"
TOTAL_ENV_STEPS="${TOTAL_ENV_STEPS:-1200000}"
NUM_ENVS="${NUM_ENVS:-8}"
VEC_ENV="${VEC_ENV:-subproc}"
MP_START_METHOD="${MP_START_METHOD:-spawn}"

BUFFER_SIZE="${BUFFER_SIZE:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LEARNING_STARTS="${LEARNING_STARTS:-20000}"
UPDATES_PER_ITER="${UPDATES_PER_ITER:-2}"
GAMMA="${GAMMA:-0.995}"
TAU="${TAU:-0.005}"
ACTOR_LR="${ACTOR_LR:-3e-4}"
CRITIC_LR="${CRITIC_LR:-3e-4}"
POLICY_DELAY="${POLICY_DELAY:-2}"
TARGET_POLICY_NOISE="${TARGET_POLICY_NOISE:-0.20}"
TARGET_NOISE_CLIP="${TARGET_NOISE_CLIP:-0.50}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
EXPLORATION_NOISE="${EXPLORATION_NOISE:-0.30}"
EXPLORATION_NOISE_FINAL="${EXPLORATION_NOISE_FINAL:-0.05}"
EXPLORATION_DECAY_STEPS="${EXPLORATION_DECAY_STEPS:-500000}"

REWARD_SCALE="${REWARD_SCALE:-800.0}"
REWARD_CLIP="${REWARD_CLIP:-20.0}"
OBS_CLIP="${OBS_CLIP:-10.0}"

SEED="${SEED:-4301}"
BACKEND="${BACKEND:-numpy}"
DEVICE="${DEVICE:-auto}"

FIXED_INIT_PROFILE="${FIXED_INIT_PROFILE:-offset_ref}"
FIXED_INIT_POSITIONS="${FIXED_INIT_POSITIONS:-}"
FIXED_INIT_VELOCITIES="${FIXED_INIT_VELOCITIES:-}"
FIXED_INIT_POS_JITTER_STD="${FIXED_INIT_POS_JITTER_STD:-0.006}"
FIXED_INIT_VEL_JITTER_STD="${FIXED_INIT_VEL_JITTER_STD:-0.004}"
FIXED_INIT_JITTER_TRIES="${FIXED_INIT_JITTER_TRIES:-64}"
EVAL_FIXED_INIT_POS_JITTER_STD="${EVAL_FIXED_INIT_POS_JITTER_STD:-0.006}"
EVAL_FIXED_INIT_VEL_JITTER_STD="${EVAL_FIXED_INIT_VEL_JITTER_STD:-0.004}"
EVAL_FIXED_INIT_JITTER_TRIES="${EVAL_FIXED_INIT_JITTER_TRIES:-64}"

HORIZON_STEPS="${HORIZON_STEPS:-500}"
ACTION_DT="${ACTION_DT:-0.05}"
INTEGRATOR_DT="${INTEGRATOR_DT:-0.001}"
PHASE_SEARCH_RADIUS="${PHASE_SEARCH_RADIUS:-35}"
MAX_ACTION_NORM="${MAX_ACTION_NORM:-1.8}"
NEAR_COLLISION_DISTANCE="${NEAR_COLLISION_DISTANCE:-0.35}"
ESCAPE_RADIUS="${ESCAPE_RADIUS:-8.0}"
INIT_MIN_PAIR_DISTANCE="${INIT_MIN_PAIR_DISTANCE:-0.25}"

W_POS="${W_POS:-20.0}"
W_VEL="${W_VEL:-12.0}"
W_FUEL="${W_FUEL:-0.0005}"
W_NEAR_COLLISION="${W_NEAR_COLLISION:-20.0}"
W_COLLISION="${W_COLLISION:-500.0}"
W_ESCAPE="${W_ESCAPE:-50.0}"
W_SWITCH="${W_SWITCH:-0.08}"
W_PHASE="${W_PHASE:-0.20}"

EVAL_EVERY_ENV_STEPS="${EVAL_EVERY_ENV_STEPS:-25000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
EVAL_POS_THRESHOLD="${EVAL_POS_THRESHOLD:-0.06}"
EVAL_VEL_THRESHOLD="${EVAL_VEL_THRESHOLD:-0.09}"
EVAL_CONSECUTIVE_CONVERGED="${EVAL_CONSECUTIVE_CONVERGED:-260}"
EVAL_MIN_TOTAL_STEPS="${EVAL_MIN_TOTAL_STEPS:-320}"
SAVE_TOPK="${SAVE_TOPK:-3}"
LOG_EVERY_ENV_STEPS="${LOG_EVERY_ENV_STEPS:-5000}"

EARLY_STOP_ON_STRICT_SUCCESS="${EARLY_STOP_ON_STRICT_SUCCESS:-0}"
EARLY_STOP_SUCCESS_RATE="${EARLY_STOP_SUCCESS_RATE:-1.0}"
EARLY_STOP_MAX_FAILURE_RATE="${EARLY_STOP_MAX_FAILURE_RATE:-0.0}"
EARLY_STOP_PATIENCE_EVALS="${EARLY_STOP_PATIENCE_EVALS:-1}"
EARLY_STOP_MIN_EVALS="${EARLY_STOP_MIN_EVALS:-1}"

EARLY_STOP_FLAG=""
if [[ "$EARLY_STOP_ON_STRICT_SUCCESS" == "1" ]]; then
  EARLY_STOP_FLAG="--early-stop-on-strict-success"
fi

python train_td3_figure8.py \
  --total-env-steps "$TOTAL_ENV_STEPS" \
  --num-envs "$NUM_ENVS" \
  --vec-env "$VEC_ENV" \
  --mp-start-method "$MP_START_METHOD" \
  --buffer-size "$BUFFER_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --learning-starts "$LEARNING_STARTS" \
  --updates-per-iter "$UPDATES_PER_ITER" \
  --gamma "$GAMMA" \
  --tau "$TAU" \
  --actor-lr "$ACTOR_LR" \
  --critic-lr "$CRITIC_LR" \
  --policy-delay "$POLICY_DELAY" \
  --target-policy-noise "$TARGET_POLICY_NOISE" \
  --target-noise-clip "$TARGET_NOISE_CLIP" \
  --hidden-size "$HIDDEN_SIZE" \
  --exploration-noise "$EXPLORATION_NOISE" \
  --exploration-noise-final "$EXPLORATION_NOISE_FINAL" \
  --exploration-decay-steps "$EXPLORATION_DECAY_STEPS" \
  --reward-scale "$REWARD_SCALE" \
  --reward-clip "$REWARD_CLIP" \
  --obs-clip "$OBS_CLIP" \
  --horizon-steps "$HORIZON_STEPS" \
  --action-dt "$ACTION_DT" \
  --integrator-dt "$INTEGRATOR_DT" \
  --phase-search-radius "$PHASE_SEARCH_RADIUS" \
  --max-action-norm "$MAX_ACTION_NORM" \
  --near-collision-distance "$NEAR_COLLISION_DISTANCE" \
  --escape-radius "$ESCAPE_RADIUS" \
  --init-min-pair-distance "$INIT_MIN_PAIR_DISTANCE" \
  --backend "$BACKEND" \
  --fixed-init-profile "$FIXED_INIT_PROFILE" \
  --fixed-init-positions "$FIXED_INIT_POSITIONS" \
  --fixed-init-velocities "$FIXED_INIT_VELOCITIES" \
  --fixed-init-pos-jitter-std "$FIXED_INIT_POS_JITTER_STD" \
  --fixed-init-vel-jitter-std "$FIXED_INIT_VEL_JITTER_STD" \
  --fixed-init-jitter-tries "$FIXED_INIT_JITTER_TRIES" \
  --eval-fixed-init-pos-jitter-std "$EVAL_FIXED_INIT_POS_JITTER_STD" \
  --eval-fixed-init-vel-jitter-std "$EVAL_FIXED_INIT_VEL_JITTER_STD" \
  --eval-fixed-init-jitter-tries "$EVAL_FIXED_INIT_JITTER_TRIES" \
  --w-pos "$W_POS" \
  --w-vel "$W_VEL" \
  --w-fuel "$W_FUEL" \
  --w-near-collision "$W_NEAR_COLLISION" \
  --w-collision "$W_COLLISION" \
  --w-escape "$W_ESCAPE" \
  --w-switch "$W_SWITCH" \
  --w-phase "$W_PHASE" \
  --eval-every-env-steps "$EVAL_EVERY_ENV_STEPS" \
  --eval-episodes "$EVAL_EPISODES" \
  --eval-strict-mode \
  --eval-lock-to-end \
  --eval-pos-threshold "$EVAL_POS_THRESHOLD" \
  --eval-vel-threshold "$EVAL_VEL_THRESHOLD" \
  --eval-consecutive-converged "$EVAL_CONSECUTIVE_CONVERGED" \
  --eval-min-total-steps "$EVAL_MIN_TOTAL_STEPS" \
  --save-topk "$SAVE_TOPK" \
  --log-every-env-steps "$LOG_EVERY_ENV_STEPS" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --run-name "$RUN_NAME" \
  --save-dir artifacts \
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
