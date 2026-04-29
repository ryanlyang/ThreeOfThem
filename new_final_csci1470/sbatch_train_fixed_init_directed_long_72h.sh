#!/usr/bin/env bash
#SBATCH --job-name=csci1470_fixedinit_directed_long
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
fi
mkdir -p logs

PIPE_TAG="${PIPE_TAG:-fixedinit_directed_long_$(date +%Y%m%d_%H%M%S)}"
PIPE_DIR="$PWD/artifacts/pipelines/$PIPE_TAG"
mkdir -p "$PIPE_DIR"

echo "=== TRAIN FIXED INIT (DIRECTED LONG) ==="
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

# Keep BLAS/OpenMP thread pools from oversubscribing per worker process.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RUN_NAME="${RUN_NAME:-fixedinit_directed_long_${PIPE_TAG}}"
UPDATES="${UPDATES:-4200}"
NUM_ENVS="${NUM_ENVS:-12}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-128}"
PPO_EPOCHS="${PPO_EPOCHS:-6}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-512}"
EVAL_EVERY="${EVAL_EVERY:-20}"
EVAL_EPISODES="${EVAL_EPISODES:-24}"
SEED="${SEED:-4301}"
VEC_ENV="${VEC_ENV:-subproc}"
MP_START_METHOD="${MP_START_METHOD:-fork}"

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
  --fixed-init-profile near_ref \
  --fixed-init-pos-jitter-std "${FIXED_INIT_POS_JITTER_STD:-0.012}" \
  --fixed-init-vel-jitter-std "${FIXED_INIT_VEL_JITTER_STD:-0.010}" \
  --fixed-init-jitter-tries "${FIXED_INIT_JITTER_TRIES:-96}" \
  --horizon-steps "${HORIZON_STEPS:-420}" \
  --max-action-norm "${MAX_ACTION_NORM:-0.14}" \
  --ent-coef "${ENT_COEF:-0.0003}" \
  --reward-scale "${REWARD_SCALE:-600.0}" \
  --w-pos "${W_POS:-1.35}" \
  --w-vel "${W_VEL:-1.05}" \
  --w-fuel "${W_FUEL:-0.06}" \
  --w-collision "${W_COLLISION:-180.0}" \
  --w-escape "${W_ESCAPE:-10.0}" \
  --w-switch "${W_SWITCH:-0.45}" \
  --w-phase "${W_PHASE:-0.10}" \
  --eval-pos-threshold "${EVAL_POS_THRESHOLD:-0.07}" \
  --eval-vel-threshold "${EVAL_VEL_THRESHOLD:-0.10}" \
  --eval-consecutive-converged "${EVAL_CONSECUTIVE_CONVERGED:-300}" \
  --eval-min-total-steps "${EVAL_MIN_TOTAL_STEPS:-360}" \
  --save-topk "${SAVE_TOPK:-3}" \
  --vec-env "$VEC_ENV" \
  --mp-start-method "$MP_START_METHOD"

RUN_DIR=$(ls -dt "$PWD"/artifacts/${RUN_NAME}_* | head -n1)
echo "$RUN_DIR" > "$PIPE_DIR/train_run_dir.txt"
echo "$RUN_DIR/checkpoint_best.pt" > "$PIPE_DIR/checkpoint_best_path.txt"
echo "$RUN_DIR/checkpoint_latest.pt" > "$PIPE_DIR/checkpoint_latest_path.txt"
echo "$RUN_DIR/checkpoint_best_rank1.pt" > "$PIPE_DIR/checkpoint_best_rank1_path.txt"
echo "$RUN_DIR/checkpoint_best_rank2.pt" > "$PIPE_DIR/checkpoint_best_rank2_path.txt"
echo "$RUN_DIR/checkpoint_best_rank3.pt" > "$PIPE_DIR/checkpoint_best_rank3_path.txt"

if [[ -f "$RUN_DIR/checkpoint_topk_manifest.json" ]]; then
  cp "$RUN_DIR/checkpoint_topk_manifest.json" "$PIPE_DIR/checkpoint_topk_manifest.json"
fi

echo "RUN_DIR=$RUN_DIR"
echo "end_time=$(date)"
