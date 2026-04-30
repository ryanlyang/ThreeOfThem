#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
mkdir -p logs

PIPE_TAG="${PIPE_TAG:-fixedinit_offset_trueconv_$(date +%Y%m%d_%H%M%S)}"

echo "Submitting TRUE-CONVERGENCE fixed-init training job with PIPE_TAG=$PIPE_TAG"
TRAIN_JOB_ID=$(
  sbatch --parsable \
    --partition=tier3 \
    --time=12:00:00 \
    --cpus-per-task=8 \
    --mem=32G \
    --export=ALL,PIPE_TAG="$PIPE_TAG",RUN_NAME="fixedinit_offset_trueconv_${PIPE_TAG}",FIXED_INIT_PROFILE=offset_ref,FIXED_INIT_POS_JITTER_STD=0.0,FIXED_INIT_VEL_JITTER_STD=0.0,FIXED_INIT_JITTER_TRIES=1,UPDATES=2200,NUM_ENVS=8,ROLLOUT_STEPS=128,PPO_EPOCHS=6,MINIBATCH_SIZE=256,EVAL_EVERY=10,EVAL_EPISODES=3,VEC_ENV=subproc,MP_START_METHOD=spawn,HORIZON_STEPS=500,MAX_ACTION_NORM=1.5,INITIAL_LOG_STD=-3.0,ENT_COEF=0.00005,REWARD_SCALE=1500.0,ESCAPE_RADIUS=8.0,W_POS=20.0,W_VEL=12.0,W_FUEL=0.000001,W_NEAR_COLLISION=50.0,W_COLLISION=1200.0,W_ESCAPE=80.0,W_SWITCH=0.08,W_PHASE=0.20,EVAL_POS_THRESHOLD_TRAIN=0.05,EVAL_VEL_THRESHOLD_TRAIN=0.08,EVAL_CONSECUTIVE_CONVERGED_TRAIN=280,EVAL_MIN_TOTAL_STEPS_TRAIN=360,SAVE_TOPK=3,EARLY_STOP_ON_STRICT_SUCCESS=1,EARLY_STOP_SUCCESS_RATE=1.0,EARLY_STOP_MAX_FAILURE_RATE=0.0,EARLY_STOP_PATIENCE_EVALS=1,EARLY_STOP_MIN_EVALS=1 \
    sbatch_train_fixed_init_quick_30m.sh
)

echo "Submitting dependent TRUE-CONVERGENCE eval job afterok:$TRAIN_JOB_ID"
EVAL_JOB_ID=$(
  sbatch --parsable \
    --partition=tier3 \
    --time=01:30:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --dependency=afterok:"$TRAIN_JOB_ID" \
    --export=ALL,PIPE_TAG="$PIPE_TAG",EVAL_EPISODES=3,NUM_SETUPS=1,MAX_STEPS=500,POS_THRESHOLD=0.05,VEL_THRESHOLD=0.08,CONSECUTIVE_CONVERGED=300,MIN_TOTAL_STEPS_FOR_CONVERGED=400,TRAIL_LEN=120,FRAME_STRIDE=1,AXIS_PAD=0.25 \
    sbatch_eval_fixed_init_after_train.sh
)

echo "TRAIN_JOB_ID=$TRAIN_JOB_ID"
echo "EVAL_JOB_ID=$EVAL_JOB_ID"
echo "PIPE_TAG=$PIPE_TAG"

echo "Track jobs:"
echo "  squeue -u \"$USER\""
echo "Logs:"
echo "  logs/csci1470_fixedinit_train_${TRAIN_JOB_ID}.out"
echo "  logs/csci1470_fixedinit_train_${TRAIN_JOB_ID}.err"
echo "  logs/csci1470_fixedinit_eval_${EVAL_JOB_ID}.out"
echo "  logs/csci1470_fixedinit_eval_${EVAL_JOB_ID}.err"
echo "Pipeline metadata dir: artifacts/pipelines/$PIPE_TAG"
