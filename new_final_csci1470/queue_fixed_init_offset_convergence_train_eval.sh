#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
mkdir -p logs

PIPE_TAG="${PIPE_TAG:-fixedinit_offset_converge_$(date +%Y%m%d_%H%M%S)}"

echo "Submitting convergence-first fixed-init training job with PIPE_TAG=$PIPE_TAG"
TRAIN_JOB_ID=$(
  sbatch --parsable \
    --export=ALL,PIPE_TAG="$PIPE_TAG",RUN_NAME="fixedinit_offset_converge_${PIPE_TAG}",FIXED_INIT_PROFILE=offset_ref,FIXED_INIT_POS_JITTER_STD=0.0,FIXED_INIT_VEL_JITTER_STD=0.0,FIXED_INIT_JITTER_TRIES=1,UPDATES=1800,NUM_ENVS=12,ROLLOUT_STEPS=128,PPO_EPOCHS=6,MINIBATCH_SIZE=512,EVAL_EVERY=10,EVAL_EPISODES=16,VEC_ENV=subproc,MP_START_METHOD=spawn,HORIZON_STEPS=420,MAX_ACTION_NORM=0.30,ENT_COEF=0.00015,REWARD_SCALE=500.0,W_POS=4.0,W_VEL=2.5,W_FUEL=0.003,W_NEAR_COLLISION=6.0,W_COLLISION=250.0,W_ESCAPE=15.0,W_SWITCH=0.20,W_PHASE=0.12,EVAL_POS_THRESHOLD_TRAIN=0.06,EVAL_VEL_THRESHOLD_TRAIN=0.09,EVAL_CONSECUTIVE_CONVERGED_TRAIN=260,EVAL_MIN_TOTAL_STEPS_TRAIN=340,SAVE_TOPK=3 \
    sbatch_train_fixed_init_quick_30m.sh
)

echo "Submitting dependent convergence-first eval job afterok:$TRAIN_JOB_ID"
EVAL_JOB_ID=$(
  sbatch --parsable \
    --dependency=afterok:"$TRAIN_JOB_ID" \
    --export=ALL,PIPE_TAG="$PIPE_TAG",NUM_SETUPS=1,MAX_STEPS=420,POS_THRESHOLD=0.06,VEL_THRESHOLD=0.09,CONSECUTIVE_CONVERGED=300,MIN_TOTAL_STEPS_FOR_CONVERGED=360,TRAIL_LEN=90,FRAME_STRIDE=1,AXIS_PAD=0.20 \
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
