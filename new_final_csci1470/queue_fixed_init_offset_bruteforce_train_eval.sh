#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
mkdir -p logs

PIPE_TAG="${PIPE_TAG:-fixedinit_offset_bruteforce_$(date +%Y%m%d_%H%M%S)}"

echo "Submitting brute-force convergence fixed-init training job with PIPE_TAG=$PIPE_TAG"
TRAIN_JOB_ID=$(
  sbatch --parsable \
    --export=ALL,PIPE_TAG="$PIPE_TAG",RUN_NAME="fixedinit_offset_bruteforce_${PIPE_TAG}",FIXED_INIT_PROFILE=offset_ref,FIXED_INIT_POS_JITTER_STD=0.0,FIXED_INIT_VEL_JITTER_STD=0.0,FIXED_INIT_JITTER_TRIES=1,UPDATES=2500,NUM_ENVS=16,ROLLOUT_STEPS=128,PPO_EPOCHS=8,MINIBATCH_SIZE=512,EVAL_EVERY=10,EVAL_EPISODES=16,VEC_ENV=subproc,MP_START_METHOD=spawn,HORIZON_STEPS=420,MAX_ACTION_NORM=2.0,INITIAL_LOG_STD=-4.0,ENT_COEF=0.0,REWARD_SCALE=900.0,ESCAPE_RADIUS=8.0,W_POS=10.0,W_VEL=7.0,W_FUEL=0.00005,W_NEAR_COLLISION=8.0,W_COLLISION=300.0,W_ESCAPE=20.0,W_SWITCH=0.15,W_PHASE=0.18,EVAL_POS_THRESHOLD_TRAIN=0.05,EVAL_VEL_THRESHOLD_TRAIN=0.08,EVAL_CONSECUTIVE_CONVERGED_TRAIN=280,EVAL_MIN_TOTAL_STEPS_TRAIN=360,SAVE_TOPK=3 \
    sbatch_train_fixed_init_quick_30m.sh
)

echo "Submitting dependent brute-force convergence eval job afterok:$TRAIN_JOB_ID"
EVAL_JOB_ID=$(
  sbatch --parsable \
    --dependency=afterok:"$TRAIN_JOB_ID" \
    --export=ALL,PIPE_TAG="$PIPE_TAG",NUM_SETUPS=1,MAX_STEPS=420,POS_THRESHOLD=0.05,VEL_THRESHOLD=0.08,CONSECUTIVE_CONVERGED=320,MIN_TOTAL_STEPS_FOR_CONVERGED=380,TRAIL_LEN=90,FRAME_STRIDE=1,AXIS_PAD=0.20 \
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
