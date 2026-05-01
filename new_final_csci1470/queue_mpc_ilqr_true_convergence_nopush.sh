#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
mkdir -p logs

PIPE_TAG="${PIPE_TAG:-mpc_ilqr_nopush_$(date +%Y%m%d_%H%M%S)}"
FIXED_INIT_PROFILE="${FIXED_INIT_PROFILE:-offset_ref}"

echo "Submitting MPC/iLQR NO-PUSH baseline job with PIPE_TAG=$PIPE_TAG"
JOB_ID=$(
  sbatch --parsable \
    --partition=debug \
    --time=02:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --export=ALL,PIPE_TAG="$PIPE_TAG",OUTDIR="$ROOT_DIR/inference_mpc_ilqr_nopush_${PIPE_TAG}",NUM_SETUPS=1,SEED=4301,MAX_STEPS=500,FIXED_INIT_PROFILE="$FIXED_INIT_PROFILE",FIXED_INIT_POS_JITTER_STD=0.0,FIXED_INIT_VEL_JITTER_STD=0.0,FIXED_INIT_JITTER_TRIES=1,HORIZON_STEPS=500,MAX_ACTION_NORM=0.0,NEAR_COLLISION_DISTANCE=0.35,ESCAPE_RADIUS=8.0,PHASE_SEARCH_RADIUS=35,W_POS_MATCH=5.0,W_VEL_MATCH=3.0,W_SWITCH_MATCH=0.20,W_PHASE_MATCH=0.15,MPC_HORIZON=1,MPC_ITERS=1,MPC_MODEL_SUBSTEPS=1,MPC_Q_POS=80.0,MPC_Q_VEL=24.0,MPC_R_ACTION=0.00005,MPC_TERMINAL_SCALE=60.0,MPC_NEAR_COLLISION_WEIGHT=100.0,MPC_NEAR_COLLISION_DISTANCE=0.35,REPLAN_EVERY=1000,POS_THRESHOLD=0.06,VEL_THRESHOLD=0.09,CONSECUTIVE_CONVERGED=300,MIN_TOTAL_STEPS_FOR_CONVERGED=400,TRAIL_LEN=120,FRAME_STRIDE=1,AXIS_PAD=0.25,LOG_EVERY=25 \
    sbatch_mpc_ilqr_fixed_init.sh
)

echo "JOB_ID=$JOB_ID"
echo "PIPE_TAG=$PIPE_TAG"
echo "FIXED_INIT_PROFILE=$FIXED_INIT_PROFILE"
echo "Track:"
echo "  squeue -u \"$USER\""
echo "Logs:"
echo "  logs/csci1470_mpc_ilqr_${JOB_ID}.out"
echo "  logs/csci1470_mpc_ilqr_${JOB_ID}.err"
echo "Output dir:"
echo "  inference_mpc_ilqr_nopush_${PIPE_TAG}"
echo "Pipeline metadata dir:"
echo "  artifacts/pipelines/$PIPE_TAG"
