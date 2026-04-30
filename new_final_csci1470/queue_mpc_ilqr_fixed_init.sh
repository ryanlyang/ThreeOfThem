#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
mkdir -p logs

PIPE_TAG="${PIPE_TAG:-mpc_ilqr_fixedinit_$(date +%Y%m%d_%H%M%S)}"

echo "Submitting MPC/iLQR fixed-init job with PIPE_TAG=$PIPE_TAG"
JOB_ID=$(
  sbatch --parsable \
    --partition=tier3 \
    --time=12:00:00 \
    --cpus-per-task=32 \
    --mem=64G \
    --export=ALL,PIPE_TAG="$PIPE_TAG",OUTDIR="$ROOT_DIR/inference_mpc_ilqr_fixed_init_${PIPE_TAG}",NUM_SETUPS=10,SEED=4301,MAX_STEPS=420,FIXED_INIT_PROFILE=offset_ref,FIXED_INIT_POS_JITTER_STD=0.0,FIXED_INIT_VEL_JITTER_STD=0.0,FIXED_INIT_JITTER_TRIES=1,HORIZON_STEPS=420,MAX_ACTION_NORM=2.0,NEAR_COLLISION_DISTANCE=0.35,ESCAPE_RADIUS=8.0,PHASE_SEARCH_RADIUS=35,W_POS_MATCH=4.0,W_VEL_MATCH=2.5,W_SWITCH_MATCH=0.2,W_PHASE_MATCH=0.12,MPC_HORIZON=42,MPC_ITERS=14,MPC_MODEL_SUBSTEPS=4,MPC_Q_POS=50.0,MPC_Q_VEL=16.0,MPC_R_ACTION=0.0002,MPC_TERMINAL_SCALE=40.0,MPC_NEAR_COLLISION_WEIGHT=40.0,MPC_NEAR_COLLISION_DISTANCE=0.28,REPLAN_EVERY=1,POS_THRESHOLD=0.06,VEL_THRESHOLD=0.09,CONSECUTIVE_CONVERGED=260,MIN_TOTAL_STEPS_FOR_CONVERGED=320,TRAIL_LEN=90,FRAME_STRIDE=1,AXIS_PAD=0.20 \
    sbatch_mpc_ilqr_fixed_init.sh
)

echo "JOB_ID=$JOB_ID"
echo "PIPE_TAG=$PIPE_TAG"
echo "Track:"
echo "  squeue -u \"$USER\""
echo "Logs:"
echo "  logs/csci1470_mpc_ilqr_${JOB_ID}.out"
echo "  logs/csci1470_mpc_ilqr_${JOB_ID}.err"
echo "Pipeline metadata dir:"
echo "  artifacts/pipelines/$PIPE_TAG"
