#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
mkdir -p logs

PIPE_TAG="${PIPE_TAG:-mpc_olws_compare_far_$(date +%Y%m%d_%H%M%S)}"
FIXED_INIT_PROFILE="${FIXED_INIT_PROFILE:-offset_ref_far}"

echo "Submitting MPC open-loop + warm-start compare (farther fixed init) with PIPE_TAG=$PIPE_TAG"
JOB_ID=$(
  sbatch --parsable \
    --partition=tier3 \
    --time=06:00:00 \
    --cpus-per-task=8 \
    --mem=32G \
    --export=ALL,PIPE_TAG="$PIPE_TAG",OUTDIR="$ROOT_DIR/inference_mpc_ilqr_openloop_warmstart_${PIPE_TAG}",SEED=4301,MAX_STEPS=500,FIXED_INIT_PROFILE="$FIXED_INIT_PROFILE",JITTER_SETUPS=10,JITTER_POS_STD=0.006,JITTER_VEL_STD=0.004,JITTER_TRIES=64,HORIZON_STEPS=500,MAX_ACTION_NORM=2.5,NEAR_COLLISION_DISTANCE=0.35,ESCAPE_RADIUS=8.0,PHASE_SEARCH_RADIUS=35,W_POS_MATCH=5.0,W_VEL_MATCH=3.0,W_SWITCH_MATCH=0.20,W_PHASE_MATCH=0.15,MPC_HORIZON=60,MPC_ITERS=20,MPC_MODEL_SUBSTEPS=4,MPC_Q_POS=80.0,MPC_Q_VEL=24.0,MPC_R_ACTION=0.00005,MPC_TERMINAL_SCALE=60.0,MPC_NEAR_COLLISION_WEIGHT=100.0,MPC_NEAR_COLLISION_DISTANCE=0.35,REPLAN_EVERY=1,OPENLOOP_CEM_ITERS=12,OPENLOOP_CEM_POPULATION=128,OPENLOOP_CEM_ELITE_FRAC=0.12,OPENLOOP_CEM_INIT_STD_FRAC=0.55,OPENLOOP_CEM_MIN_STD_FRAC=0.05,OPENLOOP_CEM_SMOOTHING=0.25,OPENLOOP_EXEC_STEPS=40,POS_THRESHOLD=0.06,VEL_THRESHOLD=0.09,CONSECUTIVE_CONVERGED=300,MIN_TOTAL_STEPS_FOR_CONVERGED=400,TRAIL_LEN=120,FRAME_STRIDE=1,AXIS_PAD=0.25,LOG_EVERY=25 \
    sbatch_mpc_ilqr_openloop_warmstart.sh
)

echo "JOB_ID=$JOB_ID"
echo "PIPE_TAG=$PIPE_TAG"
echo "Track:"
echo "  squeue -u \"$USER\""
echo "Logs:"
echo "  logs/csci1470_mpc_olws_${JOB_ID}.out"
echo "  logs/csci1470_mpc_olws_${JOB_ID}.err"
echo "Output dir:"
echo "  inference_mpc_ilqr_openloop_warmstart_${PIPE_TAG}"
echo "Pipeline metadata dir:"
echo "  artifacts/pipelines/$PIPE_TAG"
