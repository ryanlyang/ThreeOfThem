#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
mkdir -p logs

PIPE_TAG="${PIPE_TAG:-td3_mpcclone_$(date +%Y%m%d_%H%M%S)}"
PARTITION="${PARTITION:-debug}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_GB="${MEM_GB:-40G}"

# Debug-friendly defaults. The full 96-setup MPC dataset is too expensive for a
# quick proof run; this keeps the expert collection and TD3 phase bounded.
MPC_DATASET_SETUPS="${MPC_DATASET_SETUPS:-12}"
MPC_MAX_STEPS="${MPC_MAX_STEPS:-500}"
MPC_REPLAN_EVERY="${MPC_REPLAN_EVERY:-2}"
MPC_HORIZON="${MPC_HORIZON:-42}"
MPC_ITERS="${MPC_ITERS:-14}"
MPC_MODEL_SUBSTEPS="${MPC_MODEL_SUBSTEPS:-3}"
TD3_TOTAL_ENV_STEPS="${TD3_TOTAL_ENV_STEPS:-300000}"
TD3_PREFILL_MAX_SAMPLES="${TD3_PREFILL_MAX_SAMPLES:-150000}"

echo "Submitting TD3 MPC-clone pipeline"
echo "PIPE_TAG=$PIPE_TAG"
echo "PARTITION=$PARTITION TIME=$TIME_LIMIT CPU=$CPUS_PER_TASK MEM=$MEM_GB"
echo "MPC_DATASET_SETUPS=$MPC_DATASET_SETUPS MPC_REPLAN_EVERY=$MPC_REPLAN_EVERY MPC_HORIZON=$MPC_HORIZON MPC_ITERS=$MPC_ITERS"
echo "TD3_TOTAL_ENV_STEPS=$TD3_TOTAL_ENV_STEPS"

JOB_ID=$(
  sbatch --parsable \
    --partition="$PARTITION" \
    --time="$TIME_LIMIT" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM_GB" \
    --export=ALL,PIPE_TAG="$PIPE_TAG",MPC_DATASET_SETUPS="$MPC_DATASET_SETUPS",MPC_MAX_STEPS="$MPC_MAX_STEPS",MPC_REPLAN_EVERY="$MPC_REPLAN_EVERY",MPC_HORIZON="$MPC_HORIZON",MPC_ITERS="$MPC_ITERS",MPC_MODEL_SUBSTEPS="$MPC_MODEL_SUBSTEPS",TD3_TOTAL_ENV_STEPS="$TD3_TOTAL_ENV_STEPS",TD3_PREFILL_MAX_SAMPLES="$TD3_PREFILL_MAX_SAMPLES" \
    sbatch_td3_mpc_clone_pipeline.sh
)

echo "JOB_ID=$JOB_ID"
echo "PIPE_TAG=$PIPE_TAG"
echo "Track:"
echo "  squeue -j $JOB_ID"
echo "Logs:"
echo "  logs/csci1470_td3_mpcclone_${JOB_ID}.out"
echo "  logs/csci1470_td3_mpcclone_${JOB_ID}.err"
