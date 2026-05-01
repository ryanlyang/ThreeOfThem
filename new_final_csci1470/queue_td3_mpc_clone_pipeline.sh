#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
mkdir -p logs

PIPE_TAG="${PIPE_TAG:-td3_mpcclone_$(date +%Y%m%d_%H%M%S)}"
PARTITION="${PARTITION:-tier3}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_GB="${MEM_GB:-40G}"

echo "Submitting TD3 MPC-clone pipeline"
echo "PIPE_TAG=$PIPE_TAG"
echo "PARTITION=$PARTITION TIME=$TIME_LIMIT CPU=$CPUS_PER_TASK MEM=$MEM_GB"

JOB_ID=$(
  sbatch --parsable \
    --partition="$PARTITION" \
    --time="$TIME_LIMIT" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM_GB" \
    --export=ALL,PIPE_TAG="$PIPE_TAG" \
    sbatch_td3_mpc_clone_pipeline.sh
)

echo "JOB_ID=$JOB_ID"
echo "PIPE_TAG=$PIPE_TAG"
echo "Track:"
echo "  squeue -j $JOB_ID"
echo "Logs:"
echo "  logs/csci1470_td3_mpcclone_${JOB_ID}.out"
echo "  logs/csci1470_td3_mpcclone_${JOB_ID}.err"
