#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
mkdir -p logs

PIPE_TAG="${PIPE_TAG:-fixedinit_directed_long_$(date +%Y%m%d_%H%M%S)}"

echo "Submitting directed fixed-init LONG training job with PIPE_TAG=$PIPE_TAG"
TRAIN_JOB_ID=$(sbatch --parsable --export=ALL,PIPE_TAG="$PIPE_TAG" sbatch_train_fixed_init_directed_long_72h.sh)

echo "Submitting dependent eval+inference job afterok:$TRAIN_JOB_ID"
EVAL_JOB_ID=$(sbatch --parsable --dependency=afterok:"$TRAIN_JOB_ID" --export=ALL,PIPE_TAG="$PIPE_TAG" sbatch_eval_fixed_init_directed_after_long.sh)

echo "TRAIN_JOB_ID=$TRAIN_JOB_ID"
echo "EVAL_JOB_ID=$EVAL_JOB_ID"
echo "PIPE_TAG=$PIPE_TAG"

echo "Track jobs:"
echo "  squeue -u \"$USER\""
echo "Logs:"
echo "  logs/csci1470_fixedinit_directed_long_${TRAIN_JOB_ID}.out"
echo "  logs/csci1470_fixedinit_directed_long_${TRAIN_JOB_ID}.err"
echo "  logs/csci1470_fixedinit_directed_eval_${EVAL_JOB_ID}.out"
echo "  logs/csci1470_fixedinit_directed_eval_${EVAL_JOB_ID}.err"
echo "Pipeline metadata dir: artifacts/pipelines/$PIPE_TAG"
