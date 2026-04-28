#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
mkdir -p logs

PIPE_TAG="${PIPE_TAG:-pipe_$(date +%Y%m%d_%H%M%S)}"

echo "Submitting training job with PIPE_TAG=$PIPE_TAG"
TRAIN_JOB_ID=$(sbatch --parsable --export=ALL,PIPE_TAG="$PIPE_TAG" sbatch_train_ppo_long_23h.sh)

# Inference runs only if training exits successfully.
echo "Submitting dependent inference job afterok:$TRAIN_JOB_ID"
INFER_JOB_ID=$(sbatch --parsable --dependency=afterok:"$TRAIN_JOB_ID" --export=ALL,PIPE_TAG="$PIPE_TAG" sbatch_infer_best_gif_after_train.sh)

echo "TRAIN_JOB_ID=$TRAIN_JOB_ID"
echo "INFER_JOB_ID=$INFER_JOB_ID"
echo "PIPE_TAG=$PIPE_TAG"

echo "Track jobs:"
echo "  squeue -u \"$USER\""
echo "Logs:"
echo "  logs/csci1470_train_ppo_long_${TRAIN_JOB_ID}.out"
echo "  logs/csci1470_train_ppo_long_${TRAIN_JOB_ID}.err"
echo "  logs/csci1470_infer_best_gif_${INFER_JOB_ID}.out"
echo "  logs/csci1470_infer_best_gif_${INFER_JOB_ID}.err"

echo "Pipeline metadata dir: artifacts/pipelines/$PIPE_TAG"
