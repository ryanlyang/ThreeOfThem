#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
mkdir -p logs

# Distinct tag so alt profile artifacts are separated.
PIPE_TAG="${PIPE_TAG:-altw_$(date +%Y%m%d_%H%M%S)}"

# Alternate profile: stronger choreography pressure, lower fuel penalty,
# slightly higher collision penalty + slightly more exploration.
UPDATES="${UPDATES:-2200}"
NUM_ENVS="${NUM_ENVS:-8}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-128}"
PPO_EPOCHS="${PPO_EPOCHS:-8}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-256}"
EVAL_EVERY="${EVAL_EVERY:-10}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-17}"

LR="${LR:-2.5e-4}"
GAMMA="${GAMMA:-0.997}"
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
CLIP_COEF="${CLIP_COEF:-0.2}"
VF_COEF="${VF_COEF:-0.5}"
ENT_COEF="${ENT_COEF:-0.007}"

W_POS="${W_POS:-1.35}"
W_VEL="${W_VEL:-0.55}"
W_FUEL="${W_FUEL:-0.015}"
W_COLLISION="${W_COLLISION:-80.0}"
W_ESCAPE="${W_ESCAPE:-2.5}"
W_SWITCH="${W_SWITCH:-0.20}"
W_PHASE="${W_PHASE:-0.015}"

ESCAPE_RADIUS="${ESCAPE_RADIUS:-4.0}"
INIT_MIN_PAIR_DISTANCE="${INIT_MIN_PAIR_DISTANCE:-0.25}"

echo "Submitting ALT training job with PIPE_TAG=$PIPE_TAG"
TRAIN_JOB_ID=$(sbatch --parsable \
  --export=ALL,PIPE_TAG="$PIPE_TAG",UPDATES="$UPDATES",NUM_ENVS="$NUM_ENVS",ROLLOUT_STEPS="$ROLLOUT_STEPS",PPO_EPOCHS="$PPO_EPOCHS",MINIBATCH_SIZE="$MINIBATCH_SIZE",EVAL_EVERY="$EVAL_EVERY",EVAL_EPISODES="$EVAL_EPISODES",SEED="$SEED",LR="$LR",GAMMA="$GAMMA",GAE_LAMBDA="$GAE_LAMBDA",CLIP_COEF="$CLIP_COEF",VF_COEF="$VF_COEF",ENT_COEF="$ENT_COEF",W_POS="$W_POS",W_VEL="$W_VEL",W_FUEL="$W_FUEL",W_COLLISION="$W_COLLISION",W_ESCAPE="$W_ESCAPE",W_SWITCH="$W_SWITCH",W_PHASE="$W_PHASE",ESCAPE_RADIUS="$ESCAPE_RADIUS",INIT_MIN_PAIR_DISTANCE="$INIT_MIN_PAIR_DISTANCE" \
  sbatch_train_ppo_long_23h.sh)

# Same dependent inference stage.
echo "Submitting dependent inference job afterok:$TRAIN_JOB_ID"
INFER_JOB_ID=$(sbatch --parsable --dependency=afterok:"$TRAIN_JOB_ID" --export=ALL,PIPE_TAG="$PIPE_TAG" sbatch_infer_best_gif_after_train.sh)

echo "TRAIN_JOB_ID=$TRAIN_JOB_ID"
echo "INFER_JOB_ID=$INFER_JOB_ID"
echo "PIPE_TAG=$PIPE_TAG"

echo "ALT profile values:"
echo "  LR=$LR GAMMA=$GAMMA GAE_LAMBDA=$GAE_LAMBDA ENT_COEF=$ENT_COEF"
echo "  W_POS=$W_POS W_VEL=$W_VEL W_FUEL=$W_FUEL W_COLLISION=$W_COLLISION W_ESCAPE=$W_ESCAPE W_SWITCH=$W_SWITCH W_PHASE=$W_PHASE"
echo "  ESCAPE_RADIUS=$ESCAPE_RADIUS INIT_MIN_PAIR_DISTANCE=$INIT_MIN_PAIR_DISTANCE"

echo "Track jobs:"
echo "  squeue -u \"$USER\""
echo "Logs:"
echo "  logs/csci1470_train_ppo_long_${TRAIN_JOB_ID}.out"
echo "  logs/csci1470_train_ppo_long_${TRAIN_JOB_ID}.err"
echo "  logs/csci1470_infer_best_gif_${INFER_JOB_ID}.out"
echo "  logs/csci1470_infer_best_gif_${INFER_JOB_ID}.err"

echo "Pipeline metadata dir: artifacts/pipelines/$PIPE_TAG"
