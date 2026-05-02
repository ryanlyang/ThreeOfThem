#!/usr/bin/env bash
#SBATCH --job-name=csci1470_td3_mpcclone
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
fi
mkdir -p logs

PIPE_TAG="${PIPE_TAG:-td3_mpcclone_$(date +%Y%m%d_%H%M%S)}"
PIPE_DIR="$PWD/artifacts/pipelines/$PIPE_TAG"
mkdir -p "$PIPE_DIR"

echo "=== TD3 MPC-Clone Pipeline ==="
echo "PIPE_TAG=$PIPE_TAG"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "start_time=$(date)"

echo "${SLURM_JOB_ID:-none}" > "$PIPE_DIR/pipeline_job_id.txt"
echo "$PIPE_TAG" > "$PIPE_DIR/pipe_tag.txt"

VENV_DIR="${VENV_DIR:-$PWD/.venv_csci1470_smoke}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PWD/.pip_cache}"
export PIP_CACHE_DIR

bash ./bootstrap_venv.sh "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --cache-dir "$PIP_CACHE_DIR" -r requirements-train.txt

which python
python --version
nvidia-smi || true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mpl_cache}"
mkdir -p "$MPLCONFIGDIR"

SEED="${SEED:-4301}"
FIXED_INIT_PROFILE="${FIXED_INIT_PROFILE:-offset_ref_far}"
TRAIN_JITTER_POS="${TRAIN_JITTER_POS:-0.006}"
TRAIN_JITTER_VEL="${TRAIN_JITTER_VEL:-0.004}"
TRAIN_JITTER_TRIES="${TRAIN_JITTER_TRIES:-64}"

MPC_DATASET_SETUPS="${MPC_DATASET_SETUPS:-96}"
MPC_MAX_STEPS="${MPC_MAX_STEPS:-500}"
MPC_REPLAN_EVERY="${MPC_REPLAN_EVERY:-1}"
MPC_HORIZON="${MPC_HORIZON:-60}"
MPC_ITERS="${MPC_ITERS:-20}"
MPC_MODEL_SUBSTEPS="${MPC_MODEL_SUBSTEPS:-4}"
MPC_Q_POS="${MPC_Q_POS:-80.0}"
MPC_Q_VEL="${MPC_Q_VEL:-24.0}"
MPC_R_ACTION="${MPC_R_ACTION:-0.00005}"
MPC_TERMINAL_SCALE="${MPC_TERMINAL_SCALE:-60.0}"
MPC_NEAR_COLLISION_WEIGHT="${MPC_NEAR_COLLISION_WEIGHT:-100.0}"
MPC_NEAR_COLLISION_DISTANCE="${MPC_NEAR_COLLISION_DISTANCE:-0.35}"

TD3_W_POS="${TD3_W_POS:-20.0}"
TD3_W_VEL="${TD3_W_VEL:-12.0}"
TD3_W_FUEL="${TD3_W_FUEL:-0.0005}"
TD3_W_NEAR_COLLISION="${TD3_W_NEAR_COLLISION:-0.0}"
TD3_W_COLLISION="${TD3_W_COLLISION:-500.0}"
TD3_W_ESCAPE="${TD3_W_ESCAPE:-0.0}"
TD3_W_SWITCH="${TD3_W_SWITCH:-0.08}"
TD3_W_PHASE="${TD3_W_PHASE:-0.20}"
TD3_EXPL_NOISE_START="${TD3_EXPL_NOISE_START:-0.10}"
TD3_EXPL_NOISE_END="${TD3_EXPL_NOISE_END:-0.02}"
TD3_EXPL_DECAY_STEPS="${TD3_EXPL_DECAY_STEPS:-300000}"

echo "=== STAGE 1/5: Collect MPC dataset ==="
python collect_mpc_ilqr_dataset.py \
  --num-setups "$MPC_DATASET_SETUPS" \
  --seed "$SEED" \
  --max-steps "$MPC_MAX_STEPS" \
  --log-every 25 \
  --replan-every "$MPC_REPLAN_EVERY" \
  --fixed-init-profile "$FIXED_INIT_PROFILE" \
  --fixed-init-pos-jitter-std "$TRAIN_JITTER_POS" \
  --fixed-init-vel-jitter-std "$TRAIN_JITTER_VEL" \
  --fixed-init-jitter-tries "$TRAIN_JITTER_TRIES" \
  --horizon-steps 500 \
  --max-action-norm 2.5 \
  --near-collision-distance 0.35 \
  --escape-radius 8.0 \
  --phase-search-radius 35 \
  --w-pos-match 5.0 \
  --w-vel-match 3.0 \
  --w-switch-match 0.20 \
  --w-phase-match 0.15 \
  --mpc-horizon "$MPC_HORIZON" \
  --mpc-iters "$MPC_ITERS" \
  --mpc-model-substeps "$MPC_MODEL_SUBSTEPS" \
  --mpc-q-pos "$MPC_Q_POS" \
  --mpc-q-vel "$MPC_Q_VEL" \
  --mpc-r-action "$MPC_R_ACTION" \
  --mpc-terminal-scale "$MPC_TERMINAL_SCALE" \
  --mpc-near-collision-weight "$MPC_NEAR_COLLISION_WEIGHT" \
  --mpc-near-collision-distance "$MPC_NEAR_COLLISION_DISTANCE" \
  --pos-threshold 0.06 \
  --vel-threshold 0.09 \
  --consecutive-converged 300 \
  --min-total-steps-for-converged 400 \
  --strict-mode \
  --lock-to-end \
  --require-no-failure \
  --require-final-threshold \
  --keep-only-strict \
  --outdir artifacts/mpc_datasets \
  --tag "mpc_expert_${PIPE_TAG}"

DATASET_DIR="$(ls -dt "$PWD"/artifacts/mpc_datasets/mpc_expert_${PIPE_TAG}_* | head -n1)"
DATASET_NPZ="$DATASET_DIR/dataset.npz"
DATASET_META="$DATASET_DIR/dataset_meta.json"
echo "$DATASET_DIR" > "$PIPE_DIR/mpc_dataset_dir.txt"
echo "$DATASET_NPZ" > "$PIPE_DIR/mpc_dataset_npz.txt"
echo "$DATASET_META" > "$PIPE_DIR/mpc_dataset_meta.txt"
echo "DATASET_DIR=$DATASET_DIR"

echo "=== STAGE 2/5: Behavior cloning pretrain ==="
python pretrain_td3_actor_bc.py \
  --dataset "$DATASET_NPZ" \
  --dataset-meta "$DATASET_META" \
  --outdir artifacts/bc_td3 \
  --run-name "td3_bc_${PIPE_TAG}" \
  --hidden-size 256 \
  --batch-size 1024 \
  --epochs 150 \
  --learning-rate 3e-4 \
  --val-frac 0.1 \
  --early-stop-patience 25 \
  --seed "$SEED" \
  --device auto \
  --obs-clip 10.0 \
  --eval-episodes 12 \
  --eval-max-steps 500 \
  --eval-pos-threshold 0.06 \
  --eval-vel-threshold 0.09 \
  --eval-consecutive-converged 260 \
  --eval-min-total-steps 320 \
  --eval-strict-mode \
  --eval-lock-to-end

BC_RUN_DIR="$(ls -dt "$PWD"/artifacts/bc_td3/td3_bc_${PIPE_TAG}_* | head -n1)"
BC_CKPT="$BC_RUN_DIR/checkpoint_bc.pt"
echo "$BC_RUN_DIR" > "$PIPE_DIR/bc_run_dir.txt"
echo "$BC_CKPT" > "$PIPE_DIR/bc_checkpoint.txt"
echo "BC_RUN_DIR=$BC_RUN_DIR"

echo "=== STAGE 3/5: BC inference gifs ==="
python run_inference_td3_best_gif.py \
  --checkpoint "$BC_CKPT" \
  --num-setups 10 \
  --max-steps 500 \
  --seed "$SEED" \
  --fixed-init-pos-jitter-std "$TRAIN_JITTER_POS" \
  --fixed-init-vel-jitter-std "$TRAIN_JITTER_VEL" \
  --fixed-init-jitter-tries "$TRAIN_JITTER_TRIES" \
  --strict-mode \
  --lock-to-end \
  --require-no-failure \
  --require-final-threshold \
  --pos-threshold 0.06 \
  --vel-threshold 0.09 \
  --consecutive-converged 260 \
  --min-total-steps-for-converged 320 \
  --save-nohelp-baseline \
  --outdir "$PWD/inference_td3_bc_${PIPE_TAG}"

echo "$PWD/inference_td3_bc_${PIPE_TAG}" > "$PIPE_DIR/bc_inference_dir.txt"

echo "=== STAGE 4/5: TD3 warm-start from BC + replay prefill ==="
python train_td3_figure8.py \
  --total-env-steps "${TD3_TOTAL_ENV_STEPS:-500000}" \
  --num-envs "${TD3_NUM_ENVS:-8}" \
  --vec-env subproc \
  --mp-start-method spawn \
  --buffer-size "${TD3_BUFFER_SIZE:-1000000}" \
  --batch-size "${TD3_BATCH_SIZE:-512}" \
  --learning-starts "${TD3_LEARNING_STARTS:-0}" \
  --updates-per-iter "${TD3_UPDATES_PER_ITER:-2}" \
  --gamma 0.995 \
  --tau 0.005 \
  --actor-lr 3e-4 \
  --critic-lr 3e-4 \
  --policy-delay 2 \
  --target-policy-noise 0.20 \
  --target-noise-clip 0.50 \
  --hidden-size 256 \
  --init-actor-checkpoint "$BC_CKPT" \
  --init-obs-rms-from-checkpoint \
  --prefill-replay-dataset "$DATASET_NPZ" \
  --prefill-replay-max-samples "${TD3_PREFILL_MAX_SAMPLES:-300000}" \
  --exploration-noise "$TD3_EXPL_NOISE_START" \
  --exploration-noise-final "$TD3_EXPL_NOISE_END" \
  --exploration-decay-steps "$TD3_EXPL_DECAY_STEPS" \
  --reward-scale 800.0 \
  --reward-clip 20.0 \
  --obs-clip 10.0 \
  --horizon-steps 500 \
  --action-dt 0.05 \
  --integrator-dt 0.001 \
  --phase-search-radius 35 \
  --max-action-norm 2.5 \
  --near-collision-distance 0.35 \
  --escape-radius 8.0 \
  --init-min-pair-distance 0.25 \
  --backend numpy \
  --fixed-init-profile "$FIXED_INIT_PROFILE" \
  --fixed-init-pos-jitter-std "$TRAIN_JITTER_POS" \
  --fixed-init-vel-jitter-std "$TRAIN_JITTER_VEL" \
  --fixed-init-jitter-tries "$TRAIN_JITTER_TRIES" \
  --eval-fixed-init-pos-jitter-std "$TRAIN_JITTER_POS" \
  --eval-fixed-init-vel-jitter-std "$TRAIN_JITTER_VEL" \
  --eval-fixed-init-jitter-tries "$TRAIN_JITTER_TRIES" \
  --w-pos "$TD3_W_POS" \
  --w-vel "$TD3_W_VEL" \
  --w-fuel "$TD3_W_FUEL" \
  --w-near-collision "$TD3_W_NEAR_COLLISION" \
  --w-collision "$TD3_W_COLLISION" \
  --w-escape "$TD3_W_ESCAPE" \
  --w-switch "$TD3_W_SWITCH" \
  --w-phase "$TD3_W_PHASE" \
  --eval-every-env-steps 25000 \
  --eval-episodes 10 \
  --eval-strict-mode \
  --eval-lock-to-end \
  --eval-pos-threshold 0.06 \
  --eval-vel-threshold 0.09 \
  --eval-consecutive-converged 260 \
  --eval-min-total-steps 320 \
  --save-topk 3 \
  --early-stop-on-strict-success \
  --early-stop-success-rate 1.0 \
  --early-stop-max-failure-rate 0.0 \
  --early-stop-patience-evals 1 \
  --early-stop-min-evals 1 \
  --log-every-env-steps 5000 \
  --seed "$SEED" \
  --device auto \
  --run-name "td3_mpcclone_${PIPE_TAG}" \
  --save-dir artifacts

TD3_RUN_DIR="$(ls -dt "$PWD"/artifacts/td3_mpcclone_${PIPE_TAG}_* | head -n1)"
TD3_BEST_CKPT="$TD3_RUN_DIR/checkpoint_best.pt"
echo "$TD3_RUN_DIR" > "$PIPE_DIR/td3_run_dir.txt"
echo "$TD3_BEST_CKPT" > "$PIPE_DIR/td3_best_checkpoint.txt"
echo "TD3_RUN_DIR=$TD3_RUN_DIR"

echo "=== STAGE 5/5: Final eval + gifs ==="
python evaluate_td3_fixed_init.py \
  --checkpoint "$TD3_BEST_CKPT" \
  --eval-episodes 12 \
  --num-setups 10 \
  --max-steps 500 \
  --seed "$SEED" \
  --fixed-init-pos-jitter-std "$TRAIN_JITTER_POS" \
  --fixed-init-vel-jitter-std "$TRAIN_JITTER_VEL" \
  --fixed-init-jitter-tries "$TRAIN_JITTER_TRIES" \
  --strict-mode \
  --lock-to-end \
  --require-no-failure \
  --require-final-threshold \
  --pos-threshold 0.06 \
  --vel-threshold 0.09 \
  --consecutive-converged 260 \
  --min-total-steps-for-converged 320 \
  --save-nohelp-baseline \
  --outdir "$PWD/inference_td3_mpcclone_${PIPE_TAG}"

echo "$PWD/inference_td3_mpcclone_${PIPE_TAG}" > "$PIPE_DIR/final_inference_dir.txt"
echo "end_time=$(date)"
echo "PIPE_DIR=$PIPE_DIR"
