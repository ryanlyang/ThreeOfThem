#!/usr/bin/env bash
#SBATCH --job-name=csci1470_mpc_ilqr
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
fi
mkdir -p logs

PIPE_TAG="${PIPE_TAG:-mpc_ilqr_$(date +%Y%m%d_%H%M%S)}"
PIPE_DIR="$PWD/artifacts/pipelines/$PIPE_TAG"
mkdir -p "$PIPE_DIR"

echo "=== MPC/iLQR FIXED-INIT RUN ==="
echo "PIPE_TAG=$PIPE_TAG"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "start_time=$(date)"

echo "${SLURM_JOB_ID:-none}" > "$PIPE_DIR/mpc_job_id.txt"
echo "$PIPE_TAG" > "$PIPE_DIR/pipe_tag.txt"

VENV_DIR="${VENV_DIR:-$PWD/.venv_csci1470_smoke}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PWD/.pip_cache}"
export PIP_CACHE_DIR

bash ./bootstrap_venv.sh "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --cache-dir "$PIP_CACHE_DIR" -r requirements-smoke.txt

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

OUTDIR="${OUTDIR:-$PWD/inference_mpc_ilqr_fixed_init}"
mkdir -p "$OUTDIR"

SHOW_TITLE="${SHOW_TITLE:-1}"
TITLE_FLAG="--show-title"
if [[ "$SHOW_TITLE" == "0" ]]; then
  TITLE_FLAG="--no-show-title"
fi

python run_mpc_ilqr_fixed_init.py \
  --num-setups "${NUM_SETUPS:-10}" \
  --seed "${SEED:-4301}" \
  --max-steps "${MAX_STEPS:-420}" \
  --log-every "${LOG_EVERY:-20}" \
  --fixed-init-profile "${FIXED_INIT_PROFILE:-offset_ref}" \
  --fixed-init-positions "${FIXED_INIT_POSITIONS:-}" \
  --fixed-init-velocities "${FIXED_INIT_VELOCITIES:-}" \
  --fixed-init-pos-jitter-std "${FIXED_INIT_POS_JITTER_STD:-0.0}" \
  --fixed-init-vel-jitter-std "${FIXED_INIT_VEL_JITTER_STD:-0.0}" \
  --fixed-init-jitter-tries "${FIXED_INIT_JITTER_TRIES:-1}" \
  --horizon-steps "${HORIZON_STEPS:-420}" \
  --max-action-norm "${MAX_ACTION_NORM:-2.0}" \
  --near-collision-distance "${NEAR_COLLISION_DISTANCE:-0.35}" \
  --escape-radius "${ESCAPE_RADIUS:-8.0}" \
  --phase-search-radius "${PHASE_SEARCH_RADIUS:-35}" \
  --action-dt "${ACTION_DT:-0.05}" \
  --integrator-dt "${INTEGRATOR_DT:-0.001}" \
  --w-pos-match "${W_POS_MATCH:-4.0}" \
  --w-vel-match "${W_VEL_MATCH:-2.5}" \
  --w-switch-match "${W_SWITCH_MATCH:-0.2}" \
  --w-phase-match "${W_PHASE_MATCH:-0.12}" \
  --mpc-horizon "${MPC_HORIZON:-42}" \
  --mpc-iters "${MPC_ITERS:-14}" \
  --mpc-model-substeps "${MPC_MODEL_SUBSTEPS:-4}" \
  --mpc-fd-eps-state "${MPC_FD_EPS_STATE:-1e-4}" \
  --mpc-fd-eps-action "${MPC_FD_EPS_ACTION:-1e-3}" \
  --mpc-q-pos "${MPC_Q_POS:-50.0}" \
  --mpc-q-vel "${MPC_Q_VEL:-16.0}" \
  --mpc-r-action "${MPC_R_ACTION:-0.0002}" \
  --mpc-terminal-scale "${MPC_TERMINAL_SCALE:-40.0}" \
  --mpc-near-collision-weight "${MPC_NEAR_COLLISION_WEIGHT:-40.0}" \
  --mpc-near-collision-distance "${MPC_NEAR_COLLISION_DISTANCE:-0.28}" \
  --replan-every "${REPLAN_EVERY:-1}" \
  --pos-threshold "${POS_THRESHOLD:-0.06}" \
  --vel-threshold "${VEL_THRESHOLD:-0.09}" \
  --consecutive-converged "${CONSECUTIVE_CONVERGED:-260}" \
  --min-total-steps-for-converged "${MIN_TOTAL_STEPS_FOR_CONVERGED:-320}" \
  --trail-len "${TRAIL_LEN:-90}" \
  --frame-stride "${FRAME_STRIDE:-1}" \
  --axis-pad "${AXIS_PAD:-0.20}" \
  "$TITLE_FLAG" \
  --outdir "$OUTDIR"

echo "$OUTDIR" > "$PIPE_DIR/mpc_outdir.txt"
echo "end_time=$(date)"
