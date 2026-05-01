#!/usr/bin/env bash
#SBATCH --job-name=csci1470_mpc_olws
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
fi
mkdir -p logs

PIPE_TAG="${PIPE_TAG:-mpc_olws_$(date +%Y%m%d_%H%M%S)}"
PIPE_DIR="$PWD/artifacts/pipelines/$PIPE_TAG"
mkdir -p "$PIPE_DIR"

echo "=== MPC OPEN-LOOP + WARM-START RUN ==="
echo "PIPE_TAG=$PIPE_TAG"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "start_time=$(date)"

echo "${SLURM_JOB_ID:-none}" > "$PIPE_DIR/mpc_olws_job_id.txt"
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

OUTDIR="${OUTDIR:-$PWD/inference_mpc_ilqr_openloop_warmstart}"
mkdir -p "$OUTDIR"

python run_mpc_ilqr_openloop_warmstart_compare.py \
  --seed "${SEED:-4301}" \
  --max-steps "${MAX_STEPS:-420}" \
  --log-every "${LOG_EVERY:-20}" \
  --fixed-init-profile "${FIXED_INIT_PROFILE:-near_ref}" \
  --fixed-init-positions "${FIXED_INIT_POSITIONS:-}" \
  --fixed-init-velocities "${FIXED_INIT_VELOCITIES:-}" \
  --jitter-setups "${JITTER_SETUPS:-10}" \
  --jitter-pos-std "${JITTER_POS_STD:-0.008}" \
  --jitter-vel-std "${JITTER_VEL_STD:-0.006}" \
  --jitter-tries "${JITTER_TRIES:-64}" \
  --horizon-steps "${HORIZON_STEPS:-420}" \
  --max-action-norm "${MAX_ACTION_NORM:-2.5}" \
  --near-collision-distance "${NEAR_COLLISION_DISTANCE:-0.35}" \
  --escape-radius "${ESCAPE_RADIUS:-8.0}" \
  --phase-search-radius "${PHASE_SEARCH_RADIUS:-35}" \
  --action-dt "${ACTION_DT:-0.05}" \
  --integrator-dt "${INTEGRATOR_DT:-0.001}" \
  --w-pos-match "${W_POS_MATCH:-5.0}" \
  --w-vel-match "${W_VEL_MATCH:-3.0}" \
  --w-switch-match "${W_SWITCH_MATCH:-0.10}" \
  --w-phase-match "${W_PHASE_MATCH:-0.20}" \
  --mpc-horizon "${MPC_HORIZON:-42}" \
  --mpc-iters "${MPC_ITERS:-14}" \
  --mpc-model-substeps "${MPC_MODEL_SUBSTEPS:-4}" \
  --mpc-fd-eps-state "${MPC_FD_EPS_STATE:-1e-4}" \
  --mpc-fd-eps-action "${MPC_FD_EPS_ACTION:-1e-3}" \
  --mpc-q-pos "${MPC_Q_POS:-70.0}" \
  --mpc-q-vel "${MPC_Q_VEL:-24.0}" \
  --mpc-r-action "${MPC_R_ACTION:-0.000001}" \
  --mpc-terminal-scale "${MPC_TERMINAL_SCALE:-60.0}" \
  --mpc-near-collision-weight "${MPC_NEAR_COLLISION_WEIGHT:-120.0}" \
  --mpc-near-collision-distance "${MPC_NEAR_COLLISION_DISTANCE:-0.32}" \
  --replan-every "${REPLAN_EVERY:-1}" \
  --openloop-cem-iters "${OPENLOOP_CEM_ITERS:-12}" \
  --openloop-cem-population "${OPENLOOP_CEM_POPULATION:-128}" \
  --openloop-cem-elite-frac "${OPENLOOP_CEM_ELITE_FRAC:-0.12}" \
  --openloop-cem-init-std-frac "${OPENLOOP_CEM_INIT_STD_FRAC:-0.55}" \
  --openloop-cem-min-std-frac "${OPENLOOP_CEM_MIN_STD_FRAC:-0.05}" \
  --openloop-cem-smoothing "${OPENLOOP_CEM_SMOOTHING:-0.25}" \
  --openloop-exec-steps "${OPENLOOP_EXEC_STEPS:-40}" \
  --pos-threshold "${POS_THRESHOLD:-0.05}" \
  --vel-threshold "${VEL_THRESHOLD:-0.08}" \
  --consecutive-converged "${CONSECUTIVE_CONVERGED:-320}" \
  --min-total-steps-for-converged "${MIN_TOTAL_STEPS_FOR_CONVERGED:-400}" \
  --trail-len "${TRAIL_LEN:-120}" \
  --frame-stride "${FRAME_STRIDE:-1}" \
  --axis-pad "${AXIS_PAD:-0.25}" \
  --outdir "$OUTDIR"

echo "$OUTDIR" > "$PIPE_DIR/mpc_olws_outdir.txt"
echo "end_time=$(date)"
