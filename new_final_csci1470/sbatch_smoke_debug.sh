#!/usr/bin/env bash
#SBATCH --job-name=csci1470_smoke_debug
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "=== SLURM JOB INFO ==="
echo "job_id=$SLURM_JOB_ID"
echo "job_name=$SLURM_JOB_NAME"
echo "node_list=$SLURM_JOB_NODELIST"
echo "submit_dir=${SLURM_SUBMIT_DIR:-unknown}"
echo "start_time=$(date)"

echo "\n=== PATH SETUP ==="
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
fi
pwd

mkdir -p logs

# Optional environment activation hooks.
# Use by exporting one of these before sbatch:
#   export CONDA_ENV_NAME=myenv
#   export VENV_PATH=/path/to/venv
if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  echo "Activating conda env: $CONDA_ENV_NAME"
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
  else
    echo "ERROR: CONDA_ENV_NAME set but 'conda' not found"
    exit 2
  fi
elif [[ -n "${VENV_PATH:-}" ]]; then
  echo "Activating venv: $VENV_PATH"
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

echo "\n=== SYSTEM DIAGNOSTICS ==="
whoami
hostname
uname -a

echo "\n=== GPU DIAGNOSTICS ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found"
fi

echo "\n=== PYTHON DIAGNOSTICS ==="
which python3 || true
python3 --version

python3 - <<'PY'
import importlib
mods = ["numpy", "torch", "amuse", "gym", "gymnasium"]
print("Import check:")
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  {m}: OK")
    except Exception as e:
        print(f"  {m}: MISSING ({e.__class__.__name__})")
PY

echo "\n=== SYNTAX CHECK ==="
PY_FILES=(config.py reference_orbit.py simulator.py choreography_env.py run_smoke_test.py)
if [[ -f plot_reference_figure8.py ]]; then
  PY_FILES+=(plot_reference_figure8.py)
fi
python3 -m py_compile "${PY_FILES[@]}"

if [[ -f plot_reference_figure8.py ]]; then
  echo "\n=== PLOT REFERENCE FIGURE-8 ==="
  MPLCONFIGDIR=/tmp/mpl_cfg python3 plot_reference_figure8.py
fi

echo "\n=== RUN SMOKE TEST ==="
python3 run_smoke_test.py

echo "\n=== DONE ==="
echo "end_time=$(date)"
