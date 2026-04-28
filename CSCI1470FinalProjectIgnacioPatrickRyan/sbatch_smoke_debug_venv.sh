#!/usr/bin/env bash
#SBATCH --job-name=csci1470_smoke_venv
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "=== SLURM JOB INFO ==="
echo "job_id=$SLURM_JOB_ID"
echo "job_name=$SLURM_JOB_NAME"
echo "node_list=$SLURM_JOB_NODELIST"
echo "submit_dir=${SLURM_SUBMIT_DIR:-unknown}"
echo "start_time=$(date)"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
fi
pwd
mkdir -p logs

echo "=== CREATE/ACTIVATE VENV ==="
VENV_DIR="${VENV_DIR:-$PWD/.venv_csci1470_smoke}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PWD/.pip_cache}"
export PIP_CACHE_DIR

bash ./bootstrap_venv.sh "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "=== SYSTEM DIAGNOSTICS ==="
whoami
hostname
uname -a

echo "=== GPU DIAGNOSTICS ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found"
fi

echo "=== PYTHON DIAGNOSTICS ==="
which python
python --version

python - <<'PY'
import importlib
mods = ["numpy", "matplotlib", "PIL", "torch", "amuse", "gym", "gymnasium"]
print("Import check:")
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  {m}: OK")
    except Exception as e:
        print(f"  {m}: MISSING ({e.__class__.__name__})")
PY

echo "=== SYNTAX CHECK ==="
python -m py_compile \
  config.py \
  reference_orbit.py \
  simulator.py \
  choreography_env.py \
  run_smoke_test.py \
  plot_reference_figure8.py \
  animate_reference_figure8.py

echo "=== PLOT REFERENCE FIGURE-8 ==="
MPLCONFIGDIR=/tmp/mpl_cfg python plot_reference_figure8.py

echo "=== ANIMATE REFERENCE FIGURE-8 ==="
MPLCONFIGDIR=/tmp/mpl_cfg python animate_reference_figure8.py

echo "=== RUN SMOKE TEST ==="
python run_smoke_test.py

echo "=== DONE ==="
echo "end_time=$(date)"
