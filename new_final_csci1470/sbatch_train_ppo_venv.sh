#!/usr/bin/env bash
#SBATCH --job-name=csci1470_train_ppo
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
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

# Training dependencies (adds torch on top of smoke deps).
python -m pip install --cache-dir "$PIP_CACHE_DIR" -r requirements-train.txt

echo "=== PYTHON/GPU DIAGNOSTICS ==="
which python
python --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

python - <<'PY'
import importlib
mods = ["numpy", "matplotlib", "PIL", "torch"]
print("Import check:")
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  {m}: OK")
    except Exception as e:
        print(f"  {m}: MISSING ({e.__class__.__name__})")
PY

echo "=== TRAIN PPO ==="
UPDATES="${UPDATES:-120}"
NUM_ENVS="${NUM_ENVS:-8}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-128}"
PPO_EPOCHS="${PPO_EPOCHS:-8}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-256}"
EVAL_EVERY="${EVAL_EVERY:-6}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
SEED="${SEED:-7}"

python train_ppo_figure8.py \
  --updates "$UPDATES" \
  --num-envs "$NUM_ENVS" \
  --rollout-steps "$ROLLOUT_STEPS" \
  --ppo-epochs "$PPO_EPOCHS" \
  --minibatch-size "$MINIBATCH_SIZE" \
  --eval-every "$EVAL_EVERY" \
  --eval-episodes "$EVAL_EPISODES" \
  --seed "$SEED" \
  --device auto \
  --run-name ppo_figure8

echo "=== DONE ==="
echo "end_time=$(date)"
