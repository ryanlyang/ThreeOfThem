# Figure-8 Choreography Environment (CSCI1470)

This project implements the setup we discussed, from scratch, in a clean standalone package.

## What Is Implemented

1. One canonical equal-mass figure-8 reference path is precomputed.
2. A global phase index `k` is matched each step.
3. Three choreography slots are defined at `k`, `k+N/3`, `k+2N/3`.
4. Body-to-slot assignment uses full permutation matching (`3! = 6` possibilities).
5. Continuity is enforced with:
   - permutation switch penalty
   - phase jump penalty around expected forward phase stride
6. Reward combines:
   - position error to choreography targets
   - velocity direction mismatch vs figure-8 tangent
   - fuel usage penalty
   - collision penalty
7. Two simulator backends:
   - `numpy` (tested here)
   - `amuse` (style-compatible with the original `ThreeBodyProblem_astronomy` simulator, requires AMUSE installed)

## File Map

- `config.py`: environment and reward config dataclasses.
- `reference_orbit.py`: canonical figure-8 precompute + orientation convention.
- `simulator.py`: simulator backends (`NumpyThreeBodySimulator`, `AmuseThreeBodySimulator`).
- `choreography_env.py`: full phase/permutation matching and reward logic.
- `run_smoke_test.py`: quick run to verify the environment loop.
- `train_ppo_figure8.py`: PPO training with randomized starts, checkpointing, and periodic eval.
- `evaluate_trained_policy.py`: evaluate a saved checkpoint vs optional baselines.
- `sbatch_train_ppo_venv.sh`: SLURM training entrypoint (auto venv + deps).
- `sbatch_train_ppo_long_23h.sh`: long-run PPO training job (`23:00:00` on debug).
- `sbatch_infer_best_gif_after_train.sh`: dependent inference job that picks fastest convergence and saves GIF.
- `queue_train_then_infer.sh`: submits both jobs with dependency (`afterok`).
- `queue_train_then_infer_alt_hparams.sh`: same dependency pipeline with alternate reward/PPO hyperparameters.
- `run_inference_best_gif.py`: evaluates a few setups and writes the best-convergence rollout GIF.
- `mpc_ilqr.py`: standalone iLQR planner for receding-horizon Figure-8 tracking.
- `run_mpc_ilqr_fixed_init.py`: MPC/iLQR runner with strict convergence checks and GIF outputs.
- `sbatch_mpc_ilqr_fixed_init.sh`: SLURM entrypoint for the MPC/iLQR fixed-init run.
- `queue_mpc_ilqr_fixed_init.sh`: one-command tier3 submission for MPC/iLQR.

## Direction Convention

Reference trajectory is normalized to start near **upper-left** and move toward **lower-right** first.
This encodes your requested traversal direction convention.

## Run Smoke Test

```bash
python3 run_smoke_test.py
```

## Train PPO (Local / Login Node)

Create/activate venv and install training dependencies:

```bash
bash bootstrap_venv.sh
source .venv_csci1470_smoke/bin/activate
python -m pip install -r requirements-train.txt
```

Run training:

```bash
python train_ppo_figure8.py \
  --updates 120 \
  --num-envs 8 \
  --rollout-steps 128 \
  --ppo-epochs 8 \
  --minibatch-size 256 \
  --eval-every 6 \
  --eval-episodes 16 \
  --run-name ppo_figure8
```

Artifacts are saved under:

`artifacts/<run_name>_<timestamp>/`

Key files:

- `checkpoint_best.pt`
- `checkpoint_latest.pt`
- `metrics.csv`
- `metrics.png`

## Evaluate A Trained Checkpoint

```bash
python evaluate_trained_policy.py \
  --checkpoint artifacts/<run_name>/checkpoint_best.pt \
  --episodes 24 \
  --compare-baselines
```

## Plot Reference Figure-8

```bash
MPLCONFIGDIR=/tmp/mpl_cfg python3 plot_reference_figure8.py
```

This saves:

`figures/reference_figure8.png`

## Animate Reference Figure-8

```bash
MPLCONFIGDIR=/tmp/mpl_cfg python3 animate_reference_figure8.py
```

This saves:

`figures/reference_figure8_animation.gif`

## Run On SLURM (Debug Smoke Test)

From inside `new_final_csci1470`:

```bash
mkdir -p logs
sbatch sbatch_smoke_debug.sh
```

Check status and logs:

```bash
squeue -u "$USER"
tail -f logs/csci1470_smoke_debug_<JOB_ID>.out
tail -f logs/csci1470_smoke_debug_<JOB_ID>.err
```

Optional env activation before `sbatch`:

```bash
export CONDA_ENV_NAME=<your_env>
# or
export VENV_PATH=/path/to/venv
```

## Run On SLURM With Auto-Created Venv (Recommended)

This creates and reuses a dedicated project venv at:

`.venv_csci1470_smoke`

Submit:

```bash
mkdir -p logs
sbatch sbatch_smoke_debug_venv.sh
```

Monitor:

```bash
squeue -u "$USER"
tail -f logs/csci1470_smoke_venv_<JOB_ID>.out
tail -f logs/csci1470_smoke_venv_<JOB_ID>.err
```

You can also create the same venv manually (login node):

```bash
bash bootstrap_venv.sh
source .venv_csci1470_smoke/bin/activate
```

## Run PPO Training On SLURM

Submit:

```bash
mkdir -p logs
sbatch sbatch_train_ppo_venv.sh
```

Override default training size at submit time:

```bash
UPDATES=200 NUM_ENVS=10 ROLLOUT_STEPS=128 PPO_EPOCHS=8 MINIBATCH_SIZE=256 sbatch sbatch_train_ppo_venv.sh
```

Monitor:

```bash
squeue -u "$USER"
tail -f logs/csci1470_train_ppo_<JOB_ID>.out
tail -f logs/csci1470_train_ppo_<JOB_ID>.err
```

## Long 23h Train + Dependent Inference (One Command)

Submit full pipeline:

```bash
bash queue_train_then_infer.sh
```

Submit alternate-hyperparameter pipeline (for comparison):

```bash
bash queue_train_then_infer_alt_hparams.sh
```

What this does:

1. Submits `sbatch_train_ppo_long_23h.sh` (23h debug training).
2. Submits `sbatch_infer_best_gif_after_train.sh` with `afterok` dependency.
3. Inference runs on 10 different randomized setups (default), saves one GIF per setup, and also saves the fastest-converging one as a dedicated best GIF.

Pipeline metadata and resolved paths are stored in:

`artifacts/pipelines/<PIPE_TAG>/`

Main outputs:

- Training artifacts: `artifacts/ppo_long_<PIPE_TAG>_<timestamp>/`
- Best model: `checkpoint_best.pt`
- Inference summary: `inference_best_gif/inference_summary.json`
- Setup GIFs: `inference_best_gif/setup_00_seed_<seed>.gif`, ..., `setup_09_seed_<seed>.gif`
- Best GIF: `inference_best_gif/best_convergence_seed_<seed>.gif`

To compare both runs:

1. Submit baseline pipeline:
`bash queue_train_then_infer.sh`
2. Submit alternate pipeline:
`bash queue_train_then_infer_alt_hparams.sh`
3. Compare their `metrics.csv`, `checkpoint_best.pt`, and `inference_best_gif/inference_summary.json` under each run directory.

## MPC/iLQR Fixed-Init Path (Additive To PPO)

This path does not modify or disable PPO training scripts. It is a separate deterministic controller path for fixed-init convergence.

Local run:

```bash
python run_mpc_ilqr_fixed_init.py \
  --fixed-init-profile offset_ref \
  --fixed-init-pos-jitter-std 0.0 \
  --fixed-init-vel-jitter-std 0.0 \
  --num-setups 10 \
  --max-steps 420
```

Tier3 one-command run:

```bash
bash queue_mpc_ilqr_fixed_init.sh
```

Outputs:

- `inference_mpc_ilqr_fixed_init_<PIPE_TAG>/setup_00_seed_<seed>.gif` (one GIF per setup)
- `inference_mpc_ilqr_fixed_init_<PIPE_TAG>/best_convergence_seed_<seed>.gif`
- `inference_mpc_ilqr_fixed_init_<PIPE_TAG>/inference_summary.json`

## Notes

- This code does not depend on `gym`; API is `reset()` / `step(action)`.
- Action shape is `(3, 2)` (continuous acceleration for each body in 2D).
- If you want full PPO training next, this environment is ready for it.

## Planned Training Change (Directed Initialization)

We are moving away from fully randomized initial states for training.

New plan:

1. Start from one fixed base initial condition (the directed setup).
2. Add only small random perturbations around that base state:
   - slight position jitter
   - slight velocity jitter
3. Keep perturbations intentionally small so the learning problem stays local and stable.
4. Use this as the main training distribution for fixed-init convergence experiments.

Rationale:

- Gives a cleaner learning signal than global random starts.
- Keeps task difficulty manageable.
- Still adds enough variation to avoid brittle overfitting to one exact state.

Current implementation status:

- Implemented via fixed-init profile `near_ref` (see `fixed_init_profiles.py`).
- Implemented local jitter around fixed init through:
  - `fixed_init_pos_jitter_std`
  - `fixed_init_vel_jitter_std`
  - `fixed_init_jitter_tries`
- Current fixed-init training defaults in `train_fixed_init_quick.py` use this directed setup.
