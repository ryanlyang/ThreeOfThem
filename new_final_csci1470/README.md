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

## Direction Convention

Reference trajectory is normalized to start near **upper-left** and move toward **lower-right** first.
This encodes your requested traversal direction convention.

## Run Smoke Test

```bash
python3 run_smoke_test.py
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

## Notes

- This code does not depend on `gym`; API is `reset()` / `step(action)`.
- Action shape is `(3, 2)` (continuous acceleration for each body in 2D).
- If you want full PPO training next, this environment is ready for it.
