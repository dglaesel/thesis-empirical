#!/bin/bash
#SBATCH --job-name=p2-pre
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --mem=300G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/precompute_%j.out
#SBATCH --error=results/slurm/precompute_%j.err

# Phase 2 precompute — all (level, H, seed) on a single 96-core node.
# Each Python process is single-threaded; we launch up to 75 in parallel.

set -euo pipefail

# --- HPC environment ---
WS_NAME=${WS_NAME:-thesis}
PROJECT_DIR="$(ws_find ${WS_NAME})/thesis-empirical"
cd "${PROJECT_DIR}"
module load "$(cat .compiler_module)"
module load "$(cat .python_module)"
source venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONFIG_PATH=${CONFIG_PATH:-config/default.yaml}
RUN_ID=${RUN_ID:?ERROR: set RUN_ID before submitting}
SPLIT_MODE=${SPLIT_MODE:-index}
MAX_PARALLEL=${MAX_PARALLEL:-90}

mkdir -p results/slurm results/phase2

H_LIST=(0.125 0.25 0.375 0.5 0.75)
SEED_LIST=(0 1 2 3 4)

# Level → integrator mapping
declare -A INTEGRATOR_MAP=( [L1]=exact [L2]=milstein [L3]=milstein )
LEVEL_LIST=(L1 L2 L3)

RUNNING=0
FAIL=0

for LEVEL in "${LEVEL_LIST[@]}"; do
  INTEGRATOR=${INTEGRATOR_MAP[$LEVEL]}
  for H in "${H_LIST[@]}"; do
    for SEED in "${SEED_LIST[@]}"; do
      (
        echo "[precompute] level=${LEVEL} H=${H} seed=${SEED} integrator=${INTEGRATOR}"
        python scripts/phase2_precompute.py \
          --config "${CONFIG_PATH}" \
          --level "${LEVEL}" \
          --H "${H}" \
          --seed "${SEED}" \
          --integrator "${INTEGRATOR}" \
          --split-mode "${SPLIT_MODE}" \
          --run-id "${RUN_ID}"
      ) &

      RUNNING=$(( RUNNING + 1 ))
      if (( RUNNING >= MAX_PARALLEL )); then
        wait -n || FAIL=$(( FAIL + 1 ))
        RUNNING=$(( RUNNING - 1 ))
      fi
    done
  done
done

# Wait for remaining
wait || FAIL=$(( FAIL + 1 ))

if (( FAIL > 0 )); then
  echo "[precompute] WARNING: ${FAIL} job(s) failed"
  exit 1
fi
echo "[precompute] all done"
