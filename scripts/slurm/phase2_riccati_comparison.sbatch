#!/bin/bash
#SBATCH --job-name=p2-riccati
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm/riccati_%j.out
#SBATCH --error=results/slurm/riccati_%j.err

# Riccati comparison: H=0.5, L1, eta=0.001, c=0 (40 tasks, 10 concurrent).
# Uses same precomputed features as primary; only objective params differ.

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
MAX_PARALLEL=${MAX_PARALLEL:-10}

# Riccati comparison uses a separate run namespace to avoid overwriting primary results
RICCATI_RUN_ID="${RUN_ID}_riccati"

mkdir -p results/slurm

N_LIST=(1 2 3 4)
SEED_LIST=(0 1 2 3 4)
ARCH_LIST=(alin adnn)

RUNNING=0
FAIL=0
TOTAL=0

for SEED in "${SEED_LIST[@]}"; do
  for N in "${N_LIST[@]}"; do
    for ARCH in "${ARCH_LIST[@]}"; do
      (
        echo "[riccati] H=0.5 seed=${SEED} N=${N} arch=${ARCH}"
        python scripts/phase2_train.py \
          --config "${CONFIG_PATH}" \
          --level "L1" \
          --H "0.5" \
          --seed "${SEED}" \
          --N "${N}" \
          --arch "${ARCH}" \
          --integrator "exact" \
          --split-mode "${SPLIT_MODE}" \
          --run-id "${RICCATI_RUN_ID}" \
          --eta 0.001 \
          --c 0.0 \
          --device "cpu" \
          --force
      ) &

      RUNNING=$(( RUNNING + 1 ))
      TOTAL=$(( TOTAL + 1 ))
      if (( RUNNING >= MAX_PARALLEL )); then
        wait -n || FAIL=$(( FAIL + 1 ))
        RUNNING=$(( RUNNING - 1 ))
      fi
    done
  done
done

wait || FAIL=$(( FAIL + 1 ))

echo "[riccati] completed ${TOTAL} jobs, ${FAIL} failed"
if (( FAIL > 0 )); then
  exit 1
fi
