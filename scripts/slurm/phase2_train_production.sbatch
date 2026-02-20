#!/bin/bash
#SBATCH --job-name=p2-train
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=22
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/train_%j.out
#SBATCH --error=results/slurm/train_%j.err

# Phase 2 training — one level per job (200 tasks, 20 concurrent).
# Submit three times:  LEVEL=L1  LEVEL=L2  LEVEL=L3
# DNN tasks use warm restarts (3 inits × 30-epoch audition + full training).

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
MAX_PARALLEL=${MAX_PARALLEL:-20}
LEVEL=${LEVEL:?ERROR: set LEVEL (L1, L2, or L3) before submitting}

declare -A INTEGRATOR_MAP=( [L1]=exact [L2]=milstein [L3]=milstein )
INTEGRATOR=${INTEGRATOR_MAP[$LEVEL]}
if [[ -z "${INTEGRATOR}" ]]; then
  echo "ERROR: invalid LEVEL=${LEVEL}. Must be L1, L2, or L3."
  exit 1
fi

mkdir -p results/slurm

H_LIST=(0.125 0.25 0.375 0.5 0.75)
N_LIST=(1 2 3 4)
SEED_LIST=(0 1 2 3 4)
ARCH_LIST=(alin adnn)

RUNNING=0
FAIL=0
TOTAL=0

for H in "${H_LIST[@]}"; do
  for SEED in "${SEED_LIST[@]}"; do
    for N in "${N_LIST[@]}"; do
      for ARCH in "${ARCH_LIST[@]}"; do
        (
          echo "[train] level=${LEVEL} H=${H} seed=${SEED} N=${N} arch=${ARCH}"
          python scripts/phase2_train.py \
            --config "${CONFIG_PATH}" \
            --level "${LEVEL}" \
            --H "${H}" \
            --seed "${SEED}" \
            --N "${N}" \
            --arch "${ARCH}" \
            --integrator "${INTEGRATOR}" \
            --split-mode "${SPLIT_MODE}" \
            --run-id "${RUN_ID}" \
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
done

wait || FAIL=$(( FAIL + 1 ))

echo "[train] ${LEVEL}: completed ${TOTAL} jobs, ${FAIL} failed"
if (( FAIL > 0 )); then
  exit 1
fi
