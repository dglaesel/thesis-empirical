#!/bin/bash
#SBATCH --job-name=p2-train
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=384G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/train_%j.out
#SBATCH --error=results/slurm/train_%j.err

# Phase 2 training — all (level, H, N, arch, seed) on a single 96-core node.
# 600 total jobs, ~90 concurrent. Estimated ~6-7 hours.

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

mkdir -p results/slurm

H_LIST=(0.125 0.25 0.375 0.5 0.75)
N_LIST=(1 2 3 4)
SEED_LIST=(0 1 2 3 4)
ARCH_LIST=(alin adnn)

declare -A INTEGRATOR_MAP=( [L1]=exact [L2]=milstein [L3]=milstein )
LEVEL_LIST=(L1 L2 L3)

RUNNING=0
FAIL=0
TOTAL=0

for LEVEL in "${LEVEL_LIST[@]}"; do
  INTEGRATOR=${INTEGRATOR_MAP[$LEVEL]}
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
              --device "cpu"
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
done

wait || FAIL=$(( FAIL + 1 ))

echo "[train] completed ${TOTAL} jobs, ${FAIL} failed"
if (( FAIL > 0 )); then
  exit 1
fi
