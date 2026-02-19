#!/bin/bash
#SBATCH --job-name=p1-bank
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=384G
#SBATCH --time=03:00:00
#SBATCH --output=results/slurm/phase1_%j.out
#SBATCH --error=results/slurm/phase1_%j.err

# Phase 1: Bank et al. Table 1 reproduction.
# 30 jobs = 3 H x 5 N x 2 modes (alin/adnn), all on one 96-core node.
# Estimated ~1.5-2 hours.

set -euo pipefail

# --- HPC environment ---
WS_NAME=${WS_NAME:-thesis}
PROJECT_DIR="$(ws_find ${WS_NAME})/thesis-empirical"
cd "${PROJECT_DIR}"
module load "$(cat .python_module)"
source venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONFIG_PATH=${CONFIG_PATH:-config/default.yaml}

mkdir -p results/slurm results/phase1

H_LIST=(0.25 0.5 0.75)
N_LIST=(1 2 3 4 5)
MODE_LIST=(alin adnn)

RUNNING=0
FAIL=0
TOTAL=0

for H in "${H_LIST[@]}"; do
  for N in "${N_LIST[@]}"; do
    for MODE in "${MODE_LIST[@]}"; do
      (
        echo "[phase1] H=${H} N=${N} mode=${MODE}"
        python scripts/phase1_tracking.py \
          --config "${CONFIG_PATH}" \
          --H "${H}" \
          --N "${N}" \
          --mode "${MODE}"
      ) &

      RUNNING=$(( RUNNING + 1 ))
      TOTAL=$(( TOTAL + 1 ))
      if (( RUNNING >= 30 )); then
        wait -n || FAIL=$(( FAIL + 1 ))
        RUNNING=$(( RUNNING - 1 ))
      fi
    done
  done
done

wait || FAIL=$(( FAIL + 1 ))

echo "[phase1] completed ${TOTAL} jobs, ${FAIL} failed"

# Aggregate individual summaries into one CSV
python scripts/aggregate_phase1.py
echo "[phase1] aggregation done"

if (( FAIL > 0 )); then
  exit 1
fi
