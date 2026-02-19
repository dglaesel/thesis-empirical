#!/bin/bash
#SBATCH --job-name=p2-robust
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/robustness_%j.out
#SBATCH --error=results/slurm/robustness_%j.err

# Phase 2 robustness check — perturbed spread parameters.
# Two perturbations x 2 H x 2 N x 1 arch x 3 seeds x 1 level = 24 training jobs.
# Precompute: 2 perturbations x 2 H x 3 seeds = 12 jobs.
# Total: 36 jobs. Estimated < 2 hours on 96 cores.

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

RUN_ID=${RUN_ID:?ERROR: set RUN_ID before submitting}
SPLIT_MODE=${SPLIT_MODE:-index}
MAX_PARALLEL=${MAX_PARALLEL:-90}

mkdir -p results/slurm

# Perturbation configs and labels
declare -a CONFIGS=("config/robustness_low_kappa.yaml" "config/robustness_high_sigma.yaml")
declare -a LABELS=("low_kappa" "high_sigma")

H_LIST=(0.125 0.375)
N_LIST=(2 4)
SEED_LIST=(0 1 2)
LEVEL=L1
INTEGRATOR=exact
ARCH=adnn

RUNNING=0
FAIL=0

# ============================================================
# Phase A: Precompute features for each perturbation
# ============================================================
echo "=== PRECOMPUTE ==="
for i in "${!CONFIGS[@]}"; do
  CFG=${CONFIGS[$i]}
  LBL=${LABELS[$i]}
  ROBUST_RUN_ID="${RUN_ID}_robust_${LBL}"

  for H in "${H_LIST[@]}"; do
    for SEED in "${SEED_LIST[@]}"; do
      (
        echo "[precompute-robust] label=${LBL} H=${H} seed=${SEED}"
        python scripts/phase2_precompute.py \
          --config "${CFG}" \
          --level "${LEVEL}" \
          --H "${H}" \
          --seed "${SEED}" \
          --integrator "${INTEGRATOR}" \
          --split-mode "${SPLIT_MODE}" \
          --run-id "${ROBUST_RUN_ID}"
      ) &

      RUNNING=$(( RUNNING + 1 ))
      if (( RUNNING >= MAX_PARALLEL )); then
        wait -n || FAIL=$(( FAIL + 1 ))
        RUNNING=$(( RUNNING - 1 ))
      fi
    done
  done
done

wait || true
echo "[precompute-robust] done (${FAIL} failures)"

if (( FAIL > 0 )); then
  echo "FATAL: precompute failures, aborting"
  exit 1
fi

# ============================================================
# Phase B: Train policies for each perturbation
# ============================================================
echo "=== TRAINING ==="
RUNNING=0
FAIL=0
TOTAL=0

for i in "${!CONFIGS[@]}"; do
  CFG=${CONFIGS[$i]}
  LBL=${LABELS[$i]}
  ROBUST_RUN_ID="${RUN_ID}_robust_${LBL}"

  for H in "${H_LIST[@]}"; do
    for SEED in "${SEED_LIST[@]}"; do
      for N in "${N_LIST[@]}"; do
        (
          echo "[train-robust] label=${LBL} H=${H} seed=${SEED} N=${N} arch=${ARCH}"
          python scripts/phase2_train.py \
            --config "${CFG}" \
            --level "${LEVEL}" \
            --H "${H}" \
            --seed "${SEED}" \
            --N "${N}" \
            --arch "${ARCH}" \
            --integrator "${INTEGRATOR}" \
            --split-mode "${SPLIT_MODE}" \
            --run-id "${ROBUST_RUN_ID}" \
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

wait || FAIL=$(( FAIL + 1 ))

echo "[train-robust] completed ${TOTAL} jobs, ${FAIL} failed"
if (( FAIL > 0 )); then
  exit 1
fi
echo "=== ROBUSTNESS DONE ==="
