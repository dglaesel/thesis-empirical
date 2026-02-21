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

# Default Riccati comparison: H=0.5, L1, eta=0.001, c=0.
# For robustness settings, override CONFIG_PATH/H_LIST/SEED_LIST and keep
# FEATURES_RUN_ID aligned with the precompute namespace.

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
FEATURES_RUN_ID=${FEATURES_RUN_ID:-${RUN_ID}}
RICCATI_RUN_SUFFIX=${RICCATI_RUN_SUFFIX:-_riccati}
RICCATI_RUN_ID=${RICCATI_RUN_ID:-${RUN_ID}${RICCATI_RUN_SUFFIX}}
SPLIT_MODE=${SPLIT_MODE:-index}
MAX_PARALLEL=${MAX_PARALLEL:-10}

LEVEL=${LEVEL:-L1}
INTEGRATOR=${INTEGRATOR:-exact}
H_LIST=${H_LIST:-0.5}
N_LIST=${N_LIST:-"1 2 3 4"}
SEED_LIST=${SEED_LIST:-"0 1 2 3 4"}
ARCH_LIST=${ARCH_LIST:-"alin adnn"}
ETA=${ETA:-0.001}
COST_C=${COST_C:-0.0}
DEVICE=${DEVICE:-cpu}
FORCE=${FORCE:-1}
FEATURES_ROOT=${FEATURES_ROOT:-results/phase2/features}

H_LIST=${H_LIST//,/ }
N_LIST=${N_LIST//,/ }
SEED_LIST=${SEED_LIST//,/ }
ARCH_LIST=${ARCH_LIST//,/ }

mkdir -p results/slurm

read -r -a H_VALUES <<< "${H_LIST}"
read -r -a N_VALUES <<< "${N_LIST}"
read -r -a SEED_VALUES <<< "${SEED_LIST}"
read -r -a ARCH_VALUES <<< "${ARCH_LIST}"

if (( ${#H_VALUES[@]} == 0 || ${#N_VALUES[@]} == 0 || ${#SEED_VALUES[@]} == 0 || ${#ARCH_VALUES[@]} == 0 )); then
  echo "FATAL: one of H_LIST/N_LIST/SEED_LIST/ARCH_LIST is empty."
  exit 1
fi

if [[ "${RICCATI_RUN_ID}" != "${FEATURES_RUN_ID}" ]]; then
  SRC_FEATURES_DIR="${FEATURES_ROOT}/${FEATURES_RUN_ID}"
  DST_FEATURES_DIR="${FEATURES_ROOT}/${RICCATI_RUN_ID}"
  if [[ ! -d "${SRC_FEATURES_DIR}" ]]; then
    echo "FATAL: source features not found: ${SRC_FEATURES_DIR}"
    exit 1
  fi
  if [[ -L "${DST_FEATURES_DIR}" ]]; then
    echo "[riccati] feature link already exists: ${DST_FEATURES_DIR}"
  elif [[ -d "${DST_FEATURES_DIR}" ]]; then
    echo "[riccati] feature namespace already exists: ${DST_FEATURES_DIR}"
  elif [[ -e "${DST_FEATURES_DIR}" ]]; then
    echo "FATAL: cannot create feature link, path exists but is not a directory/link: ${DST_FEATURES_DIR}"
    exit 1
  else
    ln -s "${SRC_FEATURES_DIR}" "${DST_FEATURES_DIR}"
    echo "[riccati] created feature link: ${DST_FEATURES_DIR} -> ${SRC_FEATURES_DIR}"
  fi
fi

RUNNING=0
FAIL=0
TOTAL=0

for H in "${H_VALUES[@]}"; do
  for SEED in "${SEED_VALUES[@]}"; do
    for N in "${N_VALUES[@]}"; do
      for ARCH in "${ARCH_VALUES[@]}"; do
        (
          echo "[riccati] H=${H} seed=${SEED} N=${N} arch=${ARCH} run_id=${RICCATI_RUN_ID}"
          CMD=(
            python scripts/phase2_train.py
            --config "${CONFIG_PATH}"
            --level "${LEVEL}"
            --H "${H}"
            --seed "${SEED}"
            --N "${N}"
            --arch "${ARCH}"
            --integrator "${INTEGRATOR}"
            --split-mode "${SPLIT_MODE}"
            --run-id "${RICCATI_RUN_ID}"
            --eta "${ETA}"
            --c "${COST_C}"
            --device "${DEVICE}"
          )
          if [[ "${FORCE}" == "1" ]]; then
            CMD+=(--force)
          fi
          "${CMD[@]}"
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

echo "[riccati] completed ${TOTAL} jobs, ${FAIL} failed"
if (( FAIL > 0 )); then
  exit 1
fi
