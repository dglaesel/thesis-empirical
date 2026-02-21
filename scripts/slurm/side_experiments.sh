#!/bin/bash
# Submit side experiments in parallel (independent jobs):
#   1) Phase 2 robustness check
#   2) HJB/Riccati comparison (H=0.5, L1, eta=0.001, c=0)
# Optional:
#   3) Riccati-objective runs on robustness settings (low_kappa/high_sigma)
#
# Usage (one line):
#   bash scripts/slurm/side_experiments.sh
#
# Optional environment variables:
#   RUN_ID                 base timestamp id (default: now)
#   ROBUST_RUN_ID          run-id passed to robustness job (default: RUN_ID)
#   RICCATI_RUN_ID         run-id passed to riccati job (default: RUN_ID)
#   SPLIT_MODE             split mode for both jobs (default: index)
#   ROBUST_MAX_PARALLEL    robustness inner parallelism (default: 6)
#   RICCATI_MAX_PARALLEL   riccati inner parallelism (default: 8)
#   RICCATI_CPUS           sbatch override cpus for riccati (default: 12)
#   RICCATI_MEM            sbatch override mem for riccati (default: 48G)
#   RICCATI_TIME           sbatch override time for riccati (default: 04:00:00)
#   RICCATI_CONFIG_PATH    config path for riccati job (default: config/default.yaml)
#   RICCATI_RUN_SUFFIX     output suffix for riccati namespace (default: _riccati)
#   RICCATI_H_LIST         H grid for primary riccati (default: 0.5)
#   RICCATI_SEED_LIST      seed grid for primary riccati (default: "0 1 2 3 4")
#   RICCATI_N_LIST         N grid for primary riccati (default: "1 2 3 4")
#   RICCATI_ARCH_LIST      architectures for primary riccati (default: "alin adnn")
#   RICCATI_ETA            eta override for primary riccati (default: 0.001)
#   RICCATI_C              c override for primary riccati (default: 0.0)
#   RUN_ROBUST_RICCATI     set 1 to also submit robustness riccati jobs (default: 0)
#   ROBUST_RICCATI_DEPENDENCY set 1 to start robust-riccati after robustness job (default: 1)
#   ROBUST_RICCATI_MAX_PARALLEL inner parallelism for robust-riccati jobs (default: 6)
#   ROBUST_RICCATI_CPUS    sbatch cpus for robust-riccati jobs (default: 12)
#   ROBUST_RICCATI_MEM     sbatch mem for robust-riccati jobs (default: 48G)
#   ROBUST_RICCATI_TIME    sbatch time for robust-riccati jobs (default: 04:00:00)
#   ROBUST_RICCATI_H_LIST  H grid for robust-riccati (default: "0.25 0.5")
#   ROBUST_RICCATI_SEED_LIST seeds for robust-riccati (default: "0 1 2")
#   ROBUST_RICCATI_N_LIST  N grid for robust-riccati (default: "1 2 3 4")
#   ROBUST_RICCATI_ARCH_LIST arch grid for robust-riccati (default: "alin adnn")
#   ROBUST_RICCATI_ETA     eta override for robust-riccati (default: 0.001)
#   ROBUST_RICCATI_C       c override for robust-riccati (default: 0.0)
#   ROBUST_RICCATI_RUN_SUFFIX output suffix for robust-riccati namespace (default: _riccati)
#
# Note:
#   The Riccati script expects precomputed L1 exact features under FEATURES_RUN_ID.

set -euo pipefail

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
ROBUST_RUN_ID=${ROBUST_RUN_ID:-${RUN_ID}}
RICCATI_RUN_ID=${RICCATI_RUN_ID:-${RUN_ID}}
SPLIT_MODE=${SPLIT_MODE:-index}

ROBUST_MAX_PARALLEL=${ROBUST_MAX_PARALLEL:-6}
RICCATI_MAX_PARALLEL=${RICCATI_MAX_PARALLEL:-8}

RICCATI_CPUS=${RICCATI_CPUS:-12}
RICCATI_MEM=${RICCATI_MEM:-48G}
RICCATI_TIME=${RICCATI_TIME:-04:00:00}
RICCATI_CONFIG_PATH=${RICCATI_CONFIG_PATH:-config/default.yaml}
RICCATI_RUN_SUFFIX=${RICCATI_RUN_SUFFIX:-_riccati}
RICCATI_H_LIST=${RICCATI_H_LIST:-0.5}
RICCATI_SEED_LIST=${RICCATI_SEED_LIST:-"0 1 2 3 4"}
RICCATI_N_LIST=${RICCATI_N_LIST:-"1 2 3 4"}
RICCATI_ARCH_LIST=${RICCATI_ARCH_LIST:-"alin adnn"}
RICCATI_ETA=${RICCATI_ETA:-0.001}
RICCATI_C=${RICCATI_C:-0.0}

RUN_ROBUST_RICCATI=${RUN_ROBUST_RICCATI:-0}
ROBUST_RICCATI_DEPENDENCY=${ROBUST_RICCATI_DEPENDENCY:-1}
ROBUST_RICCATI_MAX_PARALLEL=${ROBUST_RICCATI_MAX_PARALLEL:-6}
ROBUST_RICCATI_CPUS=${ROBUST_RICCATI_CPUS:-12}
ROBUST_RICCATI_MEM=${ROBUST_RICCATI_MEM:-48G}
ROBUST_RICCATI_TIME=${ROBUST_RICCATI_TIME:-04:00:00}
ROBUST_RICCATI_H_LIST=${ROBUST_RICCATI_H_LIST:-"0.25 0.5"}
ROBUST_RICCATI_SEED_LIST=${ROBUST_RICCATI_SEED_LIST:-"0 1 2"}
ROBUST_RICCATI_N_LIST=${ROBUST_RICCATI_N_LIST:-"1 2 3 4"}
ROBUST_RICCATI_ARCH_LIST=${ROBUST_RICCATI_ARCH_LIST:-"alin adnn"}
ROBUST_RICCATI_ETA=${ROBUST_RICCATI_ETA:-0.001}
ROBUST_RICCATI_C=${ROBUST_RICCATI_C:-0.0}
ROBUST_RICCATI_RUN_SUFFIX=${ROBUST_RICCATI_RUN_SUFFIX:-_riccati}

echo "=== Side experiments submission ==="
echo "RUN_ID base:         ${RUN_ID}"
echo "ROBUST_RUN_ID:       ${ROBUST_RUN_ID}"
echo "RICCATI_RUN_ID:      ${RICCATI_RUN_ID}"
echo "SPLIT_MODE:          ${SPLIT_MODE}"
echo "ROBUST_MAX_PARALLEL: ${ROBUST_MAX_PARALLEL}"
echo "RICCATI_MAX_PARALLEL:${RICCATI_MAX_PARALLEL}"
echo "RICCATI resources:   cpus=${RICCATI_CPUS} mem=${RICCATI_MEM} time=${RICCATI_TIME}"
echo "RUN_ROBUST_RICCATI:  ${RUN_ROBUST_RICCATI}"
echo ""

mkdir -p results/slurm

# Submit robustness job (independent)
ROBUST_JOB=$(
  sbatch --parsable \
    --export=ALL,RUN_ID="${ROBUST_RUN_ID}",SPLIT_MODE="${SPLIT_MODE}",MAX_PARALLEL="${ROBUST_MAX_PARALLEL}" \
    scripts/slurm/phase2_robustness_production.sbatch
)

# Submit riccati job (independent)
RICCATI_JOB=$(
  sbatch --parsable \
    --cpus-per-task="${RICCATI_CPUS}" \
    --mem="${RICCATI_MEM}" \
    --time="${RICCATI_TIME}" \
    --export=ALL,RUN_ID="${RICCATI_RUN_ID}",FEATURES_RUN_ID="${RICCATI_RUN_ID}",RICCATI_RUN_SUFFIX="${RICCATI_RUN_SUFFIX}",SPLIT_MODE="${SPLIT_MODE}",MAX_PARALLEL="${RICCATI_MAX_PARALLEL}",CONFIG_PATH="${RICCATI_CONFIG_PATH}",H_LIST="${RICCATI_H_LIST}",SEED_LIST="${RICCATI_SEED_LIST}",N_LIST="${RICCATI_N_LIST}",ARCH_LIST="${RICCATI_ARCH_LIST}",ETA="${RICCATI_ETA}",COST_C="${RICCATI_C}" \
    scripts/slurm/phase2_riccati_comparison.sbatch
)

echo "[submit] robustness job: ${ROBUST_JOB}"
echo "[submit] riccati job:   ${RICCATI_JOB}"

ROBUST_RICCATI_JOBS=()
if [[ "${RUN_ROBUST_RICCATI}" == "1" ]]; then
  declare -a ROBUST_CFGS=("config/robustness_low_kappa.yaml" "config/robustness_high_sigma.yaml")
  declare -a ROBUST_LABELS=("low_kappa" "high_sigma")

  for i in "${!ROBUST_CFGS[@]}"; do
    CFG=${ROBUST_CFGS[$i]}
    LBL=${ROBUST_LABELS[$i]}
    BASE_RUN="${ROBUST_RUN_ID}_robust_${LBL}"
    DEP_ARG=""
    if [[ "${ROBUST_RICCATI_DEPENDENCY}" == "1" ]]; then
      DEP_ARG="--dependency=afterok:${ROBUST_JOB}"
    fi
    JOB=$(
      sbatch --parsable ${DEP_ARG} \
        --cpus-per-task="${ROBUST_RICCATI_CPUS}" \
        --mem="${ROBUST_RICCATI_MEM}" \
        --time="${ROBUST_RICCATI_TIME}" \
        --export=ALL,RUN_ID="${BASE_RUN}",FEATURES_RUN_ID="${BASE_RUN}",RICCATI_RUN_SUFFIX="${ROBUST_RICCATI_RUN_SUFFIX}",SPLIT_MODE="${SPLIT_MODE}",MAX_PARALLEL="${ROBUST_RICCATI_MAX_PARALLEL}",CONFIG_PATH="${CFG}",H_LIST="${ROBUST_RICCATI_H_LIST}",SEED_LIST="${ROBUST_RICCATI_SEED_LIST}",N_LIST="${ROBUST_RICCATI_N_LIST}",ARCH_LIST="${ROBUST_RICCATI_ARCH_LIST}",ETA="${ROBUST_RICCATI_ETA}",COST_C="${ROBUST_RICCATI_C}" \
        scripts/slurm/phase2_riccati_comparison.sbatch
    )
    ROBUST_RICCATI_JOBS+=("${JOB}")
    echo "[submit] robustness riccati (${LBL}): ${JOB}"
  done
fi

echo ""
if [[ "${RUN_ROBUST_RICCATI}" == "1" ]]; then
  echo "Submitted: robustness + primary riccati + robustness riccati jobs."
else
  echo "Submitted independently (no inter-job dependency)."
fi
echo "Monitor: squeue -u \$USER"
if (( ${#ROBUST_RICCATI_JOBS[@]} > 0 )); then
  echo "Cancel:  scancel ${ROBUST_JOB} ${RICCATI_JOB} ${ROBUST_RICCATI_JOBS[*]}"
else
  echo "Cancel:  scancel ${ROBUST_JOB} ${RICCATI_JOB}"
fi
