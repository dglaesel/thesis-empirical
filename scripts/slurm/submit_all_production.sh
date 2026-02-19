#!/bin/bash
# Submit ALL production jobs: Phase 1 + Phase 2 (+ optional robustness).
#
# Run from the project root inside your workspace:
#   cd $(ws_find thesis)/empiricalstudy_clean
#   bash scripts/slurm/submit_all_production.sh
#
# Optional env vars:
#   RUN_ROBUSTNESS  (default: 0, set to 1 to include robustness)
#   SKIP_PHASE1     (default: 0, set to 1 to skip Phase 1)
#   SKIP_PHASE2     (default: 0, set to 1 to skip Phase 2)
#   WS_NAME         (default: thesis)

set -euo pipefail

# Auto-generate RUN_ID from current timestamp
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_ROBUSTNESS=${RUN_ROBUSTNESS:-0}
SKIP_PHASE1=${SKIP_PHASE1:-0}
SKIP_PHASE2=${SKIP_PHASE2:-0}

echo "============================================"
echo " Production submission"
echo "============================================"
echo "  RUN_ID:         ${RUN_ID}"
echo "  RUN_ROBUSTNESS: ${RUN_ROBUSTNESS}"
echo "  SKIP_PHASE1:    ${SKIP_PHASE1}"
echo "  SKIP_PHASE2:    ${SKIP_PHASE2}"
echo ""

mkdir -p results/slurm

# ---------------------------------------------------------------
# Phase 1: Bank et al. Table 1 reproduction (~2h, 1 node)
# ---------------------------------------------------------------
if [[ "${SKIP_PHASE1}" != "1" ]]; then
  P1_JOB=$(
    sbatch --parsable \
      --export=ALL \
      scripts/slurm/phase1_production.sbatch
  )
  echo "[submit] Phase 1 job: ${P1_JOB}"
else
  echo "[submit] Phase 1: SKIPPED"
fi

# ---------------------------------------------------------------
# Phase 2: Pairs trading (precompute -> train -> optional robustness)
# ---------------------------------------------------------------
if [[ "${SKIP_PHASE2}" != "1" ]]; then
  # Step 1: Precompute features (~2h)
  PRE_JOB=$(
    sbatch --parsable \
      --export=ALL,RUN_ID="${RUN_ID}" \
      scripts/slurm/phase2_precompute_production.sbatch
  )
  echo "[submit] Phase 2 precompute job: ${PRE_JOB}"

  # Step 2: Train (depends on precompute, ~12h)
  TRAIN_JOB=$(
    sbatch --parsable \
      --dependency=afterok:${PRE_JOB} \
      --export=ALL,RUN_ID="${RUN_ID}" \
      scripts/slurm/phase2_train_production.sbatch
  )
  echo "[submit] Phase 2 train job: ${TRAIN_JOB} (depends on ${PRE_JOB})"

  # Step 3: Robustness (optional, independent — has its own precompute)
  if [[ "${RUN_ROBUSTNESS}" == "1" ]]; then
    ROBUST_JOB=$(
      sbatch --parsable \
        --export=ALL,RUN_ID="${RUN_ID}" \
        scripts/slurm/phase2_robustness_production.sbatch
    )
    echo "[submit] Phase 2 robustness job: ${ROBUST_JOB} (independent)"
  fi
else
  echo "[submit] Phase 2: SKIPPED"
fi

echo ""
echo "[submit] All jobs submitted. Monitor with: squeue -u \$USER"
