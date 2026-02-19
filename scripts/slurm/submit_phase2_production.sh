#!/bin/bash
# Submit Phase 2 production workflow only (without Phase 1):
#   1. precompute features (~2h)
#   2. train all policies (~12h, depends on precompute)
#   3. (optional) robustness check (~2h, independent)
#
# Run from the project root inside your workspace:
#   cd $(ws_find thesis)/empiricalstudy_clean
#   bash scripts/slurm/submit_phase2_production.sh
#
# To also run robustness:
#   RUN_ROBUSTNESS=1 bash scripts/slurm/submit_phase2_production.sh

set -euo pipefail

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_ROBUSTNESS=${RUN_ROBUSTNESS:-0}

echo "[submit] RUN_ID=${RUN_ID}"

mkdir -p results/slurm

# Step 1: Precompute
PRE_JOB=$(
  sbatch --parsable \
    --export=ALL,RUN_ID="${RUN_ID}" \
    scripts/slurm/phase2_precompute_production.sbatch
)
echo "[submit] precompute job: ${PRE_JOB}"

# Step 2: Train (depends on precompute)
TRAIN_JOB=$(
  sbatch --parsable \
    --dependency=afterok:${PRE_JOB} \
    --export=ALL,RUN_ID="${RUN_ID}" \
    scripts/slurm/phase2_train_production.sbatch
)
echo "[submit] train job: ${TRAIN_JOB} (depends on ${PRE_JOB})"

# Step 3: Robustness (optional, independent — has its own precompute)
if [[ "${RUN_ROBUSTNESS}" == "1" ]]; then
  ROBUST_JOB=$(
    sbatch --parsable \
      --export=ALL,RUN_ID="${RUN_ID}" \
      scripts/slurm/phase2_robustness_production.sbatch
  )
  echo "[submit] robustness job: ${ROBUST_JOB} (independent)"
fi

echo "[submit] done. Monitor with: squeue -u \$USER"
