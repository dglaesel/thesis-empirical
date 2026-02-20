#!/bin/bash
# Submit all Phase 2 jobs: precompute → training (L1, L2, L3) + Riccati comparison.
# Run this from the HPC login node after pushing code.
#
# Usage:
#   bash scripts/slurm/submit_phase2_all.sh
#
# Prerequisites:
#   - Code pushed to HPC workspace
#   - setup_hpc.sh already run (venv, .python_module, .compiler_module exist)
#   - logsigjoin available (Python 3.11)

set -euo pipefail

# --- Configuration ---
RUN_ID=$(date +%Y%m%d_%H%M%S)
export RUN_ID
export SPLIT_MODE=index

echo "=== Phase 2 submission ==="
echo "RUN_ID: ${RUN_ID}"
echo ""

# --- Step 1: Precompute (75 tasks: 3 levels × 5 H × 5 seeds) ---
echo "[1/5] Submitting precompute..."
PRECOMPUTE_JOB=$(sbatch --parsable \
  --export=ALL,RUN_ID=${RUN_ID},SPLIT_MODE=${SPLIT_MODE} \
  scripts/slurm/phase2_precompute_production.sbatch)
echo "  Precompute job: ${PRECOMPUTE_JOB}"

# --- Step 2: Training L1 (200 tasks, depends on precompute) ---
echo "[2/5] Submitting training L1..."
L1_JOB=$(sbatch --parsable \
  --dependency=afterok:${PRECOMPUTE_JOB} \
  --export=ALL,RUN_ID=${RUN_ID},LEVEL=L1,SPLIT_MODE=${SPLIT_MODE} \
  scripts/slurm/phase2_train_production.sbatch)
echo "  L1 training job: ${L1_JOB}"

# --- Step 3: Training L2 (200 tasks, depends on precompute) ---
echo "[3/5] Submitting training L2..."
L2_JOB=$(sbatch --parsable \
  --dependency=afterok:${PRECOMPUTE_JOB} \
  --export=ALL,RUN_ID=${RUN_ID},LEVEL=L2,SPLIT_MODE=${SPLIT_MODE} \
  scripts/slurm/phase2_train_production.sbatch)
echo "  L2 training job: ${L2_JOB}"

# --- Step 4: Training L3 (200 tasks, depends on precompute) ---
echo "[4/5] Submitting training L3..."
L3_JOB=$(sbatch --parsable \
  --dependency=afterok:${PRECOMPUTE_JOB} \
  --export=ALL,RUN_ID=${RUN_ID},LEVEL=L3,SPLIT_MODE=${SPLIT_MODE} \
  scripts/slurm/phase2_train_production.sbatch)
echo "  L3 training job: ${L3_JOB}"

# --- Step 5: Riccati comparison (40 tasks: H=0.5, L1, eta=0.001, c=0) ---
echo "[5/5] Submitting Riccati comparison..."
RICCATI_JOB=$(sbatch --parsable \
  --dependency=afterok:${PRECOMPUTE_JOB} \
  --export=ALL,RUN_ID=${RUN_ID},SPLIT_MODE=${SPLIT_MODE} \
  scripts/slurm/phase2_riccati_comparison.sbatch)
echo "  Riccati comparison job: ${RICCATI_JOB}"

echo ""
echo "=== All submitted ==="
echo "Precompute:  ${PRECOMPUTE_JOB}  (~2h)"
echo "L1 training: ${L1_JOB}  (~16h, starts after precompute)"
echo "L2 training: ${L2_JOB}  (~16h, starts after precompute)"
echo "L3 training: ${L3_JOB}  (~16h, starts after precompute)"
echo "Riccati:     ${RICCATI_JOB}  (~6h, starts after precompute)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Cancel all: scancel ${PRECOMPUTE_JOB} ${L1_JOB} ${L2_JOB} ${L3_JOB} ${RICCATI_JOB}"
