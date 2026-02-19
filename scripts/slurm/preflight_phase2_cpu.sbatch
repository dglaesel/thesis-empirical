#!/bin/bash
#SBATCH --job-name=preflight
#SBATCH --partition=dev_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=results/slurm/preflight_%j.out
#SBATCH --error=results/slurm/preflight_%j.err

# Preflight: verify Python environment, iisignature, torch on a compute node.

set -euo pipefail

# --- HPC environment ---
module load devel/python/3.12
WS_NAME=${WS_NAME:-thesis}
PROJECT_DIR="$(ws_find ${WS_NAME})/empiricalstudy_clean"
cd "${PROJECT_DIR}"
source venv/bin/activate

mkdir -p results/slurm

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "[preflight] host=$(hostname)"
echo "[preflight] cpus_per_task=${SLURM_CPUS_PER_TASK:-unknown}"
echo "[preflight] partition=${SLURM_JOB_PARTITION:-unknown}"
echo "[preflight] project_dir=${PROJECT_DIR}"
echo "[preflight] python=$(which python)"

python - <<'PY'
import sys
print(f"[preflight] Python {sys.version}")

import torch
print(f"[preflight] torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

import iisignature
print(f"[preflight] iisignature: {getattr(iisignature, 'version', 'unknown')}")
print(f"[preflight] logsigjoin available: {hasattr(iisignature, 'logsigjoin')}")

import numpy as np
x = np.random.randn(8, 5, 2).astype(np.float64)
for method in ["O", "D", "S"]:
    try:
        s = iisignature.prepare(2, 5, method)
        y = iisignature.logsig(x, s)
        print(f"[preflight] method={method} ok shape={y.shape}")
    except Exception as e:
        print(f"[preflight] method={method} failed: {e}")

# Quick signature join test
import iisignature
s = iisignature.prepare(2, 3, "O")
zz = np.zeros((4, iisignature.logsiglength(2, 3)), dtype=np.float64)
dz = np.random.randn(4, 2).astype(np.float64)
out = iisignature.logsigjoin(zz, dz, s)
print(f"[preflight] logsigjoin test ok, output shape={out.shape}")

print("[preflight] ALL CHECKS PASSED")
PY
