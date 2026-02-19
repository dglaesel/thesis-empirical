#!/bin/bash
# Setup Python environment on bwUniCluster 3.0.
# Run this from the project root inside your workspace:
#   cd $(ws_find thesis)/empiricalstudy_clean
#   bash setup_hpc.sh
#
# IMPORTANT: iisignature==0.24 logsigjoin only compiles correctly on Python 3.11.
# Python 3.12 builds iisignature but is MISSING logsigjoin (critical for O(K) logsig).
# Override with:  PYTHON_MODULE=devel/python/3.12 bash setup_hpc.sh
set -euo pipefail

# --- Detect Python module ---
PYTHON_MODULE=${PYTHON_MODULE:-""}

if [[ -z "${PYTHON_MODULE}" ]]; then
  # Try 3.11 first (logsigjoin works), fall back to 3.12
  if module avail devel/python/3.11 2>&1 | grep -q "devel/python/3.11"; then
    PYTHON_MODULE="devel/python/3.11"
  elif module avail devel/python/3.12 2>&1 | grep -q "devel/python/3.12"; then
    PYTHON_MODULE="devel/python/3.12"
    echo "WARNING: Python 3.11 not found, using 3.12."
    echo "         logsigjoin may NOT be available — setup will verify."
  else
    echo "ERROR: Neither devel/python/3.11 nor devel/python/3.12 found."
    echo "       Check available modules with: module avail devel/python"
    exit 1
  fi
fi

echo "Using module: ${PYTHON_MODULE}"
module load "${PYTHON_MODULE}"
python --version

# Save the module name so SLURM scripts can source it
echo "${PYTHON_MODULE}" > .python_module
echo "Saved Python module to .python_module"

echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify critical packages
python -c "
import sys
print(f'Python {sys.version}')

import torch
print(f'torch {torch.__version__}')

import iisignature
ver = iisignature.version() if callable(iisignature.version) else iisignature.version
print(f'iisignature {ver}')

has_lsj = hasattr(iisignature, 'logsigjoin')
print(f'logsigjoin available: {has_lsj}')
if not has_lsj:
    print()
    print('FATAL: logsigjoin NOT available in this iisignature build.')
    print('       logsigjoin is required for O(K) incremental log-signature computation.')
    print('       Without it, precompute uses O(K^2) fallback — far too slow for production.')
    print()
    print('       This is a known issue with iisignature 0.24 on Python 3.12.')
    print('       Fix: use Python 3.11 instead:')
    print('         rm -rf venv')
    print('         PYTHON_MODULE=devel/python/3.11 bash setup_hpc.sh')
    sys.exit(1)

# Quick functional test
import numpy as np
s = iisignature.prepare(2, 3, 'O')
dim = iisignature.logsiglength(2, 3)
zz = np.zeros((4, dim), dtype=np.float64)
dz = np.random.randn(4, 2).astype(np.float64)
out = iisignature.logsigjoin(zz, dz, s)
assert out.shape == (4, dim), f'logsigjoin shape wrong: {out.shape}'
assert np.isfinite(out).all(), 'logsigjoin produced non-finite values'
print('logsigjoin functional test: OK')

import scipy, numpy, yaml, tqdm
print('All dependencies OK')
"

echo ""
echo "============================================"
echo " Setup complete!"
echo " Python module: $(cat .python_module)"
echo "============================================"
echo ""
echo "Next step — run preflight check on a compute node:"
echo "  sbatch scripts/slurm/preflight_phase2_cpu.sbatch"
