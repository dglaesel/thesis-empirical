#!/bin/bash
# Setup Python environment on bwUniCluster 3.0.
# Run this ONCE from the project root inside your workspace:
#   cd $(ws_find thesis)/thesis-empirical
#   bash setup_hpc.sh
#
# IMPORTANT: logsigjoin was added to iisignature AFTER the 0.24 PyPI release (Dec 2019).
# The PyPI tarball does NOT have logsigjoin. We install from git (pinned commit).
#
# If something goes wrong:
#   rm -rf venv .python_module
#   bash setup_hpc.sh
set -euo pipefail

# --- Module configuration ---
# bwUniCluster 3.0 available (as of Feb 2026):
#   devel/python/3.11.7-gnu-{11.4,14.2}
#   devel/python/3.12.3-gnu-{11.4,14.2} (D)
#   devel/python/3.13.{1,3}-*
# Using 3.12 (cluster default).
COMPILER_MODULE=${COMPILER_MODULE:-"compiler/gnu/14.2"}
PYTHON_MODULE=${PYTHON_MODULE:-"devel/python/3.12.3-gnu-14.2"}

echo "============================================"
echo " Loading modules"
echo "============================================"
# Compiler must be loaded BEFORE Python so g++ is available for C++ extensions
module load "${COMPILER_MODULE}"
module load "${PYTHON_MODULE}"

echo "  compiler: $(g++ --version | head -1)"
echo "  python:   $(python --version)"

# Save module names so SLURM scripts can source them
echo "${COMPILER_MODULE}" > .compiler_module
echo "${PYTHON_MODULE}" > .python_module
echo "  Saved to .compiler_module and .python_module"

echo ""
echo "============================================"
echo " Creating virtual environment"
echo "============================================"
python -m venv venv
source venv/bin/activate
echo "  venv: $(which python)"

echo ""
echo "============================================"
echo " Installing packages"
echo "============================================"

# Step 1: Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Step 2: Install numpy first — iisignature 0.24's setup.py imports numpy
#         at the top level, but its tarball doesn't declare it in
#         pyproject.toml build-system.requires. So build isolation fails.
pip install "numpy>=1.17"

# Step 3: Build iisignature from git (post-0.24, includes logsigjoin).
#         The 0.24 PyPI tarball does NOT have logsigjoin — it was added in 2023.
#         Pinned to the same commit used in development (ee9c1ec).
#         --no-build-isolation so it sees the numpy we just installed.
echo ""
echo "Building iisignature from git (C++14 extension, includes logsigjoin)..."
pip install --no-build-isolation "iisignature @ git+https://github.com/bottler/iisignature.git@ee9c1ec3a5fde539e854b7e8a3fb92195b506817"

# Step 4: Install remaining dependencies (torch, scipy, etc.)
pip install -r requirements.txt

echo ""
echo "============================================"
echo " Verifying installation"
echo "============================================"

python -c "
import sys
print(f'Python {sys.version}')

import torch
print(f'torch {torch.__version__}')

import numpy as np
print(f'numpy {np.__version__}')

import scipy
print(f'scipy {scipy.__version__}')

import iisignature
ver = iisignature.version() if callable(iisignature.version) else iisignature.version
print(f'iisignature {ver}')

# --- logsigjoin check (CRITICAL) ---
has_lsj = hasattr(iisignature, 'logsigjoin')
print(f'logsigjoin available: {has_lsj}')
if not has_lsj:
    print()
    print('FATAL: logsigjoin NOT available in this iisignature build.')
    print('       logsigjoin is required for O(K) incremental log-signature computation.')
    print('       Without it, precompute uses O(K^2) fallback — far too slow for production.')
    print()
    print('       logsigjoin was added after the 0.24 PyPI release.')
    print('       It must be installed from git, not PyPI.')
    print('       Fix: rm -rf venv .python_module .compiler_module')
    print('            bash setup_hpc.sh')
    sys.exit(1)

# --- Functional test ---
s = iisignature.prepare(2, 3, 'O')
dim = iisignature.logsiglength(2, 3)
zz = np.zeros((4, dim), dtype=np.float64)
dz = np.random.randn(4, 2).astype(np.float64)
out = iisignature.logsigjoin(zz, dz, s)
assert out.shape == (4, dim), f'logsigjoin shape wrong: {out.shape}'
assert np.isfinite(out).all(), 'logsigjoin produced non-finite values'
print('logsigjoin functional test: OK')

import yaml, tqdm, matplotlib
print('All dependencies OK')
"

echo ""
echo "============================================"
echo " Setup complete!"
echo " Compiler: $(cat .compiler_module)"
echo " Python:   $(cat .python_module)"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Verify on login node:"
echo "     python -c \"import iisignature; print('logsigjoin:', hasattr(iisignature, 'logsigjoin'))\""
echo "  2. Run preflight on a compute node:"
echo "     sbatch scripts/slurm/preflight_phase2_cpu.sbatch"
