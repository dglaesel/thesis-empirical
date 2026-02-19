#!/bin/bash
# Setup Python environment on bwUniCluster 3.0.
# Run this from the project root inside your workspace:
#   cd $(ws_find thesis)/empiricalstudy_clean
#   bash setup_hpc.sh
set -euo pipefail

module load devel/python/3.12

echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify critical packages
python -c "
import torch
print(f'torch {torch.__version__}')

import iisignature
print(f'iisignature {iisignature.version}')
assert hasattr(iisignature, 'logsigjoin'), 'logsigjoin not available — need iisignature >= 0.24'
print('logsigjoin: OK')

import scipy, numpy, yaml, tqdm
print('All dependencies OK')
"

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "Next step — run preflight check on a compute node:"
echo "  sbatch scripts/slurm/preflight_phase2_cpu.sbatch"
