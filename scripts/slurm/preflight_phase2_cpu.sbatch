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
WS_NAME=${WS_NAME:-thesis}
PROJECT_DIR="$(ws_find ${WS_NAME})/thesis-empirical"
cd "${PROJECT_DIR}"
module load "$(cat .compiler_module)"
module load "$(cat .python_module)"
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

import numpy as np
print(f"[preflight] numpy {np.__version__}")

import scipy
print(f"[preflight] scipy {scipy.__version__}")

# ---- iisignature + logsigjoin (REQUIRED) ----
import iisignature
print(f"[preflight] iisignature: {getattr(iisignature, 'version', 'unknown')}")

if not hasattr(iisignature, 'logsigjoin'):
    print("FATAL: iisignature.logsigjoin NOT available.")
    print("       logsigjoin is required for O(K) incremental log-signature computation.")
    print("       Without it, precompute falls back to O(K^2) recomputation — too slow.")
    print("       Ensure iisignature >= 0.24 is compiled with full C++ extension support.")
    print("       Try: pip install --force-reinstall --no-cache-dir iisignature==0.24")
    sys.exit(1)
print("[preflight] logsigjoin: available")

# Functional test: logsigjoin produces correct incremental log-signatures
s = iisignature.prepare(2, 3, "O")
dim = iisignature.logsiglength(2, 3)
zz = np.zeros((4, dim), dtype=np.float64)
dz = np.random.randn(4, 2).astype(np.float64)
out = iisignature.logsigjoin(zz, dz, s)
assert out.shape == (4, dim), f"logsigjoin shape mismatch: {out.shape} vs {(4, dim)}"
assert np.isfinite(out).all(), "logsigjoin produced non-finite values"
print(f"[preflight] logsigjoin functional test OK (shape={out.shape})")

# Verify incremental == batch: logsigjoin(0, dX) should equal logsig([0, dX])
path_2pt = np.zeros((4, 2, 2), dtype=np.float64)
path_2pt[:, 1, :] = dz
batch_result = iisignature.logsig(path_2pt, s)
assert np.allclose(out, batch_result, atol=1e-12), \
    f"logsigjoin != logsig for single increment (max diff={np.abs(out - batch_result).max():.2e})"
print("[preflight] logsigjoin consistency check OK (matches batch logsig)")

# ---- iisignature methods ----
x = np.random.randn(8, 5, 2).astype(np.float64)
for method in ["O", "D", "S"]:
    try:
        s2 = iisignature.prepare(2, 5, method)
        y = iisignature.logsig(x, s2)
        print(f"[preflight] logsig method={method} OK shape={y.shape}")
    except Exception as e:
        print(f"[preflight] logsig method={method} FAILED: {e}")
        sys.exit(1)

# ---- Mini end-to-end pipeline (Phase 2 L1) ----
print("[preflight] Running mini Phase 2 pipeline...")
sys.path.insert(0, ".")
from src.phase2_core import (
    build_phase2_params, simulate_spread_paths,
    compute_logsig_pool, rollout_objective,
    build_time_augmented_path, feature_dim_from_level,
)
from src.policy import LinearPolicy

import yaml
with open("config/default.yaml") as f:
    cfg = yaml.safe_load(f)

params = build_phase2_params(cfg, "L1", 0.5)
B_H, Z = simulate_spread_paths(H=0.5, M=32, params=params, integrator="exact", seed=0)
paths_aug = build_time_augmented_path(Z, params.T, params.K)
logsig = compute_logsig_pool(paths_aug, 2)
print(f"[preflight]   simulate OK: Z={Z.shape}, logsig={logsig.shape}")

policy = LinearPolicy(feature_dim_from_level(2)).double()
J, aux = rollout_objective(Z=Z.double(), features=logsig.double(), policy=policy, params=params)
loss = -J.mean()
loss.backward()
grad_ok = any(p.grad is not None and p.grad.norm().item() > 0 for p in policy.parameters())
print(f"[preflight]   rollout OK: loss={loss.item():.6f}, grads={'OK' if grad_ok else 'ZERO'}")

print("[preflight] ALL CHECKS PASSED")
PY
