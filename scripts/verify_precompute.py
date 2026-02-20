"""Post-hoc verification of precomputed Phase 2 feature pools.

Checks all 75 = 3 levels x 5 H x 5 seeds directories for:
  - file existence and correct dtypes / shapes
  - no NaN or Inf values (sampled, not exhaustive)
  - spread paths centred near mu, not exploding
  - logsig features zero at t=0
  - train/test index integrity (no overlap, correct fractions)

Uses mmap throughout, so memory footprint stays under ~100 MB even
for M=131072 pools.  Safe to run on a login node.

Usage:
    python scripts/verify_precompute.py results/phase2/features/<run_id>
"""

from __future__ import annotations

import os
import sys

import numpy as np

# ---------- expected constants (from config/default.yaml) ----------
M = 131_072
K = 1000
LOGSIG_DIM_N4 = 8          # iisignature logsig dim for d=2, N=4
INTEGRATOR = {"L1": "exact", "L2": "milstein", "L3": "milstein"}
H_TAGS = [
    (0.125, "0p125"),
    (0.25,  "0p25"),
    (0.375, "0p375"),
    (0.5,   "0p5"),
    (0.75,  "0p75"),
]
N_SEEDS = 5
N_SAMPLE = 4096             # rows sampled for NaN/Inf spot-checks


def _sample_rows(n_total: int, n_sample: int, rng: np.random.Generator) -> np.ndarray:
    if n_sample >= n_total:
        return np.arange(n_total)
    return rng.choice(n_total, size=n_sample, replace=False)


def _check_dir(path: str, label: str, rng: np.random.Generator) -> list[str]:
    """Return list of error strings (empty = OK)."""
    errors: list[str] = []

    # --- spread pool ---
    zp = os.path.join(path, "pool_Z_fp32.npy")
    if not os.path.isfile(zp):
        return [f"{label}: pool_Z_fp32.npy missing"]
    Z = np.load(zp, mmap_mode="r")
    if Z.dtype != np.float32:
        errors.append(f"{label}: Z dtype {Z.dtype}")
    if Z.shape != (M, K + 1):
        errors.append(f"{label}: Z shape {Z.shape}")

    rows = _sample_rows(Z.shape[0], N_SAMPLE, rng)
    z_sample = np.array(Z[rows])
    if np.isnan(z_sample).any():
        errors.append(f"{label}: Z contains NaN")
    if np.isinf(z_sample).any():
        errors.append(f"{label}: Z contains Inf")

    zt = np.array(Z[:, -1])
    zm, zs = float(zt.mean()), float(zt.std())
    zlo, zhi = float(Z.min()), float(Z.max())
    if abs(zm) > 1.0:
        errors.append(f"{label}: Z terminal mean {zm:.4f}")
    if zs < 1e-4:
        errors.append(f"{label}: Z terminal std {zs:.6f}")
    if abs(zlo) > 50 or abs(zhi) > 50:
        errors.append(f"{label}: Z range [{zlo:.1f}, {zhi:.1f}]")

    print(f"  {label}: Z [{zlo:.3f}, {zhi:.3f}]  E[Z_T]={zm:.4f}  std={zs:.4f}")

    # --- logsig pool (N=4 only — largest, most likely to expose problems) ---
    lp = os.path.join(path, "pool_logsig_N4_fp32.npy")
    if not os.path.isfile(lp):
        errors.append(f"{label}: pool_logsig_N4_fp32.npy missing")
    else:
        L = np.load(lp, mmap_mode="r")
        if L.dtype != np.float32:
            errors.append(f"{label}: L dtype {L.dtype}")
        if L.shape != (M, K + 1, LOGSIG_DIM_N4):
            errors.append(f"{label}: L shape {L.shape}")

        l_sample = np.array(L[rows])
        if np.isnan(l_sample).any():
            errors.append(f"{label}: L contains NaN")
        if np.isinf(l_sample).any():
            errors.append(f"{label}: L contains Inf")

        # logsig at t=0 must be zero (no path history yet)
        l0 = np.array(L[:, 0, :])
        l0_max = float(np.abs(l0).max())
        if l0_max > 1e-6:
            errors.append(f"{label}: |L[:,0,:]| max = {l0_max:.2e}")

        print(f"  {label}: L {L.shape}  |L0|_max={l0_max:.2e}  "
              f"NaN={np.isnan(l_sample).any()}  Inf={np.isinf(l_sample).any()}")

    # --- train / test indices ---
    trp = os.path.join(path, "train_idx.npy")
    tep = os.path.join(path, "test_idx.npy")
    if not os.path.isfile(trp) or not os.path.isfile(tep):
        errors.append(f"{label}: index files missing")
    else:
        tr = np.load(trp)
        te = np.load(tep)
        n_overlap = np.intersect1d(tr, te).size
        if n_overlap:
            errors.append(f"{label}: train/test overlap ({n_overlap})")
        frac = len(tr) / M
        if not 0.70 < frac < 0.90:
            errors.append(f"{label}: train fraction {frac:.2%}")
        print(f"  {label}: train={len(tr)}  test={len(te)}  overlap={n_overlap}")

    return errors


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <features_root>")
        sys.exit(2)

    root = sys.argv[1]
    if not os.path.isdir(root):
        print(f"Not a directory: {root}")
        sys.exit(2)

    rng = np.random.default_rng(42)
    errors: list[str] = []
    n_checked = 0

    for level in ("L1", "L2", "L3"):
        integ = INTEGRATOR[level]
        ldir = os.path.join(root, f"level_{level}")
        if not os.path.isdir(ldir):
            print(f"  SKIP {level} (not found)")
            continue

        for h_val, h_tag in H_TAGS:
            for seed in range(N_SEEDS):
                d = os.path.join(ldir, f"H_{h_tag}", f"seed_{seed}", f"integrator_{integ}")
                label = f"{level}/H={h_val}/s{seed}"
                if not os.path.isdir(d):
                    errors.append(f"{label}: directory missing")
                    continue
                errors.extend(_check_dir(d, label, rng))
                n_checked += 1

    print(f"\n{'=' * 50}")
    print(f"Verified {n_checked} / 75 directories")
    if errors:
        print(f"{len(errors)} problem(s):")
        for e in errors:
            print(f"  * {e}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
