"""Sanity check for Phase 2 precomputed pools.

Run ON THE CLUSTER:
    python sanity_check_precompute.py results/phase2/features/20260220_101535

Or locally on downloaded samples:
    python sanity_check_precompute.py D:/thesis_data/spot_check
"""

import sys
import os
import numpy as np

def check_seed_dir(d, level, H, seed):
    """Check one (level, H, seed) directory."""
    tag = f"{level} H={H} seed={seed}"
    errors = []

    # --- pool_Z ---
    z_path = os.path.join(d, "pool_Z_fp32.npy")
    if not os.path.exists(z_path):
        return [f"{tag}: pool_Z_fp32.npy MISSING"]
    Z = np.load(z_path, mmap_mode="r")

    if Z.dtype != np.float32:
        errors.append(f"{tag}: Z dtype={Z.dtype}, expected float32")
    M, K1 = Z.shape
    if M != 131072:
        errors.append(f"{tag}: Z M={M}, expected 131072")
    if K1 < 2:
        errors.append(f"{tag}: Z K+1={K1}, too small")
    if np.isnan(Z).any():
        errors.append(f"{tag}: Z has NaN!")
    if np.isinf(Z).any():
        errors.append(f"{tag}: Z has Inf!")

    z_mean = float(Z[:, -1].mean())
    z_std = float(Z[:, -1].std())
    z_min = float(Z.min())
    z_max = float(Z.max())

    # Spread should be roughly centered around mu=0, not exploding
    if abs(z_mean) > 1.0:
        errors.append(f"{tag}: Z terminal mean={z_mean:.4f}, suspiciously far from 0")
    if z_std < 0.001:
        errors.append(f"{tag}: Z terminal std={z_std:.6f}, suspiciously small")
    if abs(z_min) > 50 or abs(z_max) > 50:
        errors.append(f"{tag}: Z range=[{z_min:.2f}, {z_max:.2f}], suspiciously wide")

    print(f"  {tag}: Z {Z.shape} range=[{z_min:.3f},{z_max:.3f}] "
          f"mean_ZT={z_mean:.4f} std_ZT={z_std:.4f}")

    # --- pool_logsig_N4 ---
    l_path = os.path.join(d, "pool_logsig_N4_fp32.npy")
    if not os.path.exists(l_path):
        errors.append(f"{tag}: pool_logsig_N4_fp32.npy MISSING")
    else:
        L = np.load(l_path, mmap_mode="r")
        if L.dtype != np.float32:
            errors.append(f"{tag}: L dtype={L.dtype}, expected float32")
        if L.shape[0] != M:
            errors.append(f"{tag}: L M={L.shape[0]}, expected {M}")
        if L.shape[1] != K1:
            errors.append(f"{tag}: L K+1={L.shape[1]}, expected {K1}")
        if L.shape[2] != 8:
            errors.append(f"{tag}: L dim={L.shape[2]}, expected 8")
        if np.isnan(L).any():
            errors.append(f"{tag}: L has NaN!")
        if np.isinf(L).any():
            errors.append(f"{tag}: L has Inf!")

        # Initial logsig should be zero (no path information at t=0)
        l0_max = float(np.abs(L[:, 0, :]).max())
        if l0_max > 1e-6:
            errors.append(f"{tag}: L[:,0,:] max abs={l0_max:.6e}, expected ~0")

        print(f"  {tag}: L {L.shape} NaN={np.isnan(L).any()} Inf={np.isinf(L).any()} "
              f"|L0|_max={l0_max:.2e}")

    # --- train/test indices ---
    tr_path = os.path.join(d, "train_idx.npy")
    te_path = os.path.join(d, "test_idx.npy")
    if not os.path.exists(tr_path):
        errors.append(f"{tag}: train_idx.npy MISSING")
    if not os.path.exists(te_path):
        errors.append(f"{tag}: test_idx.npy MISSING")
    if os.path.exists(tr_path) and os.path.exists(te_path):
        tr = np.load(tr_path)
        te = np.load(te_path)
        overlap = np.intersect1d(tr, te).size
        total = len(tr) + len(te)
        print(f"  {tag}: train={len(tr)} test={len(te)} total={total} overlap={overlap}")
        if overlap > 0:
            errors.append(f"{tag}: train/test OVERLAP of {overlap} indices!")
        if total > 131072:
            errors.append(f"{tag}: train+test={total} > M=131072")
        # Expect ~80% train, 20% test
        train_frac = len(tr) / 131072
        if train_frac < 0.7 or train_frac > 0.9:
            errors.append(f"{tag}: train fraction={train_frac:.2%}, expected ~80%")

    return errors


def main():
    if len(sys.argv) < 2:
        print("Usage: python sanity_check_precompute.py <features_root>")
        print("  e.g. python sanity_check_precompute.py results/phase2/features/20260220_101535")
        sys.exit(1)

    root = sys.argv[1]
    if not os.path.isdir(root):
        print(f"ERROR: directory not found: {root}")
        sys.exit(1)

    integrator_map = {"L1": "exact", "L2": "milstein", "L3": "milstein"}
    H_tags = {"0.125": "0p125", "0.25": "0p25", "0.375": "0p375", "0.5": "0p5", "0.75": "0p75"}

    all_errors = []
    checked = 0

    for level in ["L1", "L2", "L3"]:
        integ = integrator_map[level]
        level_dir = os.path.join(root, f"level_{level}")
        if not os.path.isdir(level_dir):
            print(f"SKIP: {level_dir} not found")
            continue

        for H_val, H_tag in sorted(H_tags.items()):
            for seed in range(5):
                d = os.path.join(level_dir, f"H_{H_tag}", f"seed_{seed}", f"integrator_{integ}")
                if not os.path.isdir(d):
                    all_errors.append(f"{level} H={H_val} seed={seed}: directory MISSING: {d}")
                    continue
                errs = check_seed_dir(d, level, H_val, seed)
                all_errors.extend(errs)
                checked += 1

    print(f"\n{'='*60}")
    print(f"Checked {checked}/75 directories")
    if all_errors:
        print(f"ERRORS ({len(all_errors)}):")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
