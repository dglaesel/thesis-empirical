"""Phase 2 precompute: fBm + spread simulation, log-signature extraction."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to import path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)

from src.phase2_core import (
    build_phase2_params,
    build_time_augmented_path,
    compute_logsig_pool,
    ensure_no_overlap,
    make_feature_dir,
    make_train_test_indices,
    save_npy,
    sigma_bounds,
    sigma_state,
    simulate_spread_paths,
)
from src.repro_manifest import append_manifest_line, get_code_version_id, get_python_version, utc_now_iso


def _dtype_from_name(name: str) -> np.dtype:
    if name.lower() == "float32":
        return np.dtype(np.float32)
    if name.lower() == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"Unsupported feature_dtype '{name}'. Use float32 or float64.")


def _dtype_tag(dtype: np.dtype) -> str:
    return "fp32" if dtype == np.dtype(np.float32) else "fp64"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 precompute pools.")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--level", type=str, required=True, choices=["L1", "L2", "L3"])
    p.add_argument("--H", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--integrator", type=str, default=None, choices=["euler", "milstein", "exact"])
    p.add_argument("--split-mode", type=str, default=None, choices=["index", "independent"])
    p.add_argument("--M", type=int, default=None, help="Override phase2.M for this run")
    p.add_argument("--K", type=int, default=None, help="Override top-level K for this run")
    p.add_argument("--run-id", type=str, default=None,
                    help="Run identifier for output directory (default: auto-generated timestamp).")
    p.add_argument("--force", action="store_true", help="Overwrite existing artifacts.")
    return p.parse_args()


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_integrator(p2: dict, level: str) -> str:
    """Return the default integrator for a given level from config."""
    if level == "L1":
        return str(p2.get("integrator_L1", "exact"))
    return str(p2.get("integrator_L2L3", "milstein"))


def _save_index_mode(
    feature_dir: str,
    logsigs_np: np.ndarray,
    Z_np: np.ndarray,
    *,
    split_seed: int,
    train_frac: float,
    force: bool,
    m_max: int,
    dtype_tag: str,
) -> None:
    pool_logsig_path = os.path.join(feature_dir, f"pool_logsig_N{m_max}_{dtype_tag}.npy")
    pool_z_path = os.path.join(feature_dir, "pool_Z_fp32.npy")
    train_idx_path = os.path.join(feature_dir, "train_idx.npy")
    test_idx_path = os.path.join(feature_dir, "test_idx.npy")

    if not force and any(os.path.exists(p) for p in [pool_logsig_path, pool_z_path, train_idx_path, test_idx_path]):
        raise FileExistsError(
            f"Artifacts already exist in {feature_dir}. Use --force to overwrite."
        )

    save_npy(pool_logsig_path, logsigs_np)
    save_npy(pool_z_path, Z_np.astype(np.float32, copy=False))

    train_idx, test_idx = make_train_test_indices(logsigs_np.shape[0], train_frac=train_frac, seed=split_seed)
    ensure_no_overlap(train_idx, test_idx)
    save_npy(train_idx_path, train_idx)
    save_npy(test_idx_path, test_idx)


def _save_independent_mode(
    feature_dir: str,
    *,
    logsigs_train: np.ndarray,
    Z_train: np.ndarray,
    logsigs_test: np.ndarray,
    Z_test: np.ndarray,
    force: bool,
    m_max: int,
    dtype_tag: str,
) -> None:
    paths = [
        os.path.join(feature_dir, f"train_pool_logsig_N{m_max}_{dtype_tag}.npy"),
        os.path.join(feature_dir, "train_pool_Z_fp32.npy"),
        os.path.join(feature_dir, f"test_pool_logsig_N{m_max}_{dtype_tag}.npy"),
        os.path.join(feature_dir, "test_pool_Z_fp32.npy"),
    ]
    if not force and any(os.path.exists(p) for p in paths):
        raise FileExistsError(f"Artifacts already exist in {feature_dir}. Use --force to overwrite.")

    save_npy(paths[0], logsigs_train)
    save_npy(paths[1], Z_train.astype(np.float32, copy=False))
    save_npy(paths[2], logsigs_test)
    save_npy(paths[3], Z_test.astype(np.float32, copy=False))


def _compute_dataset(
    *,
    config: dict,
    level: str,
    H: float,
    seed: int,
    integrator: str,
    M: int,
    m_max: int,
    iis_method: str,
    feature_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    params = build_phase2_params(config, level, H)
    B_H, Z = simulate_spread_paths(
        H=H,
        M=M,
        params=params,
        integrator=integrator,
        seed=seed,
        fbm_method="davies-harte",
    )
    lo, hi = sigma_bounds(params)
    sig_vals = sigma_state(Z.to(torch.float64), params)
    sig_min = float(sig_vals.min().item())
    sig_max = float(sig_vals.max().item())
    if sig_min < lo - 1e-8 or sig_max > hi + 1e-8:
        raise RuntimeError(
            f"sigma(z) bound check failed: [{sig_min:.6f},{sig_max:.6f}] outside [{lo:.6f},{hi:.6f}]"
        )

    # Features from (t, B^H_t), not (t, Z_t) — see Ch6 Remark ch6-driving-vs-state
    paths_aug = build_time_augmented_path(B_H, params.T, params.K)
    logsigs = compute_logsig_pool(paths_aug, m_max=m_max, method=iis_method)
    logsigs_np = logsigs.numpy().astype(feature_dtype, copy=False)
    z_np = Z.numpy().astype(np.float32, copy=False)
    if not np.isfinite(logsigs_np).all():
        raise FloatingPointError("Non-finite values in precomputed log-signatures.")
    if not np.isfinite(z_np).all():
        raise FloatingPointError("Non-finite values in precomputed spread paths.")
    return logsigs_np, z_np


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    p2 = config["phase2"]

    # Apply CLI overrides
    if args.K is not None:
        config["K"] = args.K

    split_mode = args.split_mode or p2["split_mode"]
    if split_mode not in {"index", "independent"}:
        raise ValueError(f"Invalid split_mode '{split_mode}'")

    level = args.level
    H = float(args.H)
    seed = int(args.seed)
    integrator = args.integrator or _default_integrator(p2, level)

    M = int(args.M) if args.M is not None else int(p2["M"])
    m_max = int(p2["m_max"])
    iis_method = str(p2["iisignature_method"])
    feature_dtype = _dtype_from_name(str(p2["feature_dtype"]))
    dtype_tag = _dtype_tag(feature_dtype)

    run_id = args.run_id
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    feature_root = str(p2["features_root"])
    feature_dir = make_feature_dir(
        feature_root,
        level=level,
        H=H,
        seed=seed,
        integrator=integrator,
        run_id=run_id,
    )
    Path(feature_dir).mkdir(parents=True, exist_ok=True)

    print(f"[phase2_precompute] run_id={run_id}")
    print(f"[phase2_precompute] level={level}, H={H}, seed={seed}, split_mode={split_mode}")
    print(f"[phase2_precompute] integrator={integrator}, M={M}, m_max={m_max}, dtype={feature_dtype.name}")

    if split_mode == "index":
        logsigs_np, Z_np = _compute_dataset(
            config=config,
            level=level,
            H=H,
            seed=seed,
            integrator=integrator,
            M=M,
            m_max=m_max,
            iis_method=iis_method,
            feature_dtype=feature_dtype,
        )
        split_seed = int(seed + int(p2.get("split_seed_offset", 12345)))
        _save_index_mode(
            feature_dir,
            logsigs_np,
            Z_np,
            split_seed=split_seed,
            train_frac=float(p2["train_frac"]),
            force=args.force,
            m_max=m_max,
            dtype_tag=dtype_tag,
        )
    else:
        logsigs_train, Z_train = _compute_dataset(
            config=config,
            level=level,
            H=H,
            seed=seed,
            integrator=integrator,
            M=M,
            m_max=m_max,
            iis_method=iis_method,
            feature_dtype=feature_dtype,
        )
        test_seed = int(seed + int(p2.get("split_seed_offset", 12345)))
        logsigs_test, Z_test = _compute_dataset(
            config=config,
            level=level,
            H=H,
            seed=test_seed,
            integrator=integrator,
            M=M,
            m_max=m_max,
            iis_method=iis_method,
            feature_dtype=feature_dtype,
        )
        _save_independent_mode(
            feature_dir,
            logsigs_train=logsigs_train,
            Z_train=Z_train,
            logsigs_test=logsigs_test,
            Z_test=Z_test,
            force=args.force,
            m_max=m_max,
            dtype_tag=dtype_tag,
        )

    manifest_path = str(p2["manifest_path"])
    append_manifest_line(
        manifest_path=manifest_path,
        created_at_utc=utc_now_iso(),
        phase="phase2_precompute",
        world=level,
        H=H,
        seed=seed,
        split_mode=split_mode,
        d=2,
        m_max=m_max,
        iisignature_method=iis_method,
        M=M,
        K=int(config["K"]),
        dtype=feature_dtype.name,
        python_version=get_python_version(),
        code_version_id=get_code_version_id(cwd=_project_root),
        config_path=args.config,
    )
    print(f"[phase2_precompute] done. Artifacts in: {feature_dir}")


if __name__ == "__main__":
    main()
