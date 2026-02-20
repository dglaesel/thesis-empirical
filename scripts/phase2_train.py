"""Phase 2 training on precomputed pools (fixed-feature pipeline)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to import path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)

from src.phase2_core import (
    LOGSIG_DIMS,
    build_phase2_params,
    feature_dim_from_level,
    format_float_tag,
    make_feature_dir,
    rollout_baseline_rule,
    rollout_objective,
)
from src.policy import DNNPolicy, LinearPolicy
from src.repro_manifest import append_manifest_line, get_code_version_id, get_python_version, utc_now_iso


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 train for a single (level,H,seed,N,arch).")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--level", type=str, required=True, choices=["L1", "L2", "L3"])
    p.add_argument("--H", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--N", type=int, required=True, choices=[1, 2, 3, 4])
    p.add_argument("--arch", type=str, required=True, choices=["alin", "adnn", "A_lin", "A_dnn"])
    p.add_argument("--integrator", type=str, default=None, choices=["euler", "milstein", "exact"])
    p.add_argument("--split-mode", type=str, default=None, choices=["index", "independent"])
    p.add_argument("--device", type=str, default="cpu", help="'cpu', 'cuda', or 'auto'")
    p.add_argument("--num-epochs", type=int, default=None, help="Optional override for phase2.num_epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Optional override for phase2.batch_size")
    p.add_argument("--eval-every", type=int, default=None, help="Optional override for phase2.eval_every")
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Optional override for phase2.eval_batch_size",
    )
    p.add_argument("--run-id", type=str, required=True,
                    help="Run identifier matching the precompute step (e.g. 20260217_190000).")
    p.add_argument("--eta", type=float, default=None, help="Override market impact parameter.")
    p.add_argument("--c", type=float, default=None, dest="prop_cost", help="Override proportional transaction cost.")
    p.add_argument("--force", action="store_true", help="Overwrite existing run directory contents.")
    return p.parse_args()


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _dtype_from_name(name: str) -> np.dtype:
    if name.lower() == "float32":
        return np.dtype(np.float32)
    if name.lower() == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"Unsupported feature_dtype '{name}'.")


def _dtype_tag(dtype: np.dtype) -> str:
    return "fp32" if dtype == np.dtype(np.float32) else "fp64"


def _load_memmap_checked(
    path: str,
    *,
    expected_ndim: int,
    expected_dtype: np.dtype,
    name: str,
) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name} file: {path}")
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.ndim != expected_ndim:
        raise ValueError(f"{name} ndim mismatch: expected {expected_ndim}, got {arr.ndim} for {path}")
    if arr.dtype != expected_dtype:
        raise ValueError(f"{name} dtype mismatch: expected {expected_dtype}, got {arr.dtype} for {path}")
    return arr


def _validate_feature_shape(
    feats: np.ndarray,
    *,
    K: int,
    m_max: int,
) -> None:
    expected_dim = feature_dim_from_level(m_max)
    if feats.shape[1] != K + 1:
        raise ValueError(f"Feature time dimension mismatch: expected {K + 1}, got {feats.shape[1]}")
    if feats.shape[2] != expected_dim:
        raise ValueError(
            f"Feature dim mismatch for m_max={m_max}: expected {expected_dim}, got {feats.shape[2]}"
        )


def _validate_z_shape(Z: np.ndarray, *, K: int) -> None:
    if Z.shape[1] != K + 1:
        raise ValueError(f"Spread path time dimension mismatch: expected {K + 1}, got {Z.shape[1]}")


def _build_policy(arch: str, dim_in: int) -> torch.nn.Module:
    if arch == "alin":
        return LinearPolicy(dim_in).to(dtype=torch.float64)
    if arch == "adnn":
        return DNNPolicy(dim_in, hidden=32).to(dtype=torch.float64)
    raise ValueError(f"Unknown arch '{arch}'")


def _canonical_arch(arch: str) -> str:
    a = arch.strip()
    if a in {"alin", "A_lin"}:
        return "alin"
    if a in {"adnn", "A_dnn"}:
        return "adnn"
    raise ValueError(f"Unknown arch '{arch}'")


def _eval_policy_on_indices(
    *,
    policy: torch.nn.Module,
    feats_mm: np.ndarray,
    z_mm: np.ndarray,
    indices: np.ndarray,
    dim_n: int,
    eval_batch_size: int,
    params,
    device: torch.device,
) -> dict[str, float]:
    total = 0
    sum_j = 0.0
    sum_w = 0.0
    sum_q2 = 0.0
    sum_turn = 0.0

    policy.eval()
    with torch.no_grad():
        for start in range(0, len(indices), eval_batch_size):
            sl = indices[start : start + eval_batch_size]
            feat = torch.tensor(np.ascontiguousarray(feats_mm[sl, :, :dim_n]),
                                dtype=torch.float64, device=device)
            z_t = torch.tensor(np.ascontiguousarray(z_mm[sl, :]),
                               dtype=torch.float64, device=device)
            j_path, aux = rollout_objective(Z=z_t, features=feat, policy=policy, params=params)

            b = len(sl)
            total += b
            sum_j += float(j_path.sum().item())
            sum_w += float(aux["W_T"].sum().item())
            sum_q2 += float((aux["Q_T"] * aux["Q_T"]).sum().item())
            sum_turn += float(aux["turnover"].sum().item())

    if total == 0:
        raise ValueError("Evaluation received empty index set.")
    return {
        "J": sum_j / total,
        "E_WT": sum_w / total,
        "E_QT2": sum_q2 / total,
        "turnover": sum_turn / total,
    }


def _eval_policy_on_arrays(
    *,
    policy: torch.nn.Module,
    feats_mm: np.ndarray,
    z_mm: np.ndarray,
    dim_n: int,
    eval_batch_size: int,
    params,
    device: torch.device,
) -> dict[str, float]:
    total = int(feats_mm.shape[0])
    if total != int(z_mm.shape[0]):
        raise ValueError(f"Feature/Spread row mismatch: {feats_mm.shape[0]} vs {z_mm.shape[0]}")
    if total == 0:
        raise ValueError("Evaluation received empty array set.")

    sum_j = 0.0
    sum_w = 0.0
    sum_q2 = 0.0
    sum_turn = 0.0

    policy.eval()
    with torch.no_grad():
        for start in range(0, total, eval_batch_size):
            sl = slice(start, min(start + eval_batch_size, total))
            feat = torch.tensor(np.ascontiguousarray(feats_mm[sl, :, :dim_n]),
                                dtype=torch.float64, device=device)
            z_t = torch.tensor(np.ascontiguousarray(z_mm[sl, :]),
                               dtype=torch.float64, device=device)
            j_path, aux = rollout_objective(Z=z_t, features=feat, policy=policy, params=params)

            b = int(feat.shape[0])
            sum_j += float(j_path.sum().item())
            sum_w += float(aux["W_T"].sum().item())
            sum_q2 += float((aux["Q_T"] * aux["Q_T"]).sum().item())
            sum_turn += float(aux["turnover"].sum().item())

    return {
        "J": sum_j / total,
        "E_WT": sum_w / total,
        "E_QT2": sum_q2 / total,
        "turnover": sum_turn / total,
    }


def _baseline_no_trade(Z: np.ndarray, params) -> dict[str, float]:
    def rule(_k: int, _z: np.ndarray, q: np.ndarray) -> np.ndarray:
        return np.zeros_like(q)

    j, aux = rollout_baseline_rule(Z=Z, params=params, control_rule=rule)
    return {
        "J": float(np.mean(j)),
        "E_WT": float(np.mean(aux["W_T"])),
        "E_QT2": float(np.mean(aux["Q_T"] ** 2)),
        "turnover": float(np.mean(aux["turnover"])),
    }


def _baseline_buyhold(Z: np.ndarray, params, q_target: float) -> dict[str, float]:
    dt = params.T / params.K

    def rule(_k: int, _z: np.ndarray, q: np.ndarray) -> np.ndarray:
        return (q_target - q) / dt

    j, aux = rollout_baseline_rule(Z=Z, params=params, control_rule=rule)
    return {
        "J": float(np.mean(j)),
        "E_WT": float(np.mean(aux["W_T"])),
        "E_QT2": float(np.mean(aux["Q_T"] ** 2)),
        "turnover": float(np.mean(aux["turnover"])),
    }


def _tune_ou_linear_baseline(
    Z_train: np.ndarray,
    Z_test: np.ndarray,
    params,
    cz_grid: list[float],
    cq_grid: list[float],
) -> dict:
    best = {"cz": None, "cq": None, "J_train": -np.inf}
    for cz in cz_grid:
        for cq in cq_grid:
            def rule(_k: int, z_k: np.ndarray, q: np.ndarray, _cz=cz, _cq=cq) -> np.ndarray:
                return -_cz * (z_k - params.mu) - _cq * q

            j_train, _ = rollout_baseline_rule(Z=Z_train, params=params, control_rule=rule)
            val = float(np.mean(j_train))
            if val > best["J_train"]:
                best.update({"cz": float(cz), "cq": float(cq), "J_train": val})

    cz = best["cz"]
    cq = best["cq"]

    def best_rule(_k: int, z_k: np.ndarray, q: np.ndarray) -> np.ndarray:
        return -cz * (z_k - params.mu) - cq * q

    j_test, aux = rollout_baseline_rule(Z=Z_test, params=params, control_rule=best_rule)
    best.update(
        {
            "J_test": float(np.mean(j_test)),
            "E_WT": float(np.mean(aux["W_T"])),
            "E_QT2": float(np.mean(aux["Q_T"] ** 2)),
            "turnover": float(np.mean(aux["turnover"])),
        }
    )
    return best


def _zscore_rollout(
    Z: np.ndarray,
    params,
    *,
    z_mean: float,
    z_std: float,
    entry: float,
    exit_: float,
    q_target: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if z_std <= 0:
        raise ValueError("z_std must be positive for z-score baseline.")
    M = Z.shape[0]
    dt = params.T / params.K
    Q = np.full(M, params.Q0, dtype=np.float64)
    W = np.zeros(M, dtype=np.float64)
    running_q2 = np.zeros(M, dtype=np.float64)
    turnover = np.zeros(M, dtype=np.float64)
    target = np.zeros(M, dtype=np.float64)

    for k in range(params.K):
        z_k = Z[:, k]
        score = (z_k - z_mean) / z_std
        enter = np.abs(score) >= entry
        leave = np.abs(score) <= exit_

        target[leave] = 0.0
        target[enter] = -np.sign(score[enter]) * q_target

        v = np.clip((target - Q) / dt, -params.v_bar, params.v_bar)
        dZ = Z[:, k + 1] - Z[:, k]
        running_q2 += Q * Q * dt
        turnover += np.abs(v) * dt
        W += Q * dZ - params.eta * v * v * dt - params.c * np.abs(v) * dt
        Q += v * dt

    J = W - params.alpha * (Q * Q) - params.phi * running_q2
    aux = {"W_T": W, "Q_T": Q, "turnover": turnover}
    return J, aux


def _tune_zscore_baseline(
    Z_train: np.ndarray,
    Z_test: np.ndarray,
    params,
    *,
    entry_grid: list[float],
    exit_grid: list[float],
    q_target: float,
) -> dict:
    z_mean = float(np.mean(Z_train))
    z_std = float(np.std(Z_train) + 1e-8)
    best = {"entry": None, "exit": None, "J_train": -np.inf}

    for entry in entry_grid:
        for exit_ in exit_grid:
            if exit_ >= entry:
                continue
            j_train, _ = _zscore_rollout(
                Z_train,
                params,
                z_mean=z_mean,
                z_std=z_std,
                entry=float(entry),
                exit_=float(exit_),
                q_target=q_target,
            )
            val = float(np.mean(j_train))
            if val > best["J_train"]:
                best.update({"entry": float(entry), "exit": float(exit_), "J_train": val})

    entry = best["entry"]
    exit_ = best["exit"]
    if entry is None or exit_ is None:
        raise RuntimeError("No valid z-score baseline configuration found.")

    j_test, aux = _zscore_rollout(
        Z_test,
        params,
        z_mean=z_mean,
        z_std=z_std,
        entry=entry,
        exit_=exit_,
        q_target=q_target,
    )
    best.update(
        {
            "J_test": float(np.mean(j_test)),
            "E_WT": float(np.mean(aux["W_T"])),
            "E_QT2": float(np.mean(aux["Q_T"] ** 2)),
            "turnover": float(np.mean(aux["turnover"])),
            "z_mean": z_mean,
            "z_std": z_std,
        }
    )
    return best


def _load_data_views(
    *,
    feature_dir: str,
    split_mode: str,
    m_max: int,
    feature_dtype: np.dtype,
    K: int,
) -> dict:
    tag = _dtype_tag(feature_dtype)
    if split_mode == "index":
        feats_mm = _load_memmap_checked(
            os.path.join(feature_dir, f"pool_logsig_N{m_max}_{tag}.npy"),
            expected_ndim=3,
            expected_dtype=feature_dtype,
            name="feature pool",
        )
        z_mm = _load_memmap_checked(
            os.path.join(feature_dir, "pool_Z_fp32.npy"),
            expected_ndim=2,
            expected_dtype=np.dtype(np.float32),
            name="spread pool",
        )
        _validate_feature_shape(feats_mm, K=K, m_max=m_max)
        _validate_z_shape(z_mm, K=K)

        train_idx_path = os.path.join(feature_dir, "train_idx.npy")
        test_idx_path = os.path.join(feature_dir, "test_idx.npy")
        if not os.path.exists(train_idx_path) or not os.path.exists(test_idx_path):
            raise FileNotFoundError(
                f"Missing train/test index files for split_mode=index in {feature_dir}"
            )
        train_idx = np.load(train_idx_path, allow_pickle=False).astype(np.int64, copy=False)
        test_idx = np.load(test_idx_path, allow_pickle=False).astype(np.int64, copy=False)
        if np.intersect1d(train_idx, test_idx).size:
            raise ValueError("train/test index overlap detected.")
        return {
            "mode": "index",
            "pool_feats": feats_mm,
            "pool_z": z_mm,
            "train_idx": train_idx,
            "test_idx": test_idx,
        }

    if split_mode == "independent":
        train_feats = _load_memmap_checked(
            os.path.join(feature_dir, f"train_pool_logsig_N{m_max}_{tag}.npy"),
            expected_ndim=3,
            expected_dtype=feature_dtype,
            name="train feature pool",
        )
        train_z = _load_memmap_checked(
            os.path.join(feature_dir, "train_pool_Z_fp32.npy"),
            expected_ndim=2,
            expected_dtype=np.dtype(np.float32),
            name="train spread pool",
        )
        test_feats = _load_memmap_checked(
            os.path.join(feature_dir, f"test_pool_logsig_N{m_max}_{tag}.npy"),
            expected_ndim=3,
            expected_dtype=feature_dtype,
            name="test feature pool",
        )
        test_z = _load_memmap_checked(
            os.path.join(feature_dir, "test_pool_Z_fp32.npy"),
            expected_ndim=2,
            expected_dtype=np.dtype(np.float32),
            name="test spread pool",
        )
        _validate_feature_shape(train_feats, K=K, m_max=m_max)
        _validate_feature_shape(test_feats, K=K, m_max=m_max)
        _validate_z_shape(train_z, K=K)
        _validate_z_shape(test_z, K=K)

        return {
            "mode": "independent",
            "train_feats": train_feats,
            "train_z": train_z,
            "test_feats": test_feats,
            "test_z": test_z,
        }

    raise ValueError(f"Unknown split_mode '{split_mode}'")


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    p2 = config["phase2"]

    level = args.level
    H = float(args.H)
    seed = int(args.seed)
    N = int(args.N)
    arch = _canonical_arch(args.arch)
    split_mode = args.split_mode or str(p2["split_mode"])
    if args.integrator is not None:
        integrator = args.integrator
    elif level == "L1":
        integrator = str(p2.get("integrator_L1", "exact"))
    else:
        integrator = str(p2.get("integrator_L2L3", "milstein"))
    m_max = int(p2["m_max"])
    if N > m_max:
        raise ValueError(f"N={N} cannot exceed m_max={m_max}.")

    feature_dtype = _dtype_from_name(str(p2["feature_dtype"]))
    params = build_phase2_params(config, level, H)
    # CLI overrides for objective parameters (Riccati comparison uses --eta 0.001 --c 0)
    import dataclasses
    overrides = {}
    if args.eta is not None:
        overrides["eta"] = args.eta
    if args.prop_cost is not None:
        overrides["c"] = args.prop_cost
    if overrides:
        params = dataclasses.replace(params, **overrides)
        print(f"[phase2_train] parameter overrides: {overrides}")
    device = _resolve_device(args.device)

    run_id = args.run_id
    feature_dir = make_feature_dir(
        str(p2["features_root"]),
        level=level,
        H=H,
        seed=seed,
        integrator=integrator,
        run_id=run_id,
    )
    if not os.path.isdir(feature_dir):
        raise FileNotFoundError(
            f"Feature directory not found: {feature_dir}. Run phase2_precompute first."
        )

    data = _load_data_views(
        feature_dir=feature_dir,
        split_mode=split_mode,
        m_max=m_max,
        feature_dtype=feature_dtype,
        K=params.K,
    )

    # --- Split training pool into train_sub / val_sub to prevent test leakage ---
    val_frac = float(p2.get("val_frac_of_train", 0.2))
    if data["mode"] == "index":
        _all_train = data["train_idx"]
    else:
        _all_train = np.arange(int(data["train_feats"].shape[0]))
    _val_rng = np.random.default_rng(seed + 999)  # deterministic, distinct from training rng
    _perm = _val_rng.permutation(len(_all_train))
    _n_val = max(1, int(round(val_frac * len(_all_train))))
    if data["mode"] == "index":
        val_idx = np.sort(_all_train[_perm[:_n_val]])
        train_sub_idx = np.sort(_all_train[_perm[_n_val:]])
    else:
        val_idx = np.sort(_perm[:_n_val])
        train_sub_idx = np.sort(_perm[_n_val:])
    _n_test = len(data["test_idx"]) if data["mode"] == "index" else int(data["test_feats"].shape[0])
    print(f"[phase2_train] train/val/test sizes: {len(train_sub_idx)}/{len(val_idx)}/{_n_test}")
    print("[phase2_train] loaded via memmap")

    dim_n = LOGSIG_DIMS[N]
    lr = float(p2["learning_rate"])
    grad_clip = float(p2["grad_clip"])
    num_epochs = int(args.num_epochs) if args.num_epochs is not None else int(p2["num_epochs"])
    batch_size = int(args.batch_size) if args.batch_size is not None else int(p2["batch_size"])
    eval_every = int(args.eval_every) if args.eval_every is not None else int(p2["eval_every"])
    eval_batch_size = int(args.eval_batch_size) if args.eval_batch_size is not None else int(p2["eval_batch_size"])

    # Warm restart config (DNN only — linear policy is convex, no restarts needed)
    n_restarts = int(p2.get("warm_restarts", 3)) if arch == "adnn" else 1
    epochs_per_audition = int(p2.get("epochs_per_audition", 30))

    runs_root = str(p2["runs_root"])
    run_dir = os.path.join(
        runs_root,
        run_id,
        f"level_{level}",
        f"H_{format_float_tag(H)}",
        f"seed_{seed}",
        f"integrator_{integrator}",
        f"N_{N}",
        f"arch_{arch}",
    )
    if os.path.isdir(run_dir) and not args.force:
        # Allow re-run into existing directory only if empty-ish.
        existing = [x for x in os.listdir(run_dir) if not x.startswith(".")]
        if existing:
            raise FileExistsError(f"Run directory already contains files: {run_dir}. Use --force.")
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    curve_path = os.path.join(run_dir, "training_curve.csv")
    t_start = time.time()
    curve_rows: list[tuple[int, float, float]] = []
    best_val_j = -float("inf")
    best_state = None

    def _val_j_now(policy_: torch.nn.Module) -> float:
        """Evaluate current policy on validation set."""
        if data["mode"] == "index":
            vm = _eval_policy_on_indices(
                policy=policy_, feats_mm=data["pool_feats"], z_mm=data["pool_z"],
                indices=val_idx, dim_n=dim_n, eval_batch_size=eval_batch_size,
                params=params, device=device)
        else:
            vm = _eval_policy_on_indices(
                policy=policy_, feats_mm=data["train_feats"], z_mm=data["train_z"],
                indices=val_idx, dim_n=dim_n, eval_batch_size=eval_batch_size,
                params=params, device=device)
        return float(vm["J"])

    global_epoch = 0  # tracks epoch counter across restarts for curve CSV

    for restart in range(n_restarts):
        # Fresh random init for each restart
        restart_seed = seed + restart * 1000
        rng = np.random.default_rng(restart_seed)
        torch.manual_seed(restart_seed)

        policy = _build_policy(arch, dim_n).to(device=device, dtype=torch.float64)
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            factor=float(p2.get("lr_factor", 0.5)),
            patience=int(p2.get("lr_patience", 15)),
            cooldown=int(p2.get("lr_cooldown", 5)),
            min_lr=float(p2.get("lr_min", 1e-5)),
        )

        is_final = (restart == n_restarts - 1)
        run_epochs = num_epochs if is_final else epochs_per_audition

        if n_restarts > 1:
            if is_final:
                print(f"[phase2_train] restart {restart+1}/{n_restarts} (FINAL — full {num_epochs} epochs)")
            else:
                print(f"[phase2_train] restart {restart+1}/{n_restarts} (audition — {run_epochs} epochs)")

        for ep in range(1, run_epochs + 1):
            global_epoch += 1

            # If this is the final restart and we're past the audition phase,
            # load the best model found across all restarts and continue.
            if is_final and ep == 1 and best_state is not None:
                policy.load_state_dict(best_state)
                # Reset optimizer and scheduler for the full run
                optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max",
                    factor=float(p2.get("lr_factor", 0.5)),
                    patience=int(p2.get("lr_patience", 15)),
                    cooldown=int(p2.get("lr_cooldown", 5)),
                    min_lr=float(p2.get("lr_min", 1e-5)),
                )
                print(f"[phase2_train] loaded best restart init (val_J={best_val_j:.6f})")

            # --- Training step: sample batch from train_sub (NOT full train pool) ---
            batch_idx = rng.choice(train_sub_idx, size=min(batch_size, len(train_sub_idx)), replace=False)
            if data["mode"] == "index":
                feat_src, z_src = data["pool_feats"], data["pool_z"]
            else:
                feat_src, z_src = data["train_feats"], data["train_z"]
            feat = torch.tensor(np.ascontiguousarray(feat_src[batch_idx, :, :dim_n]),
                                dtype=torch.float64, device=device)
            z_t = torch.tensor(np.ascontiguousarray(z_src[batch_idx, :]),
                               dtype=torch.float64, device=device)

            policy.train()
            optimizer.zero_grad(set_to_none=True)
            j_path, _ = rollout_objective(Z=z_t, features=feat, policy=policy, params=params)
            loss = -j_path.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Encountered non-finite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()

            train_j = float(j_path.mean().item())
            val_j = float("nan")
            if ep == 1 or ep % eval_every == 0 or ep == run_epochs:
                val_j = _val_j_now(policy)
                scheduler.step(val_j)
                if val_j > best_val_j:
                    best_val_j = val_j
                    best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

            curve_rows.append((global_epoch, train_j, val_j))
            if ep == 1 or ep % eval_every == 0 or ep == run_epochs:
                print(f"[phase2_train] epoch={global_epoch} (r{restart+1} e{ep}/{run_epochs}) "
                      f"train_J={train_j:.6f} val_J={val_j:.6f}")

    with open(curve_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_J", "val_J"])
        w.writerows(curve_rows)

    # Restore best model
    if best_state is not None:
        policy.load_state_dict(best_state)

    # Final metrics — evaluate on train_sub, val, and held-out test separately
    if data["mode"] == "index":
        train_metrics = _eval_policy_on_indices(
            policy=policy,
            feats_mm=data["pool_feats"],
            z_mm=data["pool_z"],
            indices=train_sub_idx,
            dim_n=dim_n,
            eval_batch_size=eval_batch_size,
            params=params,
            device=device,
        )
        val_metrics_final = _eval_policy_on_indices(
            policy=policy,
            feats_mm=data["pool_feats"],
            z_mm=data["pool_z"],
            indices=val_idx,
            dim_n=dim_n,
            eval_batch_size=eval_batch_size,
            params=params,
            device=device,
        )
        test_metrics = _eval_policy_on_indices(
            policy=policy,
            feats_mm=data["pool_feats"],
            z_mm=data["pool_z"],
            indices=data["test_idx"],
            dim_n=dim_n,
            eval_batch_size=eval_batch_size,
            params=params,
            device=device,
        )
        z_train = np.asarray(data["pool_z"][train_sub_idx, :], dtype=np.float64)
        z_test = np.asarray(data["pool_z"][data["test_idx"], :], dtype=np.float64)
        manifest_M = int(data["pool_feats"].shape[0])
    else:
        train_metrics = _eval_policy_on_indices(
            policy=policy,
            feats_mm=data["train_feats"],
            z_mm=data["train_z"],
            indices=train_sub_idx,
            dim_n=dim_n,
            eval_batch_size=eval_batch_size,
            params=params,
            device=device,
        )
        val_metrics_final = _eval_policy_on_indices(
            policy=policy,
            feats_mm=data["train_feats"],
            z_mm=data["train_z"],
            indices=val_idx,
            dim_n=dim_n,
            eval_batch_size=eval_batch_size,
            params=params,
            device=device,
        )
        test_metrics = _eval_policy_on_arrays(
            policy=policy,
            feats_mm=data["test_feats"],
            z_mm=data["test_z"],
            dim_n=dim_n,
            eval_batch_size=eval_batch_size,
            params=params,
            device=device,
        )
        z_train = np.asarray(data["train_z"][train_sub_idx, :], dtype=np.float64)
        z_test = np.asarray(data["test_z"], dtype=np.float64)
        manifest_M = int(data["train_feats"].shape[0] + data["test_feats"].shape[0])

    # Baselines (evaluate on full train/test spreads as numpy)
    base_cfg = p2["baseline"]
    q_target = float(base_cfg["q_target"])

    baseline = {
        "NoTrade_test": _baseline_no_trade(z_test, params),
        "BuyHold_test": _baseline_buyhold(z_test, params, q_target=q_target),
        "OU_linear": _tune_ou_linear_baseline(
            z_train,
            z_test,
            params,
            cz_grid=[float(x) for x in base_cfg["ou_cz_grid"]],
            cq_grid=[float(x) for x in base_cfg["ou_cq_grid"]],
        ),
        "ZScore": _tune_zscore_baseline(
            z_train,
            z_test,
            params,
            entry_grid=[float(x) for x in base_cfg["zscore_entry_grid"]],
            exit_grid=[float(x) for x in base_cfg["zscore_exit_grid"]],
            q_target=q_target,
        ),
    }

    # HJB Riccati baseline — only valid for H=0.5 with constant coefficients (L1) and eta>0
    if abs(H - 0.5) < 1e-9 and level == "L1" and params.eta > 1e-12:
        try:
            from src.hjb_pairs import solve_hjb_riccati, optimal_control
            sol = solve_hjb_riccati(
                T=params.T, K=params.K, kappa0=params.kappa0, mu=params.mu,
                sigma_min=params.sigma_min, alpha=params.alpha, phi=params.phi, eta=params.eta,
            )
            # Evaluate HJB policy on test paths
            dt = params.T / params.K
            M_test = z_test.shape[0]
            Q_hjb = np.full(M_test, params.Q0, dtype=np.float64)
            W_hjb = np.zeros(M_test, dtype=np.float64)
            rq2_hjb = np.zeros(M_test, dtype=np.float64)
            turn_hjb = np.zeros(M_test, dtype=np.float64)
            for k in range(params.K):
                v_k = optimal_control(k, z_test[:, k], Q_hjb, coeffs=sol["coeffs"], eta=params.eta)
                v_k = np.clip(v_k, -params.v_bar, params.v_bar)
                dZ_k = z_test[:, k + 1] - z_test[:, k]
                rq2_hjb += Q_hjb * Q_hjb * dt
                turn_hjb += np.abs(v_k) * dt
                W_hjb += Q_hjb * dZ_k - params.eta * v_k * v_k * dt - params.c * np.abs(v_k) * dt
                Q_hjb += v_k * dt
            J_hjb = W_hjb - params.alpha * (Q_hjb * Q_hjb) - params.phi * rq2_hjb
            baseline["HJB_Riccati"] = {
                "J_test": float(np.mean(J_hjb)),
                "E_WT": float(np.mean(W_hjb)),
                "E_QT2": float(np.mean(Q_hjb ** 2)),
                "turnover": float(np.mean(turn_hjb)),
            }
            print(f"[phase2_train] HJB Riccati baseline J_test={baseline['HJB_Riccati']['J_test']:.6f}")
        except Exception as e:
            print(f"[phase2_train] WARNING: HJB Riccati baseline failed: {e}")

    model_path = os.path.join(run_dir, "policy.pt")
    torch.save(policy.state_dict(), model_path)

    elapsed = time.time() - t_start
    summary = {
        "level": level,
        "H": H,
        "seed": seed,
        "N": N,
        "arch": arch,
        "split_mode": split_mode,
        "integrator": integrator,
        "device": str(device),
        "feature_dir": feature_dir,
        "run_dir": run_dir,
        "n_restarts": n_restarts,
        "epochs_per_audition": epochs_per_audition if n_restarts > 1 else None,
        "total_epochs": global_epoch,
        "n_train": len(train_sub_idx),
        "n_val": len(val_idx),
        "n_test": _n_test,
        "best_val_J": best_val_j,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics_final,
        "test_metrics": test_metrics,
        "baselines": baseline,
        "elapsed_seconds": elapsed,
    }
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    manifest_path = str(p2["manifest_path"])
    append_manifest_line(
        manifest_path=manifest_path,
        created_at_utc=utc_now_iso(),
        phase=f"phase2_train_{arch}_N{N}_{integrator}",
        world=level,
        H=H,
        seed=seed,
        split_mode=split_mode,
        d=2,
        m_max=m_max,
        iisignature_method=str(p2["iisignature_method"]),
        M=manifest_M,
        K=int(config["K"]),
        dtype=feature_dtype.name,
        python_version=get_python_version(),
        code_version_id=get_code_version_id(cwd=_project_root),
        config_path=args.config,
    )

    print(f"[phase2_train] done. val_J={val_metrics_final['J']:.6f} test_J={test_metrics['J']:.6f}, "
          f"restarts={n_restarts}, total_epochs={global_epoch}, elapsed={elapsed:.1f}s")
    print(f"[phase2_train] outputs: {run_dir}")


if __name__ == "__main__":
    main()
