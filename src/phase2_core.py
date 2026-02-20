"""Core utilities for Phase 2 (3D pairs-trading control)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable

import iisignature
import numpy as np
import torch

from src.fbm import simulate_fbm
from src.signatures import compute_logsignatures, time_augment

LOGSIG_DIMS = {1: 2, 2: 3, 3: 5, 4: 8, 5: 14}


@dataclass(frozen=True)
class Phase2Params:
    # Volatility: sigma(z) = sigma_min + delta_sigma * tanh(beta * (z-mu)^2)
    sigma_min: float       # > 0
    delta_sigma: float     # >= 0 (0 = constant coefficients)
    beta: float            # > 0
    # Mean-reversion: kappa(z) = kappa0 + delta_kappa * (1-exp(-gamma_k*(z-mu)^2))
    kappa0: float          # > 0
    delta_kappa: float     # >= 0 (0 = constant coefficients)
    gamma_k: float         # > 0
    # Common
    mu: float
    # Control/simulation
    T: float
    K: int
    alpha: float
    phi: float
    eta: float
    c: float               # proportional transaction cost (bid-ask half-spread)
    v_bar: float
    Q0: float


def format_float_tag(x: float) -> str:
    """Filesystem-safe float tag, e.g. 0.25 -> 0p25."""
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def sigma_state(z: torch.Tensor, p: Phase2Params) -> torch.Tensor:
    """State-dependent volatility: sigma(z) = sigma_min + delta_sigma * tanh(beta * (z-mu)^2)."""
    return p.sigma_min + p.delta_sigma * torch.tanh(p.beta * (z - p.mu) ** 2)


def sigma_state_prime(z: torch.Tensor, p: Phase2Params) -> torch.Tensor:
    """Derivative d/dz sigma(z) = delta_sigma * 2*beta*(z-mu) * sech^2(beta*(z-mu)^2)."""
    u = z - p.mu
    th = torch.tanh(p.beta * u * u)
    return p.delta_sigma * 2.0 * p.beta * u * (1.0 - th * th)


def kappa_state(z: torch.Tensor, p: Phase2Params) -> torch.Tensor:
    """State-dependent mean-reversion: kappa(z) = kappa0 + delta_kappa*(1-exp(-gamma_k*(z-mu)^2))."""
    return p.kappa0 + p.delta_kappa * (1.0 - torch.exp(-p.gamma_k * (z - p.mu) ** 2))


def kappa_state_prime(z: torch.Tensor, p: Phase2Params) -> torch.Tensor:
    """Derivative d/dz kappa(z) = delta_kappa * 2*gamma_k*(z-mu) * exp(-gamma_k*(z-mu)^2)."""
    u = z - p.mu
    return p.delta_kappa * 2.0 * p.gamma_k * u * torch.exp(-p.gamma_k * u * u)


def sigma_bounds(p: Phase2Params) -> tuple[float, float]:
    """Return uniform lower/upper bounds for sigma(z).

    tanh(beta*(z-mu)^2) in [0, 1], so sigma in [sigma_min, sigma_min + delta_sigma].
    """
    return p.sigma_min, p.sigma_min + p.delta_sigma


def build_phase2_params(config: dict, level: str, H: float) -> Phase2Params:
    """Build parameter object from config, comparison level, and Hurst exponent.

    Levels:
        L1: constant coefficients (delta_sigma=0, delta_kappa=0)
        L2: fixed steepness (beta_ref, gamma_ref from config)
        L3: variance-matched steepness (beta(H), gamma(H) via phi(H))
    """
    from src.variance_matching import beta_matched, beta_ref_from_k, gamma_matched, gamma_ref_from_k

    p2 = config["phase2"]
    sp = p2["spread_params"]

    sigma_min = float(sp["sigma_min"])
    kappa0 = float(sp["kappa0"])
    mu = float(sp["mu"])
    delta_sigma_cfg = float(sp["delta_sigma"])
    delta_kappa_cfg = float(sp["delta_kappa"])
    k_sigma = float(sp["k_sigma"])
    k_kappa = float(sp["k_kappa"])

    level = level.strip().upper()
    if level not in {"L1", "L2", "L3"}:
        raise ValueError(f"Unknown level '{level}'. Must be L1, L2, or L3.")

    if level == "L1":
        # Constant coefficients — no state dependence
        beta_val = 1.0   # placeholder (unused when delta_sigma=0)
        gamma_val = 1.0   # placeholder (unused when delta_kappa=0)
        d_sigma = 0.0
        d_kappa = 0.0
    else:
        # Compute reference steepness from activation half-widths
        b_ref = beta_ref_from_k(k_sigma, sigma_min, kappa0)
        g_ref = gamma_ref_from_k(k_kappa, sigma_min, kappa0)

        if level == "L2":
            # Fixed steepness — same beta_ref, gamma_ref for all H
            beta_val = b_ref
            gamma_val = g_ref
        else:  # L3
            # Variance-matched: steepness scaled by phi(H)
            beta_val = beta_matched(H, b_ref, kappa0)
            gamma_val = gamma_matched(H, g_ref, kappa0)
        d_sigma = delta_sigma_cfg
        d_kappa = delta_kappa_cfg

    return Phase2Params(
        sigma_min=sigma_min,
        delta_sigma=d_sigma,
        beta=beta_val,
        kappa0=kappa0,
        delta_kappa=d_kappa,
        gamma_k=gamma_val,
        mu=mu,
        T=float(config["T"]),
        K=int(config["K"]),
        alpha=float(config["alpha"]),
        phi=float(config["phi"]),
        eta=float(config["eta"]),
        c=float(config.get("c", 0.0)),
        v_bar=float(config["v_bar"]),
        Q0=float(config["Q_0"]),
    )


def simulate_spread_paths_exact(
    *,
    H: float,
    M: int,
    params: Phase2Params,
    seed: int,
    fbm_method: str = "davies-harte",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convolution recursion for constant-coefficient fOU. Returns (B_H, Z)."""
    if params.delta_sigma != 0.0 or params.delta_kappa != 0.0:
        raise ValueError(
            "Exact fOU requires constant coefficients (delta_sigma=0, delta_kappa=0). "
            f"Got delta_sigma={params.delta_sigma}, delta_kappa={params.delta_kappa}."
        )

    B_H = simulate_fbm(
        H=H,
        T=params.T,
        K=params.K,
        M=M,
        device=torch.device("cpu"),
        seed=seed,
        method=fbm_method,
    )
    dB = (B_H[:, 1:] - B_H[:, :-1]).to(torch.float64)  # [M, K]
    dt = params.T / params.K
    decay = math.exp(-params.kappa0 * dt)

    S = torch.zeros(M, dtype=torch.float64)
    Z = torch.zeros((M, params.K + 1), dtype=torch.float64)
    Z[:, 0] = params.mu

    for k in range(params.K):
        S = decay * S + dB[:, k]
        Z[:, k + 1] = params.mu + params.sigma_min * S

    B_H32 = B_H.to(torch.float32)
    Z32 = Z.to(torch.float32)
    if not torch.isfinite(B_H32).all():
        raise FloatingPointError("Non-finite values detected in simulated B_H paths.")
    if not torch.isfinite(Z32).all():
        raise FloatingPointError("Non-finite values detected in simulated Z paths.")
    return B_H32, Z32


def simulate_spread_paths(
    *,
    H: float,
    M: int,
    params: Phase2Params,
    integrator: str,
    seed: int,
    fbm_method: str = "davies-harte",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate (B^H, Z) via exact/euler/milstein. Returns float32 [M, K+1] each."""
    integrator = integrator.strip().lower()
    if integrator not in {"euler", "milstein", "exact"}:
        raise ValueError(f"integrator must be 'euler', 'milstein', or 'exact', got '{integrator}'")

    if integrator == "exact":
        return simulate_spread_paths_exact(
            H=H, M=M, params=params, seed=seed, fbm_method=fbm_method,
        )

    B_H = simulate_fbm(
        H=H,
        T=params.T,
        K=params.K,
        M=M,
        device=torch.device("cpu"),
        seed=seed,
        method=fbm_method,
    )
    dB = (B_H[:, 1:] - B_H[:, :-1]).to(torch.float64)  # [M, K]
    dt = params.T / params.K

    Z = torch.zeros((M, params.K + 1), dtype=torch.float64)
    Z[:, 0] = params.mu

    for k in range(params.K):
        z_k = Z[:, k]
        sig_k = sigma_state(z_k, params)
        kap_k = kappa_state(z_k, params)
        drift = -kap_k * (z_k - params.mu) * dt
        diff = sig_k * dB[:, k]
        if integrator == "milstein":
            b2 = 0.5 * (dB[:, k] * dB[:, k])
            sigp_k = sigma_state_prime(z_k, params)
            corr = sigp_k * sig_k * b2
        else:
            corr = 0.0
        Z[:, k + 1] = z_k + drift + diff + corr

    B_H32 = B_H.to(torch.float32)
    Z32 = Z.to(torch.float32)
    if not torch.isfinite(B_H32).all():
        raise FloatingPointError("Non-finite values detected in simulated B_H paths.")
    if not torch.isfinite(Z32).all():
        raise FloatingPointError("Non-finite values detected in simulated Z paths.")
    return B_H32, Z32


def compute_logsig_pool(
    paths_aug: torch.Tensor,
    m_max: int,
    method: str = "O",
) -> torch.Tensor:
    """Prefix log-signatures via incremental logsigjoin (O(K) per path)."""
    M, K_plus_1, d = paths_aug.shape
    if d != 2:
        raise ValueError(f"Expected path dimension d=2, got d={d}")
    if m_max not in LOGSIG_DIMS:
        raise ValueError(f"Unsupported m_max={m_max}. Supported: {sorted(LOGSIG_DIMS)}")

    if not hasattr(iisignature, "logsigjoin"):
        return compute_logsignatures(paths_aug, m_max).to(dtype=torch.float32, device=torch.device("cpu"))

    paths_np = paths_aug.detach().cpu().numpy().astype(np.float64, copy=False)
    s = iisignature.prepare(d, m_max, method)
    dim = iisignature.logsiglength(d, m_max)
    logsigs = np.zeros((M, K_plus_1, dim), dtype=np.float64)
    ZZ = np.zeros((M, dim), dtype=np.float64)

    for k in range(K_plus_1 - 1):
        dX = paths_np[:, k + 1, :] - paths_np[:, k, :]
        ZZ = iisignature.logsigjoin(ZZ, dX, s).astype(np.float64, copy=False)
        logsigs[:, k + 1, :] = ZZ

    return torch.from_numpy(logsigs.astype(np.float32, copy=False))


def make_train_test_indices(M: int, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic train/test split indices."""
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be in (0,1), got {train_frac}")
    rng = np.random.default_rng(seed)
    idx = np.arange(M, dtype=np.int64)
    rng.shuffle(idx)
    n_train = int(math.floor(train_frac * M))
    train_idx = np.sort(idx[:n_train])
    test_idx = np.sort(idx[n_train:])
    return train_idx, test_idx


def ensure_no_overlap(train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """Raise ValueError if train/test indices overlap."""
    overlap = np.intersect1d(train_idx, test_idx)
    if overlap.size > 0:
        raise ValueError(f"train/test split overlap detected ({overlap.size} indices)")


def feature_dim_from_level(N: int) -> int:
    """Return expected logsig dimension for d=2."""
    if N not in LOGSIG_DIMS:
        raise ValueError(f"Unsupported N={N}. Supported: {sorted(LOGSIG_DIMS)}")
    return LOGSIG_DIMS[N]


def make_feature_dir(
    root: str,
    *,
    level: str,
    H: float,
    seed: int,
    integrator: str,
    run_id: str | None = None,
) -> str:
    """Build canonical directory path for precomputed features.

    If *run_id* is given the layout becomes:
        <root>/<run_id>/level_L1/H_0p5/seed_0/integrator_exact/
    Otherwise the legacy flat layout is used:
        <root>/level_L1/H_0p5/seed_0/integrator_exact/
    """
    parts = [root]
    if run_id:
        parts.append(run_id)
    parts += [
        f"level_{level}",
        f"H_{format_float_tag(H)}",
        f"seed_{seed}",
        f"integrator_{integrator}",
    ]
    return os.path.join(*parts)


def rollout_objective(
    *,
    Z: torch.Tensor,
    features: torch.Tensor,
    policy: torch.nn.Module,
    params: Phase2Params,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Roll out dynamics and return per-path J = W_T - alpha*Q_T^2 - phi*int(Q^2)."""
    M, K_plus_1 = Z.shape

    dt = params.T / params.K
    Q = torch.full((M,), fill_value=params.Q0, dtype=torch.float64, device=Z.device)
    W = torch.zeros(M, dtype=torch.float64, device=Z.device)
    running_q2 = torch.zeros(M, dtype=torch.float64, device=Z.device)
    turnover = torch.zeros(M, dtype=torch.float64, device=Z.device)

    for k in range(params.K):
        feat_k = features[:, k, :]
        raw_v = policy(feat_k)
        v_k = torch.clamp(raw_v, -params.v_bar, params.v_bar)
        dZ_k = Z[:, k + 1] - Z[:, k]

        running_q2 = running_q2 + Q * Q * dt
        turnover = turnover + torch.abs(v_k) * dt
        W = W + Q * dZ_k - params.eta * v_k * v_k * dt - params.c * torch.abs(v_k) * dt
        Q = Q + v_k * dt

    J_path = W - params.alpha * (Q * Q) - params.phi * running_q2
    if not torch.isfinite(J_path).all():
        raise FloatingPointError("Non-finite objective values detected during rollout.")
    aux = {"W_T": W, "Q_T": Q, "turnover": turnover}
    return J_path, aux


def rollout_baseline_rule(
    *,
    Z: np.ndarray,
    params: Phase2Params,
    control_rule: Callable[[int, np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Roll out baseline policy with rule v_k = f(k, Z_k, Q_k), vectorized over paths."""
    if Z.ndim != 2 or Z.shape[1] != params.K + 1:
        raise ValueError(f"Expected Z shape [M,{params.K + 1}], got {Z.shape}")

    M = Z.shape[0]
    dt = params.T / params.K
    Q = np.full(M, params.Q0, dtype=np.float64)
    W = np.zeros(M, dtype=np.float64)
    running_q2 = np.zeros(M, dtype=np.float64)
    turnover = np.zeros(M, dtype=np.float64)

    for k in range(params.K):
        v_k = control_rule(k, Z[:, k], Q)
        v_k = np.clip(v_k, -params.v_bar, params.v_bar)
        dZ_k = Z[:, k + 1] - Z[:, k]
        running_q2 += Q * Q * dt
        turnover += np.abs(v_k) * dt
        W += Q * dZ_k - params.eta * v_k * v_k * dt - params.c * np.abs(v_k) * dt
        Q += v_k * dt

    J = W - params.alpha * (Q * Q) - params.phi * running_q2
    aux = {"W_T": W, "Q_T": Q, "turnover": turnover}
    return J, aux


def save_npy(path: str, arr: np.ndarray) -> None:
    """Save array to .npy ensuring parent directory exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def build_time_augmented_path(paths: torch.Tensor, T: float, K: int) -> torch.Tensor:
    """Build (t, X_t) augmented path for logsignature computation."""
    return time_augment(paths, T, K)
