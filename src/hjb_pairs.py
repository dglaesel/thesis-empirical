"""HJB benchmark for pairs trading at H=0.5 (classical OU).

Value function V(t,Z,Q) = a(t)Z^2 + b(t)Q^2 + c(t)ZQ + d(t)Z + e(t)Q + f(t).

HJB PDE (with wealth Q*dZ term included):
  V_t + max_v { -kappa0*(Z-mu)*V_Z + v*V_Q + 0.5*sigma^2*V_ZZ
                + Q*(-kappa0*(Z-mu)) - eta*v^2 - phi*Q^2 } = 0

Optimal control: v* = V_Q / (2*eta).
Terminal: a(T)=0, b(T)=-alpha, c(T)=d(T)=e(T)=f(T)=0.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def hjb_riccati_rhs(
    t: float,
    y: np.ndarray,
    *,
    kappa0: float,
    sigma_min: float,
    mu: float,
    alpha: float,
    phi: float,
    eta: float,
) -> np.ndarray:
    """RHS of the 6-dim Riccati ODE. State y = [a, b, c, d, e, f].

    Derived by substituting V = aZ^2 + bQ^2 + cZQ + dZ + eQ + f into
    the HJB (including the Q*(-kappa0*(Z-mu)) wealth drift) and matching
    polynomial coefficients. See Chapter 7, Proposition 7.7.1.
    """
    a, b, c, d, e, f = y
    s2 = sigma_min ** 2

    da = 2.0 * kappa0 * a - c * c / (4.0 * eta)
    db = -b * b / eta + phi
    dc = kappa0 * c - b * c / eta + kappa0
    dd = kappa0 * d - 2.0 * kappa0 * mu * a - c * e / (2.0 * eta)
    de_ = -kappa0 * mu * c - b * e / eta - kappa0 * mu
    df = -kappa0 * mu * d - e * e / (4.0 * eta) - s2 * a

    return np.array([da, db, dc, dd, de_, df])


def solve_hjb_riccati(
    *,
    T: float,
    kappa0: float,
    sigma_min: float,
    mu: float,
    alpha: float,
    phi: float,
    eta: float,
    K: int = 10000,
) -> dict:
    """Solve the Riccati ODE backward from T to 0."""
    # Terminal conditions at t=T
    y_T = np.array([0.0, -alpha, 0.0, 0.0, 0.0, 0.0])

    def backward_rhs(s: float, y: np.ndarray) -> np.ndarray:
        return -hjb_riccati_rhs(
            T - s, y,
            kappa0=kappa0, sigma_min=sigma_min, mu=mu,
            alpha=alpha, phi=phi, eta=eta,
        )

    sol = solve_ivp(
        backward_rhs,
        [0.0, T],
        y_T,
        method="RK45",
        dense_output=True,
        rtol=1e-10,
        atol=1e-12,
        max_step=T / K,
    )
    if not sol.success:
        raise RuntimeError(f"Riccati ODE integration failed: {sol.message}")

    t_grid = np.linspace(0.0, T, K + 1)
    s_grid = T - t_grid
    coeffs = sol.sol(s_grid).copy()

    a0, b0, c0, d0, e0, f0 = coeffs[:, 0]
    J_star = a0 * mu ** 2 + d0 * mu + f0

    return {
        "t_grid": t_grid,
        "coeffs": coeffs,
        "J_star": float(J_star),
        "terminal": {"a": coeffs[0, -1], "b": coeffs[1, -1], "c": coeffs[2, -1],
                      "d": coeffs[3, -1], "e": coeffs[4, -1], "f": coeffs[5, -1]},
    }


def optimal_control(
    t_idx: int,
    Z: np.ndarray,
    Q: np.ndarray,
    *,
    coeffs: np.ndarray,
    eta: float,
) -> np.ndarray:
    """Optimal trading rate: v* = (2bQ + cZ + e) / (2*eta)."""
    b_t = coeffs[1, t_idx]
    c_t = coeffs[2, t_idx]
    e_t = coeffs[4, t_idx]
    return (2.0 * b_t * Q + c_t * Z + e_t) / (2.0 * eta)


def hjb_benchmark(
    *,
    T: float,
    K: int,
    kappa0: float,
    sigma_min: float,
    mu: float,
    alpha: float,
    phi: float,
    eta: float,
    Q0: float = 0.0,
) -> dict:
    """HJB benchmark J* for constant-coefficient OU at H=0.5."""
    sol = solve_hjb_riccati(
        T=T, kappa0=kappa0, sigma_min=sigma_min, mu=mu,
        alpha=alpha, phi=phi, eta=eta, K=K,
    )

    a0, b0, c0, d0, e0, f0 = sol["coeffs"][:, 0]
    sol["J_star"] = float(a0 * mu ** 2 + b0 * Q0 ** 2 + c0 * mu * Q0
                          + d0 * mu + e0 * Q0 + f0)
    return sol
