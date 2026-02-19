"""Riccati ODE benchmark for the 1D tracking problem at H = 0.5.

The 1D tracking problem (Bank, Bayer, Friz, Pelizzari (2024), Section 5.2):

    dY_t = u_t dt + dW_t,    Y_0 = 0
    J(u) = E[int_0^T (Y_t^2 + u_t^2) dt]

At H = 0.5 (standard BM), the HJB equation yields the value function
V(t, y) = (1/2) P(t) y^2 + R(t), where:

    Riccati ODE:  dP/dt = (1/2) P(t)^2 - 2,   P(T) = 0
    Residual ODE: dR/dt = -(1/2) P(t),          R(T) = 0

The optimal feedback control is u*(t, y) = -(1/2) P(t) * y.
The optimal cost from Y_0 = 0 is J* = R(0).

Derivation:
    HJB PDE:  -V_t = min_u { y^2 + u^2 + u * V_y + (1/2) V_yy }
    The minimiser is u* = -(1/2) V_y, giving:
        -V_t = y^2 - (1/4) V_y^2 + (1/2) V_yy

    Ansatz V = (1/2) P(t) y^2 + R(t):
        V_y = P y,   V_yy = P,   V_t = (1/2) P' y^2 + R'

        -(1/2) P' y^2 - R' = y^2 - (1/4) P^2 y^2 + (1/2) P

    Matching y^2:   P' = (1/2) P^2 - 2
    Matching const:  R' = -(1/2) P
    Terminal:        P(T) = 0,  R(T) = 0

    Optimal control: u*(t, y) = -(1/2) P(t) y.

Note: The Riccati ODE dP/dt = (1/2)P^2 - 2 has an analytical solution via
hyperbolic functions:  P(t) = 2 * tanh(T - t).  Verification:
    P' = -2 sech^2(T-t),   (1/2)P^2 - 2 = 2 tanh^2(T-t) - 2 = -2 sech^2(T-t).
P(T) = 2 tanh(0) = 0. Check!

Reference: Bank, Bayer, Friz, Pelizzari (2024), Section 5.2.
"""

__all__ = ["solve_riccati", "optimal_cost_1d", "optimal_cost_bank", "optimal_control_gain"]

import math

import numpy as np


def _rk4_step(f, t, y, dt):
    """Single step of the classical RK4 method."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_riccati(
    T: float, num_points: int = 10000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the Riccati ODE system backwards from T to 0.

    System (in forward time t):
        dP/dt = (1/2) P^2 - 2,   P(T) = 0
        dR/dt = -(1/2) P,         R(T) = 0

    Solved by time reversal s = T - t with RK4 integration.

    Args:
        T: Time horizon.
        num_points: Number of grid points for the solution.

    Returns:
        t_grid: Time points [num_points], from 0 to T.
        P: Riccati solution P(t) at each time point.
        R: Residual cost R(t) at each time point.
    """

    def ode_reversed(s, state):
        """ODE in reversed time s = T - t.

        dP/ds = -dP/dt = 2 - (1/2)P^2
        dR/ds = -dR/dt = (1/2)P
        """
        P_val, R_val = state[0], state[1]
        return np.array([2.0 - 0.5 * P_val ** 2, 0.5 * P_val])

    # Integrate in reversed time s from 0 (=t=T) to T (=t=0)
    s_grid = np.linspace(0, T, num_points)
    ds = s_grid[1] - s_grid[0]

    state = np.array([0.0, 0.0])  # P(T) = 0, R(T) = 0
    P_reversed = np.zeros(num_points)
    R_reversed = np.zeros(num_points)
    P_reversed[0] = state[0]
    R_reversed[0] = state[1]

    for i in range(num_points - 1):
        state = _rk4_step(ode_reversed, s_grid[i], state, ds)
        P_reversed[i + 1] = state[0]
        R_reversed[i + 1] = state[1]

    # Reverse to get forward-time arrays
    t_grid = T - s_grid[::-1]
    P = P_reversed[::-1]
    R = R_reversed[::-1]

    return t_grid, P, R


def solve_riccati_analytical(
    T: float, num_points: int = 10000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytical solution of the Riccati ODE.

    P(t) = 2 tanh(T - t)
    R(t) = int_t^T tanh(T - s) ds = ln(cosh(T - t))

    This provides a cross-check against the numerical solution.

    Args:
        T: Time horizon.
        num_points: Number of grid points.

    Returns:
        t_grid, P, R: Same format as solve_riccati.
    """
    t_grid = np.linspace(0, T, num_points)
    tau = T - t_grid  # time to maturity
    P = 2.0 * np.tanh(tau)
    R = np.log(np.cosh(tau))
    return t_grid, P, R


def optimal_cost_1d(T: float) -> float:
    """Return J* = R(0), the optimal cost of the 1D tracking problem at H = 0.5.

    Uses the analytical formula: J* = ln(cosh(T)).

    This is the minimum expected cost E[int_0^T (Y_t^2 + u_t^2) dt]
    over all adapted controls u, starting from Y_0 = 0.
    """
    return np.log(np.cosh(T))


def optimal_cost_bank(T: float, kappa: float) -> float:
    """Analytical optimal cost for the Bank et al. (2024) tracking problem.

    Cost functional: L(Y, U) = 1/2 * integral_0^T (Y^2 + kappa * U^2) dt
    Dynamics: dY = U dt + d xi,  Y_0 = 0  (xi = fBM at H = 1/2)

    At H = 1/2 the problem is LQR. The Riccati ODE is
        P'(t) = P(t)^2 / kappa - 1,   P(T) = 0
    with solution P(t) = sqrt(kappa) * tanh((T - t) / sqrt(kappa)).

    The optimal cost (starting from y_0 = 0) is
        J* = R(0) = 1/2 * kappa * ln(cosh(T / sqrt(kappa))).

    For kappa = 0.1, T = 1: J* ~ 0.1235, matching Bank et al. Table 1's
    reported value of 0.124 (rounded to 3 decimal places).

    Reference: Bank et al. (2024), Theorem 5.2 specialised to H = 1/2.

    Args:
        T: Time horizon.
        kappa: Control penalty parameter (kappa > 0).

    Returns:
        Optimal expected cost J*.
    """
    return 0.5 * kappa * math.log(math.cosh(T / math.sqrt(kappa)))


def optimal_control_gain(
    T: float, num_points: int = 10000
) -> tuple[np.ndarray, np.ndarray]:
    """Return the time-dependent optimal control gain.

    The optimal control is u*(t, y) = -(1/2) P(t) * y = -tanh(T - t) * y.
    This function returns the gain g(t) = -(1/2) P(t) = -tanh(T - t) at each
    time point, so that u*(t, y) = g(t) * y.

    Returns:
        t_grid: Time points from 0 to T.
        gain: The gain function tanh(T - t) at each time point.
    """
    t_grid = np.linspace(0, T, num_points)
    gain = -np.tanh(T - t_grid)
    return t_grid, gain
