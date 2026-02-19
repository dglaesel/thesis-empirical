"""Tests for the Riccati ODE benchmark."""

import math

import numpy as np
from src.riccati import solve_riccati, solve_riccati_analytical, optimal_cost_1d, optimal_cost_bank


class TestRiccatiODE:
    """Verify the Riccati ODE solution."""

    def test_terminal_conditions(self):
        """P(T) = 0 and R(T) = 0."""
        t, P, R = solve_riccati(T=1.0)
        assert abs(P[-1]) < 1e-8, f"P(T) = {P[-1]}, expected 0"
        assert abs(R[-1]) < 1e-8, f"R(T) = {R[-1]}, expected 0"
        print("PASS: Terminal conditions P(T)=0, R(T)=0")

    def test_ode_residual(self):
        """Check that dP/dt = (1/2)P^2 - 2 along the solution."""
        t, P, R = solve_riccati(T=1.0, num_points=10000)
        dt = t[1] - t[0]
        dP_numerical = np.diff(P) / dt
        P_mid = 0.5 * (P[:-1] + P[1:])
        residual = dP_numerical - (0.5 * P_mid ** 2 - 2)
        max_residual = np.max(np.abs(residual))
        assert max_residual < 1e-4, f"Max ODE residual = {max_residual}"
        print(f"PASS: ODE residual max = {max_residual:.2e}")

    def test_optimal_cost_positive(self):
        """J* should be positive (we're minimising a non-negative cost)."""
        J_star = optimal_cost_1d(T=1.0)
        assert J_star > 0, f"J* = {J_star}, expected positive"
        print(f"PASS: J* = {J_star:.6f} > 0")

    def test_optimal_cost_reasonable(self):
        """J* should be in a reasonable range.

        The uncontrolled cost (u=0) gives J = E[int (B_t)^2 dt]
        = int t dt = T^2/2 = 0.5 for T=1.
        The optimal cost must be strictly less than 0.5.
        """
        J_star = optimal_cost_1d(T=1.0)
        assert J_star < 0.5, f"J* = {J_star}, should be < 0.5 (uncontrolled cost)"
        assert J_star > 0.1, f"J* = {J_star}, suspiciously small"
        print(f"PASS: J* = {J_star:.6f} in reasonable range (0.1, 0.5)")

    def test_P_positive(self):
        """P(t) should be positive for t < T.

        From P(t) = 2 tanh(T-t) > 0 for t < T.
        The optimal control u* = -(1/2) P y pushes Y toward zero.
        """
        t, P, R = solve_riccati(T=1.0)
        # Exclude last few points near P(T)=0
        assert np.all(P[:-10] > 0), "P(t) should be positive for t < T"
        print("PASS: P(t) > 0 for t < T")

    def test_numerical_matches_analytical(self):
        """RK4 numerical solution should match the closed-form analytical solution."""
        T = 1.0
        t_n, P_n, R_n = solve_riccati(T, num_points=10000)
        t_a, P_a, R_a = solve_riccati_analytical(T, num_points=10000)

        max_P_err = np.max(np.abs(P_n - P_a))
        max_R_err = np.max(np.abs(R_n - R_a))
        assert max_P_err < 1e-8, f"P mismatch: max|P_num-P_anal| = {max_P_err:.2e}"
        assert max_R_err < 1e-8, f"R mismatch: max|R_num-R_anal| = {max_R_err:.2e}"
        print(f"PASS: Numerical vs analytical: max|dP|={max_P_err:.2e}, max|dR|={max_R_err:.2e}")

    def test_analytical_cost_formula(self):
        """Verify J* = ln(cosh(T)) against numerical R(0)."""
        T = 1.0
        J_analytical = optimal_cost_1d(T)
        _, _, R = solve_riccati(T, num_points=100000)
        J_numerical = R[0]
        assert abs(J_analytical - J_numerical) < 1e-8, (
            f"J* analytical={J_analytical:.10f}, numerical={J_numerical:.10f}"
        )
        print(f"PASS: J* analytical = J* numerical = {J_analytical:.6f}")


class TestBankRiccati:
    """Tests for the Bank et al. (2024) cost functional with kappa."""

    def test_bank_riccati_ode(self):
        """P(t) = sqrt(kappa)*tanh((T-t)/sqrt(kappa)) satisfies P' = P^2/kappa - 1, P(T) = 0."""
        kappa = 0.1
        T = 1.0
        sqrt_k = math.sqrt(kappa)
        # Terminal condition
        assert abs(sqrt_k * math.tanh(0)) < 1e-12, "P(T) != 0"
        # ODE residual at t = 0.3
        t = 0.3
        tau = (T - t) / sqrt_k
        P = sqrt_k * math.tanh(tau)
        P_prime = -(1.0 / math.cosh(tau)) ** 2  # -sech^2(tau)
        residual = P_prime - (P ** 2 / kappa - 1)
        assert abs(residual) < 1e-10, f"ODE residual = {residual}"
        print(f"PASS: Bank Riccati ODE residual = {residual:.2e}")

    def test_bank_cost_H05(self):
        """Bank et al. Table 1: H=0.5 theoretical optimum ~ 0.124."""
        J = optimal_cost_bank(T=1.0, kappa=0.1)
        assert abs(J - 0.1235) < 0.001, f"Expected ~0.1235, got {J:.6f}"
        print(f"PASS: Bank J* = {J:.6f} (expected ~0.1235)")
