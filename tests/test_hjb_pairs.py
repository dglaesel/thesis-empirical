"""Tests for src/hjb_pairs.py — HJB benchmark for pairs trading at H=0.5."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.hjb_pairs import hjb_benchmark, hjb_riccati_rhs, optimal_control, solve_hjb_riccati

PARAMS = dict(
    T=1.0,
    kappa0=10.0,
    sigma_min=0.10,
    mu=0.0,
    alpha=10.0,
    phi=0.1,
    eta=0.01,
)


class TestTerminalConditions:
    def test_terminal_values(self):
        sol = solve_hjb_riccati(**PARAMS, K=5000)
        tc = sol["terminal"]
        assert tc["a"] == pytest.approx(0.0, abs=1e-8)
        assert tc["b"] == pytest.approx(-PARAMS["alpha"], abs=1e-8)
        assert tc["c"] == pytest.approx(0.0, abs=1e-8)
        assert tc["d"] == pytest.approx(0.0, abs=1e-8)
        assert tc["e"] == pytest.approx(0.0, abs=1e-8)
        assert tc["f"] == pytest.approx(0.0, abs=1e-8)


class TestWealthTerm:
    """Verify the +kappa0 term in dc from the Q*dZ wealth dynamics."""

    def test_dc_has_kappa0_forcing(self):
        """At terminal conditions [0, -alpha, 0, 0, 0, 0], dc should be +kappa0."""
        y_T = np.array([0.0, -PARAMS["alpha"], 0.0, 0.0, 0.0, 0.0])
        rhs = hjb_riccati_rhs(PARAMS["T"], y_T, **{k: v for k, v in PARAMS.items() if k != "T"})
        # dc = kappa0*c - b*c/eta + kappa0 = 0 - 0 + kappa0 = kappa0
        assert rhs[2] == pytest.approx(PARAMS["kappa0"], rel=1e-12)

    def test_c_nonzero_at_t0(self):
        """With the wealth term, c(0) should be nonzero even when mu=0."""
        sol = solve_hjb_riccati(**PARAMS, K=5000)
        c0 = sol["coeffs"][2, 0]
        assert abs(c0) > 0.01  # should be significantly nonzero


class TestNoTradingLimit:
    def test_no_trade_optimal(self):
        sol = solve_hjb_riccati(
            T=1.0, kappa0=10.0, sigma_min=0.10, mu=0.0,
            alpha=0.0, phi=0.0, eta=1e6, K=2000,
        )
        assert sol["J_star"] == pytest.approx(0.0, abs=1e-4)


class TestJStarFinite:
    def test_finite(self):
        sol = solve_hjb_riccati(**PARAMS, K=5000)
        assert np.isfinite(sol["J_star"])
        assert np.isfinite(sol["coeffs"]).all()


class TestConvergence:
    def test_k_convergence(self):
        j_vals = []
        for K in [1000, 5000, 10000]:
            sol = solve_hjb_riccati(**PARAMS, K=K)
            j_vals.append(sol["J_star"])
        assert j_vals[-1] == pytest.approx(j_vals[-2], rel=1e-6)


class TestHJBBenchmark:
    def test_benchmark_q0_zero(self):
        sol = hjb_benchmark(**PARAMS, K=5000, Q0=0.0)
        sol_raw = solve_hjb_riccati(**PARAMS, K=5000)
        assert sol["J_star"] == pytest.approx(sol_raw["J_star"], rel=1e-10)

    def test_benchmark_q0_nonzero(self):
        sol = hjb_benchmark(**PARAMS, K=5000, Q0=1.0)
        sol_zero = hjb_benchmark(**PARAMS, K=5000, Q0=0.0)
        assert sol["J_star"] < sol_zero["J_star"]


class TestOptimalControl:
    def test_control_shape(self):
        sol = solve_hjb_riccati(**PARAMS, K=1000)
        Z = np.array([0.0, 0.1, -0.1])
        Q = np.array([0.0, 0.5, -0.5])
        v = optimal_control(0, Z, Q, coeffs=sol["coeffs"], eta=PARAMS["eta"])
        assert v.shape == (3,)

    def test_control_at_equilibrium(self):
        """At Z=mu=0, Q=0, v should be ~0 (since e=0 when mu=0)."""
        sol = solve_hjb_riccati(**PARAMS, K=5000)
        Z = np.array([0.0])
        Q = np.array([0.0])
        v = optimal_control(0, Z, Q, coeffs=sol["coeffs"], eta=PARAMS["eta"])
        assert abs(v[0]) < 1.0


class TestSymmetry:
    """With mu=0, d and e should be ~0 (but c is nonzero due to wealth term)."""

    def test_d_e_zero_for_mu_zero(self):
        sol = solve_hjb_riccati(**PARAMS, K=5000)
        coeffs = sol["coeffs"]
        assert np.max(np.abs(coeffs[3, :])) < 1e-8  # d
        assert np.max(np.abs(coeffs[4, :])) < 1e-8  # e
