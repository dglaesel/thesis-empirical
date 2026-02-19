"""Tests for the corrected Phase 2 core (Chapter 6 aligned)."""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml

from src.phase2_core import (
    Phase2Params,
    build_phase2_params,
    kappa_state,
    kappa_state_prime,
    make_feature_dir,
    sigma_bounds,
    sigma_state,
    sigma_state_prime,
    simulate_spread_paths,
)

# ----- Helpers -----

def _make_const_params(**overrides) -> Phase2Params:
    """Build a constant-coefficient (L1-like) Phase2Params for testing."""
    defaults = dict(
        sigma_min=0.10, delta_sigma=0.0, beta=1.0,
        kappa0=10.0, delta_kappa=0.0, gamma_k=1.0,
        mu=0.0, T=1.0, K=500,
        alpha=10.0, phi=0.1, eta=0.01, v_bar=5.0, Q0=0.0,
    )
    defaults.update(overrides)
    return Phase2Params(**defaults)


def _make_sd_params(**overrides) -> Phase2Params:
    """Build a state-dependent (L2/L3-like) Phase2Params for testing."""
    defaults = dict(
        sigma_min=0.10, delta_sigma=0.15, beta=50.0,
        kappa0=10.0, delta_kappa=15.0, gamma_k=50.0,
        mu=0.0, T=1.0, K=500,
        alpha=10.0, phi=0.1, eta=0.01, v_bar=5.0, Q0=0.0,
    )
    defaults.update(overrides)
    return Phase2Params(**defaults)


def _load_test_config() -> dict:
    """Load the default config for build_phase2_params tests."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----- sigma_state tests -----

class TestSigmaState:
    def test_at_mu_equals_sigma_min(self):
        """sigma(mu) = sigma_min + delta_sigma * tanh(0) = sigma_min."""
        p = _make_sd_params()
        z = torch.tensor([p.mu], dtype=torch.float64)
        assert sigma_state(z, p).item() == pytest.approx(p.sigma_min, rel=1e-12)

    def test_far_from_mu(self):
        """For |z-mu| >> 0, sigma approaches sigma_min + delta_sigma."""
        p = _make_sd_params()
        z = torch.tensor([p.mu + 100.0])
        val = sigma_state(z, p).item()
        assert val == pytest.approx(p.sigma_min + p.delta_sigma, abs=1e-6)

    def test_symmetry(self):
        """sigma(mu+d) == sigma(mu-d) because argument is (z-mu)^2."""
        p = _make_sd_params()
        d = 0.05
        z_plus = torch.tensor([p.mu + d])
        z_minus = torch.tensor([p.mu - d])
        assert sigma_state(z_plus, p).item() == pytest.approx(
            sigma_state(z_minus, p).item(), rel=1e-12
        )

    def test_constant_when_delta_zero(self):
        """When delta_sigma=0, sigma(z) = sigma_min for all z."""
        p = _make_const_params()
        z = torch.tensor([-1.0, 0.0, 1.0, 5.0])
        vals = sigma_state(z, p)
        expected = torch.full_like(vals, p.sigma_min)
        assert torch.allclose(vals, expected, atol=1e-12)


# ----- kappa_state tests -----

class TestKappaState:
    def test_at_mu_equals_kappa0(self):
        p = _make_sd_params()
        z = torch.tensor([p.mu])
        assert kappa_state(z, p).item() == pytest.approx(p.kappa0, rel=1e-12)

    def test_far_from_mu(self):
        p = _make_sd_params()
        z = torch.tensor([p.mu + 100.0])
        val = kappa_state(z, p).item()
        assert val == pytest.approx(p.kappa0 + p.delta_kappa, abs=1e-6)

    def test_symmetry(self):
        p = _make_sd_params()
        d = 0.05
        z_plus = torch.tensor([p.mu + d])
        z_minus = torch.tensor([p.mu - d])
        assert kappa_state(z_plus, p).item() == pytest.approx(
            kappa_state(z_minus, p).item(), rel=1e-12
        )

    def test_constant_when_delta_zero(self):
        p = _make_const_params()
        z = torch.tensor([-1.0, 0.0, 1.0, 5.0])
        vals = kappa_state(z, p)
        expected = torch.full_like(vals, p.kappa0)
        assert torch.allclose(vals, expected, atol=1e-12)


# ----- Derivative finite-difference checks -----

class TestDerivatives:
    def test_sigma_prime_finite_diff(self):
        p = _make_sd_params()
        z0 = torch.tensor([0.03], dtype=torch.float64)
        eps = 1e-7
        numerical = (sigma_state(z0 + eps, p) - sigma_state(z0 - eps, p)) / (2.0 * eps)
        analytical = sigma_state_prime(z0, p)
        assert analytical.item() == pytest.approx(numerical.item(), rel=1e-5)

    def test_kappa_prime_finite_diff(self):
        p = _make_sd_params()
        z0 = torch.tensor([0.03], dtype=torch.float64)
        eps = 1e-7
        numerical = (kappa_state(z0 + eps, p) - kappa_state(z0 - eps, p)) / (2.0 * eps)
        analytical = kappa_state_prime(z0, p)
        assert analytical.item() == pytest.approx(numerical.item(), rel=1e-5)

    def test_sigma_prime_zero_at_mu(self):
        """sigma'(mu) = 0 because the quadratic argument has zero derivative at mu."""
        p = _make_sd_params()
        z = torch.tensor([p.mu], dtype=torch.float64)
        assert sigma_state_prime(z, p).item() == pytest.approx(0.0, abs=1e-14)

    def test_kappa_prime_zero_at_mu(self):
        p = _make_sd_params()
        z = torch.tensor([p.mu], dtype=torch.float64)
        assert kappa_state_prime(z, p).item() == pytest.approx(0.0, abs=1e-14)


# ----- sigma_bounds -----

class TestSigmaBounds:
    def test_bounds_constant(self):
        p = _make_const_params()
        lo, hi = sigma_bounds(p)
        assert lo == pytest.approx(p.sigma_min)
        assert hi == pytest.approx(p.sigma_min)

    def test_bounds_state_dependent(self):
        p = _make_sd_params()
        lo, hi = sigma_bounds(p)
        assert lo == pytest.approx(p.sigma_min)
        assert hi == pytest.approx(p.sigma_min + p.delta_sigma)


# ----- build_phase2_params -----

class TestBuildPhase2Params:
    def test_L1_constant(self):
        config = _load_test_config()
        p = build_phase2_params(config, "L1", 0.5)
        assert p.delta_sigma == 0.0
        assert p.delta_kappa == 0.0
        assert p.sigma_min == float(config["phase2"]["spread_params"]["sigma_min"])
        assert p.kappa0 == float(config["phase2"]["spread_params"]["kappa0"])

    def test_L3_at_half_equals_L2(self):
        """At H=0.5, phi(H)=1 so L3 steepness should match L2."""
        config = _load_test_config()
        p2 = build_phase2_params(config, "L2", 0.5)
        p3 = build_phase2_params(config, "L3", 0.5)
        assert p3.beta == pytest.approx(p2.beta, rel=1e-12)
        assert p3.gamma_k == pytest.approx(p2.gamma_k, rel=1e-12)

    def test_L2_L3_have_state_dependence(self):
        config = _load_test_config()
        for level in ["L2", "L3"]:
            p = build_phase2_params(config, level, 0.5)
            assert p.delta_sigma > 0.0
            assert p.delta_kappa > 0.0

    def test_unknown_level_raises(self):
        config = _load_test_config()
        with pytest.raises(ValueError, match="Unknown level"):
            build_phase2_params(config, "WorldA", 0.5)


# ----- make_feature_dir -----

class TestMakeFeatureDir:
    def test_level_in_path(self):
        path = make_feature_dir("root", level="L1", H=0.5, seed=0, integrator="exact")
        assert "level_L1" in path


# ----- Exact fOU simulation -----

class TestExactFOU:
    def test_stationary_variance_h05(self):
        """Empirical variance at H=0.5 should match sigma_min^2/(2*kappa0)."""
        # With kappa0=10, mixing time is ~1/kappa0=0.1, so T=5 is ~50 mixing times
        p = _make_const_params(K=5000, T=5.0)
        _, Z = simulate_spread_paths(H=0.5, M=10000, params=p, integrator="exact", seed=42)
        # Take the last time point (approximately stationary)
        z_final = Z[:, -1].double().numpy()
        empirical_var = np.var(z_final)
        theoretical_var = p.sigma_min ** 2 / (2.0 * p.kappa0)
        assert empirical_var == pytest.approx(theoretical_var, rel=0.15)

    def test_exact_vs_euler_agreement(self):
        """Exact and Euler should be close for small dt (same seed/fBM)."""
        p = _make_const_params(K=200, T=1.0)
        _, Z_exact = simulate_spread_paths(H=0.5, M=500, params=p, integrator="exact", seed=7)
        _, Z_euler = simulate_spread_paths(H=0.5, M=500, params=p, integrator="euler", seed=7)
        # They share the same fBM, so paths should be close
        max_diff = (Z_exact - Z_euler).abs().max().item()
        assert max_diff < 0.05  # should be very close

    def test_exact_rejects_state_dependent(self):
        """Exact integrator should raise for state-dependent params."""
        p = _make_sd_params()
        with pytest.raises(ValueError, match="constant coefficients"):
            simulate_spread_paths(H=0.5, M=10, params=p, integrator="exact", seed=0)

    def test_output_shapes(self):
        p = _make_const_params(K=50)
        B_H, Z = simulate_spread_paths(H=0.5, M=32, params=p, integrator="exact", seed=0)
        assert B_H.shape == (32, 51)
        assert Z.shape == (32, 51)
        assert B_H.dtype == torch.float32
        assert Z.dtype == torch.float32

    def test_initial_condition(self):
        p = _make_const_params(K=50, mu=0.5)
        _, Z = simulate_spread_paths(H=0.5, M=32, params=p, integrator="exact", seed=0)
        assert torch.allclose(Z[:, 0], torch.tensor(0.5, dtype=Z.dtype))


# ----- Euler/Milstein with state-dependent coefficients -----

class TestStateDependentSimulation:
    def test_euler_runs(self):
        p = _make_sd_params(K=50)
        B_H, Z = simulate_spread_paths(H=0.5, M=32, params=p, integrator="euler", seed=0)
        assert B_H.shape == (32, 51)
        assert Z.shape == (32, 51)
        assert torch.isfinite(Z).all()

    def test_milstein_runs(self):
        p = _make_sd_params(K=50)
        B_H, Z = simulate_spread_paths(H=0.5, M=32, params=p, integrator="milstein", seed=0)
        assert B_H.shape == (32, 51)
        assert Z.shape == (32, 51)
        assert torch.isfinite(Z).all()

    def test_invalid_integrator_raises(self):
        p = _make_const_params(K=50)
        with pytest.raises(ValueError, match="integrator"):
            simulate_spread_paths(H=0.5, M=10, params=p, integrator="rk4", seed=0)
