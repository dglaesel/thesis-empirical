"""Tests for src/variance_matching.py — variance-matching parameterization."""

from __future__ import annotations

import math
import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.variance_matching import (
    activation_halfwidth_kappa,
    activation_halfwidth_sigma,
    beta_matched,
    beta_ref_from_k,
    gamma_matched,
    gamma_ref_from_k,
    phi_ratio,
    stationary_variance,
)

KAPPA0 = 10.0
SIGMA_MIN = 0.10


class TestPhiRatio:
    """phi(H) = kappa0^{2H-1} / (2H * Gamma(2H)), phi(0.5) = 1."""

    def test_phi_half_is_one(self):
        assert phi_ratio(0.5, KAPPA0) == pytest.approx(1.0, abs=1e-14)

    def test_phi_half_any_kappa(self):
        for kappa in [1.0, 5.0, 20.0, 100.0]:
            assert phi_ratio(0.5, kappa) == pytest.approx(1.0, abs=1e-14)

    def test_phi_below_one_for_H_below_half(self):
        """phi(H) < 1 for H < 0.5 (kappa0 > 1)."""
        for H in [0.25, 0.3, 0.4]:
            assert phi_ratio(H, KAPPA0) < 1.0

    def test_phi_above_one_for_H_above_half(self):
        """phi(H) > 1 for H > 0.5 (kappa0 > 1): s(0.5)^2 > s(H)^2 since s^2 decreases."""
        for H in [0.6, 0.7, 0.75]:
            assert phi_ratio(H, KAPPA0) > 1.0

    def test_phi_monotone_in_H(self):
        """phi(H) = kappa0^{2H-1}/(2H*Gamma(2H)) increases with H for kappa0=10."""
        H_vals = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        phi_vals = [phi_ratio(H, KAPPA0) for H in H_vals]
        for i in range(len(phi_vals) - 1):
            assert phi_vals[i] < phi_vals[i + 1], (
                f"phi({H_vals[i]})={phi_vals[i]:.4f} >= phi({H_vals[i+1]})={phi_vals[i+1]:.4f}"
            )

    def test_phi_formula_values_kappa10(self):
        """Verify computed phi(H) values match the analytic formula kappa0^{2H-1}/(2H*Gamma(2H))."""
        expected = {
            0.30: 0.4456,
            0.40: 0.6775,
            0.50: 1.0000,
            0.60: 1.4385,
            0.75: 2.3788,
        }
        for H, phi_expected in expected.items():
            phi_computed = phi_ratio(H, 10.0)
            assert phi_computed == pytest.approx(phi_expected, abs=0.001), (
                f"phi({H}) = {phi_computed:.4f}, expected ~{phi_expected}"
            )


class TestStationaryVariance:
    """s(H)^2 = sigma_min^2 * H * Gamma(2H) / kappa0^{2H}."""

    def test_classical_ou_half(self):
        """At H=0.5: s^2 = sigma^2 / (2*kappa) (classical OU result)."""
        s2 = stationary_variance(0.5, SIGMA_MIN, KAPPA0)
        classical = SIGMA_MIN ** 2 / (2.0 * KAPPA0)
        assert s2 == pytest.approx(classical, rel=1e-12)
        assert s2 == pytest.approx(0.0005, rel=1e-12)

    def test_positive(self):
        for H in [0.1, 0.25, 0.5, 0.75, 0.9]:
            assert stationary_variance(H, SIGMA_MIN, KAPPA0) > 0.0


class TestBetaGammaMatched:
    def test_beta_identity_at_half(self):
        """beta_matched(0.5, beta_ref, kappa0) == beta_ref."""
        beta_ref = 42.0
        assert beta_matched(0.5, beta_ref, KAPPA0) == pytest.approx(beta_ref, rel=1e-14)

    def test_gamma_identity_at_half(self):
        """gamma_matched(0.5, gamma_ref, kappa0) == gamma_ref."""
        gamma_ref = 17.5
        assert gamma_matched(0.5, gamma_ref, KAPPA0) == pytest.approx(gamma_ref, rel=1e-14)

    def test_beta_scales_with_phi(self):
        beta_ref = 100.0
        for H in [0.3, 0.7]:
            expected = beta_ref * phi_ratio(H, KAPPA0)
            assert beta_matched(H, beta_ref, KAPPA0) == pytest.approx(expected, rel=1e-14)


class TestRefFromK:
    def test_beta_ref_roundtrip(self):
        """beta_ref_from_k -> activation_halfwidth_sigma should recover k_sigma * s(0.5)."""
        k_sigma = 1.5
        beta_ref = beta_ref_from_k(k_sigma, SIGMA_MIN, KAPPA0)
        d = activation_halfwidth_sigma(beta_ref)
        s_half = math.sqrt(stationary_variance(0.5, SIGMA_MIN, KAPPA0))
        assert d == pytest.approx(k_sigma * s_half, rel=1e-12)

    def test_gamma_ref_roundtrip(self):
        """gamma_ref_from_k -> activation_halfwidth_kappa should recover k_kappa * s(0.5)."""
        k_kappa = 1.5
        gamma_ref = gamma_ref_from_k(k_kappa, SIGMA_MIN, KAPPA0)
        d = activation_halfwidth_kappa(gamma_ref)
        s_half = math.sqrt(stationary_variance(0.5, SIGMA_MIN, KAPPA0))
        assert d == pytest.approx(k_kappa * s_half, rel=1e-12)


class TestActivationHalfwidth:
    def test_sigma_positive(self):
        assert activation_halfwidth_sigma(10.0) > 0.0

    def test_kappa_positive(self):
        assert activation_halfwidth_kappa(10.0) > 0.0

    def test_sigma_raises_nonpositive(self):
        with pytest.raises(ValueError):
            activation_halfwidth_sigma(-1.0)
        with pytest.raises(ValueError):
            activation_halfwidth_sigma(0.0)

    def test_kappa_raises_nonpositive(self):
        with pytest.raises(ValueError):
            activation_halfwidth_kappa(-1.0)
        with pytest.raises(ValueError):
            activation_halfwidth_kappa(0.0)
