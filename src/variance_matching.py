"""Variance-matching parameterization for fractional OU state-dependent coefficients.

Implements the H-dependent steepness scaling from Chapter 6:
    beta(H)  = beta_ref  * phi(H)
    gamma(H) = gamma_ref * phi(H)

where phi(H) = kappa0^{2H-1} / (2H * Gamma(2H)), normalised so phi(0.5) = 1.
"""

from __future__ import annotations

import math


def stationary_variance(H: float, sigma_min: float, kappa0: float) -> float:
    """Stationary variance of the fractional OU process.

    s(H)^2 = sigma_min^2 * H * Gamma(2H) / kappa0^{2H}
    """
    return sigma_min ** 2 * H * math.gamma(2.0 * H) / kappa0 ** (2.0 * H)


def phi_ratio(H: float, kappa0: float) -> float:
    """Variance-matching ratio phi(H) = kappa0^{2H-1} / (2H * Gamma(2H)).

    Satisfies phi(0.5) = 1 for any kappa0.
    """
    return kappa0 ** (2.0 * H - 1.0) / (2.0 * H * math.gamma(2.0 * H))


def beta_matched(H: float, beta_ref: float, kappa0: float) -> float:
    """H-dependent volatility steepness: beta(H) = beta_ref * phi(H)."""
    return beta_ref * phi_ratio(H, kappa0)


def gamma_matched(H: float, gamma_ref: float, kappa0: float) -> float:
    """H-dependent mean-reversion steepness: gamma(H) = gamma_ref * phi(H)."""
    return gamma_ref * phi_ratio(H, kappa0)


def beta_ref_from_k(k_sigma: float, sigma_min: float, kappa0: float) -> float:
    """Compute beta_ref so that sigma reaches half-activation at k_sigma s.d. from mu.

    beta_ref = arctanh(0.5) / (k_sigma^2 * s(0.5)^2)
    """
    s2_half = stationary_variance(0.5, sigma_min, kappa0)
    return math.atanh(0.5) / (k_sigma ** 2 * s2_half)


def gamma_ref_from_k(k_kappa: float, sigma_min: float, kappa0: float) -> float:
    """Compute gamma_ref so that kappa reaches half-activation at k_kappa s.d. from mu.

    gamma_ref = ln(2) / (k_kappa^2 * s(0.5)^2)
    """
    s2_half = stationary_variance(0.5, sigma_min, kappa0)
    return math.log(2.0) / (k_kappa ** 2 * s2_half)


def activation_halfwidth_sigma(beta: float) -> float:
    """Distance from mu where tanh(beta * d^2) = 0.5.

    d_sigma = sqrt(arctanh(0.5) / beta)
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    return math.sqrt(math.atanh(0.5) / beta)


def activation_halfwidth_kappa(gamma_val: float) -> float:
    """Distance from mu where 1 - exp(-gamma * d^2) = 0.5.

    d_kappa = sqrt(ln(2) / gamma)
    """
    if gamma_val <= 0.0:
        raise ValueError(f"gamma must be > 0, got {gamma_val}")
    return math.sqrt(math.log(2.0) / gamma_val)
