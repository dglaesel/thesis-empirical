"""Validation tests for the Davies-Harte fBM simulation method.

Tests cover:
  1. Variance: Var(B^H_T) ≈ T^{2H} for multiple (H, T) pairs.
  2. Cholesky match: terminal variance/mean agree between methods.
  3. BM increments: at H=0.5, consecutive increment correlations ≈ 0.
  4. Robustness: works for H ∈ {0.1, 0.3, 0.5, 0.7, 0.9} with variance within tolerance.
"""

import numpy as np
import torch
import pytest

from src.fbm import simulate_fbm_davies_harte, simulate_fbm_cholesky, clear_cache


# ---------------------------------------------------------------------------
# Test 1: Variance Var(B^H_T) ≈ T^{2H}
# ---------------------------------------------------------------------------

class TestDaviesHarteVariance:
    """Var(B^H_T) should equal T^{2H}."""

    @pytest.mark.parametrize("H,T", [
        (0.3, 1.0),
        (0.3, 2.0),
        (0.5, 1.0),
        (0.5, 2.0),
        (0.7, 1.0),
        (0.7, 2.0),
    ])
    def test_terminal_variance(self, H: float, T: float) -> None:
        clear_cache()
        K, M = 200, 10_000
        B = simulate_fbm_davies_harte(H, T, K, M, seed=42)

        empirical_var = B[:, -1].var().item()
        expected_var = T ** (2 * H)
        rel_err = abs(empirical_var - expected_var) / expected_var

        print(
            f"  H={H}, T={T}: empirical_var={empirical_var:.4f}, "
            f"expected={expected_var:.4f}, rel_err={rel_err:.3%}"
        )
        assert rel_err < 0.05, (
            f"Variance test FAILED for H={H}, T={T}: "
            f"empirical={empirical_var:.4f}, expected={expected_var:.4f}, "
            f"rel_err={rel_err:.3%}"
        )


# ---------------------------------------------------------------------------
# Test 2: Cholesky match
# ---------------------------------------------------------------------------

class TestDaviesHarteCholeksyMatch:
    """Davies-Harte and Cholesky should produce statistically equivalent paths."""

    @pytest.mark.parametrize("H", [0.3, 0.5, 0.7])
    def test_terminal_stats_match(self, H: float) -> None:
        clear_cache()
        T, K, M = 1.0, 200, 10_000

        B_dh = simulate_fbm_davies_harte(H, T, K, M, seed=100)
        B_ch = simulate_fbm_cholesky(H, T, K, M, seed=200)

        # Compare terminal variance
        var_dh = B_dh[:, -1].var().item()
        var_ch = B_ch[:, -1].var().item()
        expected_var = T ** (2 * H)

        rel_diff = abs(var_dh - var_ch) / expected_var
        print(
            f"  H={H}: DH_var={var_dh:.4f}, Chol_var={var_ch:.4f}, "
            f"expected={expected_var:.4f}, rel_diff={rel_diff:.3%}"
        )
        assert rel_diff < 0.05, (
            f"Variance mismatch for H={H}: DH={var_dh:.4f}, "
            f"Cholesky={var_ch:.4f}, diff={rel_diff:.3%}"
        )

        # Compare terminal mean (should be ~0)
        mean_dh = B_dh[:, -1].mean().item()
        mean_ch = B_ch[:, -1].mean().item()
        assert abs(mean_dh) < 0.05, f"DH mean too large: {mean_dh:.4f}"
        assert abs(mean_ch) < 0.05, f"Cholesky mean too large: {mean_ch:.4f}"
        print(
            f"  H={H}: DH_mean={mean_dh:.4f}, Chol_mean={mean_ch:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 3: BM increment correlations at H = 0.5
# ---------------------------------------------------------------------------

class TestDaviesHarteBMIncrements:
    """At H=0.5, fBM = standard BM, so increments are independent."""

    def test_increment_correlations_near_zero(self) -> None:
        clear_cache()
        H, T, K, M = 0.5, 1.0, 200, 50_000
        B = simulate_fbm_davies_harte(H, T, K, M, seed=42)

        increments = B[:, 1:] - B[:, :-1]  # [M, K]

        for lag in [1, 2, 3, 4]:
            inc_a = increments[:, :-lag]
            inc_b = increments[:, lag:]

            # Correlation at a few representative positions
            for j in [0, 50, 100, 150]:
                if j >= inc_a.shape[1]:
                    continue
                corr = torch.corrcoef(
                    torch.stack([inc_a[:, j], inc_b[:, j]])
                )[0, 1].item()
                print(f"  lag={lag}, step={j}: corr={corr:.4f}")
                assert abs(corr) < 0.05, (
                    f"Increment correlation too large at lag={lag}, step={j}: "
                    f"|rho|={abs(corr):.4f}"
                )


# ---------------------------------------------------------------------------
# Test 4: Robustness across extreme H values
# ---------------------------------------------------------------------------

class TestDaviesHarteRobustness:
    """Davies-Harte should work for the full range H ∈ (0, 1)."""

    @pytest.mark.parametrize("H", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_variance_for_extreme_H(self, H: float) -> None:
        clear_cache()
        T, K, M = 1.0, 200, 10_000
        B = simulate_fbm_davies_harte(H, T, K, M, seed=42)

        empirical_var = B[:, -1].var().item()
        expected_var = T ** (2 * H)
        rel_err = abs(empirical_var - expected_var) / expected_var

        # Relaxed tolerance (8%) for extreme H values
        tol = 0.08
        print(
            f"  H={H}: empirical_var={empirical_var:.4f}, "
            f"expected={expected_var:.4f}, rel_err={rel_err:.3%}"
        )
        assert rel_err < tol, (
            f"Robustness test FAILED for H={H}: "
            f"empirical={empirical_var:.4f}, expected={expected_var:.4f}, "
            f"rel_err={rel_err:.3%} > {tol:.0%}"
        )

    @pytest.mark.parametrize("H", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_path_starts_at_zero(self, H: float) -> None:
        clear_cache()
        B = simulate_fbm_davies_harte(H, 1.0, 100, 100, seed=0)
        assert torch.all(B[:, 0] == 0.0), "B^H_0 must be 0"

    @pytest.mark.parametrize("H", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_output_shape(self, H: float) -> None:
        K, M = 100, 64
        B = simulate_fbm_davies_harte(H, 1.0, K, M)
        assert B.shape == (M, K + 1), f"Expected shape ({M}, {K+1}), got {B.shape}"
        assert B.dtype == torch.float32, f"Expected float32, got {B.dtype}"
