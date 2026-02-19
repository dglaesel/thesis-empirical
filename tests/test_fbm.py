"""Tests for fBM simulation and signature computation."""

import numpy as np
import torch
import pytest
import iisignature

from src.fbm import simulate_fbm, clear_cholesky_cache
from src.signatures import time_augment, compute_logsignatures


class TestFBMVariance:
    """At H=0.5, fBM is standard Brownian motion with Var(B_t) = t."""

    def test_variance_matches_t(self) -> None:
        """Empirical variance of B^{0.5}_t should approximate t."""
        clear_cholesky_cache()
        H, T, K, M = 0.5, 1.0, 100, 50000
        B = simulate_fbm(H, T, K, M, seed=42)
        times = torch.linspace(0, T, K + 1)

        # Check at a few time points
        for idx in [10, 25, 50, 75, 100]:
            empirical_var = B[:, idx].var().item()
            expected_var = times[idx].item()
            # Allow ~5% relative error for Monte Carlo
            assert abs(empirical_var - expected_var) / max(expected_var, 1e-8) < 0.05, (
                f"At t={times[idx]:.2f}: empirical var={empirical_var:.4f}, "
                f"expected={expected_var:.4f}"
            )


class TestFBMIncrements:
    """At H=0.5, fBM increments should be approximately uncorrelated."""

    def test_increments_uncorrelated(self) -> None:
        """Consecutive increments of standard BM should have near-zero correlation."""
        clear_cholesky_cache()
        H, T, K, M = 0.5, 1.0, 200, 50000
        B = simulate_fbm(H, T, K, M, seed=123)

        increments = B[:, 1:] - B[:, :-1]  # [M, K]

        # Correlation between consecutive increments
        inc1 = increments[:, :-1]  # [M, K-1]
        inc2 = increments[:, 1:]   # [M, K-1]

        # Compute correlation for a few pairs
        for j in [0, 50, 100, 150]:
            corr = torch.corrcoef(torch.stack([inc1[:, j], inc2[:, j]]))[0, 1].item()
            assert abs(corr) < 0.05, (
                f"Increment correlation at step {j}: {corr:.4f} (should be ~0)"
            )


class TestSignatureStraightLine:
    """Signature of a straight line in R^d has a known form."""

    def test_sig_straight_line_1d(self) -> None:
        """For a 1D path from 0 to a, level-2 sig should be (a, a^2/2)."""
        a = 2.0
        # Path as [1, 2, 1]: batch=1, 2 points, dim=1
        path = np.array([[[0.0], [a]]])
        sig = iisignature.sig(path, 2)  # level 2
        # For d=1: sig = (a, a^2/2)
        expected = np.array([a, a ** 2 / 2.0])
        np.testing.assert_allclose(sig[0], expected, atol=1e-10)

    def test_sig_straight_line_2d(self) -> None:
        """For a 2D path (0,0)->(a,b), the level-1 sig is (a,b)."""
        a, b = 1.5, -0.7
        path = np.array([[[0.0, 0.0], [a, b]]])
        sig = iisignature.sig(path, 1)
        expected = np.array([a, b])
        np.testing.assert_allclose(sig[0], expected, atol=1e-10)


class TestChenIdentity:
    """Chen's identity: sig(path[0:t2]) == sigcombine(sig(path[0:t1]), sig(path[t1:t2]))."""

    def test_chen_identity(self) -> None:
        """Verify Chen's identity on a random 2D path."""
        np.random.seed(0)
        d, N = 2, 3
        K = 10
        path = np.cumsum(np.random.randn(1, K + 1, d), axis=1)  # [1, K+1, d]

        # Split at midpoint
        t1 = 5
        path_full = path[:, :K + 1, :]
        path_first = path[:, :t1 + 1, :]
        path_second = path[:, t1:K + 1, :]

        sig_full = iisignature.sig(path_full, N)
        sig_first = iisignature.sig(path_first, N)
        sig_second = iisignature.sig(path_second, N)
        sig_combined = iisignature.sigcombine(sig_first, sig_second, d, N)

        np.testing.assert_allclose(sig_full[0], sig_combined[0], atol=1e-8)


class TestLogsigIncremental:
    """Test that incremental logsig matches batch logsig computation."""

    def test_incremental_matches_batch(self) -> None:
        """Log-sig at final step from incremental should match direct computation."""
        np.random.seed(42)
        d, N = 2, 3
        K = 20
        M = 4

        # Random time-augmented path
        path_np = np.cumsum(np.random.randn(M, K + 1, d) * 0.1, axis=1)
        path_np[:, 0, :] = 0.0
        paths = torch.tensor(path_np, dtype=torch.float32)

        # Incremental
        logsigs_inc = compute_logsignatures(paths, N)

        # Direct computation at final step
        s = iisignature.prepare(d, N)
        logsig_direct = iisignature.logsig(path_np, s)  # [M, dim_logsig]

        np.testing.assert_allclose(
            logsigs_inc[:, -1, :].numpy(),
            logsig_direct,
            atol=1e-5,
        )
