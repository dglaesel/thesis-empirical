"""Fractional Brownian motion simulation via Cholesky and Davies-Harte methods."""

__all__ = [
    "simulate_fbm",
    "simulate_fbm_cholesky",
    "simulate_fbm_davies_harte",
    "fbm_time_grid",
    "clear_cache",
]

from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fbm_covariance(H: float, times: np.ndarray) -> np.ndarray:
    """Build the covariance matrix of fBM at given time points.

    R_H(s, t) = 0.5 * (|s|^{2H} + |t|^{2H} - |t - s|^{2H})
    """
    t = times[:, None]
    s = times[None, :]
    two_H = 2.0 * H
    return 0.5 * (np.abs(t) ** two_H + np.abs(s) ** two_H - np.abs(t - s) ** two_H)


# ---------------------------------------------------------------------------
# Cholesky method
# ---------------------------------------------------------------------------

_cholesky_cache: dict[Tuple, np.ndarray] = {}


def _get_cholesky_factor(H: float, T: float, K: int) -> np.ndarray:
    """Compute (and cache) the Cholesky factor L for fBM covariance."""
    key = (H, T, K)
    if key not in _cholesky_cache:
        times = np.linspace(0, T, K + 1)[1:]  # t_1, ..., t_K (exclude t_0=0)
        cov = _fbm_covariance(H, times)
        # Add small jitter for numerical stability
        cov += np.eye(K) * 1e-10
        L = np.linalg.cholesky(cov)
        _cholesky_cache[key] = L
    return _cholesky_cache[key]


def simulate_fbm_cholesky(
    H: float,
    T: float,
    K: int,
    M: int,
    device: torch.device = torch.device("cpu"),
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Simulate M paths of fBM via Cholesky decomposition.  O(K^3 + M*K^2).

    Args:
        H: Hurst parameter in (0, 1).
        T: Time horizon.
        K: Number of time steps.
        M: Number of paths.
        device: Torch device for the output tensor.
        seed: Optional random seed.

    Returns:
        Tensor of shape [M, K+1] with B^H_0 = 0.
    """
    if seed is not None:
        np.random.seed(seed)

    L = _get_cholesky_factor(H, T, K)  # [K, K]
    Z = np.random.randn(M, K)
    B_H_inner = Z @ L.T  # [M, K]

    B_H = np.zeros((M, K + 1))
    B_H[:, 1:] = B_H_inner
    return torch.tensor(B_H, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Davies-Harte (circulant embedding) method
# ---------------------------------------------------------------------------

_dh_cache: dict[Tuple, np.ndarray] = {}


def _get_dh_sqrt_eigenvalues(H: float, T: float, K: int) -> np.ndarray:
    """Compute (and cache) sqrt of circulant eigenvalues for Davies-Harte.

    The method embeds the K x K covariance matrix of fBM *increments* into a
    circulant matrix of size 2K, whose eigenvalues are obtained via FFT.

    Returns:
        Array of shape [K+1] — sqrt eigenvalues in rfft format (first K+1 of 2K).
        Raises ValueError if any eigenvalue is negative (should not happen for fBM).
    """
    key = (H, T, K)
    if key not in _dh_cache:
        dt = T / K
        two_H = 2.0 * H

        # Autocovariance of fBM increments at lag j:
        # gamma(j) = (dt^{2H} / 2) * (|j-1|^{2H} - 2|j|^{2H} + |j+1|^{2H})
        j = np.arange(K)
        gamma = 0.5 * dt ** two_H * (
            np.abs(j - 1) ** two_H - 2.0 * np.abs(j) ** two_H + (j + 1) ** two_H
        )

        # Build first row of the 2K x 2K circulant matrix:
        # [gamma(0), gamma(1), ..., gamma(K-1), gamma(K), gamma(K-1), ..., gamma(1)]
        gamma_K = 0.5 * dt ** two_H * (
            (K - 1) ** two_H - 2.0 * K ** two_H + (K + 1) ** two_H
        )
        # For rfft, we only need the first K+1 elements of the symmetric first row
        # rfft of length 2K returns K+1 complex values (the eigenvalues are real
        # because the first row is symmetric).
        row = np.zeros(2 * K)
        row[:K] = gamma
        row[K] = gamma_K
        row[K + 1:] = gamma[1:][::-1]

        # Eigenvalues via rfft (real-input FFT, returns K+1 values)
        eigenvalues = np.fft.rfft(row).real  # [K+1], real since row is symmetric

        if np.any(eigenvalues < -1e-10):
            raise ValueError(
                f"Davies-Harte failed: negative eigenvalue "
                f"(min={eigenvalues.min():.2e}) for H={H}, K={K}. "
                f"Use Cholesky method instead."
            )
        eigenvalues = np.maximum(eigenvalues, 0.0)
        _dh_cache[key] = np.sqrt(eigenvalues)

    return _dh_cache[key]


def simulate_fbm_davies_harte(
    H: float,
    T: float,
    K: int,
    M: int,
    device: torch.device = torch.device("cpu"),
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Simulate M paths of fBM via the Davies-Harte (circulant embedding) method.

    Complexity: O(K log K) for setup, O(M * K log K) for sampling.

    Uses the Hermitian FFT approach (irfft) for efficiency: the circulant
    matrix is real-symmetric, so its eigenvalues are real and the spectral
    sampling can use real-input FFTs, halving memory and computation.

    References: Davies & Harte (1987), Dieker (2004), Wood & Chan (1994).

    Args:
        H: Hurst parameter in (0, 1).
        T: Time horizon.
        K: Number of time steps.
        M: Number of paths.
        device: Torch device for the output tensor.
        seed: Optional random seed.

    Returns:
        Tensor of shape [M, K+1] with B^H_0 = 0.
    """
    if seed is not None:
        np.random.seed(seed)

    n = 2 * K
    sqrt_eig = _get_dh_sqrt_eigenvalues(H, T, K)  # [K+1]

    # Construct Hermitian-symmetric noise in rfft format: (M, K+1) complex.
    # Boundary terms (DC and Nyquist) are real; interior terms are complex.
    Z = np.empty((M, K + 1), dtype=np.complex128)
    Z[:, 0] = np.random.randn(M)                 # DC: real
    Z[:, K] = np.random.randn(M)                 # Nyquist: real
    Z[:, 1:K] = (
        np.random.randn(M, K - 1) + 1j * np.random.randn(M, K - 1)
    ) / np.sqrt(2)

    # Multiply by sqrt eigenvalues in frequency domain.
    # irfft normalises by 1/n, so scale by sqrt(n) to compensate.
    Z *= (sqrt_eig * np.sqrt(n))[np.newaxis, :]

    # irfft produces real output of length n=2K with correct covariance
    increments = np.fft.irfft(Z, n=n, axis=1)[:, :K]

    # Cumulative sum to get fBM values, prepend B^H_0 = 0
    B_H = np.zeros((M, K + 1))
    B_H[:, 1:] = np.cumsum(increments, axis=1)

    return torch.tensor(B_H, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def simulate_fbm(
    H: float,
    T: float,
    K: int,
    M: int,
    device: torch.device = torch.device("cpu"),
    seed: Optional[int] = None,
    method: str = "davies-harte",
) -> torch.Tensor:
    """Simulate M paths of fractional Brownian motion.

    Args:
        H: Hurst parameter in (0, 1).
        T: Time horizon.
        K: Number of time steps.
        M: Number of paths.
        device: Torch device for the output tensor.
        seed: Optional random seed.
        method: "davies-harte" (default, O(MK log K)) or "cholesky" (O(K^3 + MK^2)).

    Returns:
        Tensor of shape [M, K+1] with B^H_0 = 0.
    """
    if method == "davies-harte":
        return simulate_fbm_davies_harte(H, T, K, M, device=device, seed=seed)
    elif method == "cholesky":
        return simulate_fbm_cholesky(H, T, K, M, device=device, seed=seed)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'davies-harte' or 'cholesky'.")


def clear_cache() -> None:
    """Clear all cached decomposition factors."""
    _cholesky_cache.clear()
    _dh_cache.clear()


# Keep old name for backward compatibility with tests
clear_cholesky_cache = clear_cache


def fbm_time_grid(T: float, K: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Return the time grid [0, dt, 2*dt, ..., T] as a tensor."""
    return torch.linspace(0, T, K + 1, device=device)
