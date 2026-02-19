"""Log-signature computation using iisignature."""

__all__ = ["time_augment", "compute_logsignatures", "logsig_dim", "sig_dim", "prepare_join"]

import numpy as np
import torch
import iisignature
try:
    from tqdm import trange
except ImportError:
    def trange(n, **kwargs):
        return range(n)


def time_augment(
    paths: torch.Tensor,
    T: float,
    K: int,
) -> torch.Tensor:
    """Augment [M, K+1] paths with time channel -> [M, K+1, 2]."""
    M = paths.shape[0]
    device = paths.device
    times = torch.linspace(0, T, K + 1, device=device).unsqueeze(0).expand(M, -1)  # [M, K+1]
    return torch.stack([times, paths], dim=-1)  # [M, K+1, 2]


def compute_logsignatures(
    paths: torch.Tensor,
    N: int,
) -> torch.Tensor:
    """Prefix log-signatures via sub-path recomputation (O(K^2) fallback)."""
    M, K_plus_1, d = paths.shape
    K = K_plus_1 - 1
    device = paths.device

    paths_np = paths.detach().cpu().numpy().astype(np.float64)

    s = iisignature.prepare(d, N)
    dim_logsig = iisignature.logsiglength(d, N)

    logsigs = np.zeros((M, K_plus_1, dim_logsig), dtype=np.float32)
    # logsigs[:, 0, :] = 0  (trivial path at t=0)

    for k in trange(K, desc=f"Computing logsig (N={N})", leave=False):
        # Compute logsig of the sub-path paths[:, 0:k+2, :]
        logsigs[:, k + 1, :] = iisignature.logsig(paths_np[:, :k + 2, :], s)

    return torch.from_numpy(logsigs).to(device)


def logsig_dim(d: int, N: int) -> int:
    """Return the dimension of the level-N log-signature for d-dimensional paths."""
    return iisignature.logsiglength(d, N)


def sig_dim(d: int, N: int) -> int:
    """Return the dimension of the level-N (standard) signature for d-dimensional paths."""
    return iisignature.siglength(d, N)


def prepare_join(d: int, N: int, space: str):
    """Return (update_fn, dim, prep_obj) for incremental sig/logsig updates."""
    if space == "sig":
        feat_dim = iisignature.siglength(d, N)
        def join_fn(ZZ_np, dZ_np):
            return iisignature.sigjoin(ZZ_np, dZ_np, N).astype(np.float64, copy=False)
        return join_fn, feat_dim, N

    if space == "log":
        s = iisignature.prepare(d, N, "O")
        feat_dim = iisignature.logsiglength(d, N)
        has_logsigjoin = hasattr(iisignature, "logsigjoin")
        state = {"ZZ": None, "n_points": None}

        def logsig_fn(arg1, arg2):
            # Direct incremental: logsig_fn(ZZ_np, dZ_np)
            if isinstance(arg2, np.ndarray):
                ZZ_np = np.asarray(arg1, dtype=np.float64)
                dZ_np = np.asarray(arg2, dtype=np.float64)
                if has_logsigjoin:
                    return iisignature.logsigjoin(ZZ_np, dZ_np, s).astype(np.float64, copy=False)
                raise RuntimeError("logsigjoin unavailable")

            # Prefix-based: logsig_fn(path_np, n_points)
            path_np = np.asarray(arg1, dtype=np.float64)
            n_points = int(arg2)
            if n_points <= 1:
                return np.zeros((path_np.shape[0], feat_dim), dtype=np.float64)

            if has_logsigjoin:
                if state["ZZ"] is None or n_points == 2 or state["ZZ"].shape[0] != path_np.shape[0]:
                    state["ZZ"] = np.zeros((path_np.shape[0], feat_dim), dtype=np.float64)
                    state["n_points"] = 1
                if state["n_points"] is None or n_points != state["n_points"] + 1:
                    state["ZZ"] = iisignature.logsig(path_np[:, :n_points, :], s).astype(np.float64, copy=False)
                    state["n_points"] = n_points
                    return state["ZZ"]
                dZ_np = path_np[:, n_points - 1, :] - path_np[:, n_points - 2, :]
                state["ZZ"] = iisignature.logsigjoin(state["ZZ"], dZ_np, s).astype(np.float64, copy=False)
                state["n_points"] = n_points
                return state["ZZ"]

            return iisignature.logsig(path_np[:, :n_points, :], s).astype(np.float64, copy=False)

        return logsig_fn, feat_dim, s

    raise ValueError(f"space must be 'sig' or 'log', got '{space}'")
