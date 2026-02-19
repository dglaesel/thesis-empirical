"""Phase 1: Reproduce Bank et al. (2024) Table 1 — 1D tracking problem.

State:     Y_t in R (scalar tracking variable)
Dynamics:  dY_t = u_t dt + dB^H_t
Cost:      J(u) = E[1/2 * integral_0^T (Y_t^2 + kappa * u_t^2) dt]
Control:   u_t = F_theta(LogSig^{<=N}(X_hat)_{0,t})  where X_hat = (t, B^H_t)

At H = 1/2, the optimal cost is J* = 1/2 * kappa * ln(cosh(T / sqrt(kappa)))
(Riccati benchmark, Bank et al. Theorem 5.2).
"""

import argparse
import csv
import gc
import io
import sys
import os
import time
from datetime import datetime


class _Tee:
    """Duplicate stdout to a StringIO buffer."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

import copy

import numpy as np
import torch
import torch.nn as nn
import yaml

# Add project root to path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)

from src.fbm import simulate_fbm
from src.signatures import time_augment, compute_logsignatures, logsig_dim, prepare_join
from src.policy import LinearPolicy, DNNStrategy
from src.riccati import optimal_cost_bank


# ---------------------------------------------------------------------------
# Log-signature dimension lookup (Witt formula, d=2)
# ---------------------------------------------------------------------------

LOGSIG_DIMS = {1: 2, 2: 3, 3: 5, 4: 8, 5: 14}


# ---------------------------------------------------------------------------
# Bank et al. Table 1 reference values
# ---------------------------------------------------------------------------

BANK_THEORETICAL = {0.25: 0.206, 0.50: 0.124, 0.75: 0.071}
BANK_ALIN = {
    (0.25, 1): 0.223, (0.25, 2): 0.211, (0.25, 3): 0.210, (0.25, 4): 0.209, (0.25, 5): 0.209,
    (0.50, 1): 0.146, (0.50, 2): 0.127, (0.50, 3): 0.124, (0.50, 4): 0.124, (0.50, 5): 0.124,
    (0.75, 1): 0.101, (0.75, 2): 0.076, (0.75, 3): 0.073, (0.75, 4): 0.073, (0.75, 5): 0.073,
}
BANK_ADNN = {
    (0.25, 1): 0.221, (0.25, 2): 0.209, (0.25, 3): 0.208, (0.25, 4): 0.208, (0.25, 5): 0.208,
    (0.50, 1): 0.142, (0.50, 2): 0.125, (0.50, 3): 0.124, (0.50, 4): 0.124, (0.50, 5): 0.124,
    (0.75, 1): 0.096, (0.75, 2): 0.074, (0.75, 3): 0.072, (0.75, 4): 0.072, (0.75, 5): 0.072,
}


# ---------------------------------------------------------------------------
# Config / device helpers
# ---------------------------------------------------------------------------

def load_config(path: str = None) -> dict:
    """Load YAML config file."""
    if path is None:
        path = os.path.join(_project_root, "config", "default.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(config: dict) -> torch.device:
    """Resolve device from config."""
    dev = config.get("device", "auto")
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Bank et al. Table 1 reproduction"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file. Defaults to config/default.yaml."
    )
    parser.add_argument(
        "--H", type=float, default=None,
        help="Run a single H value (e.g. --H 0.25). "
             "If omitted, run all H values from config."
    )
    parser.add_argument(
        "--N", type=int, default=None,
        help="Run a single truncation level (e.g. --N 3). "
             "If omitted, run all N values from config."
    )
    parser.add_argument(
        "--mode", choices=["alin", "adnn", "both", "legacy"],
        default="both",
        help="alin: Bank A_lin (linear on sig), adnn: Bank A_dnn (DNN on logsig), "
             "both: run A_lin then A_dnn, legacy: original open-loop approach"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward_pass_1d(
    B_H: torch.Tensor,
    logsigs: torch.Tensor,
    policy: LinearPolicy,
    T: float,
    K: int,
    kappa: float = 0.1,
) -> torch.Tensor:
    """Vectorised forward pass for the 1D tracking problem.

    Euler discretisation of dY_t = u_t dt + dB^H_t with left-endpoint
    Riemann sum for J = E[1/2 * integral_0^T (Y_t^2 + kappa * u_t^2) dt].

    Args:
        B_H: fBM paths [M, K+1].
        logsigs: Log-signatures [M, K+1, dim_logsig].
        policy: Linear policy mapping features -> control.
        T: Time horizon.
        K: Number of time steps.
        kappa: Control penalty parameter.

    Returns:
        Scalar mean cost (loss to minimise).
    """
    M = B_H.shape[0]
    dt = T / K
    device = B_H.device

    Y = torch.zeros(M, device=device)
    cost = torch.zeros(M, device=device)

    for k in range(K):
        # At k=0, logsig is zero (trivial path) so u_0 = bias only.
        feat = logsigs[:, k, :]       # [M, dim_logsig]
        u = policy(feat)               # [M]

        dB = B_H[:, k + 1] - B_H[:, k]  # fBM increment

        cost = cost + 0.5 * (Y ** 2 + kappa * u ** 2) * dt
        Y = Y + u * dt + dB

    return cost.mean()


# ---------------------------------------------------------------------------
# Pre-computation
# ---------------------------------------------------------------------------

def precompute_paths_and_logsigs(
    H: float,
    N_max: int,
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate fBM and compute log-signatures at N_max (once per H).

    Args:
        H: Hurst parameter.
        N_max: Maximum truncation level.
        config: Full config dict.

    Returns:
        (B_H, paths_aug, logsigs_full):
            B_H [M, K+1], paths_aug [M, K+1, 2], logsigs_full [M, K+1, dim_N_max].
    """
    p1 = config["phase1"]
    T = p1["T"]
    K = p1["K"]
    M = p1["M"]
    seed = config.get("seed", 42)

    # Step 1: Simulate fBM paths
    print(f"  Simulating fBM paths (H={H}, M={M}, K={K})...")
    t0 = time.time()
    B_H = simulate_fbm(H, T, K, M, device=torch.device("cpu"), seed=seed)
    t_fbm = time.time() - t0
    print(f"    Done in {t_fbm:.1f}s, shape={B_H.shape}")

    # Step 2: Time-augment
    paths_aug = time_augment(B_H, T, K)  # [M, K+1, 2]

    # Step 3: Compute log-signatures at N_max
    print(f"  Computing log-signatures at N_max={N_max} (CPU, one-time)...")
    t0 = time.time()
    logsigs_full = compute_logsignatures(paths_aug, N_max)  # [M, K+1, dim_N_max]
    t_logsig = time.time() - t0
    print(f"    Done in {t_logsig:.1f}s, logsig dim={logsigs_full.shape[-1]}")

    return B_H, paths_aug, logsigs_full


def verify_logsig_prefix(paths_aug: torch.Tensor, N_max: int) -> None:
    """Verify that log-signature at level N is a prefix of level N_max.

    Uses a single path to confirm iisignature's Lyndon basis ordering
    gives the prefix property.
    """
    import iisignature as iis

    test_path_np = paths_aug[0].numpy()  # single path [K+1, 2]
    d = test_path_np.shape[-1]
    s_low = iis.prepare(d, N_max - 1)
    s_high = iis.prepare(d, N_max)
    ls_low = iis.logsig(test_path_np, s_low)   # [K, dim_{N_max-1}]
    ls_high = iis.logsig(test_path_np, s_high)  # [K, dim_{N_max}]
    dim_low = LOGSIG_DIMS[N_max - 1]
    assert np.allclose(ls_low, ls_high[:dim_low], atol=1e-6), \
        "Log-signature prefix property VIOLATED — cannot use slice optimisation"
    print("  [check] Log-signature prefix property verified")


# ---------------------------------------------------------------------------
# Training for a single (H, N) pair
# ---------------------------------------------------------------------------

def train_single(
    B_H: torch.Tensor,
    logsigs: torch.Tensor,
    H: float,
    N: int,
    config: dict,
    device: torch.device,
) -> dict:
    """Train Phase 1 for a single (H, N) pair using pre-computed data.

    Args:
        B_H: fBM paths [M, K+1] (CPU).
        logsigs: Log-signatures [M, K+1, dim_N] already sliced for this N (CPU).
        H: Hurst parameter.
        N: Truncation level.
        config: Full config dict.
        device: Torch device for training.

    Returns:
        Dictionary with keys: final_cost, training_curve, policy_state_dict, timing.
    """
    p1 = config["phase1"]
    T = p1["T"]
    K = p1["K"]
    M = p1["M"]
    kappa = p1["kappa"]
    batch_size = p1["batch_size"]
    num_epochs = p1["num_epochs"]
    lr = p1["learning_rate"]
    grad_clip = config.get("grad_clip", 1.0)
    eval_interval = 50

    dim_ls = logsigs.shape[-1]
    run_start = time.time()

    print(f"\n  --- Training H={H}, N={N} (logsig dim={dim_ls}) ---")

    # Move to device
    B_H_dev = B_H.to(device)
    logsigs_dev = logsigs.to(device)

    # Setup policy and optimiser
    policy = LinearPolicy(input_dim=dim_ls).to(device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)

    num_params = sum(p.numel() for p in policy.parameters())
    print(f"    Policy parameters: {num_params}")

    # Training loop
    print("    Training...")
    t0 = time.time()
    training_curve = []

    for epoch in range(num_epochs):
        # Sample batch
        idx = torch.randint(0, M, (batch_size,))
        B_H_batch = B_H_dev[idx]
        logsig_batch = logsigs_dev[idx]

        # Forward + backward
        optimiser.zero_grad()
        loss = forward_pass_1d(B_H_batch, logsig_batch, policy, T, K, kappa)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimiser.step()

        batch_cost = loss.item()

        # Periodic full-dataset evaluation
        full_eval_cost = None
        if (epoch + 1) % eval_interval == 0 or epoch == 0:
            policy.eval()
            with torch.no_grad():
                full_eval_cost = forward_pass_1d(
                    B_H_dev, logsigs_dev, policy, T, K, kappa
                ).item()
            policy.train()
            print(
                f"    Epoch {epoch+1:4d}/{num_epochs} | "
                f"batch={batch_cost:.6f} | full_eval={full_eval_cost:.6f}"
            )

        training_curve.append((epoch + 1, batch_cost, full_eval_cost))

    t_training = time.time() - t0

    # Final evaluation on full dataset
    policy.eval()
    with torch.no_grad():
        final_cost = forward_pass_1d(
            B_H_dev, logsigs_dev, policy, T, K, kappa
        ).item()

    t_total = time.time() - run_start

    print(f"    Final cost (full dataset): {final_cost:.6f}")
    print(f"    Policy weights: {policy.linear.weight.data.cpu().numpy().flatten()}")
    print(f"    Policy bias:    {policy.linear.bias.data.cpu().item():.6f}")
    print(f"    Training time: {t_training:.1f}s, total: {t_total:.1f}s")

    return {
        "final_cost": final_cost,
        "training_curve": training_curve,
        "policy_state_dict": {
            k: v.cpu() for k, v in policy.state_dict().items()
        },
        "timing": {
            "training_s": round(t_training, 1),
            "total_s": round(t_total, 1),
        },
    }


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def _save_config_snapshot(config: dict, out_dir: str) -> None:
    """Save a frozen copy of the config used for this run."""
    path = os.path.join(out_dir, "config_snapshot.yaml")
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _save_training_curve(
    training_curve: list, H: float, N: int, out_dir: str
) -> None:
    """Save training curve as CSV."""
    path = os.path.join(out_dir, f"training_curve_H{H}_N{N}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "batch_cost", "full_eval_cost"])
        for epoch, batch_cost, full_eval_cost in training_curve:
            writer.writerow([
                epoch,
                f"{batch_cost:.8f}",
                f"{full_eval_cost:.8f}" if full_eval_cost is not None else "",
            ])


def _save_policy(
    state_dict: dict, H: float, N: int, out_dir: str, mode_tag: str | None = None
) -> None:
    """Save policy weights as .pt file."""
    suffix = f"_{mode_tag}" if mode_tag else ""
    path = os.path.join(out_dir, f"policy_H{H}_N{N}{suffix}.pt")
    torch.save(state_dict, path)


def _save_summary_csv(
    results: dict,
    riccati_cost: float,
    H_values: list,
    N_values: list,
    out_dir: str,
) -> None:
    """Save summary table as CSV with Bank et al. comparison columns."""
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "H", "N", "our_cost", "riccati_cost",
            "bank_theoretical", "bank_alin",
            "rel_error_vs_theory", "rel_error_vs_bank",
            "time_training_s", "time_total_s",
        ])
        for H in H_values:
            bank_theo = BANK_THEORETICAL.get(H, "")
            for N in N_values:
                info = results.get((H, N))
                if info is None:
                    continue
                cost = info["final_cost"]
                t = info["timing"]
                bank_alin = BANK_ALIN.get((H, N), "")

                # Relative error vs Riccati (H=0.5 only)
                if abs(H - 0.5) < 1e-6:
                    rel_theory = abs(cost - riccati_cost) / riccati_cost
                    rel_theory_str = f"{rel_theory:.6f}"
                else:
                    rel_theory_str = ""

                # Relative error vs Bank A_lin
                if bank_alin != "":
                    rel_bank = abs(cost - bank_alin) / bank_alin
                    rel_bank_str = f"{rel_bank:.6f}"
                else:
                    rel_bank_str = ""

                writer.writerow([
                    f"{H:.3f}", N, f"{cost:.8f}",
                    f"{riccati_cost:.8f}" if abs(H - 0.5) < 1e-6 else "",
                    f"{bank_theo}" if bank_theo != "" else "",
                    f"{bank_alin}" if bank_alin != "" else "",
                    rel_theory_str, rel_bank_str,
                    t["training_s"], t["total_s"],
                ])


def _save_bank_summary_csv(
    results: dict,
    riccati_cost: float,
    H_values: list,
    N_values: list,
    modes_run: list,
    out_dir: str,
) -> None:
    """Save Bank-faithful summary as CSV."""
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "H", "N", "mode", "our_cost", "our_var", "best_val_cost",
            "bank_ref", "rel_error_vs_bank", "time_total_s",
        ])
        for H in H_values:
            for space, nn_hidden in modes_run:
                label = "A_lin" if space == "sig" else "A_dnn"
                ref = BANK_ALIN if space == "sig" else BANK_ADNN
                for N in N_values:
                    info = results.get((H, N, space))
                    if info is None:
                        continue
                    cost = info["final_cost"]
                    var = info.get("final_var", "")
                    best_val = info.get("best_val_cost", "")
                    t = info["timing"]
                    bank_val = ref.get((H, N), "")

                    if bank_val != "":
                        rel = abs(cost - bank_val) / bank_val
                        rel_str = f"{rel:.6f}"
                    else:
                        rel_str = ""

                    writer.writerow([
                        f"{H:.3f}", N, label, f"{cost:.8f}",
                        f"{var:.8f}" if var != "" else "",
                        f"{best_val:.8f}" if best_val != "" else "",
                        f"{bank_val}" if bank_val != "" else "",
                        rel_str, t["total_s"],
                    ])


# ---------------------------------------------------------------------------
# Validation criteria
# ---------------------------------------------------------------------------

def _validate_results(
    results: dict,
    riccati_cost: float,
    H_values: list,
    N_values: list,
) -> bool:
    """Run 4 PASS/FAIL criteria from Step 8.

    Returns True if overall PASS.
    """
    all_pass = True

    print("\nVALIDATION:")

    # [1] Riccati match: best cost at H=0.5 within 2% of J*
    PASS_THRESHOLD = 0.02
    h05_costs = {N: results[(0.5, N)]["final_cost"]
                 for N in N_values if (0.5, N) in results}
    if h05_costs:
        best_cost = min(h05_costs.values())
        best_N = min(h05_costs, key=h05_costs.get)
        rel_err = abs(best_cost - riccati_cost) / riccati_cost
        status = "PASS" if rel_err < PASS_THRESHOLD else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [1] Riccati (H=0.5): best cost = {best_cost:.6f} "
              f"(N={best_N}) vs J*={riccati_cost:.4f} "
              f"-> {rel_err:.1%} -> {status}")
    else:
        print("  [1] Riccati (H=0.5): SKIPPED (H=0.5 not in grid)")

    # [2] N-convergence: for each H, cost at N=3 < cost at N=1
    print("  [2] N-convergence:")
    for H in H_values:
        c1 = results.get((H, 1), {}).get("final_cost")
        c3 = results.get((H, 3), {}).get("final_cost")
        if c1 is not None and c3 is not None:
            status = "PASS" if c3 < c1 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"      H={H}: N=1 ({c1:.4f}) > N=3 ({c3:.4f}) -> {status}")
        else:
            print(f"      H={H}: SKIPPED (missing N=1 or N=3)")

    # [3] H-monotonicity: for each N, cost(H=0.25) > cost(H=0.5) > cost(H=0.75)
    print("  [3] H-monotonicity:")
    for N in N_values:
        c_lo = results.get((0.25, N), {}).get("final_cost")
        c_mid = results.get((0.50, N), {}).get("final_cost")
        c_hi = results.get((0.75, N), {}).get("final_cost")
        if c_lo is not None and c_mid is not None and c_hi is not None:
            ok = c_lo > c_mid > c_hi
            status = "PASS" if ok else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"      N={N}: {c_lo:.4f} > {c_mid:.4f} > {c_hi:.4f} -> {status}")
        else:
            print(f"      N={N}: SKIPPED (missing H values)")

    # [4] Within 15% of Bank A_lin
    print("  [4] Bank comparison:")
    max_dev = 0.0
    max_dev_key = None
    for H in H_values:
        for N in N_values:
            bank_val = BANK_ALIN.get((H, N))
            our_val = results.get((H, N), {}).get("final_cost")
            if bank_val is not None and our_val is not None:
                dev = abs(our_val - bank_val) / bank_val
                if dev > max_dev:
                    max_dev = dev
                    max_dev_key = (H, N)
    if max_dev_key is not None:
        status = "PASS" if max_dev < 0.15 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"      max deviation = {max_dev:.1%} at (H={max_dev_key[0]}, "
              f"N={max_dev_key[1]}) -> {status}")
    else:
        print("      SKIPPED (no Bank reference values matched)")

    print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ---------------------------------------------------------------------------
# Bank-faithful closed-loop mode (A_lin / A_dnn)
# ---------------------------------------------------------------------------

def generate_training_data(H, K, T, MC, seed=None):
    """Generate time-augmented fBM paths [MC, K+1, 2] in float64."""
    B_H = simulate_fbm(H, T, K, MC, device=torch.device("cpu"), seed=seed)
    paths_aug = time_augment(B_H, T, K)
    return paths_aug.to(dtype=torch.float64)


def forward_pass_bank(
    X: torch.Tensor,
    policy: nn.Module,
    T: float,
    K: int,
    kappa: float,
    update_fn,
    feat_dim: int,
    space: str = "sig",
) -> tuple:
    """Closed-loop forward pass faithful to Bank et al. rde.py.

    Computes Sig/LogSig of the controlled state (t, Y) on-the-fly.
    The signature/logsig is detached (no grad flows through it).

    Feature updates:
    - 'sig': incremental Chen update via iisignature.sigjoin
    - 'log': logsig recomputation from the controlled path prefix

    Args:
        X: Time-augmented fBM paths [M, K+1, 2] (float64).
        policy: DNNStrategy mapping (log-)signature features to control.
        T: Time horizon.
        K: Number of time steps.
        kappa: Control penalty parameter.
        update_fn: From prepare_join.
            For 'sig': update_fn(ZZ_np, dZ_np) -> updated ZZ_np
            For 'log': update_fn(path_np, n_points) -> logsig_np
        feat_dim: Dimension of the (log-)signature features.
        space: 'sig' or 'log'.

    Returns:
        (loss, var): mean cost and variance of costs across batch.
    """
    if space not in {"sig", "log"}:
        raise ValueError(f"space must be 'sig' or 'log', got '{space}'")

    M = X.shape[0]
    dt = T / K

    Y = torch.zeros(M, dtype=torch.float64)
    cost = torch.zeros(M, dtype=torch.float64)

    if space == "sig":
        ZZ = np.zeros((M, feat_dim), dtype=np.float64)
    else:
        ctrl_path = np.zeros((M, K + 1, 2), dtype=np.float64)
        ZZ = np.zeros((M, feat_dim), dtype=np.float64)

    for k in range(K):
        # Control from current signature/logsig (detached — no grad)
        ZZ_t = torch.from_numpy(ZZ)  # [M, feat_dim], no grad
        u = policy(ZZ_t)             # [M], grad through policy params

        # Cost accumulation (left-endpoint Riemann sum)
        cost = cost + 0.5 * (Y ** 2 + kappa * u ** 2) * dt

        # Euler dynamics: dY = u*dt + dB^H
        dB = X[:, k + 1, 1] - X[:, k, 1]
        dY = u * dt + dB
        Y = Y + dY

        # Update features — detached, no grad
        with torch.no_grad():
            dY_np = dY.detach().cpu().numpy()
            if space == "sig":
                dZ = np.column_stack([
                    np.full(M, dt, dtype=np.float64),
                    dY_np,
                ])
                ZZ = update_fn(ZZ, dZ)
            else:
                ctrl_path[:, k + 1, 0] = ctrl_path[:, k, 0] + dt
                ctrl_path[:, k + 1, 1] = ctrl_path[:, k, 1] + dY_np
                ZZ = update_fn(ctrl_path, k + 2)

    loss = cost.mean()
    var = cost.var()
    return loss, var


def train_bank(
    H: float,
    N: int,
    space: str,
    nn_hidden: int,
    config: dict,
    device: torch.device,
) -> dict:
    """Train Phase 1 with Bank-faithful closed-loop approach.

    Args:
        H: Hurst parameter.
        N: Signature truncation level.
        space: 'sig' for A_lin, 'log' for A_dnn.
        nn_hidden: 0 for A_lin (linear), 2 for A_dnn (DNN).
        config: Full config dict.
        device: Torch device (CPU for Bank-faithful mode).

    Returns:
        dict with final_cost, final_var, best_val_cost, timing.
    """
    p1 = config["phase1"]
    pb = config["phase1_bank"]
    T = p1["T"]
    K = p1["K"]
    kappa = p1["kappa"]
    seed = config.get("seed", 42)

    batch_size = pb["batch_size"]
    n_batches = pb["n_batches"]
    lr = pb["learning_rate"]
    grad_clip = pb.get("grad_clip", config.get("grad_clip", 1.0))
    base_epochs = pb["base_epochs"]
    epochs_per_N = pb["epochs_per_N"]
    restarts = pb["restarts"]
    steps_per_restart = pb["steps_per_restart"]
    mc_eval = pb["mc_eval"]
    validation_size = pb["validation_size"]
    sig_comp = pb.get("sig_comp", "tY")
    train_data_mode = str(pb.get("train_data_mode", "stream")).strip().lower()

    if sig_comp != "tY":
        raise ValueError(
            "Current Bank-faithful implementation supports sig_comp='tY' only."
        )
    if train_data_mode not in {"stream", "preload"}:
        raise ValueError(
            f"phase1_bank.train_data_mode must be 'stream' or 'preload', got '{train_data_mode}'"
        )

    total_epochs = base_epochs + N * epochs_per_N
    mode_label = "A_lin" if space == "sig" else "A_dnn"
    update_fn, feat_dim, _ = prepare_join(d=len(sig_comp), N=N, space=space)

    run_start = time.time()
    time_train = 0.0
    time_val = 0.0

    print(f"\n  --- Training {mode_label}: H={H}, N={N} "
          f"(feat_dim={feat_dim}, epochs={total_epochs}, train_data_mode={train_data_mode}) ---")

    def _evaluate_model(model: nn.Module, data: torch.Tensor) -> float:
        model.eval()
        with torch.no_grad():
            losses = []
            eval_batch_size = min(batch_size, data.shape[0])
            n_eval_batches = max(1, data.shape[0] // eval_batch_size)
            for b in range(n_eval_batches):
                xb = data[b * eval_batch_size:(b + 1) * eval_batch_size]
                l_, _ = forward_pass_bank(
                    xb, model, T, K, kappa, update_fn, feat_dim, space=space
                )
                losses.append(l_.item())
        val = float(np.mean(losses))
        if not np.isfinite(val):
            return float("inf")
        return val

    best_val_cost = float("inf")
    best_state = None
    best_optim_state = None

    for restart in range(restarts):
        print(f"    Restart {restart + 1}/{restarts}...")

        strategy = DNNStrategy(feat_dim, nn_hidden=nn_hidden, nn_dropout=0.0)
        strategy.to(dtype=torch.float64)
        optimizer = torch.optim.Adam(strategy.parameters(), lr=lr)
        if best_state is None:
            # Keep a finite fallback checkpoint so warm-restart logic cannot fail.
            best_state = copy.deepcopy(strategy.state_dict())
            best_optim_state = copy.deepcopy(optimizer.state_dict())

        # Use a large restart stride so streamed per-batch seeds do not overlap.
        restart_seed_base = seed + restart * 10_000_000
        train_seed = restart_seed_base
        val_seed = restart_seed_base + 1
        if train_data_mode == "preload":
            print(f"      Generating training data ({batch_size * n_batches} paths, "
                  f"seed={train_seed})...")
            train_data = generate_training_data(H, K, T, batch_size * n_batches, seed=train_seed)
        else:
            print(f"      Streaming training data ({n_batches} x {batch_size} paths per epoch, "
                  f"seed_base={train_seed})...")
            train_data = None
        print(f"      Generating validation data ({validation_size} paths, "
              f"seed={val_seed})...")
        val_data = generate_training_data(H, K, T, validation_size, seed=val_seed)

        for epoch in range(total_epochs):
            t0 = time.time()
            val_cost = _evaluate_model(strategy, val_data)
            time_val += time.time() - t0

            print(f"      Restart {restart + 1}, epoch {epoch}: "
                  f"val={val_cost:.6f} (best={best_val_cost:.6f})")

            if np.isfinite(val_cost) and val_cost < best_val_cost:
                best_val_cost = val_cost
                best_state = copy.deepcopy(strategy.state_dict())
                best_optim_state = copy.deepcopy(optimizer.state_dict())
            elif not np.isfinite(val_cost):
                print("      Warning: non-finite validation cost encountered; "
                      "ignoring this checkpoint.")

            # Bank-faithful warm-restart behavior from run_environment.py:
            # after `steps_per_restart` validation checkpoints, either break
            # (early restarts) or load the best model and continue training.
            if epoch == steps_per_restart:
                if restart == restarts - 1:
                    strategy = DNNStrategy(feat_dim, nn_hidden=nn_hidden, nn_dropout=0.0)
                    strategy.to(dtype=torch.float64)
                    strategy.load_state_dict(best_state)
                    optimizer = torch.optim.Adam(strategy.parameters(), lr=lr)
                    if best_optim_state is not None:
                        optimizer.load_state_dict(best_optim_state)
                    print("      Loaded best restart model; continuing full training...")
                else:
                    break
            elif epoch == total_epochs - 1:
                break

            strategy.train()
            t0 = time.time()
            epoch_loss = 0.0
            batches_done = 0
            non_finite_train = False
            for b in range(n_batches):
                if train_data_mode == "preload":
                    xb = train_data[b * batch_size:(b + 1) * batch_size]
                else:
                    # Deterministic streamed batches for reproducibility.
                    batch_seed = train_seed + epoch * n_batches + b
                    xb = generate_training_data(H, K, T, batch_size, seed=batch_seed)
                optimizer.zero_grad()
                loss, _ = forward_pass_bank(
                    xb, strategy, T, K, kappa, update_fn, feat_dim, space=space
                )
                if not torch.isfinite(loss):
                    print(f"      Warning: non-finite training loss at "
                          f"restart={restart + 1}, epoch={epoch}, batch={b}; "
                          "stopping this restart.")
                    non_finite_train = True
                    break
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(strategy.parameters(), grad_clip)
                optimizer.step()
                epoch_loss += loss.item()
                batches_done += 1
            if non_finite_train:
                break
            if batches_done == 0:
                print("      Warning: no valid batches processed in epoch; "
                      "stopping this restart.")
                break
            epoch_loss /= batches_done
            time_train += time.time() - t0

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"      Restart {restart + 1}, epoch {epoch + 1}/{total_epochs}: "
                      f"train={epoch_loss:.6f}")

        if train_data is not None:
            del train_data
        del val_data
        gc.collect()

    if best_state is None:
        raise RuntimeError("Bank training failed to produce a best model state.")

    strategy = DNNStrategy(feat_dim, nn_hidden=nn_hidden, nn_dropout=0.0)
    strategy.to(dtype=torch.float64)
    strategy.load_state_dict(best_state)
    strategy.eval()

    print(f"    Final evaluation ({mc_eval} fresh paths)...")
    eval_data = generate_training_data(H, K, T, mc_eval, seed=seed + 8000)
    with torch.no_grad():
        eval_losses = []
        eval_vars = []
        eval_batch_size = min(batch_size, mc_eval)
        n_eval_batches = max(1, mc_eval // eval_batch_size)
        for b in range(n_eval_batches):
            eb = eval_data[b * eval_batch_size:(b + 1) * eval_batch_size]
            el, ev = forward_pass_bank(
                eb, strategy, T, K, kappa, update_fn, feat_dim, space=space
            )
            eval_losses.append(el.item())
            eval_vars.append(ev.item())
        final_cost = float(np.mean(eval_losses))
        final_var = float(np.mean(eval_vars))

    t_total = time.time() - run_start
    print(f"    Final cost: {final_cost:.6f} (var={final_var:.6f})")
    print(f"    Total time: {t_total:.1f}s ({t_total / 60:.1f} min)")

    del eval_data
    gc.collect()

    return {
        "final_cost": final_cost,
        "final_var": final_var,
        "best_val_cost": best_val_cost,
        "timing": {
            "total_s": round(t_total, 1),
            "training_s": round(time_train, 1),
            "validation_s": round(time_val, 1),
        },
        "policy_state_dict": {
            k: v.cpu() for k, v in best_state.items()
        },
    }


def _print_bank_table(
    results: dict,
    H_values: list,
    N_values: list,
    kappa: float,
    T: float,
    K: int,
    modes_run: list,
) -> None:
    """Print Bank et al. comparison table for A_lin and/or A_dnn."""
    print(f"\n{'='*80}")
    print("Phase 1 Results — Bank et al. Table 1 Reproduction (Bank-faithful)")
    print(f"{'='*80}")
    print(f"Cost: 1/2 * integral(Y^2 + {kappa}*u^2) dt, T={T}, K={K}")

    # Header
    n_cols = "".join(f" | {'N='+str(n):>6s}" for n in N_values)
    print(f"\n         {n_cols} | th.opt")
    print(f"---------" + "+--------" * len(N_values) + "+-------")

    for H in H_values:
        bank_theo = BANK_THEORETICAL.get(H, float("nan"))

        for space, nn_hidden in modes_run:
            label = "A_lin" if space == "sig" else "A_dnn"
            ref = BANK_ALIN if space == "sig" else BANK_ADNN

            # Our results
            our_vals = ""
            for N in N_values:
                cost = results.get((H, N, space), {}).get("final_cost")
                if cost is not None:
                    our_vals += f" | {cost:6.3f}"
                else:
                    our_vals += " |    n/a"
            print(f"H={H:<4} {label:>5}{our_vals} | {bank_theo:.3f}")

            # Bank reference
            bank_vals = ""
            for N in N_values:
                bv = ref.get((H, N))
                if bv is not None:
                    bank_vals += f" | {bv:6.3f}"
                else:
                    bank_vals += " |    n/a"
            print(f"  Bank {label:>5}{bank_vals} |")

        print(f"---------" + "+--------" * len(N_values) + "+-------")

    print(f"{'='*80}")


def _validate_bank_results(
    results: dict,
    riccati_cost: float,
    H_values: list,
    N_values: list,
    modes_run: list,
) -> bool:
    """Validate Bank-faithful results against reference values."""
    all_pass = True
    print("\nVALIDATION (Bank-faithful):")

    for space, nn_hidden in modes_run:
        label = "A_lin" if space == "sig" else "A_dnn"
        ref = BANK_ALIN if space == "sig" else BANK_ADNN

        # [1] N-convergence: cost at N=3 < cost at N=1
        print(f"  [{label}] N-convergence:")
        for H in H_values:
            c1 = results.get((H, 1, space), {}).get("final_cost")
            c3 = results.get((H, 3, space), {}).get("final_cost")
            if c1 is not None and c3 is not None:
                status = "PASS" if c3 < c1 else "FAIL"
                if status == "FAIL":
                    all_pass = False
                print(f"      H={H}: N=1 ({c1:.4f}) > N=3 ({c3:.4f}) -> {status}")

        # [2] H-monotonicity
        print(f"  [{label}] H-monotonicity:")
        for N in N_values:
            c_lo = results.get((0.25, N, space), {}).get("final_cost")
            c_mid = results.get((0.50, N, space), {}).get("final_cost")
            c_hi = results.get((0.75, N, space), {}).get("final_cost")
            if c_lo is not None and c_mid is not None and c_hi is not None:
                ok = c_lo > c_mid > c_hi
                status = "PASS" if ok else "FAIL"
                if status == "FAIL":
                    all_pass = False
                print(f"      N={N}: {c_lo:.4f} > {c_mid:.4f} > {c_hi:.4f} -> {status}")

        # [3] Within 15% of Bank reference
        print(f"  [{label}] Bank comparison:")
        max_dev = 0.0
        max_dev_key = None
        for H in H_values:
            for N in N_values:
                bank_val = ref.get((H, N))
                our_val = results.get((H, N, space), {}).get("final_cost")
                if bank_val is not None and our_val is not None:
                    dev = abs(our_val - bank_val) / bank_val
                    if dev > max_dev:
                        max_dev = dev
                        max_dev_key = (H, N)
        if max_dev_key is not None:
            status = "PASS" if max_dev < 0.15 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"      max deviation = {max_dev:.1%} at "
                  f"(H={max_dev_key[0]}, N={max_dev_key[1]}) -> {status}")

    # [4] Riccati match at H=0.5
    for space, nn_hidden in modes_run:
        label = "A_lin" if space == "sig" else "A_dnn"
        h05_costs = {N: results[(0.5, N, space)]["final_cost"]
                     for N in N_values if (0.5, N, space) in results}
        if h05_costs:
            best_cost = min(h05_costs.values())
            best_N = min(h05_costs, key=h05_costs.get)
            rel_err = abs(best_cost - riccati_cost) / riccati_cost
            status = "PASS" if rel_err < 0.02 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  [{label}] Riccati (H=0.5): best={best_cost:.6f} "
                  f"(N={best_N}) vs J*={riccati_cost:.4f} "
                  f"-> {rel_err:.1%} -> {status}")

    print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Phase 1 experiments over the H x N grid."""
    overall_start = time.time()
    args = parse_args()
    config = load_config(args.config)
    device = get_device(config)

    p1 = config["phase1"]
    T = p1["T"]
    kappa = p1["kappa"]
    N_values = p1["N_values"]

    # Determine H values: single-H mode or full grid
    if args.H is not None:
        H_values = [args.H]
    else:
        H_values = p1["H_values"]

    # Determine N values: single-N mode or full grid
    if args.N is not None:
        N_values = [args.N]

    N_max = max(N_values)

    # Timestamped output directory (never overwrite previous results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = args.mode
    tag_parts = []
    if mode != "legacy":
        tag_parts.append(f"bank_{mode}")
    if args.H is not None:
        tag_parts.append(f"H{args.H}")
    if args.N is not None:
        tag_parts.append(f"N{args.N}")
    tag = "_".join(tag_parts) if tag_parts else ""
    out_dir = os.path.join(
        _project_root, "results", "phase1",
        f"run_{timestamp}_{tag}" if tag else f"run_{timestamp}"
    )
    os.makedirs(out_dir, exist_ok=True)

    log_buffer = io.StringIO()
    sys.stdout = _Tee(sys.__stdout__, log_buffer)

    if args.H is not None:
        print(f"Single-H mode: running H={args.H} only")
    else:
        print(f"Full grid mode: H_values={H_values}")
    if args.N is not None:
        print(f"Single-N mode: running N={args.N} only")
    else:
        print(f"Full N grid: N_values={N_values}")

    print(f"Mode: {mode}")
    print(f"Device: {device}")
    print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results directory: {os.path.abspath(out_dir)}")

    # Save config snapshot
    _save_config_snapshot(config, out_dir)
    print(f"Config snapshot saved to {out_dir}/config_snapshot.yaml")

    # Riccati benchmark for H = 0.5
    riccati_cost = optimal_cost_bank(T, kappa)
    print(f"Riccati benchmark (H=0.5, T={T}, kappa={kappa}): "
          f"J* = {riccati_cost:.6f}")

    # ---------------------------------------------------------------
    # Legacy mode: original open-loop approach
    # ---------------------------------------------------------------
    if mode == "legacy":
        results = {}

        for H_idx, H in enumerate(H_values):
            print(f"\n{'='*60}")
            print(f"H = {H}  (pre-computing fBM + log-signatures at N_max={N_max})")
            print(f"{'='*60}")

            # Free previous H's tensors to avoid OOM
            if H_idx > 0:
                del B_H, paths_aug, logsigs_full
                gc.collect()

            # Pre-compute fBM + logsigs once for this H
            B_H, paths_aug, logsigs_full = precompute_paths_and_logsigs(
                H, N_max, config
            )

            # Verify prefix property (first H only)
            if H_idx == 0 and N_max > 1:
                verify_logsig_prefix(paths_aug, N_max)

            # Train for each N using sliced logsigs
            for N in N_values:
                dim_N = LOGSIG_DIMS[N]
                logsigs_N = logsigs_full[:, :, :dim_N]

                info = train_single(B_H, logsigs_N, H, N, config, device)
                results[(H, N)] = info

                # Save per-run outputs
                _save_training_curve(info["training_curve"], H, N, out_dir)
                _save_policy(info["policy_state_dict"], H, N, out_dir)

        # Save summary CSV
        _save_summary_csv(results, riccati_cost, H_values, N_values, out_dir)

        # Print Bank et al. comparison table
        print(f"\n{'='*72}")
        print("Phase 1 Results — Bank et al. Table 1 Reproduction")
        print(f"{'='*72}")
        print(f"Cost: 1/2 * integral(Y^2 + {kappa}*u^2) dt,  "
              f"T={T}, K={p1['K']}, M={p1['M']}")

        # Header
        n_cols = "".join(f" | {'N='+str(n):>6s}" for n in N_values)
        print(f"\n       {n_cols} | th.opt")
        print(f"-------" + "+--------" * len(N_values) + "+-------")

        for H in H_values:
            # Our results row
            our_vals = ""
            for N in N_values:
                cost = results.get((H, N), {}).get("final_cost")
                if cost is not None:
                    our_vals += f" | {cost:6.3f}"
                else:
                    our_vals += " |    n/a"
            bank_theo = BANK_THEORETICAL.get(H, float("nan"))
            print(f"H={H:<4}{our_vals} | {bank_theo:.3f}")

            # Bank reference row
            bank_vals = ""
            for N in N_values:
                bv = BANK_ALIN.get((H, N))
                if bv is not None:
                    bank_vals += f" | {bv:6.3f}"
                else:
                    bank_vals += " |    n/a"
            print(f" Bank:{bank_vals} |")

            print(f"-------" + "+--------" * len(N_values) + "+-------")

        print(f"{'='*72}")

        # Validation
        _validate_results(results, riccati_cost, H_values, N_values)

    # ---------------------------------------------------------------
    # Bank-faithful mode: A_lin, A_dnn, or both
    # ---------------------------------------------------------------
    else:
        modes_run = []
        if mode == "alin":
            modes_run = [("sig", 0)]
        elif mode == "adnn":
            modes_run = [("log", 2)]
        else:  # "both"
            modes_run = [("sig", 0), ("log", 2)]

        results = {}

        for H in H_values:
            print(f"\n{'='*60}")
            print(f"H = {H}  (Bank-faithful closed-loop)")
            print(f"{'='*60}")

            for space, nn_hidden in modes_run:
                for N in N_values:
                    result = train_bank(
                        H, N, space, nn_hidden, config, device
                    )
                    results[(H, N, space)] = result

                    # Save policy
                    label = "alin" if space == "sig" else "adnn"
                    _save_policy(
                        result["policy_state_dict"], H, N,
                        out_dir, mode_tag=label
                    )

        # Print comparison table
        _print_bank_table(
            results, H_values, N_values, kappa, T, p1["K"], modes_run
        )

        # Validation
        _validate_bank_results(
            results, riccati_cost, H_values, N_values, modes_run
        )

        # Save summary CSV for Bank mode
        _save_bank_summary_csv(
            results, riccati_cost, H_values, N_values, modes_run, out_dir
        )

    overall_elapsed = time.time() - overall_start
    print(f"\nTotal wall-clock time: {overall_elapsed:.0f}s "
          f"({overall_elapsed/60:.1f} min)")
    print(f"Run finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to {os.path.abspath(out_dir)}")

    # Save log to run directory
    sys.stdout = sys.__stdout__  # restore
    log_path = os.path.join(out_dir, "run_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_buffer.getvalue())
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
