"""Create dedicated, interpretable H=0.25 plots for a Phase 2 run.

Usage:
  ./.venv-phase2/Scripts/python scripts/plot_phase2_h025_dashboard.py --run-id 20260218_013716
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot H=0.25 dashboard for Phase 2 run.")
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Default: notes/plots/phase2_smoke_<run_id>",
    )
    return p.parse_args()


def _load_rows(run_root: str) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(run_root, "**", "summary.json"), recursive=True))
    rows: list[dict] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
        rows.append(
            {
                "H": float(j["H"]),
                "N": int(j["N"]),
                "seed": int(j["seed"]),
                "J_test": float(j["test_metrics"]["J"]),
                "E_WT": float(j["test_metrics"]["E_WT"]),
                "E_QT2": float(j["test_metrics"]["E_QT2"]),
                "turnover": float(j["test_metrics"]["turnover"]),
                "J_OU": float(j["baselines"]["OU_linear"]["J_test"]),
                "E_WT_OU": float(j["baselines"]["OU_linear"]["E_WT"]),
                "E_QT2_OU": float(j["baselines"]["OU_linear"]["E_QT2"]),
                "turnover_OU": float(j["baselines"]["OU_linear"]["turnover"]),
            }
        )
    rows.sort(key=lambda r: (r["H"], r["N"], r["seed"]))
    return rows


def _load_curves(run_root: str, target_h: float = 0.25) -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
    out: dict[int, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    paths = sorted(glob.glob(os.path.join(run_root, "**", "training_curve.csv"), recursive=True))
    for p in paths:
        parts = p.replace("/", "\\").split("\\")
        h_tag = next(x for x in parts if x.startswith("H_"))
        n_tag = next(x for x in parts if x.startswith("N_"))
        H = float(h_tag.replace("H_", "").replace("p", "."))
        N = int(n_tag.split("_")[1])
        if abs(H - target_h) > 1e-12:
            continue
        epochs: list[int] = []
        test_j: list[float] = []
        with open(p, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for row in rd:
                tj = row["test_J"]
                if tj.lower() == "nan":
                    continue
                epochs.append(int(row["epoch"]))
                test_j.append(float(tj))
        out[N].append((np.array(epochs, dtype=int), np.array(test_j, dtype=float)))
    return out


def _mean_sd(vals: list[float]) -> tuple[float, float]:
    arr = np.array(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def _build_h025_table(rows: list[dict]) -> dict[int, dict]:
    h_rows = [r for r in rows if abs(r["H"] - 0.25) < 1e-12]
    table: dict[int, dict] = {}
    for n in sorted(set(r["N"] for r in h_rows)):
        g = [r for r in h_rows if r["N"] == n]
        j_mean, j_sd = _mean_sd([x["J_test"] for x in g])
        d = [x["J_test"] - x["J_OU"] for x in g]
        d_mean, d_sd = _mean_sd(d)
        ewt_mean, ewt_sd = _mean_sd([x["E_WT"] for x in g])
        eq_mean, eq_sd = _mean_sd([x["E_QT2"] for x in g])
        to_mean, to_sd = _mean_sd([x["turnover"] for x in g])
        table[n] = {
            "rows": g,
            "J_mean": j_mean,
            "J_sd": j_sd,
            "dOU_mean": d_mean,
            "dOU_sd": d_sd,
            "wins": int(np.sum(np.array(d, dtype=float) > 0)),
            "EWT_mean": ewt_mean,
            "EWT_sd": ewt_sd,
            "EQT2_mean": eq_mean,
            "EQT2_sd": eq_sd,
            "TO_mean": to_mean,
            "TO_sd": to_sd,
            "J_OU": float(np.mean([x["J_OU"] for x in g])),
            "EWT_OU": float(np.mean([x["E_WT_OU"] for x in g])),
            "EQT2_OU": float(np.mean([x["E_QT2_OU"] for x in g])),
            "TO_OU": float(np.mean([x["turnover_OU"] for x in g])),
        }
    return table


def _plot_dashboard(table: dict[int, dict], out_path: str) -> None:
    Ns = sorted(table.keys())
    x = np.array(Ns, dtype=float)

    j_mean = np.array([table[n]["J_mean"] for n in Ns], dtype=float)
    j_sd = np.array([table[n]["J_sd"] for n in Ns], dtype=float)
    j_ou = float(np.mean([table[n]["J_OU"] for n in Ns]))
    d_mean = np.array([table[n]["dOU_mean"] for n in Ns], dtype=float)
    d_sd = np.array([table[n]["dOU_sd"] for n in Ns], dtype=float)
    wins = np.array([table[n]["wins"] for n in Ns], dtype=float)

    ewt_mean = np.array([table[n]["EWT_mean"] for n in Ns], dtype=float)
    ewt_sd = np.array([table[n]["EWT_sd"] for n in Ns], dtype=float)
    ewt_ou = float(np.mean([table[n]["EWT_OU"] for n in Ns]))

    eq_mean = np.array([table[n]["EQT2_mean"] for n in Ns], dtype=float)
    eq_sd = np.array([table[n]["EQT2_sd"] for n in Ns], dtype=float)
    eq_ou = float(np.mean([table[n]["EQT2_OU"] for n in Ns]))

    to_mean = np.array([table[n]["TO_mean"] for n in Ns], dtype=float)
    to_sd = np.array([table[n]["TO_sd"] for n in Ns], dtype=float)
    to_ou = float(np.mean([table[n]["TO_OU"] for n in Ns]))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("H=0.25 Dashboard: A_DNN Improvement and Metric Trade-offs", fontsize=16)

    # (1) J_test vs N
    ax = axes[0, 0]
    ax.plot(x, j_mean, marker="o", linewidth=2.2, label="A_DNN mean J_test")
    ax.fill_between(x, j_mean - j_sd, j_mean + j_sd, alpha=0.2, label="±1 sd")
    ax.axhline(j_ou, linestyle="--", linewidth=1.8, label="OU mean")
    ax.set_title("Performance")
    ax.set_xlabel("N")
    ax.set_ylabel("J_test")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    # (2) Delta vs OU
    ax = axes[0, 1]
    ax.bar(x, d_mean, color=np.where(d_mean >= 0, "#2ca02c", "#d62728"), alpha=0.75)
    ax.errorbar(x, d_mean, yerr=d_sd, fmt="none", ecolor="black", capsize=4, linewidth=1)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Delta vs OU (J_test - J_OU)")
    ax.set_xlabel("N")
    ax.set_ylabel("Delta J")
    ax.grid(alpha=0.25)

    # (3) Seed wins vs OU
    ax = axes[0, 2]
    ax.bar(x, wins, color="#1f77b4", alpha=0.85)
    ax.set_ylim(0, 3.2)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_title("Seed-Level Wins vs OU")
    ax.set_xlabel("N")
    ax.set_ylabel("Wins out of 3")
    for xi, wi in zip(x, wins):
        ax.text(xi, wi + 0.07, f"{int(wi)}/3", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    # (4) E[W_T]
    ax = axes[1, 0]
    ax.plot(x, ewt_mean, marker="o", linewidth=2.0, label="A_DNN")
    ax.fill_between(x, ewt_mean - ewt_sd, ewt_mean + ewt_sd, alpha=0.2)
    ax.axhline(ewt_ou, linestyle="--", linewidth=1.8, label="OU")
    ax.set_title("Reward Component: E[W_T]")
    ax.set_xlabel("N")
    ax.set_ylabel("E[W_T]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    # (5) E[Q_T^2]
    ax = axes[1, 1]
    ax.plot(x, eq_mean, marker="o", linewidth=2.0, label="A_DNN")
    ax.fill_between(x, eq_mean - eq_sd, eq_mean + eq_sd, alpha=0.2)
    ax.axhline(eq_ou, linestyle="--", linewidth=1.8, label="OU")
    ax.set_title("Risk Component: E[Q_T^2]")
    ax.set_xlabel("N")
    ax.set_ylabel("E[Q_T^2]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    # (6) Turnover
    ax = axes[1, 2]
    ax.plot(x, to_mean, marker="o", linewidth=2.0, label="A_DNN")
    ax.fill_between(x, to_mean - to_sd, to_mean + to_sd, alpha=0.2)
    ax.axhline(to_ou, linestyle="--", linewidth=1.8, label="OU")
    ax.set_title("Execution Intensity: Turnover")
    ax.set_xlabel("N")
    ax.set_ylabel("Turnover")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_learning(curves: dict[int, list[tuple[np.ndarray, np.ndarray]]], out_path: str) -> None:
    Ns = sorted(curves.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)
    axes = axes.ravel()
    for ax, n in zip(axes, Ns):
        items = curves[n]
        mats = []
        epochs = None
        for e, t in items:
            epochs = e
            mats.append(t)
            ax.plot(e, t, linewidth=1.0, alpha=0.35)
        mat = np.vstack(mats)
        m = np.mean(mat, axis=0)
        ax.plot(epochs, m, color="black", linewidth=2.0, label="Mean")
        gain = float(m[-1] - m[-5]) if len(m) >= 5 else float(m[-1] - m[0])
        ax.set_title(f"N={n} (mean gain 920->1000: {gain:+.5f})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test J")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("H=0.25 Learning Curves (A_DNN)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    run_root = os.path.join("results", "phase2", "runs", args.run_id)
    if not os.path.isdir(run_root):
        raise FileNotFoundError(f"Run directory not found: {run_root}")

    out_dir = args.out_dir or os.path.join("notes", "plots", f"phase2_smoke_{args.run_id}")
    os.makedirs(out_dir, exist_ok=True)

    rows = _load_rows(run_root)
    table = _build_h025_table(rows)
    curves = _load_curves(run_root, target_h=0.25)

    dash_path = os.path.join(out_dir, "07_h025_dashboard.png")
    learn_path = os.path.join(out_dir, "08_h025_learning_curves.png")
    _plot_dashboard(table, dash_path)
    _plot_learning(curves, learn_path)

    print(dash_path.replace("\\", "/"))
    print(learn_path.replace("\\", "/"))


if __name__ == "__main__":
    main()

