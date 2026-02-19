"""Create insight plots for a completed Phase 1 Bank-mode run.

Reads a run summary CSV (Bank mode) and writes a set of plots + a short
markdown interpretation into an output directory.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


BANK_THEORETICAL = {0.25: 0.206, 0.50: 0.124, 0.75: 0.071}


def _safe_float(x: str) -> float:
    x = x.strip().lower()
    if x in {"nan", "+nan", "-nan"}:
        return float("nan")
    if x in {"inf", "+inf"}:
        return float("inf")
    if x == "-inf":
        return float("-inf")
    return float(x)


def load_rows(summary_csv: Path) -> List[dict]:
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        raw = list(csv.DictReader(f))

    rows = []
    for r in raw:
        rows.append(
            {
                "H": _safe_float(r["H"]),
                "N": int(r["N"]),
                "mode": r.get("mode", ""),
                "our_cost": _safe_float(r["our_cost"]),
                "our_var": _safe_float(r["our_var"]) if r.get("our_var", "") else float("nan"),
                "best_val_cost": _safe_float(r["best_val_cost"]) if r.get("best_val_cost", "") else float("nan"),
                "bank_ref": _safe_float(r["bank_ref"]) if r.get("bank_ref", "") else float("nan"),
                "rel_error_vs_bank": _safe_float(r["rel_error_vs_bank"]) if r.get("rel_error_vs_bank", "") else float("nan"),
                "time_total_s": _safe_float(r["time_total_s"]) if r.get("time_total_s", "") else float("nan"),
            }
        )
    return rows


def build_maps(rows: List[dict]) -> Tuple[List[float], List[int], Dict[Tuple[float, int], dict]]:
    hs = sorted({r["H"] for r in rows})
    ns = sorted({r["N"] for r in rows})
    by_key = {(r["H"], r["N"]): r for r in rows}
    return hs, ns, by_key


def _vals(by_key: Dict[Tuple[float, int], dict], h: float, ns: List[int], field: str) -> np.ndarray:
    out = []
    for n in ns:
        rec = by_key.get((h, n))
        out.append(float("nan") if rec is None else rec[field])
    return np.array(out, dtype=float)


def plot_cost_vs_n(hs: List[float], ns: List[int], by_key: Dict[Tuple[float, int], dict], out: Path) -> None:
    fig, axes = plt.subplots(len(hs), 1, figsize=(10, 3.8 * len(hs)), sharex=True, squeeze=False)
    for i, h in enumerate(hs):
        ax = axes[i, 0]
        y_our = _vals(by_key, h, ns, "our_cost")
        y_bank = _vals(by_key, h, ns, "bank_ref")
        y_best = _vals(by_key, h, ns, "best_val_cost")

        ax.plot(ns, y_our, marker="o", linewidth=1.8, label="Our final cost")
        ax.plot(ns, y_best, marker="x", linewidth=1.2, linestyle="--", label="Best validation cost")
        ax.plot(ns, y_bank, marker="s", linewidth=1.4, linestyle=":", label="Bank A_lin")
        if h in BANK_THEORETICAL:
            ax.axhline(BANK_THEORETICAL[h], color="gray", linestyle="-.", linewidth=1.1, label=f"th.opt={BANK_THEORETICAL[h]:.3f}")

        ax.set_title(f"H = {h}")
        ax.set_ylabel("Cost")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    axes[-1, 0].set_xlabel("N")
    fig.suptitle("Cost vs N (per H)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_h_monotonicity(hs: List[float], ns: List[int], by_key: Dict[Tuple[float, int], dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for n in ns:
        y = np.array([by_key[(h, n)]["our_cost"] for h in hs], dtype=float)
        ax.plot(hs, y, marker="o", linewidth=1.6, label=f"Our N={n}")

        yb = np.array([by_key[(h, n)]["bank_ref"] for h in hs], dtype=float)
        ax.plot(hs, yb, linewidth=1.0, linestyle=":", alpha=0.7, label=f"Bank N={n}")

    ax.set_title("H-monotonicity profile")
    ax.set_xlabel("H")
    ax.set_ylabel("Cost")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _heat_matrix(hs: List[float], ns: List[int], by_key: Dict[Tuple[float, int], dict], field: str, scale: float = 1.0) -> np.ndarray:
    mat = np.full((len(hs), len(ns)), np.nan, dtype=float)
    for i, h in enumerate(hs):
        for j, n in enumerate(ns):
            mat[i, j] = by_key[(h, n)][field] * scale
    return mat


def _plot_heatmap(mat: np.ndarray, hs: List[float], ns: List[int], title: str, cbar: str, out: Path, fmt: str = "{:.1f}") -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(mat, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels(ns)
    ax.set_yticks(range(len(hs)))
    ax.set_yticklabels([f"{h:g}" for h in hs])
    ax.set_xlabel("N")
    ax.set_ylabel("H")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = "nan" if not np.isfinite(v) else fmt.format(v)
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white" if np.isfinite(v) and v > np.nanmean(mat) else "black")

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_runtime_breakdown(hs: List[float], ns: List[int], by_key: Dict[Tuple[float, int], dict], out: Path) -> None:
    by_n = []
    for n in ns:
        vals = [by_key[(h, n)]["time_total_s"] / 60.0 for h in hs]
        by_n.append(np.nanmean(vals))

    by_h = []
    for h in hs:
        vals = [by_key[(h, n)]["time_total_s"] / 60.0 for n in ns]
        by_h.append(np.nanmean(vals))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar([str(n) for n in ns], by_n)
    axes[0].set_title("Avg runtime per N")
    axes[0].set_ylabel("Minutes")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar([f"{h:g}" for h in hs], by_h)
    axes[1].set_title("Avg runtime per H")
    axes[1].set_ylabel("Minutes")
    axes[1].grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_bestval_vs_final(rows: List[dict], out: Path) -> None:
    x = np.array([r["best_val_cost"] for r in rows], dtype=float)
    y = np.array([r["our_cost"] for r in rows], dtype=float)
    hs = [r["H"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = ax.scatter(x, y, c=hs, cmap="viridis", s=45)
    lo = np.nanmin(np.concatenate([x, y]))
    hi = np.nanmax(np.concatenate([x, y]))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.0)
    ax.set_title("Best validation vs final evaluation")
    ax.set_xlabel("Best validation cost")
    ax.set_ylabel("Final cost")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("H")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_summary_md(rows: List[dict], hs: List[float], ns: List[int], by_key: Dict[Tuple[float, int], dict], out: Path) -> None:
    finite = [r for r in rows if np.isfinite(r["our_cost"])]
    total_s = np.nansum([r["time_total_s"] for r in rows])
    avg_pair = np.nanmean([r["time_total_s"] for r in rows])

    worst = max(finite, key=lambda r: r["rel_error_vs_bank"])
    best = min(finite, key=lambda r: r["rel_error_vs_bank"])

    # N-convergence checks: N=3 < N=1 for each H
    nconv = []
    for h in hs:
        c1 = by_key[(h, 1)]["our_cost"]
        c3 = by_key[(h, 3)]["our_cost"]
        nconv.append((h, c3 < c1, c1, c3))

    # H-monotonicity checks: H=0.25 > 0.5 > 0.75 for each N
    hmono = []
    if set(hs) >= {0.25, 0.5, 0.75}:
        for n in ns:
            c025 = by_key[(0.25, n)]["our_cost"]
            c05 = by_key[(0.5, n)]["our_cost"]
            c075 = by_key[(0.75, n)]["our_cost"]
            hmono.append((n, c025 > c05 > c075, c025, c05, c075))

    lines = []
    lines.append("# Phase 1 Bank-Mode Insights")
    lines.append("")
    lines.append(f"- Total wall-clock (sum over pairs): `{total_s/3600:.2f} h`")
    lines.append(f"- Average pair runtime: `{avg_pair/60:.1f} min`")
    lines.append(f"- Best relative error vs Bank: `{best['rel_error_vs_bank']*100:.2f}%` at `(H={best['H']}, N={best['N']})`")
    lines.append(f"- Worst relative error vs Bank: `{worst['rel_error_vs_bank']*100:.2f}%` at `(H={worst['H']}, N={worst['N']})`")
    lines.append("")
    lines.append("## N-Convergence Check")
    for h, ok, c1, c3 in nconv:
        lines.append(f"- H={h}: N=1 ({c1:.4f}) > N=3 ({c3:.4f}) -> {'PASS' if ok else 'FAIL'}")
    lines.append("")
    lines.append("## H-Monotonicity Check")
    for n, ok, c025, c05, c075 in hmono:
        lines.append(f"- N={n}: {c025:.4f} > {c05:.4f} > {c075:.4f} -> {'PASS' if ok else 'FAIL'}")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot insights for a Phase 1 Bank-mode run.")
    p.add_argument("--summary", required=True, help="Path to run summary.csv")
    p.add_argument("--outdir", default=None, help="Output directory (default: <run_dir>/insights)")
    args = p.parse_args()

    summary = Path(args.summary)
    if not summary.exists():
        raise FileNotFoundError(f"Missing summary file: {summary}")
    outdir = Path(args.outdir) if args.outdir else summary.parent / "insights"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(summary)
    hs, ns, by_key = build_maps(rows)

    plot_cost_vs_n(hs, ns, by_key, outdir / "01_cost_vs_n_by_h.png")
    plot_h_monotonicity(hs, ns, by_key, outdir / "02_cost_vs_h_by_n.png")
    rel_mat = _heat_matrix(hs, ns, by_key, "rel_error_vs_bank", scale=100.0)
    _plot_heatmap(rel_mat, hs, ns, "Relative Error vs Bank", "Percent", outdir / "03_rel_error_heatmap.png", fmt="{:.1f}%")
    t_mat = _heat_matrix(hs, ns, by_key, "time_total_s", scale=1.0 / 60.0)
    _plot_heatmap(t_mat, hs, ns, "Runtime per Pair", "Minutes", outdir / "04_runtime_heatmap.png", fmt="{:.1f}")
    plot_runtime_breakdown(hs, ns, by_key, outdir / "05_runtime_breakdown.png")
    plot_bestval_vs_final(rows, outdir / "06_bestval_vs_final.png")
    write_summary_md(rows, hs, ns, by_key, outdir / "insights_summary.md")

    print(f"Saved insights to: {outdir}")


if __name__ == "__main__":
    main()

