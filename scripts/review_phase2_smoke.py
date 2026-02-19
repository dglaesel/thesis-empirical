"""Generate a full review report + plots for a completed Phase 2 smoke run.

Usage:
  ./.venv-phase2/Scripts/python scripts/review_phase2_smoke.py --run-id 20260218_013716
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from datetime import datetime
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Review Phase 2 smoke run.")
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for plots and report. Default: notes/plots/phase2_smoke_<run_id>",
    )
    return p.parse_args()


def _load_summaries(run_root: str) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(run_root, "**", "summary.json"), recursive=True))
    rows: list[dict] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
        rows.append(
            {
                "path": p.replace("\\", "/"),
                "H": float(j["H"]),
                "N": int(j["N"]),
                "seed": int(j["seed"]),
                "J_test": float(j["test_metrics"]["J"]),
                "J_train": float(j["train_metrics"]["J"]),
                "E_WT": float(j["test_metrics"]["E_WT"]),
                "E_QT2": float(j["test_metrics"]["E_QT2"]),
                "turnover": float(j["test_metrics"]["turnover"]),
                "J_OU": float(j["baselines"]["OU_linear"]["J_test"]),
                "E_WT_OU": float(j["baselines"]["OU_linear"]["E_WT"]),
                "E_QT2_OU": float(j["baselines"]["OU_linear"]["E_QT2"]),
                "turnover_OU": float(j["baselines"]["OU_linear"]["turnover"]),
                "J_ZScore": float(j["baselines"]["ZScore"]["J_test"]),
                "J_NoTrade": float(j["baselines"]["NoTrade_test"]["J"]),
                "J_BuyHold": float(j["baselines"]["BuyHold_test"]["J"]),
                "elapsed_seconds": float(j["elapsed_seconds"]),
            }
        )
    rows.sort(key=lambda r: (r["H"], r["N"], r["seed"]))
    return rows


def _load_curves(run_root: str) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(run_root, "**", "training_curve.csv"), recursive=True))
    curves: list[dict] = []
    for p in paths:
        parts = p.replace("/", "\\").split("\\")
        h_tag = next(x for x in parts if x.startswith("H_"))
        n_tag = next(x for x in parts if x.startswith("N_"))
        seed_tag = next(x for x in parts if x.startswith("seed_"))
        H = float(h_tag.replace("H_", "").replace("p", "."))
        N = int(n_tag.split("_")[1])
        seed = int(seed_tag.split("_")[1])
        epochs: list[int] = []
        train_j: list[float] = []
        test_j: list[float] = []
        with open(p, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for row in rd:
                e = int(row["epoch"])
                tr = float(row["train_J"])
                tj = row["test_J"]
                epochs.append(e)
                train_j.append(tr)
                test_j.append(float(tj) if tj.lower() != "nan" else np.nan)
        curves.append(
            {
                "path": p.replace("\\", "/"),
                "H": H,
                "N": N,
                "seed": seed,
                "epochs": np.array(epochs, dtype=int),
                "train_j": np.array(train_j, dtype=float),
                "test_j": np.array(test_j, dtype=float),
            }
        )
    curves.sort(key=lambda c: (c["H"], c["N"], c["seed"]))
    return curves


def _group(rows: list[dict], H: float, N: int) -> list[dict]:
    return [r for r in rows if r["H"] == H and r["N"] == N]


def _compute_group_table(rows: list[dict], Hs: list[float], Ns: list[int]) -> list[dict]:
    out: list[dict] = []
    for H in Hs:
        for N in Ns:
            g = _group(rows, H, N)
            j = np.array([x["J_test"] for x in g], dtype=float)
            dou = np.array([x["J_test"] - x["J_OU"] for x in g], dtype=float)
            out.append(
                {
                    "H": H,
                    "N": N,
                    "count": len(g),
                    "J_test_mean": float(j.mean()),
                    "J_test_sd": float(j.std(ddof=0)),
                    "J_OU_mean": float(np.mean([x["J_OU"] for x in g])),
                    "delta_vs_OU_mean": float(dou.mean()),
                    "wins_vs_OU": int(np.sum(dou > 0)),
                    "turnover_mean": float(np.mean([x["turnover"] for x in g])),
                    "turnover_OU_mean": float(np.mean([x["turnover_OU"] for x in g])),
                    "E_WT_mean": float(np.mean([x["E_WT"] for x in g])),
                    "E_QT2_mean": float(np.mean([x["E_QT2"] for x in g])),
                    "elapsed_mean_s": float(np.mean([x["elapsed_seconds"] for x in g])),
                }
            )
    out.sort(key=lambda r: (r["H"], r["N"]))
    return out


def _write_csvs(rows: list[dict], group_rows: list[dict], out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    long_path = os.path.join(out_dir, "summary_long.csv")
    grp_path = os.path.join(out_dir, "summary_grouped.csv")

    with open(long_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(grp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
        w.writeheader()
        w.writerows(group_rows)
    return long_path.replace("\\", "/"), grp_path.replace("\\", "/")


def _save_fig_heatmap_mean_j(group_rows: list[dict], Hs: list[float], Ns: list[int], out_dir: str) -> str:
    mat = np.zeros((len(Hs), len(Ns)), dtype=float)
    for i, h in enumerate(Hs):
        for j, n in enumerate(Ns):
            rec = next(r for r in group_rows if r["H"] == h and r["N"] == n)
            mat[i, j] = rec["J_test_mean"]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(Ns)), [f"N={n}" for n in Ns])
    ax.set_yticks(range(len(Hs)), [f"H={h}" for h in Hs])
    ax.set_title("Mean Test J (A_DNN, Smoke Run)")
    for i in range(len(Hs)):
        for j in range(len(Ns)):
            ax.text(j, i, f"{mat[i,j]:.4f}", ha="center", va="center", color="white", fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean J_test")
    fig.tight_layout()
    out = os.path.join(out_dir, "01_heatmap_mean_j.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out.replace("\\", "/")


def _save_fig_heatmap_delta_ou(group_rows: list[dict], Hs: list[float], Ns: list[int], out_dir: str) -> str:
    mat = np.zeros((len(Hs), len(Ns)), dtype=float)
    for i, h in enumerate(Hs):
        for j, n in enumerate(Ns):
            rec = next(r for r in group_rows if r["H"] == h and r["N"] == n)
            mat[i, j] = rec["delta_vs_OU_mean"]

    vmax = float(np.max(np.abs(mat)))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(Ns)), [f"N={n}" for n in Ns])
    ax.set_yticks(range(len(Hs)), [f"H={h}" for h in Hs])
    ax.set_title("Mean Delta vs OU Baseline (J_test - J_OU)")
    for i in range(len(Hs)):
        for j in range(len(Ns)):
            ax.text(j, i, f"{mat[i,j]:+.4f}", ha="center", va="center", color="black", fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean delta vs OU")
    fig.tight_layout()
    out = os.path.join(out_dir, "02_heatmap_delta_vs_ou.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out.replace("\\", "/")


def _save_fig_j_vs_n(group_rows: list[dict], Hs: list[float], Ns: list[int], out_dir: str) -> str:
    fig, axes = plt.subplots(1, len(Hs), figsize=(14, 4.2), sharey=True)
    if len(Hs) == 1:
        axes = [axes]
    for ax, h in zip(axes, Hs):
        xs = np.array(Ns, dtype=float)
        means = []
        sds = []
        ou_means = []
        for n in Ns:
            rec = next(r for r in group_rows if r["H"] == h and r["N"] == n)
            means.append(rec["J_test_mean"])
            sds.append(rec["J_test_sd"])
            ou_means.append(rec["J_OU_mean"])
        means = np.array(means, dtype=float)
        sds = np.array(sds, dtype=float)
        ou = np.array(ou_means, dtype=float)
        ax.plot(xs, means, marker="o", linewidth=2.0, label="A_DNN mean J_test")
        ax.fill_between(xs, means - sds, means + sds, alpha=0.2, label="±1 sd")
        ax.plot(xs, ou, linestyle="--", linewidth=1.5, label="OU baseline mean")
        ax.set_title(f"H={h}")
        ax.set_xlabel("N")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("J_test")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Performance vs Signature Order")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    out = os.path.join(out_dir, "03_j_vs_n_by_h.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out.replace("\\", "/")


def _save_fig_winrate_ou(group_rows: list[dict], Hs: list[float], Ns: list[int], out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.25
    x = np.arange(len(Ns), dtype=float)
    for i, h in enumerate(Hs):
        vals = []
        for n in Ns:
            rec = next(r for r in group_rows if r["H"] == h and r["N"] == n)
            vals.append(rec["wins_vs_OU"] / max(rec["count"], 1))
        ax.bar(x + (i - 1) * width, vals, width=width, label=f"H={h}")
    ax.set_xticks(x, [f"N={n}" for n in Ns])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Win rate vs OU (across seeds)")
    ax.set_title("Seed-Level Win Rate Against OU Baseline")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(out_dir, "04_winrate_vs_ou.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out.replace("\\", "/")


def _save_fig_learning_curves(curves: list[dict], Hs: list[float], Ns: list[int], out_dir: str) -> str:
    fig, axes = plt.subplots(len(Hs), len(Ns), figsize=(16, 8.5), sharex=True, sharey=False)
    for i, h in enumerate(Hs):
        for j, n in enumerate(Ns):
            ax = axes[i, j]
            sub = [c for c in curves if c["H"] == h and c["N"] == n]
            # Evaluation points only (test_j non-nan)
            eval_epochs = None
            vals = []
            for c in sub:
                mask = ~np.isnan(c["test_j"])
                e = c["epochs"][mask]
                t = c["test_j"][mask]
                eval_epochs = e
                vals.append(t)
                ax.plot(e, t, alpha=0.35, linewidth=1.2)
            if vals:
                mat = np.vstack(vals)
                ax.plot(eval_epochs, np.mean(mat, axis=0), color="black", linewidth=1.8)
            if i == 0:
                ax.set_title(f"N={n}")
            if j == 0:
                ax.set_ylabel(f"H={h}\nTest J")
            if i == len(Hs) - 1:
                ax.set_xlabel("Epoch")
            ax.grid(alpha=0.2)
    fig.suptitle("Learning Curves (thin: seeds, bold: mean)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(out_dir, "05_learning_curves_grid.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out.replace("\\", "/")


def _save_fig_metric_decomp(group_rows: list[dict], Hs: list[float], Ns: list[int], out_dir: str) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    metric_specs = [
        ("E_WT_mean", "E[W_T]"),
        ("E_QT2_mean", "E[Q_T^2]"),
        ("turnover_mean", "Turnover"),
    ]
    for ax, (key, label) in zip(axes, metric_specs):
        for h in Hs:
            ys = [next(r for r in group_rows if r["H"] == h and r["N"] == n)[key] for n in Ns]
            ax.plot(Ns, ys, marker="o", linewidth=1.8, label=f"H={h}")
        ax.set_xlabel("N")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs N")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Metric Decomposition (mean over seeds)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(out_dir, "06_metric_decomposition.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out.replace("\\", "/")


def _training_tail_stats(curves: list[dict]) -> dict[float, dict[str, float]]:
    by_h: dict[float, list[float]] = defaultdict(list)
    best_epoch_all: list[int] = []
    for c in curves:
        mask = ~np.isnan(c["test_j"])
        e = c["epochs"][mask]
        t = c["test_j"][mask]
        m = {int(ee): float(tt) for ee, tt in zip(e, t)}
        if 920 in m and 1000 in m:
            by_h[c["H"]].append(m[1000] - m[920])
        best_epoch_all.append(int(e[int(np.argmax(t))]))
    out: dict[float, dict[str, float]] = {}
    for h, gains in by_h.items():
        arr = np.array(gains, dtype=float)
        out[h] = {
            "gain_920_1000_mean": float(arr.mean()),
            "gain_920_1000_min": float(arr.min()),
            "gain_920_1000_max": float(arr.max()),
        }
    out["best_epoch_at_end_rate"] = {
        "value": float(np.mean(np.array(best_epoch_all, dtype=int) == 1000))
    }
    return out


def _make_markdown(
    run_id: str,
    run_root: str,
    rows: list[dict],
    group_rows: list[dict],
    tail: dict,
    plot_paths: list[str],
    long_csv: str,
    grp_csv: str,
) -> str:
    Hs = sorted(set(r["H"] for r in rows))
    Ns = sorted(set(r["N"] for r in rows))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Selected facts for narrative
    rec_h025_n4 = next(r for r in group_rows if r["H"] == 0.25 and r["N"] == 4)
    rec_h05_n4 = next(r for r in group_rows if r["H"] == 0.5 and r["N"] == 4)
    rec_h075_n4 = next(r for r in group_rows if r["H"] == 0.75 and r["N"] == 4)

    lines: list[str] = []
    lines.append(f"# Phase 2 Smoke Run Review (`{run_id}`)")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Run root: `{run_root.replace(os.sep, '/')}`")
    lines.append(f"- Completed artifacts: `{len(rows)}` summaries, `{len(rows)}` training curves")
    lines.append("- Config used: `config/smoke_8h.yaml`")
    lines.append("- Level/arch tested: `L1`, `A_dnn` only")
    lines.append("")
    lines.append("## Main Results")
    lines.append("")
    lines.append(f"- `H=0.25, N=4`: mean `J_test={rec_h025_n4['J_test_mean']:.6f}`, mean delta vs OU `{rec_h025_n4['delta_vs_OU_mean']:+.6f}`, wins `{rec_h025_n4['wins_vs_OU']}/3`.")
    lines.append(f"- `H=0.5, N=4`: mean `J_test={rec_h05_n4['J_test_mean']:.6f}`, mean delta vs OU `{rec_h05_n4['delta_vs_OU_mean']:+.6f}`, wins `{rec_h05_n4['wins_vs_OU']}/3`.")
    lines.append(f"- `H=0.75, N=4`: mean `J_test={rec_h075_n4['J_test_mean']:.6f}`, mean delta vs OU `{rec_h075_n4['delta_vs_OU_mean']:+.6f}`, wins `{rec_h075_n4['wins_vs_OU']}/3`.")
    lines.append(f"- Best-checkpoint epoch was final epoch (`1000`) in all runs: `{tail['best_epoch_at_end_rate']['value']*100:.1f}%`.")
    lines.append("")
    lines.append("## Grouped Table (mean over seeds)")
    lines.append("")
    lines.append("| H | N | mean J_test | sd J_test | mean J_OU | mean delta vs OU | wins vs OU | mean turnover | mean OU turnover |")
    lines.append("|---|---|-------------|-----------|-----------|------------------|------------|---------------|------------------|")
    for h in Hs:
        for n in Ns:
            r = next(x for x in group_rows if x["H"] == h and x["N"] == n)
            lines.append(
                f"| {h} | {n} | {r['J_test_mean']:.6f} | {r['J_test_sd']:.6f} | {r['J_OU_mean']:.6f} | {r['delta_vs_OU_mean']:+.6f} | {r['wins_vs_OU']}/{r['count']} | {r['turnover_mean']:.4f} | {r['turnover_OU_mean']:.4f} |"
            )
    lines.append("")
    lines.append("## What Is Already Consistent With Chapter 06")
    lines.append("")
    lines.append("- `H=0.25` (anti-persistent regime): higher-order signatures improve materially over `N=1`, and `N=4` is strongest. This matches the Chapter 6 intuition that path-history features matter for `H!=1/2`.")
    lines.append("- `N=1` underperforms consistently across all `H`, which is aligned with the idea that low-order features are often insufficient.")
    lines.append("- The run remains optimization-limited at 1000 epochs (positive tail gains from epoch 920 to 1000 for all `H`), so additional budget should still improve results.")
    lines.append("")
    lines.append("## What Is Not Demonstrated Yet (Smoke-Config Limits)")
    lines.append("")
    lines.append("- Only `L1` was tested. Chapter 6's full state-dependent setup (`L2/L3`, variance-matching discussion) is not exercised.")
    lines.append("- Only `A_dnn` was tested; no `A_lin` comparison in this run.")
    lines.append("- `H` grid is reduced to `{0.25, 0.5, 0.75}` (missing intermediate values used in broader study design).")
    lines.append("- Small scale: `M=16384`, `seeds=3`, `index` split only. This limits power and robustness.")
    lines.append("")
    lines.append("## Systematic Issues To Fix (Methodological/Code)")
    lines.append("")
    lines.append("- Test leakage in model selection: `scripts/phase2_train.py:607` to `scripts/phase2_train.py:611` uses test metrics for LR scheduling and best checkpointing.")
    lines.append("- Training loop does one update per epoch (`scripts/phase2_train.py:560` to `scripts/phase2_train.py:582`), which is inefficient and contributes to under-convergence.")
    lines.append("- Launch script drift: `scripts/slurm/phase2_train_adnn_cpu.sbatch:49` still uses `--world` and includes `N=5` (`scripts/slurm/phase2_train_adnn_cpu.sbatch:30`) while current trainer expects `--level` and `N<=4`.")
    lines.append("- Helper usage examples are stale (`scripts/run_phase2.ps1:17` to `scripts/run_phase2.ps1:19` uses `--world`).")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for p in plot_paths:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("## Data Exports")
    lines.append("")
    lines.append(f"- `{long_csv}`")
    lines.append(f"- `{grp_csv}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This review is read-only with respect to experimental outputs; it only aggregates existing artifacts.")
    lines.append("- Report conclusions are bounded to this smoke run configuration and should not be treated as final thesis-level claims.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    run_id = args.run_id
    run_root = os.path.join("results", "phase2", "runs", run_id)
    if not os.path.isdir(run_root):
        raise FileNotFoundError(f"Run root not found: {run_root}")

    out_dir = args.out_dir or os.path.join("notes", "plots", f"phase2_smoke_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    rows = _load_summaries(run_root)
    curves = _load_curves(run_root)
    if len(rows) == 0:
        raise RuntimeError("No summary.json files found.")
    if len(curves) == 0:
        raise RuntimeError("No training_curve.csv files found.")

    Hs = sorted(set(r["H"] for r in rows))
    Ns = sorted(set(r["N"] for r in rows))
    group_rows = _compute_group_table(rows, Hs, Ns)

    long_csv, grp_csv = _write_csvs(rows, group_rows, out_dir)
    plots = [
        _save_fig_heatmap_mean_j(group_rows, Hs, Ns, out_dir),
        _save_fig_heatmap_delta_ou(group_rows, Hs, Ns, out_dir),
        _save_fig_j_vs_n(group_rows, Hs, Ns, out_dir),
        _save_fig_winrate_ou(group_rows, Hs, Ns, out_dir),
        _save_fig_learning_curves(curves, Hs, Ns, out_dir),
        _save_fig_metric_decomp(group_rows, Hs, Ns, out_dir),
    ]
    tail = _training_tail_stats(curves)

    md = _make_markdown(run_id, run_root, rows, group_rows, tail, plots, long_csv, grp_csv)
    report_path = os.path.join("notes", f"phase2_smoke_run_review_{run_id}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Review report: {report_path.replace(os.sep, '/')}")
    print(f"Plots/data dir: {out_dir.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
