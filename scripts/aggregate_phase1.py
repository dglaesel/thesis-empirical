"""Aggregate Phase 1 results from parallel per-(H, N, mode) runs.

Scans results/phase1/ for individual run directories, each containing a
summary.csv with one or more rows, and merges them into a single
results/phase1/summary.csv suitable for plot_phase1_bank_insights.py.

Usage:
    python scripts/aggregate_phase1.py [--results-dir results/phase1]
"""

import argparse
import csv
import os
import sys
from pathlib import Path

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)

HEADER = [
    "H", "N", "mode", "our_cost", "our_var", "best_val_cost",
    "bank_ref", "rel_error_vs_bank", "time_total_s",
]


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Phase 1 per-job summary CSVs into one file."
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(_project_root, "results", "phase1"),
        help="Root directory containing Phase 1 run_* subdirectories.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default: <results-dir>/summary.csv",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"ERROR: results directory not found: {results_dir}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else results_dir / "summary.csv"

    # Collect all rows from per-job summary CSVs
    all_rows = []
    seen_keys = set()
    run_dirs = sorted(results_dir.glob("run_*"))

    if not run_dirs:
        print(f"ERROR: no run_* directories found in {results_dir}")
        sys.exit(1)

    for run_dir in run_dirs:
        csv_path = run_dir / "summary.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["H"], row["N"], row["mode"])
                if key in seen_keys:
                    # Keep latest (run_dirs sorted by timestamp)
                    all_rows = [r for r in all_rows
                                if (r["H"], r["N"], r["mode"]) != key]
                seen_keys.add(key)
                all_rows.append(row)

    if not all_rows:
        print(f"ERROR: no summary.csv rows found in {results_dir}/run_*/")
        sys.exit(1)

    # Sort by (H, mode, N)
    def sort_key(row):
        return (float(row["H"]), row["mode"], int(row["N"]))

    all_rows.sort(key=sort_key)

    # Write merged CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Aggregated {len(all_rows)} rows from {len(run_dirs)} run directories")
    print(f"Output: {output_path}")

    # Print quick summary table
    print(f"\n{'H':>6s} {'N':>3s} {'mode':>5s} {'our_cost':>10s} {'bank_ref':>10s} {'rel_err':>8s}")
    print("-" * 48)
    for row in all_rows:
        rel = row.get("rel_error_vs_bank", "")
        rel_str = f"{float(rel)*100:.1f}%" if rel else ""
        print(f"{row['H']:>6s} {row['N']:>3s} {row['mode']:>5s} "
              f"{row['our_cost']:>10s} {row['bank_ref']:>10s} {rel_str:>8s}")


if __name__ == "__main__":
    main()
