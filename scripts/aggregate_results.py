"""
aggregate_results.py — Phase 8.1

Reads all drift_metrics.csv files from logs/runs/ and produces:
  - logs/summary/all_results.csv : one row per completed run
  - logs/summary/seed_agg.csv    : mean ± std across seeds, one row per cell

A "cell" is the combination (dataset, alpha, model, method).

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --logs-dir logs/runs --out-dir logs/summary
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

FLBENCH_ROOT = Path(__file__).parent.parent

# Columns present in drift_metrics.csv that we want at the final round
FINAL_ROUND_COLS = [
    "global_acc", "global_f1", "global_precision", "global_recall",
    # Raw L2 drift
    "drift_norm_mean",    "drift_norm_std",
    "drift_feature_mean", "drift_feature_std",
    "drift_head_mean",    "drift_head_std",
    # Normalised (RMS) drift — primary metric for cross-group/cross-arch comparisons
    "drift_norm_norm_mean",    "drift_norm_norm_std",
    "drift_feature_norm_mean", "drift_feature_norm_std",
    "drift_head_norm_mean",    "drift_head_norm_std",
    "interference_norm", "interference_feature", "interference_head",
    "fairness_gap", "client_acc_min", "client_acc_max", "client_acc_std",
]

# Run directory name pattern produced by run_experiments.sh
# Note: dataset uses .+ (not [^_]+) because dataset names like
# "brain_tumor" contain underscores. The literal "_alpha" token
# is the unambiguous separator between dataset and alpha.
RUN_PATTERN = re.compile(
    r"^(?P<dataset>.+)_alpha(?P<alpha>[^_]+)_(?P<model>.+)"
    r"_(?P<method>cka\w+)_seed(?P<seed>\d+)$"
)

CONVERGENCE_THRESHOLDS: Dict[str, float] = {
    "cifar10": 70.0,
    "brain_tumor": 80.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_run_name(name: str) -> Optional[Dict[str, str]]:
    m = RUN_PATTERN.match(name)
    return m.groupdict() if m else None


def convergence_round(acc_series: pd.Series, threshold: float, window: int = 5) -> int:
    """First round where accuracy stays >= threshold for `window` consecutive rounds."""
    vals = acc_series.tolist()
    for t in range(len(vals) - window + 1):
        if all(v >= threshold for v in vals[t: t + window]):
            return t + 1  # 1-indexed
    return -1


def load_drift_csv(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load drift_metrics.csv (primary) or fall back to metrics.csv."""
    for fname in ("drift_metrics.csv", "metrics.csv"):
        p = run_dir / fname
        if p.exists():
            try:
                df = pd.read_csv(p)
                if not df.empty and "round" in df.columns:
                    return df
            except Exception as exc:
                print(f"  [WARN] Cannot read {p}: {exc}")
    return None


def summarise_run(run_dir: Path, meta: Dict[str, str]) -> Optional[Dict]:
    df = load_drift_csv(run_dir)
    if df is None:
        return None

    final = df.iloc[-1]
    row: Dict = {**meta, "n_rounds": len(df)}

    for col in FINAL_ROUND_COLS:
        row[f"{col}@final"] = float(final[col]) if col in final.index else float("nan")

    threshold = CONVERGENCE_THRESHOLDS.get(meta["dataset"], 70.0)
    row["convergence_round"] = (
        convergence_round(df["global_acc"], threshold)
        if "global_acc" in df.columns
        else -1
    )
    return row


def aggregate_seeds(all_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over seeds: mean ± std per (dataset, alpha, model, method) cell."""
    cell_cols = ["dataset", "alpha", "model", "method"]
    metric_cols = [
        c for c in all_df.columns
        if c.endswith("@final") or c == "convergence_round"
    ]

    rows: List[Dict] = []
    for keys, group in all_df.groupby(cell_cols):
        row: Dict = dict(zip(cell_cols, keys))
        row["n_seeds"] = len(group)
        for col in metric_cols:
            vals = group[col].dropna().values
            row[f"{col}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"{col}_std"]  = float(np.std(vals))  if len(vals) else float("nan")
        acc_std = row.get("global_acc@final_std", 0.0) or 0.0
        row["high_variance_flag"] = int(acc_std > 3.0)
        rows.append(row)

    return pd.DataFrame(rows)


def print_summary_table(seed_agg: pd.DataFrame) -> None:
    """Print a human-readable per-cell summary to stdout.

    Shows: Acc@final (mean±std), convergence round, raw drift-norm and
    normalised drift-norm (mean), and interference for the norm layer group.
    Falls back gracefully when normalised columns are absent (old CSV runs).
    """
    header = (
        f"{'Dataset':<13} {'α':<7} {'Model':<16} {'Method':<16} "
        f"{'Seeds':>5} {'Acc@final':>11} {'ConvRnd':>8} "
        f"{'RawDrift-norm':>14} {'NormDrift-norm':>15} {'Interf-norm':>12}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    has_norm_drift = "drift_norm_norm_mean@final_mean" in seed_agg.columns

    for _, r in seed_agg.iterrows():
        acc_str   = f"{r.get('global_acc@final_mean', float('nan')):6.1f}±{r.get('global_acc@final_std', float('nan')):.1f}"
        conv_str  = str(int(r.get("convergence_round_mean", -1) or -1))
        raw_d     = r.get("drift_norm_mean@final_mean", float("nan"))
        norm_d    = r.get("drift_norm_norm_mean@final_mean", float("nan")) if has_norm_drift else float("nan")
        interf    = r.get("interference_norm@final_mean", float("nan"))
        print(
            f"{str(r['dataset']):<13} {str(r['alpha']):<7} {str(r['model']):<16} "
            f"{str(r['method']):<16} {int(r.get('n_seeds', 0)):>5} "
            f"{acc_str:>11} {conv_str:>8} "
            f"{raw_d:>14.4f} {norm_d:>15.4f} {interf:>12.4f}"
        )
    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate drift experiment results.")
    parser.add_argument("--logs-dir", default=str(FLBENCH_ROOT / "logs" / "runs"),
                        help="Directory containing per-run subdirectories.")
    parser.add_argument("--out-dir",  default=str(FLBENCH_ROOT / "logs" / "summary"),
                        help="Output directory for CSV summaries.")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not logs_dir.exists():
        print(f"[ERROR] logs dir not found: {logs_dir}")
        sys.exit(1)

    run_dirs = sorted(d for d in logs_dir.iterdir() if d.is_dir())
    print(f"Found {len(run_dirs)} run directories in {logs_dir}")

    all_rows: List[Dict] = []
    skipped = 0

    for run_dir in run_dirs:
        meta = parse_run_name(run_dir.name)
        if meta is None:
            print(f"  [SKIP] Unrecognised name: {run_dir.name}")
            skipped += 1
            continue
        row = summarise_run(run_dir, meta)
        if row is None:
            print(f"  [SKIP] No valid CSV: {run_dir.name}")
            skipped += 1
            continue
        all_rows.append(row)

    if not all_rows:
        print("[ERROR] No valid runs found.")
        sys.exit(1)

    all_df = pd.DataFrame(all_rows)
    all_path = out_dir / "all_results.csv"
    all_df.to_csv(all_path, index=False)
    print(f"\n[OK] {len(all_df)} runs  →  {all_path}")

    seed_agg = aggregate_seeds(all_df)
    agg_path = out_dir / "seed_agg.csv"
    seed_agg.to_csv(agg_path, index=False)
    print(f"[OK] {len(seed_agg)} cells →  {agg_path}")

    print_summary_table(seed_agg)

    hv = seed_agg[seed_agg["high_variance_flag"] == 1]
    if not hv.empty:
        print(f"\n[WARN] {len(hv)} high-variance cell(s) (acc_std > 3%):")
        for _, r in hv.iterrows():
            std_val = r.get("global_acc@final_std", float("nan"))
            print(f"  {r['dataset']} α={r['alpha']} {r['model']} {r['method']}"
                  f"  acc_std={std_val:.2f}%")

    if skipped:
        print(f"\n[INFO] {skipped} directories skipped.")


if __name__ == "__main__":
    main()
