"""
sanity_check.py — Phase 7.3 / 8.2

Flags runs where:
  1. drift_metrics.csv is missing or unreadable
  2. Required columns are absent
  3. Fewer rows than expected (truncated run)
  4. Final global_acc < 10% (likely crashed)
  5. drift_norm_mean = 0.0 for ALL rounds (metric not logging)
  6. Cross-seed accuracy std > 5% (high variance)

Usage:
    python scripts/sanity_check.py
    python scripts/sanity_check.py --logs-dir logs/runs --expected-rounds 40
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

FLBENCH_ROOT = Path(__file__).parent.parent

RUN_PATTERN = re.compile(
    r"^(?P<dataset>.+)_alpha(?P<alpha>[^_]+)_(?P<model>.+)"
    r"_(?P<method>drift\w+)_seed(?P<seed>\d+)$"
)

REQUIRED_COLS = [
    "round", "global_acc", "global_f1",
    "drift_norm_mean", "drift_feature_mean", "drift_head_mean",
    "interference_norm", "interference_feature", "interference_head",
    "fairness_gap", "client_acc_std",
]


def _load_csv(run_dir: Path) -> Tuple[pd.DataFrame, str]:
    """Return (dataframe, filename) for the best available CSV, or (empty, '')."""
    for fname in ("drift_metrics.csv", "metrics.csv"):
        p = run_dir / fname
        if p.exists():
            try:
                df = pd.read_csv(p)
                return df, fname
            except Exception:
                pass
    return pd.DataFrame(), ""


def check_run(run_dir: Path, expected_rounds: int) -> List[str]:
    issues: List[str] = []
    df, fname = _load_csv(run_dir)

    if fname == "":
        return ["drift_metrics.csv missing"]

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns in {fname}: {missing_cols}")

    if df.empty:
        return issues + [f"{fname} is empty"]

    if len(df) < expected_rounds:
        issues.append(f"Only {len(df)}/{expected_rounds} rounds in {fname}")

    if "global_acc" in df.columns:
        final_acc = float(df["global_acc"].iloc[-1])
        if final_acc < 10.0:
            issues.append(f"Final accuracy = {final_acc:.2f}% (< 10%, likely crashed)")

    if "drift_norm_mean" in df.columns:
        if (df["drift_norm_mean"] == 0.0).all():
            issues.append("drift_norm_mean = 0.0 for ALL rounds (metric not logging)")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check all experiment runs.")
    parser.add_argument("--logs-dir", default=str(FLBENCH_ROOT / "logs" / "runs"))
    parser.add_argument("--expected-rounds", type=int, default=40)
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"[ERROR] logs dir not found: {logs_dir}")
        sys.exit(1)

    run_dirs = sorted(d for d in logs_dir.iterdir() if d.is_dir())
    print(f"Checking {len(run_dirs)} runs in {logs_dir}\n")

    all_issues: Dict[str, List[str]] = {}
    ok_count = 0

    for run_dir in run_dirs:
        issues = check_run(run_dir, args.expected_rounds)
        if issues:
            all_issues[run_dir.name] = issues
        else:
            ok_count += 1

    # ── Cross-seed variance check ─────────────────────────────────────────
    cell_data: Dict[Tuple, List[float]] = {}
    for run_dir in run_dirs:
        m = RUN_PATTERN.match(run_dir.name)
        if not m:
            continue
        df, _ = _load_csv(run_dir)
        if df.empty or "global_acc" not in df.columns:
            continue
        final_acc = float(df["global_acc"].iloc[-1])
        cell = (m.group("dataset"), m.group("alpha"),
                m.group("model"), m.group("method"))
        cell_data.setdefault(cell, []).append(final_acc)

    high_var: List[Tuple] = [
        (cell, accs)
        for cell, accs in cell_data.items()
        if len(accs) >= 2 and float(np.std(accs)) > 5.0
    ]

    # ── Report ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Sanity Check Results")
    print("=" * 60)
    print(f"  OK:                  {ok_count}")
    print(f"  Runs with issues:    {len(all_issues)}")
    print(f"  High-variance cells: {len(high_var)}")
    print()

    if all_issues:
        print("── Per-run issues " + "─" * 42)
        for name, issues in sorted(all_issues.items()):
            print(f"  {name}")
            for issue in issues:
                print(f"    • {issue}")
        print()

    if high_var:
        print("── High-variance cells (acc std > 5% across seeds) " + "─" * 9)
        for (ds, alpha, model, method), accs in sorted(high_var):
            print(f"  {ds} α={alpha} {model} {method}  "
                  f"accs={[round(a, 1) for a in accs]}  "
                  f"std={float(np.std(accs)):.2f}%")
        print()

    if all_issues or high_var:
        sys.exit(1)
    else:
        print("All runs passed sanity checks.")


if __name__ == "__main__":
    main()
