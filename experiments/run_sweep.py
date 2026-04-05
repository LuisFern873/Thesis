"""
Multi-seed experiment sweep for FL-bench thesis experiments.

Runs the full experiment matrix:
  2 FL methods   × {fedavg, fedprox}
  3 architectures × {efficient0, deit, mamba}
  3 seeds        × {42, 123, 456}
  3 α values     × {0.1, 0.5, 1.0}
  = 54 total runs

Usage:
    python experiments/run_sweep.py                     # run everything
    python experiments/run_sweep.py --dry-run            # preview commands
    python experiments/run_sweep.py --methods fedavg     # subset
    python experiments/run_sweep.py --alphas 0.1 0.5     # subset
    python experiments/run_sweep.py --aggregate-only     # just aggregate existing results
"""

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FLBENCH_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = FLBENCH_ROOT / "data" / "tumor"
RESULTS_DIR = FLBENCH_ROOT / "out" / "sweep_results"

# ─── Experiment Matrix ────────────────────────────────────────────────────────
DEFAULT_METHODS = ["fedavg", "fedprox"]
DEFAULT_MODELS = ["efficient0", "deit", "mamba"]
DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_ALPHAS = [0.1, 0.5, 1.0]

# Fixed hyperparameters (shared across all runs)
GLOBAL_EPOCHS = 100
JOIN_RATIO = 0.5
LOCAL_EPOCH = 5
BATCH_SIZE = 32
CLIENT_NUM = 10
OPTIMIZER = "sgd"
LR = 0.01
FEDPROX_MU = 0.01


def partition_key(alpha: float) -> str:
    """Generate a human-readable partition directory name."""
    return f"alpha_{alpha:.1f}"


def generate_partition(alpha: float, seed: int = 42) -> Path:
    """Generate a Dirichlet partition for the given alpha, then store it
    in a named subdirectory so we can switch between partitions."""
    partition_dir = DATA_DIR / partition_key(alpha)
    partition_dir.mkdir(parents=True, exist_ok=True)

    # Check if partition already exists
    if (partition_dir / "partition.pkl").exists():
        print(f"  [skip] Partition α={alpha} already exists at {partition_dir}")
        return partition_dir

    print(f"  [gen]  Generating partition α={alpha} ...")
    cmd = [
        sys.executable, str(FLBENCH_ROOT / "generate_data.py"),
        "-d", "tumor",
        "-a", str(alpha),
        "-cn", str(CLIENT_NUM),
        "--seed", str(seed),
        "--split", "sample",
        "--test_ratio", "0.2",
        "--plot_distribution", "1",
    ]
    subprocess.run(cmd, cwd=str(FLBENCH_ROOT), check=True)

    # Move generated files from data/tumor/ to the named subdirectory
    for fname in ["partition.pkl", "all_stats.json", "args.json",
                  "partition_md5.txt", "class_distribution.png"]:
        src = DATA_DIR / fname
        if src.exists():
            shutil.move(str(src), str(partition_dir / fname))

    print(f"  [done] Partition α={alpha} saved to {partition_dir}")
    return partition_dir


def activate_partition(alpha: float):
    """Copy the named partition files into data/tumor/ so FL-bench loads them."""
    partition_dir = DATA_DIR / partition_key(alpha)
    if not partition_dir.exists():
        raise FileNotFoundError(
            f"Partition directory {partition_dir} not found. "
            f"Run with --generate-partitions first."
        )
    for fname in ["partition.pkl", "all_stats.json", "args.json", "partition_md5.txt"]:
        src = partition_dir / fname
        dst = DATA_DIR / fname
        if src.exists():
            shutil.copy2(str(src), str(dst))
    print(f"  [activated] Partition α={alpha}")


def run_id(method: str, model: str, alpha: float, seed: int) -> str:
    """Unique human-readable identifier for a single run."""
    return f"{method}__{model}__alpha{alpha}__seed{seed}"


def run_single_experiment(method: str, model: str, alpha: float, seed: int,
                          dry_run: bool = False) -> dict:
    """Run a single FL-bench experiment and return metadata."""
    rid = run_id(method, model, alpha, seed)
    output_dir = RESULTS_DIR / rid

    # Skip if already completed
    if output_dir.exists() and (output_dir / "metrics.csv").exists():
        print(f"  [skip] {rid} already completed")
        return {"run_id": rid, "status": "skipped", "output_dir": str(output_dir)}

    cmd = [
        sys.executable, str(FLBENCH_ROOT / "main.py"),
        f"method={method}",
        f"model.name={model}",
        f"dataset.name=tumor",
        f"common.global_epoch={GLOBAL_EPOCHS}",
        f"common.join_ratio={JOIN_RATIO}",
        f"common.local_epoch={LOCAL_EPOCH}",
        f"common.batch_size={BATCH_SIZE}",
        f"common.seed={seed}",
        f"optimizer.name={OPTIMIZER}",
        f"optimizer.lr={LR}",
        f"model.use_torchvision_pretrained_weights=false",
        f"common.monitor=null",
        f"common.save_log=true",
        f"common.save_metrics=true",
        f"common.save_learning_curve_plot=false",
        f"common.delete_useless_run=false",
        f"hydra.run.dir={output_dir.as_posix()}",
    ]

    if method == "fedprox":
        cmd.append(f"fedprox.mu={FEDPROX_MU}")

    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return {"run_id": rid, "status": "dry-run", "cmd": " ".join(cmd)}

    print(f"\n{'='*60}")
    print(f"  Running: {rid}")
    print(f"  Method={method}  Model={model}  α={alpha}  Seed={seed}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(FLBENCH_ROOT))
    elapsed = time.time() - t0

    status = "completed" if result.returncode == 0 else "failed"
    print(f"  [{status}] {rid} in {elapsed/60:.1f} min (exit code: {result.returncode})")

    return {
        "run_id": rid,
        "status": status,
        "elapsed_min": round(elapsed / 60, 2),
        "output_dir": str(output_dir),
        "returncode": result.returncode,
    }


def aggregate_results():
    """Aggregate all completed sweep results into a summary CSV."""
    print("\n" + "="*60)
    print("  Aggregating results...")
    print("="*60)

    rows = []
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_file = run_dir / "metrics.csv"
        if not metrics_file.exists():
            continue

        # Parse run_id: method__model__alphaX.X__seedY
        parts = run_dir.name.split("__")
        if len(parts) != 4:
            continue
        method = parts[0]
        model = parts[1]
        alpha = float(parts[2].replace("alpha", ""))
        seed = int(parts[3].replace("seed", ""))

        df = pd.read_csv(metrics_file)
        # Get the last row (final round) metrics
        if len(df) == 0:
            continue

        last = df.iloc[-1]
        row = {
            "method": method,
            "model": model,
            "alpha": alpha,
            "seed": seed,
        }

        # Extract available metrics from the CSV
        for col in df.columns:
            if col != "epoch":
                row[f"final_{col}"] = last[col]
                # Also record the max value across all rounds
                row[f"max_{col}"] = df[col].max()

        rows.append(row)

    if len(rows) == 0:
        print("  No completed runs found.")
        return

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "all_runs.csv", index=False)
    print(f"  Saved {len(rows)} individual run results to all_runs.csv")

    # Compute aggregated statistics (mean ± std) across seeds
    agg_rows = []
    metric_cols = [c for c in results_df.columns
                   if c.startswith("final_") or c.startswith("max_")]
    group_cols = ["method", "model", "alpha"]

    for keys, group in results_df.groupby(group_cols):
        agg_row = dict(zip(group_cols, keys))
        agg_row["n_seeds"] = len(group)

        for col in metric_cols:
            values = group[col].dropna()
            if len(values) > 0:
                agg_row[f"{col}_mean"] = round(values.mean(), 4)
                agg_row[f"{col}_std"] = round(values.std(), 4)

        agg_rows.append(agg_row)

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(RESULTS_DIR / "aggregated_results.csv", index=False)
    print(f"  Saved {len(agg_rows)} aggregated configs to aggregated_results.csv")

    # Print summary table
    print("\n" + "="*80)
    print("  RESULTS SUMMARY (Final Test Accuracy: mean ± std)")
    print("="*80)

    acc_col = None
    for candidate in ["final_accuracy_test_after", "final_accuracy_test_before"]:
        if f"{candidate}_mean" in agg_df.columns:
            acc_col = candidate
            break

    if acc_col:
        for _, row in agg_df.iterrows():
            mean_val = row.get(f"{acc_col}_mean", "N/A")
            std_val = row.get(f"{acc_col}_std", "N/A")
            print(f"  {row['method']:>8s} | {row['model']:>12s} | α={row['alpha']:.1f} | "
                  f"seeds={row['n_seeds']} | {mean_val:.2f} ± {std_val:.2f}%")
    else:
        print("  (accuracy columns not found; check metrics.csv format)")


def main():
    parser = argparse.ArgumentParser(description="FL-bench thesis experiment sweep")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        choices=DEFAULT_METHODS, help="FL methods to run")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Model architectures to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Random seeds")
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS,
                        help="Dirichlet alpha values")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only aggregate existing results")
    parser.add_argument("--generate-partitions", action="store_true",
                        help="Only generate partitions, don't run experiments")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate_results()
        return

    # ─── Phase 1: Generate all needed partitions ──────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 1: Generating data partitions")
    print("="*60)
    for alpha in args.alphas:
        generate_partition(alpha)

    if args.generate_partitions:
        print("\nPartitions generated. Exiting (--generate-partitions mode).")
        return

    # ─── Phase 2: Run all experiments ─────────────────────────────────────────
    total = len(args.methods) * len(args.models) * len(args.seeds) * len(args.alphas)
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Running {total} experiments")
    print(f"  Methods: {args.methods}")
    print(f"  Models:  {args.models}")
    print(f"  Seeds:   {args.seeds}")
    print(f"  Alphas:  {args.alphas}")
    print(f"{'='*60}")

    run_log = []
    completed = 0
    failed = 0

    # Group experiments by alpha to minimize partition switching
    for alpha in args.alphas:
        print(f"\n--- Activating partition α={alpha} ---")
        activate_partition(alpha)

        for method, model, seed in itertools.product(
            args.methods, args.models, args.seeds
        ):
            result = run_single_experiment(
                method, model, alpha, seed, dry_run=args.dry_run
            )
            run_log.append(result)
            if result["status"] == "completed":
                completed += 1
            elif result["status"] == "failed":
                failed += 1

    # Save run log
    log_path = RESULTS_DIR / "run_log.json"
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  SWEEP COMPLETE")
    print(f"  Total: {total}  Completed: {completed}  Failed: {failed}  "
          f"Skipped: {total - completed - failed}")
    print(f"  Run log saved to: {log_path}")
    print(f"{'='*60}")

    # ─── Phase 3: Aggregate results ───────────────────────────────────────────
    if not args.dry_run:
        aggregate_results()


if __name__ == "__main__":
    main()
