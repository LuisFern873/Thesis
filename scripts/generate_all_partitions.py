"""
Generate all data partitions required for the drift experiment.

Experiment matrix (Section 1.2 of methodology):
  - Datasets  : cifar10, brain_tumor
  - α values  : 1000 (IID baseline), 1.0, 0.3, 0.1, 0.03
  - Seeds     : 42, 123, 456
  - Clients   : 10
  Total       : 2 × 5 × 3 = 30 partition runs

Output layout (per run):
  data/{dataset}/partitions/alpha_{α}/seed_{seed}/
    ├── partition.pkl       # FL-bench partition object
    ├── args.json           # run configuration snapshot
    ├── all_stats.json      # raw counts + Hellinger Distance + entropy
    ├── class_distribution.png
    └── partition_md5.txt   # idempotency guard

Usage:
    # Generate everything
    python scripts/generate_all_partitions.py

    # Dry-run: print commands without executing
    python scripts/generate_all_partitions.py --dry-run

    # Single dataset
    python scripts/generate_all_partitions.py --dataset cifar10
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------
DATASETS = ["cifar10", "brain_tumor"]
ALPHAS = [1000, 1.0, 0.3, 0.1, 0.03]
SEEDS = [42, 123, 456]
CLIENT_NUM = 10

FLBENCH_ROOT = Path(__file__).parent.parent.absolute()


def build_command(dataset: str, alpha: float, seed: int) -> list[str]:
    """Build the generate_data.py command for one (dataset, α, seed) cell."""
    return [
        sys.executable,
        str(FLBENCH_ROOT / "generate_data.py"),
        "-d", dataset,
        "-a", str(alpha),
        "-cn", str(CLIENT_NUM),
        "--seed", str(seed),
        "--split", "sample",
        "--test_ratio", "0.25",
        "--val_ratio", "0.0",
        "--min_samples_per_client", "10",
        "--plot_distribution", "1",
    ]


def alpha_str(alpha: float) -> str:
    """Return the alpha string exactly as generate_data.py formats it.

    generate_data.py uses f"alpha_{args.alpha}" where args.alpha is a float,
    so 1000 → "alpha_1000.0", 0.03 → "alpha_0.03", etc.
    """
    return f"alpha_{float(alpha)}"


def expected_output_dir(dataset: str, alpha: float, seed: int) -> Path:
    """Return the directory where generate_data.py will write its outputs."""
    return (
        FLBENCH_ROOT
        / "data"
        / dataset
        / "partitions"
        / alpha_str(alpha)
        / f"seed_{seed}"
    )


def is_already_done(dataset: str, alpha: float, seed: int) -> bool:
    """Return True if this partition was already generated (idempotency check)."""
    out_dir = expected_output_dir(dataset, alpha, seed)
    return (out_dir / "partition.pkl").exists() and (out_dir / "all_stats.json").exists()


def main():
    parser = argparse.ArgumentParser(description="Generate all FL drift experiment partitions.")
    parser.add_argument(
        "--dataset",
        choices=DATASETS + ["all"],
        default="all",
        help="Which dataset to generate partitions for (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate even if partition already exists.",
    )
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    total = len(datasets) * len(ALPHAS) * len(SEEDS)
    done = 0
    skipped = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f"FL-bench Partition Generator — Drift Experiment")
    print(f"Datasets : {datasets}")
    print(f"α values : {ALPHAS}")
    print(f"Seeds    : {SEEDS}")
    print(f"Clients  : {CLIENT_NUM}")
    print(f"Total    : {total} runs")
    print(f"{'='*60}\n")

    for dataset in datasets:
        for alpha in ALPHAS:
            for seed in SEEDS:
                run_label = f"{dataset} | α={alpha:<6} | seed={seed}"

                if not args.force and is_already_done(dataset, alpha, seed):
                    print(f"[SKIP]  {run_label}  (already exists)")
                    skipped += 1
                    continue

                cmd = build_command(dataset, alpha, seed)

                if args.dry_run:
                    print(f"[DRY]   {run_label}")
                    print(f"        {' '.join(cmd)}\n")
                    done += 1
                    continue

                print(f"[RUN]   {run_label}")
                result = subprocess.run(
                    cmd,
                    cwd=str(FLBENCH_ROOT),
                    capture_output=False,
                )

                if result.returncode != 0:
                    print(f"[FAIL]  {run_label}  (exit code {result.returncode})")
                    failed += 1
                else:
                    print(f"[OK]    {run_label}")
                    done += 1

    print(f"\n{'='*60}")
    print(f"Summary: {done} done | {skipped} skipped | {failed} failed")
    print(f"{'='*60}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
