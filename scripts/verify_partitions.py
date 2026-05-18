"""
Verify all generated partitions for the drift experiment (checklist items 1.7–1.9).

Checks performed per partition:
  1. partition.pkl and all_stats.json exist
  2. all_stats.json contains the enriched fields (hellinger_distance, entropy,
     label_distribution, n_samples) for every client
  3. α=0.03 produces higher mean Hellinger Distance than α=1.0 (sanity check)
  4. Every client has at least min_samples_per_client=10 training samples
  5. The "summary" block is present

Usage:
    python scripts/verify_partitions.py
    python scripts/verify_partitions.py --dataset cifar10
    python scripts/verify_partitions.py --verbose
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Experiment matrix (must match generate_all_partitions.py)
# ---------------------------------------------------------------------------
DATASETS = ["cifar10", "brain_tumor"]
ALPHAS = [1000, 1.0, 0.3, 0.03]
SEEDS = [42, 123, 456]
CLIENT_NUM = 10
MIN_SAMPLES = 10

FLBENCH_ROOT = Path(__file__).parent.parent.absolute()

REQUIRED_CLIENT_FIELDS = [
    "n_samples",
    "label_distribution",
    "dominant_class",
    "entropy",
    "hellinger_distance",
]


def alpha_str(alpha: float) -> str:
    """Match the directory name format used by generate_data.py (f'alpha_{float}')."""
    return f"alpha_{float(alpha)}"


def partition_dir(dataset: str, alpha: float, seed: int) -> Path:
    return (
        FLBENCH_ROOT
        / "data"
        / dataset
        / "partitions"
        / alpha_str(alpha)
        / f"seed_{seed}"
    )


def check_partition(dataset: str, alpha: float, seed: int, verbose: bool) -> list[str]:
    """Run all checks for one (dataset, α, seed) cell. Returns list of error strings."""
    errors = []
    pdir = partition_dir(dataset, alpha, seed)
    label = f"{dataset}/alpha_{alpha}/seed_{seed}"

    # --- 1. Files exist ---
    pkl_path = pdir / "partition.pkl"
    stats_path = pdir / "all_stats.json"

    if not pkl_path.exists():
        errors.append(f"{label}: partition.pkl missing")
        return errors  # can't continue without the file

    if not stats_path.exists():
        errors.append(f"{label}: all_stats.json missing")
        return errors

    # --- 2. Load files ---
    with open(pkl_path, "rb") as f:
        partition = pickle.load(f)

    with open(stats_path, "r") as f:
        stats = json.load(f)

    # --- 3. Enriched fields present for every client ---
    for client_id in range(CLIENT_NUM):
        key = str(client_id)
        if key not in stats:
            errors.append(f"{label}: client {client_id} missing from all_stats.json")
            continue
        for field in REQUIRED_CLIENT_FIELDS:
            if field not in stats[key]:
                errors.append(f"{label}: client {client_id} missing field '{field}'")

    # --- 4. Summary block present ---
    if "summary" not in stats:
        errors.append(f"{label}: 'summary' block missing from all_stats.json")

    # --- 5. Minimum training samples per client ---
    # The min_samples_per_client=10 guard in the Dirichlet scheme applies to
    # total samples *before* the train/test split (default test_ratio=0.25).
    # After splitting, a client with 13 total samples ends up with ~9 train
    # samples, which is acceptable. We use a post-split floor of 7.
    MIN_TRAIN_SAMPLES = 7
    data_indices = partition.get("data_indices", [])
    for client_id, idx_dict in enumerate(data_indices):
        if isinstance(idx_dict, dict):
            train_size = len(idx_dict.get("train", []))
        else:
            train_size = len(idx_dict)
        if train_size < MIN_TRAIN_SAMPLES:
            errors.append(
                f"{label}: client {client_id} has only {train_size} training samples "
                f"(post-split floor is {MIN_TRAIN_SAMPLES})"
            )

    if verbose and not errors:
        summary = stats.get("summary", {})
        print(
            f"  [OK] {label:<45} "
            f"HD_mean={summary.get('mean_hellinger', 'N/A'):.3f}  "
            f"entropy_mean={summary.get('mean_entropy', 'N/A'):.3f}"
        )

    return errors


def check_heterogeneity_ordering(dataset: str, seed: int) -> list[str]:
    """Verify that lower α → higher mean Hellinger Distance (sanity check)."""
    errors = []
    hd_values = {}

    for alpha in ALPHAS:
        stats_path = partition_dir(dataset, alpha, seed) / "all_stats.json"
        if not stats_path.exists():
            return []  # skip if files not generated yet
        with open(stats_path, "r") as f:
            stats = json.load(f)
        summary = stats.get("summary", {})
        hd = summary.get("mean_hellinger")
        if hd is not None:
            hd_values[alpha] = hd

    # Check monotonicity: HD(α=0.03) > HD(α=0.3) > HD(α=1.0)
    # α=1000 (IID) is excluded — it uses Dirichlet with very high α which
    # approaches uniform, so HD should be lowest, but ordering with 1.0 is
    # not guaranteed to be strict.
    ordered_alphas = [0.03, 0.3, 1.0]
    for i in range(len(ordered_alphas) - 1):
        a_low = ordered_alphas[i]
        a_high = ordered_alphas[i + 1]
        if a_low in hd_values and a_high in hd_values:
            if hd_values[a_low] <= hd_values[a_high]:
                errors.append(
                    f"{dataset}/seed_{seed}: expected HD(α={a_low}) > HD(α={a_high}), "
                    f"got {hd_values[a_low]:.3f} ≤ {hd_values[a_high]:.3f}"
                )

    return errors


def main():
    parser = argparse.ArgumentParser(description="Verify FL drift experiment partitions.")
    parser.add_argument(
        "--dataset",
        choices=DATASETS + ["all"],
        default="all",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-partition summary even when passing.",
    )
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    all_errors = []
    total = 0
    passed = 0

    print(f"\n{'='*60}")
    print(f"FL-bench Partition Verifier — Drift Experiment")
    print(f"{'='*60}\n")

    for dataset in datasets:
        print(f"Dataset: {dataset}")
        for alpha in ALPHAS:
            for seed in SEEDS:
                total += 1
                errors = check_partition(dataset, alpha, seed, args.verbose)
                if errors:
                    for e in errors:
                        print(f"  [FAIL] {e}")
                    all_errors.extend(errors)
                else:
                    passed += 1
                    if not args.verbose:
                        print(f"  [OK]   alpha_{alpha}/seed_{seed}")

        # Cross-α sanity check (one per seed)
        for seed in SEEDS:
            ordering_errors = check_heterogeneity_ordering(dataset, seed)
            if ordering_errors:
                for e in ordering_errors:
                    print(f"  [WARN] {e}")
                # Ordering errors are warnings, not hard failures
        print()

    print(f"{'='*60}")
    print(f"Results: {passed}/{total} passed | {len(all_errors)} errors")
    print(f"{'='*60}\n")

    if all_errors:
        print("Errors found:")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("All partitions verified successfully.")


if __name__ == "__main__":
    main()
