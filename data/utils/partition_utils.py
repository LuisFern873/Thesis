"""
Partition statistics utilities for FL-bench drift experiments.

Computes per-client label distribution metrics (Hellinger Distance, entropy,
dominant class) and enriches the all_stats.json produced by generate_data.py.

Usage (called automatically by generate_data.py after partitioning):
    from data.utils.partition_utils import compute_partition_stats, enrich_stats

Reference: Jimenez et al. (2025) for the α → Hellinger Distance mapping.
"""

import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np


# ---------------------------------------------------------------------------
# Core metric: per-client statistics
# ---------------------------------------------------------------------------

def compute_partition_stats(
    data_indices: List[Dict[str, Union[List[int], np.ndarray]]],
    targets: np.ndarray,
    num_classes: int,
) -> Dict:
    """Compute per-client label distribution statistics for a given partition.

    For each client the function computes:
      - n_samples        : total number of samples assigned to the client
      - label_distribution: fraction of each class (length = num_classes)
      - dominant_class   : class index with the highest fraction
      - entropy          : Shannon entropy of the label distribution (nats)
      - hellinger_distance: Hellinger Distance from the uniform distribution

    The Hellinger Distance from uniform is defined as:
        HD(p, u) = (1/√2) * ||√p - √u||_2
    where u = [1/C, ..., 1/C] is the uniform distribution over C classes.
    HD = 0 means perfectly IID; HD → 1 means maximally non-IID.

    Args:
        data_indices: List of per-client dicts with key "train" (and optionally
                      "val", "test") holding integer index arrays.  Matches the
                      format written by generate_data.py (split="sample").
        targets:      Full target array for the dataset (numpy int32/int64).
        num_classes:  Total number of classes C.

    Returns:
        A dict with integer client keys (0 … K-1) plus a "summary" key:
        {
            0: {
                "n_samples": int,
                "label_distribution": [float, ...],   # length num_classes
                "dominant_class": int,
                "entropy": float,
                "hellinger_distance": float,
            },
            ...
            "summary": {
                "mean_hellinger": float,
                "std_hellinger": float,
                "mean_entropy": float,
                "std_entropy": float,
                "min_samples": int,
                "max_samples": int,
            }
        }
    """
    uniform_dist = np.ones(num_classes) / num_classes
    stats: Dict = {}
    hellinger_values: List[float] = []
    entropy_values: List[float] = []
    sample_counts: List[int] = []

    for client_id, indices_dict in enumerate(data_indices):
        # Collect all indices for this client (train + val + test)
        if isinstance(indices_dict, dict):
            all_idx = np.concatenate(
                [np.asarray(v) for v in indices_dict.values() if len(v) > 0]
            ).astype(np.int64)
        else:
            # Legacy format: plain list / array
            all_idx = np.asarray(indices_dict, dtype=np.int64)

        if len(all_idx) == 0:
            # Client has no data — record zeros
            stats[client_id] = {
                "n_samples": 0,
                "label_distribution": [0.0] * num_classes,
                "dominant_class": -1,
                "entropy": 0.0,
                "hellinger_distance": 1.0,
            }
            hellinger_values.append(1.0)
            entropy_values.append(0.0)
            sample_counts.append(0)
            continue

        client_labels = targets[all_idx]
        counts = np.bincount(client_labels, minlength=num_classes)
        dist = counts / counts.sum()  # normalised label distribution

        # Shannon entropy (nats); add epsilon to avoid log(0)
        entropy = float(-np.sum(dist * np.log(dist + 1e-12)))

        # Hellinger Distance from uniform
        hd = float(
            np.sqrt(np.sum((np.sqrt(dist) - np.sqrt(uniform_dist)) ** 2)) / np.sqrt(2)
        )

        stats[client_id] = {
            "n_samples": int(len(all_idx)),
            "label_distribution": dist.tolist(),
            "dominant_class": int(np.argmax(dist)),
            "entropy": entropy,
            "hellinger_distance": hd,
        }

        hellinger_values.append(hd)
        entropy_values.append(entropy)
        sample_counts.append(len(all_idx))

    # Aggregate summary across clients
    stats["summary"] = {
        "mean_hellinger": float(np.mean(hellinger_values)),
        "std_hellinger": float(np.std(hellinger_values)),
        "mean_entropy": float(np.mean(entropy_values)),
        "std_entropy": float(np.std(entropy_values)),
        "min_samples": int(np.min(sample_counts)) if sample_counts else 0,
        "max_samples": int(np.max(sample_counts)) if sample_counts else 0,
    }

    return stats


# ---------------------------------------------------------------------------
# Enrichment helper: merge into existing all_stats.json
# ---------------------------------------------------------------------------

def enrich_stats(
    save_dir: Path,
    data_indices: List[Dict],
    targets: np.ndarray,
    num_classes: int,
) -> None:
    """Compute partition stats and merge them into the existing all_stats.json.

    The function reads the raw stats written by generate_data.py (which only
    contains "x" and "y" keys per client), computes the richer metrics via
    ``compute_partition_stats``, and writes the merged result back to disk.

    Args:
        save_dir:    Directory containing all_stats.json (written by generate_data.py).
        data_indices: Per-client index dicts (from partition["data_indices"]).
        targets:     Full target array for the dataset.
        num_classes: Total number of classes.
    """
    stats_path = save_dir / "all_stats.json"

    # Load existing raw stats (may not exist yet during first write)
    existing: Dict = {}
    if stats_path.exists():
        with open(stats_path, "r") as f:
            existing = json.load(f)

    # Compute rich metrics
    rich_stats = compute_partition_stats(data_indices, targets, num_classes)

    # Merge: add rich fields into each client entry
    for key, value in rich_stats.items():
        if key == "summary":
            existing["summary"] = value
        else:
            client_key = str(key)
            if client_key in existing:
                existing[client_key].update(value)
            else:
                existing[client_key] = value

    with open(stats_path, "w") as f:
        json.dump(existing, f, indent=4)
