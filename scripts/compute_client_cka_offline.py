"""compute_client_cka_offline.py — Offline pairwise client-to-client CKA computation.

Reads the model checkpoints saved by :class:`CKADriftFedAvgServer` (or
:class:`CKADriftFedProxServer`) and computes CKA similarity between every
*pair* of client models (client_i vs. client_j, i < j) at each scheduled
round — without involving the global model.

This complements ``compute_cka_offline.py`` (which measures client-vs-global
drift) by revealing how much clients' internal representations diverge from
*each other*.

Output per run directory
------------------------

    <run_dir>/
        client_cka_metrics.csv      — one row per (round, client_i, client_j, layer)

CSV schema
----------
    round, client_i, client_j, model_name, dataset, alpha,
    layer_index, layer_name, cka_diagonal

No full matrices are written because pairwise matrices grow as O(K²) in the
number of clients.  If you need the full matrix, add ``--save-matrices``.

Usage examples
--------------
  # Single run directory
  python scripts/compute_client_cka_offline.py \\
      --run-dir logs/runs/brain_tumor_alpha0.1_efficient1_ckadriftfedavg_seed42

  # All runs under a parent directory
  python scripts/compute_client_cka_offline.py --logs-dir logs/runs

  # Restrict to specific rounds and sample at most 6 clients per round
  python scripts/compute_client_cka_offline.py \\
      --run-dir logs/runs/... --rounds 1 10 50 --client-sample 6

  # Run on GPU (default: CPU)
  python scripts/compute_client_cka_offline.py --run-dir logs/runs/... --device cuda

  # Resume a partially-completed run (skip rounds already in the CSV)
  python scripts/compute_client_cka_offline.py --run-dir logs/runs/... --skip-existing

  # Also persist the full (L × L) matrix per client pair
  python scripts/compute_client_cka_offline.py --run-dir logs/runs/... --save-matrices
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from itertools import combinations, islice
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
FLBENCH_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(FLBENCH_ROOT))
sys.path.insert(0, str(FLBENCH_ROOT / "simtorch"))

# ---------------------------------------------------------------------------
# FL-bench imports
# ---------------------------------------------------------------------------
from src.utils.models import MODELS  # noqa: E402
from src.utils.cka_drift import (  # noqa: E402
    SIMTORCH_AVAILABLE,
    build_probe_loader,
    get_layer_spec,
)
from data.utils.datasets import DATASETS  # noqa: E402

if SIMTORCH_AVAILABLE:
    from simtorch.model.sim_model import SimilarityModel
    from simtorch.similarity.cka import CKA
else:
    print(
        "ERROR: simtorch not found. Install it before running this script.\n"
        f"  Expected location: {FLBENCH_ROOT / 'simtorch'}"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------
_CSV_HEADERS = [
    "round",
    "client_i",
    "client_j",
    "model_name",
    "dataset",
    "alpha",
    "layer_index",
    "layer_name",
    "cka_diagonal",
]


# ---------------------------------------------------------------------------
# Model construction (mirrors compute_cka_offline.py)
# ---------------------------------------------------------------------------

def _build_model(model_name: str, dataset_name: str, pretrained: bool = False) -> nn.Module:
    from argparse import Namespace

    model = MODELS[model_name](dataset=dataset_name, pretrained=pretrained)
    args_stub = Namespace(
        common=Namespace(buffers="global"),
        model=Namespace(
            name=model_name,
            use_torchvision_pretrained_weights=pretrained,
            external_model_weights_path=None,
        ),
    )
    model.check_and_preprocess(args_stub)
    model.eval()
    return model


def _load_model(model_name: str, dataset_name: str, ckpt_path: Path, pretrained: bool = False) -> nn.Module:
    model = _build_model(model_name, dataset_name, pretrained)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Dataset + probe loader construction
# ---------------------------------------------------------------------------

def _build_testset(dataset_name: str):
    from src.utils.constants import DATA_MEAN, DATA_STD
    from argparse import Namespace

    root = FLBENCH_ROOT / "data" / dataset_name
    mean = DATA_MEAN.get(dataset_name, [0.5, 0.5, 0.5])
    std = DATA_STD.get(dataset_name, [0.5, 0.5, 0.5])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=mean, std=std),
    ])
    args_stub = Namespace(name=dataset_name)
    dataset = DATASETS[dataset_name](
        root=root,
        args=args_stub,
        test_data_transform=test_transform,
        test_target_transform=transforms.Compose([]),
        train_data_transform=test_transform,
        train_target_transform=transforms.Compose([]),
    )
    dataset.eval()
    return dataset


# ---------------------------------------------------------------------------
# Pairwise CKA computation
# ---------------------------------------------------------------------------

def compute_cka_for_pair(
    model_i: nn.Module,
    model_j: nn.Module,
    layer_spec: List[str],
    probe_loader: DataLoader,
    probe_batches: int,
    device: torch.device,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    """Compute the CKA similarity matrix between two client models.

    Returns
    -------
    (diagonal, matrix, layer_names) — or (None, None, None) on failure.
    """
    try:
        model_i.to(device).eval()
        model_j.to(device).eval()

        with torch.no_grad():
            if probe_batches >= 1:
                data_iter = [
                    (x.cpu(), *rest)
                    for x, *rest in islice(probe_loader, probe_batches)
                ]
            else:
                data_iter = list(probe_loader)

        try:
            sim_i = SimilarityModel(model_i, layers_to_include=layer_spec, device=device)
            sim_j = SimilarityModel(model_j, layers_to_include=layer_spec, device=device)
            cka = CKA(sim_i, sim_j, device=device)
            matrix: np.ndarray = cka.compute(data_iter)

            diagonal: np.ndarray = np.diag(matrix)
            layer_names = list(sim_j.model_activations.keys())
            return diagonal, matrix, layer_names

        finally:
            del data_iter
            try:
                del sim_i, sim_j, cka
            except NameError:
                pass
            if device.type == "cuda":
                torch.cuda.empty_cache()

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        print(f"    [WARN] CKA pair failed: {type(exc).__name__}: {exc}")
        return None, None, None

    finally:
        model_i.cpu()
        model_j.cpu()


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _init_csv(csv_path: Path) -> None:
    """Write the header row if the file does not exist yet."""
    if not csv_path.exists():
        with open(csv_path, "w") as f:
            f.write(",".join(_CSV_HEADERS) + "\n")


def _append_csv_rows(
    csv_path: Path,
    round_idx: int,
    client_i: int,
    client_j: int,
    model_name: str,
    dataset_name: str,
    alpha: str,
    diagonal: np.ndarray,
    layer_names: List[str],
) -> None:
    try:
        with open(csv_path, "a") as f:
            for layer_index, (layer_name, value) in enumerate(zip(layer_names, diagonal)):
                f.write(
                    f"{round_idx},{client_i},{client_j},{model_name},{dataset_name},"
                    f"{alpha},{layer_index},{layer_name},{value:.6f}\n"
                )
    except Exception as e:
        print(f"\n    [WARN] Failed to write CSV rows: {e}")


def _already_computed(csv_path: Path, round_idx: int, ci: int, cj: int) -> bool:
    """Check whether a (round, client_i, client_j) row is already in the CSV."""
    if not csv_path.exists():
        return False
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, usecols=["round", "client_i", "client_j"])
        return bool(((df["round"] == round_idx) & (df["client_i"] == ci) & (df["client_j"] == cj)).any())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def process_run(
    run_dir: Path,
    rounds_override: Optional[Set[int]],
    client_sample_override: Optional[int],
    probe_batches_override: Optional[int],
    probe_batch_size_override: Optional[int],
    device: torch.device,
    skip_existing: bool,
    save_matrices: bool,
    verbose: bool,
) -> int:
    """Compute pairwise client-to-client CKA for all scheduled rounds.

    Returns the number of (round, i, j) pairs successfully computed.
    """
    checkpoints_dir = run_dir / "checkpoints"
    metadata_path = checkpoints_dir / "run_metadata.json"

    if not metadata_path.exists():
        print(f"  [SKIP] No run_metadata.json in {run_dir} — not a CKA run")
        return 0

    with open(metadata_path) as f:
        meta = json.load(f)

    model_name = meta["model_name"]
    dataset_name = meta["dataset_name"]
    alpha = meta["alpha"]
    seed = meta.get("seed", 42)
    cka_rounds: List[int] = meta["cka_rounds"]
    client_sample = client_sample_override or meta["client_sample"]
    probe_batches = probe_batches_override if probe_batches_override is not None else meta["probe_batches"]
    probe_batch_size = probe_batch_size_override or meta["probe_batch_size"]
    use_pretrained = meta.get("use_pretrained", False)

    if rounds_override:
        cka_rounds = [r for r in cka_rounds if r in rounds_override]
    if not cka_rounds:
        print(f"  [SKIP] No matching rounds in {run_dir.name}")
        return 0

    print(f"  model={model_name}  dataset={dataset_name}  α={alpha}  seed={seed}")
    print(f"  rounds={cka_rounds}  client_sample={client_sample}  probe_batches={probe_batches}")

    # Output paths
    csv_path = run_dir / "client_cka_metrics.csv"
    _init_csv(csv_path)

    matrices_dir: Optional[Path] = None
    if save_matrices:
        matrices_dir = run_dir / "client_cka_matrices"
        matrices_dir.mkdir(exist_ok=True)

    # Build probe loader once
    print("  Building testset and probe loader...", end=" ", flush=True)
    try:
        testset = _build_testset(dataset_name)
        probe_loader = build_probe_loader(
            testset,
            batch_size=probe_batch_size,
            dataset_name=dataset_name,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        return 0

    layer_spec = get_layer_spec(model_name)

    total_computed = 0
    rng = random.Random(seed)

    for round_idx in sorted(cka_rounds):
        round_dir = checkpoints_dir / f"round_{round_idx:04d}"

        client_ckpts = sorted(round_dir.glob("client_*.pt"))
        if not client_ckpts:
            print(f"  [WARN] Round {round_idx}: no client checkpoints — skipping")
            continue

        # Parse available client IDs
        available_clients: List[int] = []
        for p in client_ckpts:
            try:
                cid = int(p.stem.split("_")[1])
                available_clients.append(cid)
            except (IndexError, ValueError):
                pass

        # Sample a subset of clients (same deterministic seed as training)
        k = min(client_sample, len(available_clients))
        sampled_clients = rng.sample(available_clients, k)

        # All ordered pairs (i, j) with i < j
        pairs = list(combinations(sorted(sampled_clients), 2))
        print(
            f"  Round {round_idx:3d}: {len(sampled_clients)} clients → "
            f"{len(pairs)} pairs",
            end="",
            flush=True,
        )

        if not pairs:
            print(" | skipped (need ≥ 2 clients)")
            continue

        # Pre-load all sampled client models for this round to avoid
        # reloading model_j multiple times across pairs.
        loaded_models: Dict[int, nn.Module] = {}
        for cid in sampled_clients:
            ckpt = round_dir / f"client_{cid:03d}.pt"
            try:
                loaded_models[cid] = _load_model(model_name, dataset_name, ckpt, use_pretrained)
            except Exception as e:
                print(f"\n    [WARN] Client {cid}: load failed: {e}")

        all_diagonals: List[np.ndarray] = []

        for ci, cj in pairs:
            if ci not in loaded_models or cj not in loaded_models:
                print("x", end="", flush=True)
                continue

            # Skip already-computed pairs when --skip-existing is set
            if skip_existing and _already_computed(csv_path, round_idx, ci, cj):
                print(".", end="", flush=True)
                continue

            # Deep-copy so CKA hooks don't accumulate across pairs
            mi = copy.deepcopy(loaded_models[ci])
            mj = copy.deepcopy(loaded_models[cj])

            diagonal, matrix, layer_names = compute_cka_for_pair(
                model_i=mi,
                model_j=mj,
                layer_spec=layer_spec,
                probe_loader=probe_loader,
                probe_batches=probe_batches,
                device=device,
            )

            del mi, mj

            if diagonal is None:
                print("x", end="", flush=True)
                continue

            _append_csv_rows(
                csv_path=csv_path,
                round_idx=round_idx,
                client_i=ci,
                client_j=cj,
                model_name=model_name,
                dataset_name=dataset_name,
                alpha=alpha,
                diagonal=diagonal,
                layer_names=layer_names,
            )

            if save_matrices and matrices_dir is not None:
                npz_path = matrices_dir / f"round_{round_idx:04d}_ci{ci:03d}_cj{cj:03d}.npz"
                try:
                    np.savez_compressed(
                        npz_path,
                        matrix=matrix.astype(np.float32),
                        layer_names=np.array(layer_names, dtype=object),
                    )
                except Exception as e:
                    print(f"\n    [WARN] Failed to save matrix: {e}")

            all_diagonals.append(diagonal)
            total_computed += 1
            print(".", end="", flush=True)

        # Free all pre-loaded models for this round
        for m in loaded_models.values():
            del m
        loaded_models.clear()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if all_diagonals:
            mean_cka = float(np.mean(np.concatenate(all_diagonals)))
            print(f" | mean pairwise CKA={mean_cka:.4f}")
        else:
            print(" | no results")

    return total_computed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline pairwise client-to-client CKA computation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--run-dir",
        type=Path,
        metavar="DIR",
        help="Path to a single run directory (must contain checkpoints/run_metadata.json).",
    )
    src_group.add_argument(
        "--logs-dir",
        type=Path,
        metavar="DIR",
        help="Parent directory of run subdirectories.  All subdirs are processed.",
    )

    parser.add_argument(
        "--rounds",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Only process these 1-based round indices (default: all from metadata).",
    )
    parser.add_argument(
        "--client-sample",
        type=int,
        default=None,
        metavar="N",
        help="Max clients to sample per round (default: value from run_metadata.json).",
    )
    parser.add_argument(
        "--probe-batches",
        type=int,
        default=None,
        metavar="N",
        help="Max probe batches (default: value from run_metadata.json).",
    )
    parser.add_argument(
        "--probe-batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Probe batch size (default: value from run_metadata.json).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        metavar="DEVICE",
        help="Torch device for CKA forward passes (default: cpu).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip (round, i, j) pairs already present in client_cka_metrics.csv.",
    )
    parser.add_argument(
        "--save-matrices",
        action="store_true",
        help="Also persist the full (L×L) CKA matrix per client pair as .npz files "
             "in <run_dir>/client_cka_matrices/.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-pair progress dots.",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    rounds_override = set(args.rounds) if args.rounds else None

    if args.run_dir is not None:
        run_dirs = [args.run_dir.resolve()]
    else:
        logs_dir = args.logs_dir.resolve()
        if not logs_dir.exists():
            print(f"ERROR: --logs-dir does not exist: {logs_dir}")
            sys.exit(1)
        run_dirs = sorted(d for d in logs_dir.iterdir() if d.is_dir())

    total = 0
    for run_dir in run_dirs:
        print(f"\nProcessing: {run_dir.name}")
        n = process_run(
            run_dir=run_dir,
            rounds_override=rounds_override,
            client_sample_override=args.client_sample,
            probe_batches_override=args.probe_batches,
            probe_batch_size_override=args.probe_batch_size,
            device=device,
            skip_existing=args.skip_existing,
            save_matrices=args.save_matrices,
            verbose=not args.quiet,
        )
        total += n

    print(f"\nDone — {total} pairwise CKA matrices computed across {len(run_dirs)} run(s).")


if __name__ == "__main__":
    main()
