"""compute_cka_offline.py — Offline CKA computation from training checkpoints.

Reads the model checkpoints saved by :class:`CKADriftFedAvgServer` (or
:class:`CKADriftFedProxServer`) and computes all CKA metrics without any
impact on training throughput or GPU memory.

Background
----------
During training, the server saves only state dicts:

    <run_dir>/checkpoints/
        run_metadata.json
        round_0001/
            global.pt
            client_000.pt  client_001.pt  ...
        round_0002/
        ...

This script reconstructs the models from those state dicts, runs the CKA
probe forward passes, and writes the same outputs that the inline version
would have produced:

    <run_dir>/
        cka_metrics.csv           — one row per (round, client, layer)
        cka_matrices/
            round_0001_client_000.npz
            ...

Usage examples
--------------
  # Single run directory (auto-detects checkpoints/run_metadata.json)
  python scripts/compute_cka_offline.py --run-dir logs/runs/brain_tumor_alpha1.0_res9_ckadriftfedavg_seed42

  # All runs under a parent directory
  python scripts/compute_cka_offline.py --logs-dir logs/runs

  # Override probe settings (useful for quick re-runs)
  python scripts/compute_cka_offline.py --run-dir logs/runs/... --probe-batches 10 --probe-batch-size 8

  # Restrict to specific rounds or clients
  python scripts/compute_cka_offline.py --run-dir logs/runs/... --rounds 1 10 50 --client-sample 5

  # Run on GPU (default is CPU — safer on shared clusters)
  python scripts/compute_cka_offline.py --run-dir logs/runs/... --device cuda

  # Skip rounds already computed (idempotent re-runs)
  python scripts/compute_cka_offline.py --run-dir logs/runs/... --skip-existing
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# ---------------------------------------------------------------------------
# Repo root on sys.path so we can import src.* and data.*
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
FLBENCH_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(FLBENCH_ROOT))
sys.path.insert(0, str(FLBENCH_ROOT / "simtorch"))

# ---------------------------------------------------------------------------
# FL-bench imports (after path setup)
# ---------------------------------------------------------------------------
from src.utils.models import MODELS  # noqa: E402
from src.utils.cka_drift import (  # noqa: E402
    ARCHITECTURE_LAYER_MAP,
    SIMTORCH_AVAILABLE,
    build_probe_loader,
    extract_alpha,
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
# CSV header (mirrors CKADriftFedAvgServer._CKA_CSV_HEADERS)
# ---------------------------------------------------------------------------
_CKA_CSV_HEADERS = [
    "round",
    "client_id",
    "model_name",
    "dataset",
    "alpha",
    "layer_index",
    "layer_name",
    "cka_diagonal",
]


# ---------------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------------

def build_model(model_name: str, dataset_name: str, pretrained: bool = False) -> nn.Module:
    """Instantiate a model from the MODELS registry and return it in eval mode.

    Uses ``model.check_and_preprocess()`` with a minimal args stub that sets
    ``buffers='global'`` — matching the training configuration used in all
    CKA experiments (BN running stats are treated as parameters and included
    in the state dict).

    Args:
        model_name:   Key in the ``MODELS`` registry (e.g. ``"res9"``).
        dataset_name: Dataset name passed to the model constructor
                      (determines ``num_classes`` and ``in_channels``).
        pretrained:   Whether to load torchvision pretrained weights.
                      Should match ``model.use_torchvision_pretrained_weights``
                      from the original run.  Almost always ``False`` for FL
                      experiments.

    Returns:
        The model instance in ``eval()`` mode on CPU.
    """
    from argparse import Namespace

    model = MODELS[model_name](dataset=dataset_name, pretrained=pretrained)

    # check_and_preprocess converts BN buffers → nn.Parameter when
    # buffers="global", matching the training config.
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


# ---------------------------------------------------------------------------
# Dataset reconstruction
# ---------------------------------------------------------------------------

def build_testset(dataset_name: str) -> object:
    """Load the full dataset and return it configured for test-time use.

    Applies the same transform pipeline used by ``FedAvgServer`` during
    training: ``Resize(224) → Normalize``.

    Args:
        dataset_name: Key in the ``DATASETS`` registry.

    Returns:
        A ``BaseDataset`` instance with ``data_transform`` set to the
        test-time transform (``Resize(224) → Normalize``).
    """
    from src.utils.constants import DATA_MEAN, DATA_STD
    from argparse import Namespace

    root = FLBENCH_ROOT / "data" / dataset_name

    mean = DATA_MEAN.get(dataset_name, [0.5, 0.5, 0.5])
    std = DATA_STD.get(dataset_name, [0.5, 0.5, 0.5])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=mean, std=std),
    ])
    # Minimal args stub — only dataset name is needed by most dataset classes
    args_stub = Namespace(name=dataset_name)

    dataset = DATASETS[dataset_name](
        root=root,
        args=args_stub,
        test_data_transform=test_transform,
        test_target_transform=transforms.Compose([]),
        train_data_transform=test_transform,   # probe uses test transform
        train_target_transform=transforms.Compose([]),
    )
    dataset.eval()   # activates test_data_transform
    return dataset


# ---------------------------------------------------------------------------
# CKA computation for a single (global, client) pair
# ---------------------------------------------------------------------------

def compute_cka_for_pair(
    global_model: nn.Module,
    client_model: nn.Module,
    layer_spec: List[str],
    probe_loader: DataLoader,
    probe_batches: int,
    device: torch.device,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    """Compute the CKA similarity matrix for one global–client model pair.

    Args:
        global_model:  Global model in ``eval()`` mode.
        client_model:  Client model in ``eval()`` mode.
        layer_spec:    Exact submodule paths to hook (from ``ARCHITECTURE_LAYER_MAP``).
        probe_loader:  Deterministic probe DataLoader.
        probe_batches: Max batches to consume; -1/0 for all.
        device:        Target device for forward passes.

    Returns:
        ``(diagonal, matrix, layer_names)`` or ``(None, None, None)`` on error.
    """
    try:
        global_model.to(device).eval()
        client_model.to(device).eval()

        # Build data iterator — move batches to target device inline during
        # the CKA compute loop (simtorch does X.to(device) internally via
        # the CKA constructor's model placement, so we just supply raw
        # CPU tensors here).
        with torch.no_grad():
            if probe_batches >= 1:
                data_iter = [
                    (x.cpu(), *rest)
                    for x, *rest in islice(probe_loader, probe_batches)
                ]
            else:
                data_iter = list(probe_loader)

        try:
            sim_global = SimilarityModel(
                global_model, layers_to_include=layer_spec, device=device
            )
            sim_client = SimilarityModel(
                client_model, layers_to_include=layer_spec, device=device
            )
            cka = CKA(sim_global, sim_client, device=device)
            matrix: np.ndarray = cka.compute(data_iter)

            diagonal: np.ndarray = np.diag(matrix)
            layer_names = list(sim_client.model_activations.keys())
            return diagonal, matrix, layer_names

        finally:
            del data_iter
            try:
                del sim_global, sim_client, cka
            except NameError:
                pass
            if device.type == "cuda":
                torch.cuda.empty_cache()

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        print(f"    [WARN] CKA computation failed: {type(exc).__name__}: {exc}")
        return None, None, None

    finally:
        # Always move models back to CPU so the caller's pool is not
        # accumulating GPU memory between clients.
        global_model.cpu()
        client_model.cpu()


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
    verbose: bool,
) -> int:
    """Compute CKA for all scheduled rounds in one run directory.

    Args:
        run_dir:                   Path to the run output directory.
        rounds_override:           If set, process only these 1-based rounds
                                   (instead of the rounds listed in metadata).
        client_sample_override:    If set, sample at most this many clients per
                                   round (instead of ``client_sample`` in metadata).
        probe_batches_override:    Override ``probe_batches`` from metadata.
        probe_batch_size_override: Override ``probe_batch_size`` from metadata.
        device:                    Torch device for CKA forward passes.
        skip_existing:             If True, skip (round, client) pairs whose
                                   ``.npz`` file already exists.
        verbose:                   Print per-file progress.

    Returns:
        Number of (round, client) CKA matrices successfully computed.
    """
    checkpoints_dir = run_dir / "checkpoints"
    metadata_path = checkpoints_dir / "run_metadata.json"

    if not metadata_path.exists():
        print(f"  [SKIP] No run_metadata.json in {run_dir} — not a CKA run")
        return 0

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------
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

    # Apply round filter
    if rounds_override:
        cka_rounds = [r for r in cka_rounds if r in rounds_override]
    if not cka_rounds:
        print(f"  [SKIP] No matching rounds in {run_dir.name}")
        return 0

    print(f"  model={model_name}  dataset={dataset_name}  α={alpha}  seed={seed}")
    print(f"  rounds={cka_rounds}  client_sample={client_sample}  probe_batches={probe_batches}")

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    cka_matrices_dir = run_dir / "cka_matrices"
    cka_matrices_dir.mkdir(exist_ok=True)

    csv_path = run_dir / "cka_metrics.csv"
    # Write header only if the file doesn't exist yet
    if not csv_path.exists():
        with open(csv_path, "w") as f:
            f.write(",".join(_CKA_CSV_HEADERS) + "\n")

    # ------------------------------------------------------------------
    # Dataset + probe loader (built once, shared across all rounds)
    # ------------------------------------------------------------------
    print("  Building testset and probe loader...", end=" ", flush=True)
    try:
        testset = build_testset(dataset_name)
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

    # ------------------------------------------------------------------
    # Layer spec (resolved once from model name)
    # ------------------------------------------------------------------
    layer_spec = get_layer_spec(model_name)   # no model instance needed

    # ------------------------------------------------------------------
    # Process each round
    # ------------------------------------------------------------------
    total_computed = 0
    rng = random.Random(seed)  # deterministic client sampling (matches training)

    for round_idx in sorted(cka_rounds):
        round_dir = checkpoints_dir / f"round_{round_idx:04d}"
        global_ckpt = round_dir / "global.pt"

        if not global_ckpt.exists():
            print(f"  [WARN] Round {round_idx}: global.pt missing — skipping")
            continue

        # Discover available client checkpoints for this round
        client_ckpts = sorted(round_dir.glob("client_*.pt"))
        if not client_ckpts:
            print(f"  [WARN] Round {round_idx}: no client checkpoints — skipping")
            continue

        # Parse client IDs from filenames
        available_clients = []
        for p in client_ckpts:
            try:
                cid = int(p.stem.split("_")[1])
                available_clients.append(cid)
            except (IndexError, ValueError):
                pass

        # Sample clients (matches training behaviour: without replacement,
        # seeded per-round for reproducibility)
        k = min(client_sample, len(available_clients))
        sampled_clients = rng.sample(available_clients, k)

        print(f"  Round {round_idx:3d}: {len(sampled_clients)} clients", end="", flush=True)

        # Load global model once per round
        try:
            global_model = build_model(model_name, dataset_name, pretrained=use_pretrained)
            global_state = torch.load(global_ckpt, map_location="cpu", weights_only=True)
            global_model.load_state_dict(global_state, strict=False)
            global_model.eval()
        except Exception as e:
            print(f" — FAILED to load global model: {e}")
            continue

        all_diagonals: List[np.ndarray] = []

        for client_id in sampled_clients:
            client_ckpt = round_dir / f"client_{client_id:03d}.pt"

            # Skip if already computed
            npz_path = cka_matrices_dir / f"round_{round_idx:04d}_client_{client_id:03d}.npz"
            if skip_existing and npz_path.exists():
                print(".", end="", flush=True)
                continue

            try:
                client_model = build_model(model_name, dataset_name, pretrained=use_pretrained)
                client_state = torch.load(client_ckpt, map_location="cpu", weights_only=True)
                client_model.load_state_dict(client_state, strict=False)
                client_model.eval()
            except Exception as e:
                print(f"\n    [WARN] Client {client_id}: failed to load: {e}")
                continue

            diagonal, matrix, layer_names = compute_cka_for_pair(
                global_model=global_model,
                client_model=client_model,
                layer_spec=layer_spec,
                probe_loader=probe_loader,
                probe_batches=probe_batches,
                device=device,
            )

            if diagonal is None:
                print("x", end="", flush=True)
                continue

            # Write CSV rows
            _append_csv_rows(
                csv_path=csv_path,
                round_idx=round_idx,
                client_id=client_id,
                model_name=model_name,
                dataset_name=dataset_name,
                alpha=alpha,
                diagonal=diagonal,
                layer_names=layer_names,
            )

            # Persist full matrix
            try:
                np.savez_compressed(
                    npz_path,
                    matrix=matrix.astype(np.float32),
                    layer_names=np.array(layer_names, dtype=object),
                )
            except Exception as e:
                print(f"\n    [WARN] Client {client_id}: failed to save .npz: {e}")

            all_diagonals.append(diagonal)
            total_computed += 1
            print(".", end="", flush=True)

            del client_model

        del global_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Per-round summary
        if all_diagonals:
            mean_cka = float(np.mean(np.concatenate(all_diagonals)))
            print(f" | mean CKA={mean_cka:.4f}")
        else:
            print(" | no results")

    return total_computed


# ---------------------------------------------------------------------------
# CSV append helper
# ---------------------------------------------------------------------------

def _append_csv_rows(
    csv_path: Path,
    round_idx: int,
    client_id: int,
    model_name: str,
    dataset_name: str,
    alpha: str,
    diagonal: np.ndarray,
    layer_names: List[str],
) -> None:
    """Append one CSV row per layer to *csv_path*."""
    try:
        with open(csv_path, "a") as f:
            for layer_index, (layer_name, value) in enumerate(zip(layer_names, diagonal)):
                f.write(
                    f"{round_idx},{client_id},{model_name},{dataset_name},"
                    f"{alpha},{layer_index},{layer_name},{value:.6f}\n"
                )
    except Exception as e:
        print(f"\n    [WARN] Failed to write CSV rows: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline CKA computation from CKADriftFedAvg/FedProx checkpoints.",
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
        help="Parent directory of run subdirectories.  All subdirectories are processed.",
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
        help="Torch device for CKA forward passes (default: cpu).  "
             "Use 'cuda' or 'cuda:N' when the GPU is free.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip (round, client) pairs whose .npz file already exists.  "
             "Useful for resuming interrupted runs.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-file progress dots.",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    rounds_override = set(args.rounds) if args.rounds else None

    # Collect run directories
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
            verbose=not args.quiet,
        )
        total += n

    print(f"\nDone — {total} CKA matrix/matrices computed across {len(run_dirs)} run(s).")


if __name__ == "__main__":
    main()
