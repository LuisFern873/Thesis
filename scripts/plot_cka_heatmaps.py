"""
plot_cka_heatmaps.py — offline CKA heatmap generation

Reads the full N×N CKA similarity matrices stored by CKADriftFedAvgServer
(or CKADriftFedProxServer) under ``cka_matrices/`` in each run directory
and renders heatmap PNGs.  Run this after training completes.

Why a separate script?
----------------------
Generating heatmaps during training adds matplotlib overhead to every CKA
round and couples visualisation configuration (colour map, figure size,
annotations) to the training code.  Storing the raw matrices as compressed
NumPy archives instead lets you:

  * regenerate heatmaps with a different colour map or annotation style
    without touching training code or re-running experiments;
  * select specific rounds or clients for closer inspection;
  * batch-render heatmaps across all run directories in one pass.

Input format
------------
Each ``cka_matrices/round_NNNN_client_MMM.npz`` file contains:
  ``matrix``      — float32 array, shape (N, N); CKA similarity values in [0, 1]
  ``layer_names`` — object array of strings, shape (N,)

Outputs
-------
By default PNGs are written to ``<run_dir>/cka_heatmaps/``.  Pass
``--out-dir`` to redirect all outputs to a single directory (useful for
aggregating across runs).

Usage examples
--------------
  # Render all heatmaps for a single run
  python scripts/plot_cka_heatmaps.py --run-dir logs/runs/cifar10_alpha0.1_efficient0_ckadriftfedavg_seed42

  # Render heatmaps for every run under logs/runs/
  python scripts/plot_cka_heatmaps.py --logs-dir logs/runs

  # Only rounds 1, 10, 50; only client 0; custom colour map
  python scripts/plot_cka_heatmaps.py --logs-dir logs/runs --rounds 1 10 50 --clients 0 --cmap viridis

  # Save all outputs to a central figures directory
  python scripts/plot_cka_heatmaps.py --logs-dir logs/runs --out-dir logs/figures/cka_heatmaps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless / cluster use
import matplotlib.pyplot as plt
import numpy as np

FLBENCH_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Default rendering options
# ---------------------------------------------------------------------------
DEFAULT_CMAP = "RdYlGn"          # red = low similarity, green = high similarity
DEFAULT_FIG_SIZE = (8, 6)        # inches
DEFAULT_DPI = 150


# ---------------------------------------------------------------------------
# Core rendering function
# ---------------------------------------------------------------------------

def render_heatmap(
    matrix: np.ndarray,
    layer_names: List[str],
    title: str,
    out_path: Path,
    cmap: str = DEFAULT_CMAP,
    fig_size: tuple = DEFAULT_FIG_SIZE,
    dpi: int = DEFAULT_DPI,
    annotate: bool = True,
) -> None:
    """Render a single CKA similarity heatmap and save to *out_path*.

    Args:
        matrix:      2-D float array of shape (N, N); values in [0, 1].
        layer_names: Layer name strings for axis tick labels (length N).
        title:       Figure title.
        out_path:    Destination file path (PNG).
        cmap:        Matplotlib colour map name.
        fig_size:    Figure size in inches ``(width, height)``.
        dpi:         Output resolution.
        annotate:    If True, write the numeric value of each cell.
                     Recommended only for N ≤ 20.
    """
    n = matrix.shape[0]
    fig, ax = plt.subplots(figsize=fig_size)

    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="CKA similarity")

    # Tick labels — abbreviate long module paths to keep the figure readable
    short_names = [_shorten(name) for name in layer_names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=max(5, 9 - n // 4))
    ax.set_yticklabels(short_names, fontsize=max(5, 9 - n // 4))

    ax.set_xlabel("Client model layers →")
    ax.set_ylabel("Global model layers →")
    ax.set_title(title, fontsize=10, pad=8)

    # Optional cell annotations (skip when N is large to avoid clutter)
    if annotate and n <= 20:
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                # Dark text on light cells, light text on dark cells
                text_color = "black" if val > 0.5 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=max(4, 8 - n // 3), color=text_color)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _shorten(name: str, max_len: int = 22) -> str:
    """Abbreviate a dotted module path for use as a tick label."""
    if len(name) <= max_len:
        return name
    parts = name.split(".")
    # Keep first + last token, insert ellipsis
    if len(parts) <= 2:
        return name[:max_len]
    return f"{parts[0]}.…{parts[-1]}"


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def process_run_dir(
    run_dir: Path,
    out_dir: Optional[Path],
    rounds_filter: Optional[set],
    clients_filter: Optional[set],
    cmap: str,
    fig_size: tuple,
    dpi: int,
    annotate: bool,
    verbose: bool,
) -> int:
    """Process all ``.npz`` matrices in *run_dir*/cka_matrices/.

    Returns the number of heatmaps written.
    """
    matrices_dir = run_dir / "cka_matrices"
    if not matrices_dir.exists():
        if verbose:
            print(f"  [SKIP] No cka_matrices/ in {run_dir.name}")
        return 0

    # Default output: run_dir/cka_heatmaps/
    target_dir = out_dir if out_dir is not None else (run_dir / "cka_heatmaps")

    npz_files = sorted(matrices_dir.glob("round_*.npz"))
    if not npz_files:
        if verbose:
            print(f"  [SKIP] cka_matrices/ is empty in {run_dir.name}")
        return 0

    written = 0
    for npz_path in npz_files:
        # Parse round and client from filename: round_NNNN_client_MMM.npz
        stem = npz_path.stem  # e.g. "round_0001_client_000"
        parts = stem.split("_")
        try:
            round_idx  = int(parts[1])
            client_id  = int(parts[3])
        except (IndexError, ValueError):
            if verbose:
                print(f"  [WARN] Cannot parse filename: {npz_path.name} — skipping")
            continue

        if rounds_filter is not None and round_idx not in rounds_filter:
            continue
        if clients_filter is not None and client_id not in clients_filter:
            continue

        # Load matrix and layer names
        try:
            data = np.load(npz_path, allow_pickle=True)
            matrix: np.ndarray = data["matrix"].astype(np.float64)
            layer_names: List[str] = list(data["layer_names"])
        except Exception as e:
            print(f"  [ERROR] Cannot load {npz_path}: {e}")
            continue

        # Build a descriptive title from the run directory name
        title = _build_title(run_dir.name, round_idx, client_id)

        # Output filename mirrors the input stem
        if out_dir is not None:
            # When redirecting to a central dir, prefix with run name to avoid
            # filename collisions across runs
            png_name = f"{run_dir.name}__{stem}.png"
        else:
            png_name = f"{stem}.png"

        out_path = target_dir / png_name

        render_heatmap(
            matrix=matrix,
            layer_names=layer_names,
            title=title,
            out_path=out_path,
            cmap=cmap,
            fig_size=fig_size,
            dpi=dpi,
            annotate=annotate,
        )
        written += 1
        if verbose:
            print(f"  [OK] {out_path.relative_to(FLBENCH_ROOT)}")

    return written


def _build_title(run_name: str, round_idx: int, client_id: int) -> str:
    """Construct a heatmap title from the run directory name.

    Expected run name format (produced by run_experiments.sh):
      <dataset>_alpha<α>_<model>_<method>_seed<seed>
    """
    # Try to extract key fields for a compact title
    import re
    m = re.match(
        r"(?P<dataset>.+?)_alpha(?P<alpha>[^_]+)_(?P<model>.+?)"
        r"_(?P<method>cka\w+)_seed(?P<seed>\d+)",
        run_name,
    )
    if m:
        return (
            f"CKA | {m.group('model')} | {m.group('dataset')} α={m.group('alpha')} "
            f"| Round {round_idx} | Client {client_id}"
        )
    # Fallback: use run name as-is
    return f"{run_name} | Round {round_idx} | Client {client_id}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CKA heatmap PNGs from stored .npz matrix files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Source selection (mutually exclusive: single run vs. all runs)
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--run-dir",
        type=Path,
        metavar="DIR",
        help="Path to a single run directory (e.g. logs/runs/<name>).",
    )
    src_group.add_argument(
        "--logs-dir",
        type=Path,
        metavar="DIR",
        help="Parent directory of run subdirectories (e.g. logs/runs/).  "
             "All subdirectories are processed.",
    )

    # Output
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Redirect all PNG outputs to this directory instead of writing "
             "cka_heatmaps/ inside each run directory.  Useful for aggregating "
             "all heatmaps in one place.",
    )

    # Filters
    parser.add_argument(
        "--rounds",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Only render heatmaps for these 1-based round indices.  "
             "Renders all rounds when omitted.",
    )
    parser.add_argument(
        "--clients",
        nargs="+",
        type=int,
        default=None,
        metavar="ID",
        help="Only render heatmaps for these client IDs.  "
             "Renders all clients when omitted.",
    )

    # Visual options
    parser.add_argument(
        "--cmap",
        default=DEFAULT_CMAP,
        metavar="NAME",
        help=f"Matplotlib colour map name (default: {DEFAULT_CMAP!r}).  "
             "Examples: viridis, plasma, RdYlBu, coolwarm.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=DEFAULT_FIG_SIZE[0],
        metavar="W",
        help=f"Figure width in inches (default: {DEFAULT_FIG_SIZE[0]}).",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=DEFAULT_FIG_SIZE[1],
        metavar="H",
        help=f"Figure height in inches (default: {DEFAULT_FIG_SIZE[1]}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        metavar="N",
        help=f"Output resolution in DPI (default: {DEFAULT_DPI}).",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Suppress per-cell numeric annotations (always suppressed when N > 20).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-file progress messages.",
    )

    args = parser.parse_args()

    fig_size = (args.fig_width, args.fig_height)
    rounds_filter  = set(args.rounds)  if args.rounds  else None
    clients_filter = set(args.clients) if args.clients else None
    verbose = not args.quiet

    # Collect run directories to process
    if args.run_dir is not None:
        run_dirs = [args.run_dir.resolve()]
    else:
        logs_dir = args.logs_dir.resolve()
        if not logs_dir.exists():
            print(f"[ERROR] --logs-dir does not exist: {logs_dir}")
            sys.exit(1)
        run_dirs = sorted(d for d in logs_dir.iterdir() if d.is_dir())
        if not run_dirs:
            print(f"[ERROR] No subdirectories found in {logs_dir}")
            sys.exit(1)

    out_dir = args.out_dir.resolve() if args.out_dir else None

    total_written = 0
    for run_dir in run_dirs:
        if verbose:
            print(f"Processing: {run_dir.name}")
        n = process_run_dir(
            run_dir=run_dir,
            out_dir=out_dir,
            rounds_filter=rounds_filter,
            clients_filter=clients_filter,
            cmap=args.cmap,
            fig_size=fig_size,
            dpi=args.dpi,
            annotate=not args.no_annotate,
            verbose=verbose,
        )
        total_written += n

    dest = str(out_dir) if out_dir else "cka_heatmaps/ in each run dir"
    print(f"\nDone — {total_written} heatmap(s) written to {dest}")


if __name__ == "__main__":
    main()
