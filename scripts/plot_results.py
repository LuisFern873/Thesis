"""
plot_results.py — Phase 8.3–8.7

Generates all figures described in methodology Section 8.
Run this after all experiments complete.

Reads data from the nested logs/runs/ hierarchy:
    logs/runs/<dataset>/<method>/<model>/alpha<X>/seed<N>/

Figures are saved to a mirrored nested structure:
    logs/figures/<dataset>/<method>/

Figures produced:
  fig1_accuracy_vs_round_{dataset}_{method}.png
      Accuracy vs. round for all models × α values.
      One subplot per α, mean ± std shading across 3 seeds.

  fig2_drift_vs_round_{dataset}_alpha{α}_{method}.png
      Per-layer drift (norm / feature / head) vs. round for all models.

  fig3_interference_vs_round_{dataset}_alpha{α}_{method}.png
      Gradient cosine similarity vs. round for all models.

  fig4_normalization_ablation_{dataset}_{method}.png
      Bar chart: drift@final and accuracy@final for
      EfficientNet-BN vs. GN vs. LN at each α level.

  fig5_fairness_vs_alpha_{dataset}_{method}.png
      Fairness gap (max-min per-client accuracy) vs. α, one line per model.

  fig6_b0_vs_b1_{dataset}_{method}.png
      EfficientNet-B0 vs B1 side-by-side comparison.

  table1_comparison_{dataset}_{method}.txt
      Primary comparison table (methodology Section 5.5) as plain text.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --logs-dir logs/runs --out-dir logs/figures
    python scripts/plot_results.py --dataset cifar10
    python scripts/plot_results.py --dataset brain_tumor --alpha 0.03 --method fedavg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for cluster use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FLBENCH_ROOT = Path(__file__).parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = ["cifar10", "brain_tumor"]
ALPHAS   = ["1000.0", "1.0", "0.3", "0.1", "0.03"]
SEEDS    = [42, 123, 456]

# Method names as they appear in logs/runs/<dataset>/<method>/
# Primary study runs (ckadrift* → fedavg / fedprox)
# No-CKA preliminary runs get the _nodcka suffix
METHODS_ALL = ["fedavg", "fedprox", "fedavg_nodcka", "fedprox_nodcka", "scaffold_nodcka"]
METHODS_PRIMARY = ["fedavg", "fedprox"]  # default for --method all

MODEL_LABELS: Dict[str, str] = {
    "efficient0":    "EfficientNet-B0",
    "efficient0_gn": "EfficientNet-B0 (GN)",
    "efficient0_ln": "EfficientNet-B0 (LN)",
    "efficient1":    "EfficientNet-B1",
    "efficient1_gn": "EfficientNet-B1 (GN)",
    "efficient1_ln": "EfficientNet-B1 (LN)",
    "vit_tiny":      "ViT-Tiny",
    "vim_tiny":      "Vim-Tiny",
    "vig_tiny":      "ViG-Tiny",
    "res9":          "ResNet-9",
}

# Consistent colour palette across all figures
MODEL_COLORS: Dict[str, str] = {
    "efficient0":    "#e41a1c",   # red
    "efficient0_gn": "#ff7f00",   # orange
    "efficient0_ln": "#f0c040",   # yellow

    "efficient1":    "#CC79A7",   # reddish purple
    "efficient1_gn": "#a65628",   # brown
    "efficient1_ln": "#f781bf",   # pink

    "vit_tiny":      "#0072B2",   # blue
    "vim_tiny":      "#009E73",   # bluish green
    "vig_tiny":      "#E69F00",   # orange
    "res9":          "#D55E00",   # vermillion
}

ALPHA_LABELS: Dict[str, str] = {
    "1000.0": "IID (α=1000)",
    "1.0":    "Low het. (α=1.0)",
    "0.3":    "Mid het. (α=0.3)",
    "0.1":    "High het. (α=0.1)",
    "0.03":   "Extreme het. (α=0.03)",
}

LAYER_GROUPS = ["norm", "feature", "head"]
LAYER_LABELS = {"norm": "Norm layers", "feature": "Feature layers", "head": "Head"}

# Convergence thresholds (methodology Section 5.2)
CONVERGENCE_THRESHOLDS: Dict[str, float] = {
    "cifar10":     70.0,
    "brain_tumor": 80.0,
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — nested hierarchy
# ─────────────────────────────────────────────────────────────────────────────

def load_all_runs(logs_dir: Path) -> Dict[Tuple, pd.DataFrame]:
    """
    Scan the nested logs/runs/ hierarchy and load CSV data for each run.

    Expected layout:
        logs_dir/<dataset>/<method>/<model>/alpha<X>/seed<N>/

    Returns:
        { (dataset, alpha, model, method, seed): DataFrame }

    The DataFrame is loaded from drift_metrics.csv if present, else metrics.csv.
    Only DataFrames with a 'round' column are included.
    """
    runs: Dict[Tuple, pd.DataFrame] = {}
    if not logs_dir.exists():
        print(f"[WARN] logs_dir not found: {logs_dir}", file=sys.stderr)
        return runs

    for dataset_dir in sorted(logs_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name

        for method_dir in sorted(dataset_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method = method_dir.name

            for model_dir in sorted(method_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                model = model_dir.name

                for alpha_dir in sorted(model_dir.iterdir()):
                    if not alpha_dir.is_dir():
                        continue
                    # Expect "alpha<value>" — strip the prefix
                    alpha_raw = alpha_dir.name
                    if not alpha_raw.startswith("alpha"):
                        continue
                    alpha = alpha_raw[len("alpha"):]  # e.g. "0.1", "1000.0"

                    for seed_dir in sorted(alpha_dir.iterdir()):
                        if not seed_dir.is_dir():
                            continue
                        seed_raw = seed_dir.name
                        if not seed_raw.startswith("seed"):
                            continue
                        try:
                            seed = int(seed_raw[len("seed"):])
                        except ValueError:
                            continue

                        # Load CSV: prefer drift_metrics, fall back to metrics
                        for fname in ("drift_metrics.csv", "metrics.csv"):
                            csv_path = seed_dir / fname
                            if csv_path.exists():
                                try:
                                    df = pd.read_csv(csv_path)
                                    if not df.empty and "round" in df.columns:
                                        key = (dataset, alpha, model, method, seed)
                                        runs[key] = df
                                        break
                                except Exception as exc:
                                    print(
                                        f"[WARN] Could not read {csv_path}: {exc}",
                                        file=sys.stderr,
                                    )
    return runs


def get_series(
    runs: Dict[Tuple, pd.DataFrame],
    dataset: str,
    alpha: str,
    model: str,
    method: str,
    col: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return (rounds, mean, std) across seeds for a given column.
    Returns None if no seed has data for this column.
    """
    seed_series: List[pd.Series] = []
    for seed in SEEDS:
        key = (dataset, alpha, model, method, seed)
        if key in runs and col in runs[key].columns:
            seed_series.append(runs[key].set_index("round")[col])

    if not seed_series:
        return None

    combined = pd.concat(seed_series, axis=1)
    rounds = combined.index.values
    mean   = combined.mean(axis=1).values
    std    = combined.std(axis=1).fillna(0).values
    return rounds, mean, std


def _convergence_round(
    acc_series: pd.Series, threshold: float, window: int = 5
) -> int:
    """First round where accuracy stays >= threshold for `window` consecutive rounds."""
    vals = acc_series.tolist()
    for t in range(len(vals) - window + 1):
        if all(v >= threshold for v in vals[t : t + window]):
            return t + 1  # 1-indexed
    return -1


def _method_out_dir(out_dir: Path, dataset: str, method: str) -> Path:
    """Return (and create) the output subdirectory for a dataset/method pair."""
    d = out_dir / dataset / method
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Accuracy vs. Round
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_vs_round(
    runs: Dict[Tuple, pd.DataFrame],
    dataset: str,
    method: str,
    out_dir: Path,
) -> None:
    models  = list(MODEL_LABELS.keys())
    n_alpha = len(ALPHAS)
    fig, axes = plt.subplots(1, n_alpha, figsize=(4.5 * n_alpha, 4.5), sharey=True)
    if n_alpha == 1:
        axes = [axes]

    for ax, alpha in zip(axes, ALPHAS):
        for model in models:
            result = get_series(runs, dataset, alpha, model, method, "global_acc")
            if result is None:
                continue
            rounds, mean, std = result
            color = MODEL_COLORS[model]
            ax.plot(rounds, mean, label=MODEL_LABELS[model], color=color, linewidth=1.8)
            ax.fill_between(rounds, mean - std, mean + std, alpha=0.10, color=color)

        ax.set_title(ALPHA_LABELS.get(alpha, f"α={alpha}"))
        ax.set_xlabel("Communication Round")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Global Test Accuracy (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(models), 5),
                   bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle(f"Accuracy vs. Round — {dataset} ({method})", fontsize=12, y=1.01)
    fig.tight_layout()

    dest = _method_out_dir(out_dir, dataset, method)
    fname = dest / f"fig1_accuracy_vs_round_{dataset}_{method}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fname.relative_to(out_dir)}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Per-Layer Drift vs. Round
# ─────────────────────────────────────────────────────────────────────────────

def plot_drift_vs_round(
    runs: Dict[Tuple, pd.DataFrame],
    dataset: str,
    alpha: str,
    method: str,
    out_dir: Path,
) -> None:
    """Three subplots (one per layer group) showing normalised RMS drift.

    All three axes share the same Y scale so magnitudes are directly comparable
    across layer groups. Falls back to raw L2 drift with a note in the title
    when normalised columns are absent (older CSV files).
    """
    models = list(MODEL_LABELS.keys())

    # Decide which column set to use: prefer normalised, fall back to raw.
    has_norm = any(
        get_series(runs, dataset, alpha, model, method,
                   f"drift_{group}_norm_mean") is not None
        for model in models
        for group in LAYER_GROUPS
    )
    col_suffix = "_norm_mean" if has_norm else "_mean"
    ylabel     = "RMS Drift  ||Δθ||₂ / √N  (per-parameter)" if has_norm else "Raw L2 Drift  ||Δθ||₂"
    title_tag  = "Normalised RMS Drift" if has_norm else "Raw L2 Drift (no norm data)"

    # First pass: collect data for a shared Y limit
    all_upper: List[float] = []
    series_cache: Dict[Tuple[str, str], Optional[Tuple]] = {}
    for group in LAYER_GROUPS:
        col = f"drift_{group}{col_suffix}"
        for model in models:
            result = get_series(runs, dataset, alpha, model, method, col)
            series_cache[(group, model)] = result
            if result is not None:
                rounds, mean, std = result
                all_upper.extend((mean + std).tolist())

    y_max = max(all_upper) * 1.12 if all_upper else 1.0  # 12 % headroom

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, group in zip(axes, LAYER_GROUPS):
        any_data = False
        for model in models:
            result = series_cache[(group, model)]
            if result is None:
                continue
            rounds, mean, std = result
            any_data = True
            color = MODEL_COLORS[model]
            ax.plot(rounds, mean, label=MODEL_LABELS[model],
                    color=color, linewidth=1.8)
            ax.fill_between(rounds, mean - std, mean + std,
                            alpha=0.15, color=color)

        ax.set_title(f"{title_tag}\n{LAYER_LABELS[group]}")
        ax.set_xlabel("Communication Round")
        ax.set_ylim(0, y_max)
        ax.grid(True, linestyle="--", alpha=0.4)

        if not any_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="grey", fontsize=9)

    axes[0].set_ylabel(ylabel)

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(models), 5),
                   bbox_to_anchor=(0.5, -0.08), frameon=False)

    alpha_label = ALPHA_LABELS.get(alpha, f"α={alpha}")
    fig.suptitle(f"Per-Layer Drift — {dataset}, {alpha_label} ({method})",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    dest = _method_out_dir(out_dir, dataset, method)
    fname = dest / f"fig2_drift_vs_round_{dataset}_alpha{alpha}_{method}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fname.relative_to(out_dir)}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Gradient Alignment (Interference) vs. Round
# ─────────────────────────────────────────────────────────────────────────────

def plot_interference_vs_round(
    runs: Dict[Tuple, pd.DataFrame],
    dataset: str,
    alpha: str,
    method: str,
    out_dir: Path,
) -> None:
    models = list(MODEL_LABELS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, group in zip(axes, LAYER_GROUPS):
        col = f"interference_{group}"
        for model in models:
            result = get_series(runs, dataset, alpha, model, method, col)
            if result is None:
                continue
            rounds, mean, std = result
            color = MODEL_COLORS[model]
            ax.plot(rounds, mean, label=MODEL_LABELS[model], color=color, linewidth=1.8)
            ax.fill_between(rounds, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_title(f"Gradient Alignment — {LAYER_LABELS[group]}")
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Mean Pairwise Cosine Similarity")
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(models), 5),
                   bbox_to_anchor=(0.5, -0.08), frameon=False)
    alpha_label = ALPHA_LABELS.get(alpha, f"α={alpha}")
    fig.suptitle(f"Gradient Alignment — {dataset}, {alpha_label} ({method})",
                 fontsize=12, y=1.01)
    fig.tight_layout()

    dest = _method_out_dir(out_dir, dataset, method)
    fname = dest / f"fig3_interference_vs_round_{dataset}_alpha{alpha}_{method}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fname.relative_to(out_dir)}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Normalization Ablation Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_normalization_ablation(
    runs: Dict[Tuple, pd.DataFrame],
    dataset: str,
    method: str,
    out_dir: Path,
) -> None:
    """Bar chart comparing BN / GN / LN for both B0 and B1 at each alpha level.

    Two rows of bar charts:
      Row 1 (B0): EfficientNet-B0 BN vs GN vs LN — accuracy and norm-layer drift
      Row 2 (B1): EfficientNet-B1 BN vs GN vs LN — accuracy and norm-layer drift

    The drift bars show normalised (RMS) drift, falling back to raw L2 for
    older CSV files.
    """
    backbone_groups = [
        ("B0", ["efficient0",    "efficient0_gn", "efficient0_ln"]),
        ("B1", ["efficient1",    "efficient1_gn", "efficient1_ln"]),
    ]
    x     = np.arange(len(ALPHAS))
    width = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for row, (backbone, ablation_models) in enumerate(backbone_groups):
        ax_acc   = axes[row][0]
        ax_drift = axes[row][1]

        norm_col = "drift_norm_norm_mean"
        raw_col  = "drift_norm_mean"

        for i, model in enumerate(ablation_models):
            accs, drifts = [], []
            for alpha in ALPHAS:
                r_acc   = get_series(runs, dataset, alpha, model, method, "global_acc")
                r_drift = get_series(runs, dataset, alpha, model, method, norm_col)
                if r_drift is None:
                    r_drift = get_series(runs, dataset, alpha, model, method, raw_col)

                accs.append(  float(r_acc[1][-1])   if r_acc   is not None else 0.0)
                drifts.append(float(r_drift[1][-1]) if r_drift is not None else 0.0)

            offset = (i - 1) * width
            color  = MODEL_COLORS[model]
            label  = MODEL_LABELS[model]
            ax_acc.bar(x + offset,   accs,   width, label=label, color=color, alpha=0.85)
            ax_drift.bar(x + offset, drifts, width, label=label, color=color, alpha=0.85)

        any_norm = any(
            get_series(runs, dataset, alpha, model, method, norm_col) is not None
            for alpha in ALPHAS
            for model in ablation_models
        )
        drift_ylabel = (
            "Norm-Layer RMS Drift@Final  (||Δθ||₂/√N)"
            if any_norm else
            "Norm-Layer Raw L2 Drift@Final  (||Δθ||₂)"
        )

        for ax, ylabel, title in [
            (ax_acc,   "Accuracy@Final (%)",
             f"EfficientNet-{backbone}: Accuracy — Norm Ablation"),
            (ax_drift, drift_ylabel,
             f"EfficientNet-{backbone}: Norm-Layer Drift — Norm Ablation"),
        ]:
            ax.set_xticks(x)
            ax.set_xticklabels(
                [ALPHA_LABELS.get(a, a) for a in ALPHAS], rotation=15, ha="right"
            )
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(frameon=False)
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(f"Normalization Ablation: B0 vs B1 — {dataset}", fontsize=12)
    fig.tight_layout()

    dest = _method_out_dir(out_dir, dataset, method)
    fname = dest / f"fig4_normalization_ablation_{dataset}_{method}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fname.relative_to(out_dir)}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Fairness Gap vs. Alpha
# ─────────────────────────────────────────────────────────────────────────────

def plot_fairness_vs_alpha(
    runs: Dict[Tuple, pd.DataFrame],
    dataset: str,
    method: str,
    out_dir: Path,
) -> None:
    models = list(MODEL_LABELS.keys())
    sorted_pairs = sorted(zip([float(a) for a in ALPHAS], ALPHAS))

    fig, ax = plt.subplots(figsize=(7, 5))

    for model in models:
        gaps: List[float] = []
        for _, alpha in sorted_pairs:
            result = get_series(runs, dataset, alpha, model, method, "fairness_gap")
            gaps.append(float(result[1][-1]) if result is not None else float("nan"))

        x_vals = [p[0] for p in sorted_pairs]
        ax.plot(x_vals, gaps, marker="o", label=MODEL_LABELS[model],
                color=MODEL_COLORS[model], linewidth=1.8)

    ax.set_xscale("log")
    ax.set_xlabel("Dirichlet α (log scale)")
    ax.set_ylabel("Fairness Gap (max − min client accuracy, %)")
    ax.set_title(f"Fairness Gap vs. Heterogeneity — {dataset} ({method})")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    dest = _method_out_dir(out_dir, dataset, method)
    fname = dest / f"fig5_fairness_vs_alpha_{dataset}_{method}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fname.relative_to(out_dir)}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — EfficientNet-B0 vs B1 side-by-side comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_b0_vs_b1(
    runs: Dict[Tuple, pd.DataFrame],
    dataset: str,
    method: str,
    out_dir: Path,
) -> None:
    """Side-by-side comparison of EfficientNet-B0 (BN) and EfficientNet-B1 (BN).

    Four panels:
      Row 1: Accuracy vs. round at α=0.03 (extreme) and α=1000 (IID)
      Row 2: Norm-layer drift vs. round at the same two α levels
    """
    compare_alphas = ["1000.0", "0.03"]
    b0, b1 = "efficient0", "efficient1"
    colors  = {b0: MODEL_COLORS[b0], b1: MODEL_COLORS[b1]}
    labels  = {b0: MODEL_LABELS[b0],  b1: MODEL_LABELS[b1]}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row")

    for col, alpha in enumerate(compare_alphas):
        alpha_label = ALPHA_LABELS.get(alpha, f"α={alpha}")

        # Row 0: Accuracy
        ax_acc = axes[0][col]
        for model in [b0, b1]:
            result = get_series(runs, dataset, alpha, model, method, "global_acc")
            if result is None:
                continue
            rounds, mean, std = result
            ax_acc.plot(rounds, mean, label=labels[model],
                        color=colors[model], linewidth=2.0)
            ax_acc.fill_between(rounds, mean - std, mean + std,
                                alpha=0.15, color=colors[model])
        ax_acc.set_title(f"Accuracy — {alpha_label}")
        ax_acc.set_xlabel("Communication Round")
        ax_acc.set_ylim(0, 100)
        ax_acc.grid(True, linestyle="--", alpha=0.4)
        ax_acc.legend(frameon=False)

        # Row 1: Norm-layer drift (normalised, fallback to raw)
        ax_drift = axes[1][col]
        drift_col = "drift_norm_norm_mean"
        has_norm = any(
            get_series(runs, dataset, alpha, m, method, drift_col) is not None
            for m in [b0, b1]
        )
        if not has_norm:
            drift_col = "drift_norm_mean"

        drift_ylabel = (
            "RMS Drift (||Δθ||₂/√N)" if has_norm else "Raw L2 Drift (||Δθ||₂)"
        )

        for model in [b0, b1]:
            result = get_series(runs, dataset, alpha, model, method, drift_col)
            if result is None:
                continue
            rounds, mean, std = result
            ax_drift.plot(rounds, mean, label=labels[model],
                          color=colors[model], linewidth=2.0)
            ax_drift.fill_between(rounds, mean - std, mean + std,
                                  alpha=0.15, color=colors[model])
        ax_drift.set_title(f"Norm-Layer Drift — {alpha_label}")
        ax_drift.set_xlabel("Communication Round")
        ax_drift.set_ylabel(drift_ylabel)
        ax_drift.grid(True, linestyle="--", alpha=0.4)
        ax_drift.legend(frameon=False)

    axes[0][0].set_ylabel("Global Test Accuracy (%)")
    axes[1][0].set_ylabel("Norm-Layer Drift  (RMS if available, else L2)")

    b0_params = 4.01
    b1_params = 6.51
    fig.suptitle(
        f"EfficientNet-B0 ({b0_params:.2f}M params) vs. "
        f"B1 ({b1_params:.2f}M params) — {dataset} ({method})",
        fontsize=12,
    )
    fig.tight_layout()

    dest = _method_out_dir(out_dir, dataset, method)
    fname = dest / f"fig6_b0_vs_b1_{dataset}_{method}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fname.relative_to(out_dir)}")


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 — Primary Comparison Table (Section 5.5)
# ─────────────────────────────────────────────────────────────────────────────

def write_comparison_table(
    runs: Dict[Tuple, pd.DataFrame],
    dataset: str,
    method: str,
    out_dir: Path,
) -> None:
    """Primary comparison table (methodology Section 5.5).

    Columns:
      Acc@final±std, Conv.Rnd,
      NormDrift per layer group (norm / feature / head),
      Interference per layer group,
      Fairness gap.
    """
    header = (
        f"{'Model':<20} {'α':<8} {'Acc@final±std':>15} {'Conv.Rnd':>10} "
        f"{'NormDrift-norm':>15} {'NormDrift-feat':>15} {'NormDrift-head':>15} "
        f"{'Interf-norm':>13} {'Interf-feat':>13} {'Interf-head':>13} "
        f"{'Fairness':>10}"
    )
    sep = "-" * len(header)
    rows: List[str] = [
        f"Primary Comparison Table — {dataset} ({method})", sep, header, sep
    ]

    threshold = CONVERGENCE_THRESHOLDS.get(dataset, 70.0)

    def _final_mean(seed_dfs: List[pd.DataFrame], col: str) -> float:
        vals = []
        for df in seed_dfs:
            v = df.iloc[-1].get(col, None)
            if v is not None:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        return float(np.nanmean(vals)) if vals else float("nan")

    def _fmt(v: float, width: int = 13, decimals: int = 4) -> str:
        return f"{v:>{width}.{decimals}f}" if not np.isnan(v) else f"{'n/a':>{width}}"

    for alpha in ALPHAS:
        for model in list(MODEL_LABELS.keys()):
            seed_dfs: List[pd.DataFrame] = []
            conv_rounds: List[int] = []

            for seed in SEEDS:
                key = (dataset, alpha, model, method, seed)
                if key not in runs:
                    continue
                df = runs[key]
                if df.empty:
                    continue
                seed_dfs.append(df)
                if "global_acc" in df.columns:
                    conv_rounds.append(_convergence_round(df["global_acc"], threshold))

            if not seed_dfs:
                continue

            accs     = [float(df.iloc[-1].get("global_acc", float("nan"))) for df in seed_dfs]
            acc_mean = float(np.nanmean(accs))
            acc_std  = float(np.nanstd(accs))
            conv_m   = int(round(float(np.nanmean(conv_rounds)))) if conv_rounds else -1

            nd_norm    = _final_mean(seed_dfs, "drift_norm_norm_mean")
            nd_feature = _final_mean(seed_dfs, "drift_feature_norm_mean")
            nd_head    = _final_mean(seed_dfs, "drift_head_norm_mean")

            interf_norm    = _final_mean(seed_dfs, "interference_norm")
            interf_feature = _final_mean(seed_dfs, "interference_feature")
            interf_head    = _final_mean(seed_dfs, "interference_head")

            fair_m = _final_mean(seed_dfs, "fairness_gap")

            rows.append(
                f"{MODEL_LABELS[model]:<20} {alpha:<8} "
                f"{acc_mean:>6.1f}±{acc_std:<6.1f} "
                f"{conv_m:>8}   "
                f"{_fmt(nd_norm, 13, 4)}   "
                f"{_fmt(nd_feature, 13, 4)}   "
                f"{_fmt(nd_head, 13, 4)}   "
                f"{_fmt(interf_norm, 11, 4)}   "
                f"{_fmt(interf_feature, 11, 4)}   "
                f"{_fmt(interf_head, 11, 4)}   "
                f"{fair_m:>8.1f}"
            )
        rows.append("")  # blank line between alpha groups

    rows.append(sep)
    rows.append("")
    rows.append("Columns:")
    rows.append("  NormDrift-norm  = ||Δθ_norm||₂    / √N_norm    (RMS per parameter — norm layers)")
    rows.append("  NormDrift-feat  = ||Δθ_feature||₂ / √N_feature (RMS per parameter — feature layers)")
    rows.append("  NormDrift-head  = ||Δθ_head||₂    / √N_head    (RMS per parameter — head layers)")
    rows.append("  Interf-norm     = mean pairwise cosine similarity of pseudo-gradients (norm layers)")
    rows.append("  Interf-feat     = mean pairwise cosine similarity of pseudo-gradients (feature layers)")
    rows.append("  Interf-head     = mean pairwise cosine similarity of pseudo-gradients (head layers)")
    rows.append("  Fairness        = max − min per-client accuracy (%)")

    dest = _method_out_dir(out_dir, dataset, method)
    fname = dest / f"table1_comparison_{dataset}_{method}.txt"
    fname.write_text("\n".join(rows), encoding="utf-8")
    print(f"  [OK] {fname.relative_to(out_dir)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all result figures.")
    parser.add_argument(
        "--logs-dir", default=str(FLBENCH_ROOT / "logs" / "runs"),
        help="Root of the nested logs/runs/ hierarchy.",
    )
    parser.add_argument(
        "--out-dir", default=str(FLBENCH_ROOT / "logs" / "figures"),
        help="Root output directory. Figures are placed in <out-dir>/<dataset>/<method>/.",
    )
    parser.add_argument(
        "--dataset", default="all",
        choices=DATASETS + ["all"],
    )
    parser.add_argument(
        "--alpha", default="all",
        help="Alpha value to plot (e.g. 0.03), or 'all'.",
    )
    parser.add_argument(
        "--method", default="all",
        choices=METHODS_ALL + ["all"],
        help=(
            "Method folder name as it appears in logs/runs/<dataset>/<method>/. "
            "Primary runs: fedavg, fedprox. "
            "Preliminary (no CKA): fedavg_nodcka, fedprox_nodcka, scaffold_nodcka. "
            "Use 'all' to generate figures for every method that has data."
        ),
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading runs from {logs_dir} ...")
    runs = load_all_runs(logs_dir)
    if not runs:
        print("[ERROR] No runs found. Check that experiments have been run and "
              "that logs_dir points to the correct location.")
        sys.exit(1)

    # Report what was found
    found_keys = set((d, m) for d, _a, _mo, m, _s in runs)
    print(f"Loaded {len(runs)} run(s) across {len(found_keys)} (dataset, method) pairs:")
    for d, m in sorted(found_keys):
        count = sum(1 for k in runs if k[0] == d and k[3] == m)
        print(f"  {d}/{m}: {count} seed-run(s)")
    print()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    alphas   = ALPHAS   if args.alpha   == "all" else [args.alpha]

    # When --method all, iterate only methods that actually have data
    if args.method == "all":
        available_methods = sorted({m for _d, _a, _mo, m, _s in runs})
    else:
        available_methods = [args.method]

    for dataset in datasets:
        for method in available_methods:
            # Check if there is any data for this (dataset, method) pair
            pair_runs = {k: v for k, v in runs.items() if k[0] == dataset and k[3] == method}
            if not pair_runs:
                continue

            print(f"-- {dataset} / {method} " + "-" * 30)

            plot_accuracy_vs_round(runs, dataset, method, out_dir)

            for alpha in alphas:
                plot_drift_vs_round(runs, dataset, alpha, method, out_dir)
                plot_interference_vs_round(runs, dataset, alpha, method, out_dir)

            plot_normalization_ablation(runs, dataset, method, out_dir)
            plot_fairness_vs_alpha(runs, dataset, method, out_dir)
            plot_b0_vs_b1(runs, dataset, method, out_dir)
            write_comparison_table(runs, dataset, method, out_dir)

    print(f"\nAll outputs saved under {out_dir}/")
    print("Structure: <out-dir>/<dataset>/<method>/<figure_file>")


if __name__ == "__main__":
    main()
