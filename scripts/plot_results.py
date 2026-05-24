"""
plot_results.py — Phase 8.3–8.7

Generates all figures described in methodology Section 8.
Run this after all experiments complete.

Figures produced (saved to logs/figures/):
  fig1_accuracy_vs_round_{dataset}_{method}.png
      Accuracy vs. round for all 5 models × 4 α values.
      One subplot per α, mean ± std shading across 3 seeds.

  fig2_drift_vs_round_{dataset}_alpha{α}_{method}.png
      Per-layer drift (norm / feature / head) vs. round for all models.
      One subplot per layer group.

  fig3_interference_vs_round_{dataset}_alpha{α}_{method}.png
      Gradient cosine similarity vs. round for all models.
      One subplot per layer group.

  fig4_normalization_ablation_{dataset}_{method}.png
      Bar chart: drift@final and accuracy@final for
      EfficientNet-BN vs. GN vs. LN at each α level.

  fig5_fairness_vs_alpha_{dataset}_{method}.png
      Fairness gap (max-min per-client accuracy) vs. α,
      one line per model.

  table1_comparison_{dataset}_{method}.txt
      Primary comparison table (methodology Section 5.5) as plain text.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --logs-dir logs/runs --out-dir logs/figures
    python scripts/plot_results.py --dataset cifar10
    python scripts/plot_results.py --dataset cifar10 --alpha 0.03 --method driftfedavg
"""

from __future__ import annotations

import argparse
import re
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
ALPHAS   = ["1000.0", "1.0", "0.3", "0.03"]
SEEDS    = [42, 123, 456]

MODEL_LABELS: Dict[str, str] = {
    "efficient0":    "EfficientNet-B0 (BN)",
    "efficient0_gn": "EfficientNet-B0 (GN)",
    "efficient0_ln": "EfficientNet-B0 (LN)",
    "efficient1":    "EfficientNet-B1 (BN)",
    "efficient1_gn": "EfficientNet-B1 (GN)",
    "efficient1_ln": "EfficientNet-B1 (LN)",
    "vit_tiny":      "ViT-Tiny",
    "vim_tiny":      "Vim-Tiny",
}

# Consistent colour palette across all figures
MODEL_COLORS: Dict[str, str] = {
    "efficient0":    "#e41a1c",   # red
    "efficient0_gn": "#ff7f00",   # orange
    "efficient0_ln": "#f0c040",   # yellow
    "efficient1":    "#984ea3",   # purple
    "efficient1_gn": "#a65628",   # brown
    "efficient1_ln": "#f781bf",   # pink
    "vit_tiny":      "#377eb8",   # blue
    "vim_tiny":      "#4daf4a",   # green
}

ALPHA_LABELS: Dict[str, str] = {
    "1000.0": "IID (α=1000)",
    "1.0":    "Low het. (α=1.0)",
    "0.3":    "High het. (α=0.3)",
    "0.03":   "Extreme het. (α=0.03)",
}

LAYER_GROUPS = ["norm", "feature", "head"]
LAYER_LABELS = {"norm": "Norm layers", "feature": "Feature layers", "head": "Head"}

# Convergence thresholds (methodology Section 5.2)
CONVERGENCE_THRESHOLDS: Dict[str, float] = {
    "cifar10": 70.0,
    "brain_tumor": 80.0,
}

RUN_PATTERN = re.compile(
    r"^(?P<dataset>.+)_alpha(?P<alpha>[^_]+)_(?P<model>.+)"
    r"_(?P<method>drift\w+)_seed(?P<seed>\d+)$"
)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _convergence_round(
    acc_series: pd.Series, threshold: float, window: int = 5
) -> int:
    """First round where accuracy stays >= threshold for `window` consecutive rounds."""
    vals = acc_series.tolist()
    for t in range(len(vals) - window + 1):
        if all(v >= threshold for v in vals[t: t + window]):
            return t + 1  # 1-indexed
    return -1


def load_all_runs(logs_dir: Path) -> Dict[Tuple, pd.DataFrame]:
    """
    Scan logs_dir for run subdirectories and load their drift_metrics.csv.
    Returns {(dataset, alpha, model, method, seed): DataFrame}.
    """
    runs: Dict[Tuple, pd.DataFrame] = {}
    if not logs_dir.exists():
        return runs

    for run_dir in sorted(logs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        m = RUN_PATTERN.match(run_dir.name)
        if not m:
            continue
        key = (
            m.group("dataset"), m.group("alpha"),
            m.group("model"),   m.group("method"),
            int(m.group("seed")),
        )
        # Prefer drift_metrics.csv (primary); fall back to metrics.csv
        for fname in ("drift_metrics.csv", "metrics.csv"):
            p = run_dir / fname
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    if not df.empty and "round" in df.columns:
                        runs[key] = df
                        break
                except Exception:
                    pass
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
    Returns None if no data is available for any seed.
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
            ax.fill_between(rounds, mean - std, mean + std, alpha=0.15, color=color)

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

    fname = out_dir / f"fig1_accuracy_vs_round_{dataset}_{method}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fname.name}")


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
    models = list(MODEL_LABELS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, group in zip(axes, LAYER_GROUPS):
        col = f"drift_{group}_mean"
        for model in models:
            result = get_series(runs, dataset, alpha, model, method, col)
            if result is None:
                continue
            rounds, mean, std = result
            color = MODEL_COLORS[model]
            ax.plot(rounds, mean, label=MODEL_LABELS[model], color=color, linewidth=1.8)
            ax.fill_between(rounds, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_title(f"Drift — {LAYER_LABELS[group]}")
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("L2 Drift")
        ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(models), 5),
                   bbox_to_anchor=(0.5, -0.08), frameon=False)
    alpha_label = ALPHA_LABELS.get(alpha, f"α={alpha}")
    fig.suptitle(f"Per-Layer Drift — {dataset}, {alpha_label} ({method})",
                 fontsize=12, y=1.01)
    fig.tight_layout()

    fname = out_dir / f"fig2_drift_vs_round_{dataset}_alpha{alpha}_{method}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fname.name}")


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

    fname = out_dir / f"fig3_interference_vs_round_{dataset}_alpha{alpha}_{method}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fname.name}")


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

    Within each row: normalization effect (BN vs GN vs LN, same capacity).
    Between rows:    capacity effect (B0 vs B1, same normalization type).
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

        for i, model in enumerate(ablation_models):
            accs, drifts = [], []
            for alpha in ALPHAS:
                r_acc   = get_series(runs, dataset, alpha, model, method, "global_acc")
                r_drift = get_series(runs, dataset, alpha, model, method, "drift_norm_mean")
                accs.append(  float(r_acc[1][-1])   if r_acc   is not None else 0.0)
                drifts.append(float(r_drift[1][-1]) if r_drift is not None else 0.0)

            offset = (i - 1) * width
            color  = MODEL_COLORS[model]
            label  = MODEL_LABELS[model]
            ax_acc.bar(x + offset,   accs,   width, label=label, color=color, alpha=0.85)
            ax_drift.bar(x + offset, drifts, width, label=label, color=color, alpha=0.85)

        for ax, ylabel, title in [
            (ax_acc,   "Accuracy@Final (%)",
             f"EfficientNet-{backbone}: Accuracy — Norm Ablation"),
            (ax_drift, "Drift-Norm@Final",
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

    fname = out_dir / f"fig4_normalization_ablation_{dataset}_{method}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fname.name}")


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
    # Sort alphas by numeric value for a meaningful x-axis
    sorted_pairs = sorted(zip([float(a) for a in ALPHAS], ALPHAS))

    fig, ax = plt.subplots(figsize=(7, 5))

    for model in models:
        gaps: List[float] = []
        for _, alpha in sorted_pairs:
            result = get_series(runs, dataset, alpha, model, method, "fairness_gap")
            # Use mean at final round
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

    fname = out_dir / f"fig5_fairness_vs_alpha_{dataset}_{method}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fname.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 — Primary Comparison Table (Section 5.5)
# ─────────────────────────────────────────────────────────────────────────────

def write_comparison_table(
    runs: Dict[Tuple, pd.DataFrame],
    dataset: str,
    method: str,
    out_dir: Path,
) -> None:
    header = (
        f"{'Model':<20} {'α':<8} {'Acc@final±std':>15} {'Conv.Rnd':>10} "
        f"{'Drift-norm':>12} {'Interference':>14} {'Fairness':>10}"
    )
    sep = "-" * len(header)
    rows: List[str] = [
        f"Primary Comparison Table — {dataset} ({method})", sep, header, sep
    ]

    threshold = CONVERGENCE_THRESHOLDS.get(dataset, 70.0)

    for alpha in ALPHAS:
        for model in list(MODEL_LABELS.keys()):
            accs, drifts, interfs, fairs, conv_rounds = [], [], [], [], []
            for seed in SEEDS:
                key = (dataset, alpha, model, method, seed)
                if key not in runs:
                    continue
                df = runs[key]
                if df.empty:
                    continue
                final = df.iloc[-1]
                accs.append(   float(final.get("global_acc",          float("nan"))))
                drifts.append( float(final.get("drift_norm_mean",     float("nan"))))
                interfs.append(float(final.get("interference_feature",float("nan"))))
                fairs.append(  float(final.get("fairness_gap",        float("nan"))))
                if "global_acc" in df.columns:
                    conv_rounds.append(_convergence_round(df["global_acc"], threshold))

            if not accs:
                continue

            acc_mean = float(np.nanmean(accs))
            acc_std  = float(np.nanstd(accs))
            drift_m  = float(np.nanmean(drifts))
            interf_m = float(np.nanmean(interfs))
            fair_m   = float(np.nanmean(fairs))
            conv_m   = int(round(float(np.nanmean(conv_rounds)))) if conv_rounds else -1

            rows.append(
                f"{MODEL_LABELS[model]:<20} {alpha:<8} "
                f"{acc_mean:>6.1f}±{acc_std:<6.1f} "
                f"{conv_m:>8}   "
                f"{drift_m:>10.4f}   "
                f"{interf_m:>12.4f}   "
                f"{fair_m:>8.1f}"
            )
        rows.append("")  # blank line between alpha groups

    rows.append(sep)
    fname = out_dir / f"table1_comparison_{dataset}_{method}.txt"
    fname.write_text("\n".join(rows), encoding="utf-8")
    print(f"  [OK] {fname.name}")


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

    Four panels per figure:
      Row 1: Accuracy vs. round at α=0.03 (extreme) and α=1000 (IID)
      Row 2: Norm-layer drift vs. round at the same two α levels

    This directly answers: does scaling from B0 → B1 (+2.5M params) change
    the drift profile or convergence behaviour under non-IID FL?
    """
    compare_alphas = ["1000.0", "0.03"]   # IID baseline + extreme non-IID
    b0, b1 = "efficient0", "efficient1"
    colors  = {b0: MODEL_COLORS[b0], b1: MODEL_COLORS[b1]}
    labels  = {b0: MODEL_LABELS[b0],  b1: MODEL_LABELS[b1]}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row")

    for col, alpha in enumerate(compare_alphas):
        alpha_label = ALPHA_LABELS.get(alpha, f"α={alpha}")

        # ── Row 0: Accuracy ──────────────────────────────────────────────
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

        # ── Row 1: Norm-layer drift ───────────────────────────────────────
        ax_drift = axes[1][col]
        for model in [b0, b1]:
            result = get_series(runs, dataset, alpha, model, method, "drift_norm_mean")
            if result is None:
                continue
            rounds, mean, std = result
            ax_drift.plot(rounds, mean, label=labels[model],
                          color=colors[model], linewidth=2.0)
            ax_drift.fill_between(rounds, mean - std, mean + std,
                                  alpha=0.15, color=colors[model])
        ax_drift.set_title(f"Norm-Layer Drift — {alpha_label}")
        ax_drift.set_xlabel("Communication Round")
        ax_drift.set_ylabel("L2 Drift")
        ax_drift.grid(True, linestyle="--", alpha=0.4)
        ax_drift.legend(frameon=False)

    axes[0][0].set_ylabel("Global Test Accuracy (%)")
    axes[1][0].set_ylabel("L2 Drift (norm layers)")

    # Parameter count annotation
    b0_params = 4.01   # base params in M (brain_tumor head)
    b1_params = 6.51
    fig.suptitle(
        f"EfficientNet-B0 ({b0_params:.2f}M params) vs. "
        f"B1 ({b1_params:.2f}M params) — {dataset} ({method})",
        fontsize=12,
    )
    fig.tight_layout()

    fname = out_dir / f"fig6_b0_vs_b1_{dataset}_{method}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fname.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all result figures.")
    parser.add_argument("--logs-dir", default=str(FLBENCH_ROOT / "logs" / "runs"),
                        help="Directory containing per-run subdirectories.")
    parser.add_argument("--out-dir",  default=str(FLBENCH_ROOT / "logs" / "figures"),
                        help="Output directory for figures and tables.")
    parser.add_argument("--dataset",  default="all",
                        choices=DATASETS + ["all"])
    parser.add_argument("--alpha",    default="all",
                        help="Alpha value to plot (e.g. 0.03), or 'all'.")
    parser.add_argument("--method",   default="driftfedavg",
                        choices=["driftfedavg", "driftfedprox", "all"])
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading runs from {logs_dir} ...")
    runs = load_all_runs(logs_dir)
    if not runs:
        print("[ERROR] No runs found. Run experiments first.")
        sys.exit(1)
    print(f"Loaded {len(runs)} run(s).\n")

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    methods  = ["driftfedavg", "driftfedprox"] if args.method == "all" else [args.method]
    alphas   = ALPHAS if args.alpha == "all" else [args.alpha]

    for dataset in datasets:
        for method in methods:
            print(f"── {dataset} / {method} " + "─" * 30)

            plot_accuracy_vs_round(runs, dataset, method, out_dir)

            for alpha in alphas:
                plot_drift_vs_round(runs, dataset, alpha, method, out_dir)
                plot_interference_vs_round(runs, dataset, alpha, method, out_dir)

            plot_normalization_ablation(runs, dataset, method, out_dir)
            plot_fairness_vs_alpha(runs, dataset, method, out_dir)
            plot_b0_vs_b1(runs, dataset, method, out_dir)
            write_comparison_table(runs, dataset, method, out_dir)

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
