#!/bin/bash

set -euo pipefail

LOG_FILE="shell/cifar10/fedlategram/seed42_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo " FedLateGram Experiments"
echo " Started: $(date)"
echo "========================================"

run_experiment() {
  local desc="$1"
  local run_dir="$2"
  shift 2

  # Si ya existe metrics.csv, omitir el experimento
  if [[ -f "${run_dir}/metrics.csv" ]]; then
    echo ""
    echo "----------------------------------------"
    echo " Skipping: $desc"
    echo " Reason:   ${run_dir}/metrics.csv already exists"
    echo "----------------------------------------"
    return 0
  fi

  echo ""
  echo "----------------------------------------"
  echo " Running: $desc"
  echo " Time:    $(date)"
  echo "----------------------------------------"

  "$@"

  echo " Done:    $desc"
}

# ══════════════════════════════════════════════════════════════════════
# Hyperparameters per architecture (from grampenaltyanalysis.md)
#
# Regime mapping:
#   near-IID  → alpha >= 1.0  (alpha=1000.0, alpha=1.0)
#   non-IID   → alpha <= 0.3  (alpha=0.3, alpha=0.1, alpha=0.03)
#
# ┌────────────┬────────────────┬──────────────┬─────────────────┬──────────┬──────┐
# │ arch       │ tau_cka (IID)  │ lf (IID)     │ tau_cka (0.1)   │ lf (0.1) │ strat│
# ├────────────┼────────────────┼──────────────┼─────────────────┼──────────┼──────┤
# │ res9       │ 0.85 / lf=0.50 │ T_warm=20 λ=0.15 │ 0.85 / lf=0.65 T_warm=15 λ=0.20 │ warm_then_freeze │
# │ vim_tiny   │ 0.85 / lf=0.15 │ T_warm=15 λ=0.10 │ 0.85 / lf=0.45 T_warm=15 λ=0.15 │ warm→freeze/slow │
# │ vig_tiny   │ 0.92 / lf=0.15 │ T_warm=25 λ=0.05 │ 0.87 / lf=0.20 T_warm=20 λ=0.08 │ full_freeze      │
# │ vit_tiny   │ 0.70 / lf=0.60 │ T_warm=8  λ=0.20 │ 0.65 / lf=0.75 T_warm=5  λ=0.25 │ slow_update      │
# └────────────┴────────────────┴──────────────┴─────────────────┴──────────┴──────┘
# ══════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────
# ── res9 ──────────────────────────────────────────
# warm_then_freeze  |  IID: lf=0.50 τ=0.85 T=20 λ=0.15
#                   |  non-IID: lf=0.65 τ=0.85 T=15 λ=0.20
# ──────────────────────────────────────────────────

# ── Alpha 1000.0  (near-IID) ──────────────────────
run_experiment \
  "cifar10 | res9 | alpha=1000.0 | seed=42" \
  "logs/runs/cifar10/fedlategram/res9/alpha1000.0/seed42" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1000.0/seed_42 \
    model.name=res9 \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/res9/alpha1000.0/seed42" \
    fedlategram.tau_cka=0.85 \
    fedlategram.late_fraction=0.50 \
    fedlategram.freeze_strategy=warm_then_freeze \
    fedlategram.T_warm=20 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true

# ── Alpha 1.0  (near-IID) ─────────────────────────
run_experiment \
  "cifar10 | res9 | alpha=1.0 | seed=42" \
  "logs/runs/cifar10/fedlategram/res9/alpha1.0/seed42" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1.0/seed_42 \
    model.name=res9 \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/res9/alpha1.0/seed42" \
    fedlategram.tau_cka=0.85 \
    fedlategram.late_fraction=0.50 \
    fedlategram.freeze_strategy=warm_then_freeze \
    fedlategram.T_warm=20 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true

# ── Alpha 0.3  (non-IID) ──────────────────────────
run_experiment \
  "cifar10 | res9 | alpha=0.3 | seed=42" \
  "logs/runs/cifar10/fedlategram/res9/alpha0.3/seed42" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.3/seed_42 \
    model.name=res9 \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/res9/alpha0.3/seed42" \
    fedlategram.tau_cka=0.85 \
    fedlategram.late_fraction=0.65 \
    fedlategram.freeze_strategy=warm_then_freeze \
    fedlategram.T_warm=15 \
    fedlategram.lam=0.20 \
    fedlategram.lam_adaptive=true

# ── Alpha 0.1  (non-IID) ──────────────────────────
run_experiment \
  "cifar10 | res9 | alpha=0.1 | seed=42" \
  "logs/runs/cifar10/fedlategram/res9/alpha0.1/seed42" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.1/seed_42 \
    model.name=res9 \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/res9/alpha0.1/seed42" \
    fedlategram.tau_cka=0.85 \
    fedlategram.late_fraction=0.65 \
    fedlategram.freeze_strategy=warm_then_freeze \
    fedlategram.T_warm=15 \
    fedlategram.lam=0.20 \
    fedlategram.lam_adaptive=true

# ── Alpha 0.03  (non-IID) ─────────────────────────
run_experiment \
  "cifar10 | res9 | alpha=0.03 | seed=42" \
  "logs/runs/cifar10/fedlategram/res9/alpha0.03/seed42" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.03/seed_42 \
    model.name=res9 \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/res9/alpha0.03/seed42" \
    fedlategram.tau_cka=0.85 \
    fedlategram.late_fraction=0.65 \
    fedlategram.freeze_strategy=warm_then_freeze \
    fedlategram.T_warm=15 \
    fedlategram.lam=0.20 \
    fedlategram.lam_adaptive=true


# ──────────────────────────────────────────────────
# ── vim_tiny ──────────────────────────────────────
# IID: warm_then_freeze  lf=0.15 τ=0.85 T=15 λ=0.10
# non-IID: slow_update   lf=0.45 τ=0.85 T=15 λ=0.15 α_lr=0.05
# ──────────────────────────────────────────────────

# ── Alpha 1000.0  (near-IID) ──────────────────────
run_experiment \
  "cifar10 | vim_tiny | alpha=1000.0 | seed=42" \
  "logs/runs/cifar10/fedlategram/vim_tiny/alpha1000.0/seed42" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1000.0/seed_42 \
    model.name=vim_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vim_tiny/alpha1000.0/seed42" \
    fedlategram.tau_cka=0.85 \
    fedlategram.late_fraction=0.15 \
    fedlategram.freeze_strategy=warm_then_freeze \
    fedlategram.T_warm=15 \
    fedlategram.lam=0.10 \
    fedlategram.lam_adaptive=true

# ── Alpha 1.0  (near-IID) ─────────────────────────
run_experiment \
  "cifar10 | vim_tiny | alpha=1.0 | seed=42" \
  "logs/runs/cifar10/fedlategram/vim_tiny/alpha1.0/seed42" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1.0/seed_42 \
    model.name=vim_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vim_tiny/alpha1.0/seed42" \
    fedlategram.tau_cka=0.85 \
    fedlategram.late_fraction=0.15 \
    fedlategram.freeze_strategy=warm_then_freeze \
    fedlategram.T_warm=15 \
    fedlategram.lam=0.10 \
    fedlategram.lam_adaptive=true

# ── Alpha 0.3  (non-IID) ──────────────────────────
run_experiment \
  "cifar10 | vim_tiny | alpha=0.3 | seed=42" \
  "logs/runs/cifar10/fedlategram/vim_tiny/alpha0.3/seed42" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.3/seed_42 \
    model.name=vim_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vim_tiny/alpha0.3/seed42" \
    fedlategram.tau_cka=0.85 \
    fedlategram.late_fraction=0.45 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.alpha_early_lr=0.05 \
    fedlategram.T_warm=15 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true

# ── Alpha 0.1  (non-IID) ──────────────────────────
run_experiment \
  "cifar10 | vim_tiny | alpha=0.1 | seed=42" \
  "logs/runs/cifar10/fedlategram/vim_tiny/alpha0.1/seed42" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.1/seed_42 \
    model.name=vim_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vim_tiny/alpha0.1/seed42" \
    fedlategram.tau_cka=0.85 \
    fedlategram.late_fraction=0.45 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.alpha_early_lr=0.05 \
    fedlategram.T_warm=15 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true

# ── Alpha 0.03  (non-IID) ─────────────────────────
run_experiment \
  "cifar10 | vim_tiny | alpha=0.03 | seed=42" \
  "logs/runs/cifar10/fedlategram/vim_tiny/alpha0.03/seed42" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.03/seed_42 \
    model.name=vim_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vim_tiny/alpha0.03/seed42" \
    fedlategram.tau_cka=0.85 \
    fedlategram.late_fraction=0.45 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.alpha_early_lr=0.05 \
    fedlategram.T_warm=15 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true


# ──────────────────────────────────────────────────
# ── vig_tiny ──────────────────────────────────────
# full_freeze  |  IID: lf=0.15 τ=0.92 T=25 λ=0.05
#              |  non-IID: lf=0.20 τ=0.87 T=20 λ=0.08
# ──────────────────────────────────────────────────

# ── Alpha 1000.0  (near-IID) ──────────────────────
run_experiment \
  "cifar10 | vig_tiny | alpha=1000.0 | seed=42" \
  "logs/runs/cifar10/fedlategram/vig_tiny/alpha1000.0/seed42" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1000.0/seed_42 \
    model.name=vig_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vig_tiny/alpha1000.0/seed42" \
    fedlategram.tau_cka=0.92 \
    fedlategram.late_fraction=0.15 \
    fedlategram.freeze_strategy=full_freeze \
    fedlategram.T_warm=25 \
    fedlategram.lam=0.05 \
    fedlategram.lam_adaptive=true

# ── Alpha 1.0  (near-IID) ─────────────────────────
run_experiment \
  "cifar10 | vig_tiny | alpha=1.0 | seed=42" \
  "logs/runs/cifar10/fedlategram/vig_tiny/alpha1.0/seed42" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1.0/seed_42 \
    model.name=vig_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vig_tiny/alpha1.0/seed42" \
    fedlategram.tau_cka=0.92 \
    fedlategram.late_fraction=0.15 \
    fedlategram.freeze_strategy=full_freeze \
    fedlategram.T_warm=25 \
    fedlategram.lam=0.05 \
    fedlategram.lam_adaptive=true

# ── Alpha 0.3  (non-IID) ──────────────────────────
run_experiment \
  "cifar10 | vig_tiny | alpha=0.3 | seed=42" \
  "logs/runs/cifar10/fedlategram/vig_tiny/alpha0.3/seed42" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.3/seed_42 \
    model.name=vig_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vig_tiny/alpha0.3/seed42" \
    fedlategram.tau_cka=0.87 \
    fedlategram.late_fraction=0.20 \
    fedlategram.freeze_strategy=full_freeze \
    fedlategram.T_warm=20 \
    fedlategram.lam=0.08 \
    fedlategram.lam_adaptive=true

# ── Alpha 0.1  (non-IID) ──────────────────────────
run_experiment \
  "cifar10 | vig_tiny | alpha=0.1 | seed=42" \
  "logs/runs/cifar10/fedlategram/vig_tiny/alpha0.1/seed42" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.1/seed_42 \
    model.name=vig_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vig_tiny/alpha0.1/seed42" \
    fedlategram.tau_cka=0.87 \
    fedlategram.late_fraction=0.20 \
    fedlategram.freeze_strategy=full_freeze \
    fedlategram.T_warm=20 \
    fedlategram.lam=0.08 \
    fedlategram.lam_adaptive=true

# ── Alpha 0.03  (non-IID) ─────────────────────────
run_experiment \
  "cifar10 | vig_tiny | alpha=0.03 | seed=42" \
  "logs/runs/cifar10/fedlategram/vig_tiny/alpha0.03/seed42" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.03/seed_42 \
    model.name=vig_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vig_tiny/alpha0.03/seed42" \
    fedlategram.tau_cka=0.87 \
    fedlategram.late_fraction=0.20 \
    fedlategram.freeze_strategy=full_freeze \
    fedlategram.T_warm=20 \
    fedlategram.lam=0.08 \
    fedlategram.lam_adaptive=true


# ──────────────────────────────────────────────────
# ── vit_tiny ──────────────────────────────────────
# slow_update  |  IID: lf=0.60 τ=0.70 T=8  λ=0.20 α_lr=0.10
#              |  non-IID: lf=0.75 τ=0.65 T=5 λ=0.25 α_lr=0.05
# NOTE: ViT-Tiny is the case where early-layer stability assumption
#       breaks down — all layers drift even under IID (CKA starts at 0.80).
# ──────────────────────────────────────────────────

# ── Alpha 1000.0  (near-IID) ──────────────────────
run_experiment \
  "cifar10 | vit_tiny | alpha=1000.0 | seed=42" \
  "logs/runs/cifar10/fedlategram/vit_tiny/alpha1000.0/seed42" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1000.0/seed_42 \
    model.name=vit_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vit_tiny/alpha1000.0/seed42" \
    fedlategram.tau_cka=0.70 \
    fedlategram.late_fraction=0.60 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.alpha_early_lr=0.10 \
    fedlategram.T_warm=8 \
    fedlategram.lam=0.20 \
    fedlategram.lam_adaptive=true

# ── Alpha 1.0  (near-IID) ─────────────────────────
run_experiment \
  "cifar10 | vit_tiny | alpha=1.0 | seed=42" \
  "logs/runs/cifar10/fedlategram/vit_tiny/alpha1.0/seed42" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1.0/seed_42 \
    model.name=vit_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vit_tiny/alpha1.0/seed42" \
    fedlategram.tau_cka=0.70 \
    fedlategram.late_fraction=0.60 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.alpha_early_lr=0.10 \
    fedlategram.T_warm=8 \
    fedlategram.lam=0.20 \
    fedlategram.lam_adaptive=true

# ── Alpha 0.3  (non-IID) ──────────────────────────
run_experiment \
  "cifar10 | vit_tiny | alpha=0.3 | seed=42" \
  "logs/runs/cifar10/fedlategram/vit_tiny/alpha0.3/seed42" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.3/seed_42 \
    model.name=vit_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vit_tiny/alpha0.3/seed42" \
    fedlategram.tau_cka=0.65 \
    fedlategram.late_fraction=0.75 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.alpha_early_lr=0.05 \
    fedlategram.T_warm=5 \
    fedlategram.lam=0.25 \
    fedlategram.lam_adaptive=true

# ── Alpha 0.1  (non-IID) ──────────────────────────
run_experiment \
  "cifar10 | vit_tiny | alpha=0.1 | seed=42" \
  "logs/runs/cifar10/fedlategram/vit_tiny/alpha0.1/seed42" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.1/seed_42 \
    model.name=vit_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vit_tiny/alpha0.1/seed42" \
    fedlategram.tau_cka=0.65 \
    fedlategram.late_fraction=0.75 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.alpha_early_lr=0.05 \
    fedlategram.T_warm=5 \
    fedlategram.lam=0.25 \
    fedlategram.lam_adaptive=true

# ── Alpha 0.03  (non-IID) ─────────────────────────
run_experiment \
  "cifar10 | vit_tiny | alpha=0.03 | seed=42" \
  "logs/runs/cifar10/fedlategram/vit_tiny/alpha0.03/seed42" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.03/seed_42 \
    model.name=vit_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vit_tiny/alpha0.03/seed42" \
    fedlategram.tau_cka=0.65 \
    fedlategram.late_fraction=0.75 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.alpha_early_lr=0.05 \
    fedlategram.T_warm=5 \
    fedlategram.lam=0.25 \
    fedlategram.lam_adaptive=true


echo ""
echo "========================================"
echo " All experiments completed!"
echo " Finished: $(date)"
echo " Log saved to: $LOG_FILE"
echo "========================================"
