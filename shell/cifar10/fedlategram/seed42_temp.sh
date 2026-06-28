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

# ──────────────────────────────────────────────────
# ── vim_tiny ──────────────────────────────────────
# IID: warm_then_freeze  lf=0.15 τ=0.85 T=15 λ=0.10
# non-IID: slow_update   lf=0.45 τ=0.85 T=15 λ=0.15 α_lr=0.05
# ──────────────────────────────────────────────────

# ── Alpha 1000.0  (near-IID) ──────────────────────
# run_experiment \
#   "cifar10 | vim_tiny | alpha=1000.0 | seed=42" \
#   "logs/runs/cifar10/fedlategram/vim_tiny/alpha1000.0/seed42" \
#   python main.py --config-name fedlategram \
#     dataset.name=cifar10 \
#     dataset.partition_dir=partitions/alpha_1000.0/seed_42 \
#     model.name=vim_tiny \
#     common.seed=42 \
#     common.global_epoch=50 \
#     mode=serial \
#     "hydra.run.dir=logs/runs/cifar10/fedlategram/vim_tiny/alpha1000.0/seed42" \
#     fedlategram.tau_cka=0.85 \
#     fedlategram.late_fraction=0.15 \
#     fedlategram.freeze_strategy=warm_then_freeze \
#     fedlategram.T_warm=15 \
#     fedlategram.lam=0.10 \
#     fedlategram.lam_adaptive=true

# ── Alpha 1.0  (near-IID) ─────────────────────────
# run_experiment \
#   "cifar10 | vim_tiny | alpha=1.0 | seed=42" \
#   "logs/runs/cifar10/fedlategram/vim_tiny/alpha1.0/seed42" \
#   python main.py --config-name fedlategram \
#     dataset.name=cifar10 \
#     dataset.partition_dir=partitions/alpha_1.0/seed_42 \
#     model.name=vim_tiny \
#     common.seed=42 \
#     common.global_epoch=50 \
#     mode=serial \
#     "hydra.run.dir=logs/runs/cifar10/fedlategram/vim_tiny/alpha1.0/seed42" \
#     fedlategram.tau_cka=0.85 \
#     fedlategram.late_fraction=0.15 \
#     fedlategram.freeze_strategy=warm_then_freeze \
#     fedlategram.T_warm=15 \
#     fedlategram.lam=0.10 \
#     fedlategram.lam_adaptive=true

# ── Alpha 0.3  (non-IID) ──────────────────────────

# freeze_strategy=slow_update
# freq_early=2
run_experiment \
  "cifar10 | vim_tiny | alpha=0.3 | seed=42" \
  "logs/runs/cifar10/fedlategram/vim_tiny/alpha0.3/seed42_fourth" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.3/seed_42 \
    model.name=vim_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vim_tiny/alpha0.3/seed42_fourth" \
    fedlategram.tau_cka=0.85 \
    fedlategram.late_fraction=0.45 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.freq_early=2 \
    fedlategram.alpha_early_lr=0.05 \
    fedlategram.T_warm=15 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true

# freeze_strategy=slow_update
# freq_early=3
run_experiment \
  "cifar10 | vim_tiny | alpha=0.3 | seed=42" \
  "logs/runs/cifar10/fedlategram/vim_tiny/alpha0.3/seed42_fifth" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.3/seed_42 \
    model.name=vim_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vim_tiny/alpha0.3/seed42_fifth" \
    fedlategram.tau_cka=0.85 \
    fedlategram.late_fraction=0.45 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.freq_early=3 \
    fedlategram.alpha_early_lr=0.05 \
    fedlategram.T_warm=15 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true

# freeze_strategy=slow_update
# freq_early=1
run_experiment \
  "cifar10 | vim_tiny | alpha=0.3 | seed=42" \
  "logs/runs/cifar10/fedlategram/vim_tiny/alpha0.3/seed42_sixth" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.3/seed_42 \
    model.name=vim_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vim_tiny/alpha0.3/seed42_sixth" \
    fedlategram.tau_cka=0.85 \
    fedlategram.late_fraction=0.45 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.freq_early=1 \
    fedlategram.alpha_early_lr=0.05 \
    fedlategram.T_warm=15 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true

# ── Alpha 0.1  (non-IID) ──────────────────────────

# freeze_strategy=slow_update
# freq_early=2
run_experiment \
  "cifar10 | vim_tiny | alpha=0.1 | seed=42" \
  "logs/runs/cifar10/fedlategram/vim_tiny/alpha0.1/seed42_fourth" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.1/seed_42 \
    model.name=vim_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vim_tiny/alpha0.1/seed42_fourth" \
    fedlategram.tau_cka=0.85 \
    fedlategram.late_fraction=0.45 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.freq_early=2 \
    fedlategram.alpha_early_lr=0.05 \
    fedlategram.T_warm=15 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true

# freeze_strategy=slow_update
# freq_early=3

run_experiment \
  "cifar10 | vim_tiny | alpha=0.1 | seed=42" \
  "logs/runs/cifar10/fedlategram/vim_tiny/alpha0.1/seed42_fifth" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.1/seed_42 \
    model.name=vim_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vim_tiny/alpha0.1/seed42_fifth" \
    fedlategram.tau_cka=0.85 \
    fedlategram.late_fraction=0.45 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.freq_early=3 \
    fedlategram.alpha_early_lr=0.05 \
    fedlategram.T_warm=15 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true

# ── Alpha 0.03  (non-IID) ─────────────────────────
# run_experiment \
#   "cifar10 | vim_tiny | alpha=0.03 | seed=42" \
#   "logs/runs/cifar10/fedlategram/vim_tiny/alpha0.03/seed42" \
#   python main.py --config-name fedlategram \
#     dataset.name=cifar10 \
#     dataset.partition_dir=partitions/alpha_0.03/seed_42 \
#     model.name=vim_tiny \
#     common.seed=42 \
#     common.global_epoch=50 \
#     mode=serial \
#     "hydra.run.dir=logs/runs/cifar10/fedlategram/vim_tiny/alpha0.03/seed42" \
#     fedlategram.tau_cka=0.85 \
#     fedlategram.late_fraction=0.80 \
#     fedlategram.freeze_strategy=slow_update \
#     fedlategram.alpha_early_lr=0.05 \
#     fedlategram.T_warm=15 \
#     fedlategram.lam=0.15 \
#     fedlategram.lam_adaptive=true


# ──────────────────────────────────────────────────
# ── res9 ──────────────────────────────────────
# ──────────────────────────────────────────────────

run_experiment \
  "cifar10 | res9 | alpha=0.3 | seed=42" \
  "logs/runs/cifar10/fedlategram/res9/alpha0.3/seed42_second" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.3/seed_42 \
    model.name=res9 \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/res9/alpha0.3/seed42_second" \
    fedlategram.tau_cka=0.85 \
    fedlategram.late_fraction=0.45 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.freq_early=1 \
    fedlategram.alpha_early_lr=0.05 \
    fedlategram.T_warm=20 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true

echo ""
echo "========================================"
echo " All experiments completed!"
echo " Finished: $(date)"
echo " Log saved to: $LOG_FILE"
echo "========================================"
