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


# vim_tiny


run_experiment \
  "cifar10 | vim_tiny | alpha=0.1 | seed=123" \
  "logs/runs/cifar10/fedlategram/vim_tiny/alpha0.1/seed123_second" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1000.0/seed_42 \
    model.name=vim_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vim_tiny/alpha1000.0/seed42" \
    fedlategram.late_fraction=0.45 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.freq_early=1 \
    fedlategram.alpha_early_lr=1 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true \
    fedlategram.mu_clf=0.01

  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1000.0/seed_123 \
    model.name=vim_tiny \
    common.seed=123 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vim_tiny/alpha1000.0/seed123" \
    fedlategram.late_fraction=0.45 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.freq_early=1 \
    fedlategram.alpha_early_lr=1 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true \
    fedlategram.mu_clf=0.01


  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1.0/seed_42 \
    model.name=vim_tiny \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vim_tiny/alpha1.0/seed42" \
    fedlategram.late_fraction=0.45 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.freq_early=1 \
    fedlategram.alpha_early_lr=1 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true \
    fedlategram.mu_clf=0.01


  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1.0/seed_123 \
    model.name=vim_tiny \
    common.seed=123 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vim_tiny/alpha1.0/seed123" \
    fedlategram.late_fraction=0.45 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.freq_early=1 \
    fedlategram.alpha_early_lr=1 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true \
    fedlategram.mu_clf=0.01


  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1.0/seed_42 \
    model.name=res9 \
    common.seed=42 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/res9/alpha1.0/seed42" \
    fedlategram.late_fraction=0.45 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.freq_early=1 \
    fedlategram.alpha_early_lr=1 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true \
    fedlategram.mu_clf=0.01


# ── Alpha 0.1  (non-IID) ──────────────────────────

run_experiment \
  "cifar10 | vim_tiny | alpha=0.1 | seed=123" \
  "logs/runs/cifar10/fedlategram/vim_tiny/alpha0.1/seed123_second" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.1/seed_123 \
    model.name=vim_tiny \
    common.seed=123 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vim_tiny/alpha0.1/seed123_second" \
    fedlategram.late_fraction=0.45 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.freq_early=1 \
    fedlategram.alpha_early_lr=1 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true \
    fedlategram.mu_clf=0.1

# ── Alpha 0.03  (non-IID) ──────────────────────────

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
    fedlategram.late_fraction=0.45 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.freq_early=1 \
    fedlategram.alpha_early_lr=1 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true \
    fedlategram.mu_clf=0.1

run_experiment \
  "cifar10 | vim_tiny | alpha=0.03 | seed=123" \
  "logs/runs/cifar10/fedlategram/vim_tiny/alpha0.03/seed123" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.03/seed_123 \
    model.name=vim_tiny \
    common.seed=123 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vim_tiny/alpha0.03/seed123" \
    fedlategram.late_fraction=0.45 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.freq_early=1 \
    fedlategram.alpha_early_lr=1 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true \
    fedlategram.mu_clf=0.1

# vig_tiny



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
    fedlategram.late_fraction=0.1 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.freq_early=1 \
    fedlategram.alpha_early_lr=1 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true \
    fedlategram.mu_clf=0.1

run_experiment \
  "cifar10 | vig_tiny | alpha=0.3 | seed=123" \
  "logs/runs/cifar10/fedlategram/vig_tiny/alpha0.3/seed123" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.3/seed_123 \
    model.name=vig_tiny \
    common.seed=123 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vig_tiny/alpha0.3/seed123" \
    fedlategram.late_fraction=0.1 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.freq_early=1 \
    fedlategram.alpha_early_lr=1 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true \
    fedlategram.mu_clf=0.1


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
    fedlategram.late_fraction=0.1 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.freq_early=1 \
    fedlategram.alpha_early_lr=1 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true \
    fedlategram.mu_clf=0.1

run_experiment \
  "cifar10 | vig_tiny | alpha=0.1 | seed=123" \
  "logs/runs/cifar10/fedlategram/vig_tiny/alpha0.1/seed123" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.1/seed_123 \
    model.name=vig_tiny \
    common.seed=123 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vig_tiny/alpha0.1/seed123" \
    fedlategram.late_fraction=0.1 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.freq_early=1 \
    fedlategram.alpha_early_lr=1 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true \
    fedlategram.mu_clf=0.1

# ── Alpha 0.03  (non-IID) ──────────────────────────

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
    fedlategram.late_fraction=0.1 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.freq_early=1 \
    fedlategram.alpha_early_lr=1 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true \
    fedlategram.mu_clf=0.1

run_experiment \
  "cifar10 | vig_tiny | alpha=0.03 | seed=123" \
  "logs/runs/cifar10/fedlategram/vig_tiny/alpha0.03/seed123" \
  python main.py --config-name fedlategram \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.03/seed_123 \
    model.name=vig_tiny \
    common.seed=123 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedlategram/vig_tiny/alpha0.03/seed123" \
    fedlategram.late_fraction=0.1 \
    fedlategram.freeze_strategy=slow_update \
    fedlategram.freq_early=1 \
    fedlategram.alpha_early_lr=1 \
    fedlategram.lam=0.15 \
    fedlategram.lam_adaptive=true \
    fedlategram.mu_clf=0.1

echo ""
echo "========================================"
echo " All experiments completed!"
echo " Finished: $(date)"
echo " Log saved to: $LOG_FILE"
echo "========================================"
