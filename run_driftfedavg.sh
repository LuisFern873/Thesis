#!/bin/bash

# ============================================================
# DriftFedAvg Experiment Runner
# Runs all experiments sequentially on the remote cluster
# ============================================================

set -euo pipefail

LOG_FILE="run_driftfedavg_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo " DriftFedAvg Experiments"
echo " Started: $(date)"
echo "========================================"

run_experiment() {
  local desc="$1"
  shift
  echo ""
  echo "----------------------------------------"
  echo " Running: $desc"
  echo " Time:    $(date)"
  echo "----------------------------------------"
  "$@"
  echo " Done:    $desc"
}

# ── Alpha 1000.0 ──────────────────────────────────────────
run_experiment "brain_tumor | alpha=1000.0 | vit_tiny" \
  python main.py --config-name driftfedavg method=driftfedavg \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_1000.0/seed_42 \
    model.name=vit_tiny \
    common.seed=42 common.global_epoch=40 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha1000.0_vit_tiny_driftfedavg_seed42"

# ── Alpha 1.0 ─────────────────────────────────────────────
run_experiment "brain_tumor | alpha=1.0 | efficient0" \
  python main.py --config-name driftfedavg method=driftfedavg \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_1.0/seed_42 \
    model.name=efficient0 \
    common.seed=42 common.global_epoch=40 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha1.0_efficient0_driftfedavg_seed42"


run_experiment "brain_tumor | alpha=1.0 | vit_tiny" \
  python main.py --config-name driftfedavg method=driftfedavg \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_1.0/seed_42 \
    model.name=vit_tiny \
    common.seed=42 common.global_epoch=40 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha1.0_vit_tiny_driftfedavg_seed42"

# ── Alpha 0.3 ─────────────────────────────────────────────
run_experiment "brain_tumor | alpha=0.3 | efficient0" \
  python main.py --config-name driftfedavg method=driftfedavg \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_0.3/seed_42 \
    model.name=efficient0 \
    common.seed=42 common.global_epoch=40 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha0.3_efficient0_driftfedavg_seed42"

run_experiment "brain_tumor | alpha=0.3 | vit_tiny" \
  python main.py --config-name driftfedavg method=driftfedavg \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_0.3/seed_42 \
    model.name=vit_tiny \
    common.seed=42 common.global_epoch=40 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha0.3_vit_tiny_driftfedavg_seed42"

# ── Alpha 0.03 ────────────────────────────────────────────
run_experiment "brain_tumor | alpha=0.03 | efficient0" \
  python main.py --config-name driftfedavg method=driftfedavg \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_0.03/seed_42 \
    model.name=efficient0 \
    common.seed=42 common.global_epoch=40 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha0.03_efficient0_driftfedavg_seed42"

run_experiment "brain_tumor | alpha=0.03 | vit_tiny" \
  python main.py --config-name driftfedavg method=driftfedavg \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_0.03/seed_42 \
    model.name=vit_tiny \
    common.seed=42 common.global_epoch=40 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha0.03_vit_tiny_driftfedavg_seed42"

echo ""
echo "========================================"
echo " All experiments completed!"
echo " Finished: $(date)"
echo " Log saved to: $LOG_FILE"
echo "========================================"
