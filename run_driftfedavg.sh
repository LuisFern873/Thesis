#!/bin/bash

# ============================================================
# driftfedprox Experiment Runner
# Runs all experiments sequentially on the remote cluster
# ============================================================

set -euo pipefail

LOG_FILE="run_driftfedprox_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo " driftfedprox Experiments"
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

run_experiment "brain_tumor | alpha=1000.0 | vig_tiny" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_1000.0/seed_42 \
    model.name=vig_tiny \
    common.seed=42 common.global_epoch=50 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha1000.0_vig_tiny_ckadriftfedavg_seed42"



# ── vit_tiny ──────────────────────────────────────────

# ── Alpha 1000.0 ──────────────────────────────────────────
run_experiment "brain_tumor | alpha=1000.0 | vit_tiny" \
  python main.py --config-name driftfedprox method=driftfedprox \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_1000.0/seed_42 \
    model.name=vit_tiny \
    common.seed=42 common.global_epoch=50 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha1000.0_vit_tiny_driftfedprox_seed42"

# ── Alpha 1.0 ─────────────────────────────────────────────
run_experiment "brain_tumor | alpha=1.0 | vit_tiny" \
  python main.py --config-name driftfedprox method=driftfedprox \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_1.0/seed_42 \
    model.name=vit_tiny \
    common.seed=42 common.global_epoch=50 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha1.0_vit_tiny_driftfedprox_seed42"

# ── Alpha 0.3 ─────────────────────────────────────────────
run_experiment "brain_tumor | alpha=0.3 | vit_tiny" \
  python main.py --config-name driftfedprox method=driftfedprox \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_0.3/seed_42 \
    model.name=vit_tiny \
    common.seed=42 common.global_epoch=50 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha0.3_vit_tiny_driftfedprox_seed42"

# ── Alpha 0.03 ────────────────────────────────────────────
run_experiment "brain_tumor | alpha=0.03 | vit_tiny" \
  python main.py --config-name driftfedprox method=driftfedprox \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_0.03/seed_42 \
    model.name=vit_tiny \
    common.seed=42 common.global_epoch=50 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha0.03_vit_tiny_driftfedprox_seed42"

# ── efficient1 ──────────────────────────────────────────

# ── Alpha 1000.0 ──────────────────────────────────────────
run_experiment "brain_tumor | alpha=1000.0 | efficient1" \
  python main.py --config-name driftfedprox method=driftfedprox \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_1000.0/seed_42 \
    model.name=efficient1 \
    common.seed=42 common.global_epoch=50 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha1000.0_efficient1_driftfedprox_seed42"

# ── Alpha 1.0 ─────────────────────────────────────────────
run_experiment "brain_tumor | alpha=1.0 | efficient1" \
  python main.py --config-name driftfedprox method=driftfedprox \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_1.0/seed_42 \
    model.name=efficient1 \
    common.seed=42 common.global_epoch=50 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha1.0_efficient1_driftfedprox_seed42"

# ── Alpha 0.3 ─────────────────────────────────────────────
run_experiment "brain_tumor | alpha=0.3 | efficient1" \
  python main.py --config-name driftfedprox method=driftfedprox \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_0.3/seed_42 \
    model.name=efficient1 \
    common.seed=42 common.global_epoch=50 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha0.3_efficient1_driftfedprox_seed42"

# ── Alpha 0.03 ────────────────────────────────────────────
run_experiment "brain_tumor | alpha=0.03 | efficient1" \
  python main.py --config-name driftfedprox method=driftfedprox \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_0.03/seed_42 \
    model.name=efficient1 \
    common.seed=42 common.global_epoch=50 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha0.03_efficient1_driftfedprox_seed42"



  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_0.1/seed_42 \
    model.name=res9 \
    common.seed=42 common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/brain_tumor_alpha0.1_res9_ckadriftfedavg_seed42"

echo ""
echo "========================================"
echo " All experiments completed!"
echo " Finished: $(date)"
echo " Log saved to: $LOG_FILE"
echo "========================================"