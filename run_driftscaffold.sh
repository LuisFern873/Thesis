#!/bin/bash

# ============================================================
# driftscaffold Experiment Runner
# Runs all experiments sequentially on the remote cluster
# ============================================================

set -euo pipefail

LOG_FILE="run_driftscaffold_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo " driftscaffold Experiments"
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


# ── vim_tiny ──────────────────────────────────────────

# ── Alpha 1000.0 ──────────────────────────────────────────
# run_experiment "brain_tumor | alpha=1000.0 | vim_tiny" \
#   python main.py --config-name driftscaffold method=driftscaffold \
#     dataset.name=brain_tumor \
#     dataset.partition_dir=partitions/alpha_1000.0/seed_42 \
#     model.name=vim_tiny \
#     common.seed=42 common.global_epoch=40 \
#     "hydra.run.dir=logs/runs/brain_tumor_alpha1000.0_vim_tiny_driftscaffold_seed42"

# ── Alpha 1.0 ─────────────────────────────────────────────
# run_experiment "brain_tumor | alpha=1.0 | vim_tiny" \
#   python main.py --config-name driftscaffold method=driftscaffold \
#     dataset.name=brain_tumor \
#     dataset.partition_dir=partitions/alpha_1.0/seed_42 \
#     model.name=vim_tiny \
#     common.seed=42 common.global_epoch=40 \
#     "hydra.run.dir=logs/runs/brain_tumor_alpha1.0_vim_tiny_driftscaffold_seed42"

# ── Alpha 0.3 ─────────────────────────────────────────────
# run_experiment "brain_tumor | alpha=0.3 | vim_tiny" \
#   python main.py --config-name driftscaffold method=driftscaffold \
#     dataset.name=brain_tumor \
#     dataset.partition_dir=partitions/alpha_0.3/seed_42 \
#     model.name=vim_tiny \
#     common.seed=42 common.global_epoch=40 \
#     "hydra.run.dir=logs/runs/brain_tumor_alpha0.3_vim_tiny_driftscaffold_seed42"

# ── Alpha 0.03 ────────────────────────────────────────────
# run_experiment "brain_tumor | alpha=0.03 | vim_tiny" \
#   python main.py --config-name driftscaffold method=driftscaffold \
#     dataset.name=brain_tumor \
#     dataset.partition_dir=partitions/alpha_0.03/seed_42 \
#     model.name=vim_tiny \
#     common.seed=42 common.global_epoch=40 \
#     "hydra.run.dir=logs/runs/brain_tumor_alpha0.03_vim_tiny_driftscaffold_seed42"



# ── vit_tiny ──────────────────────────────────────────

# ── Alpha 1000.0 ──────────────────────────────────────────
run_experiment "brain_tumor | alpha=1000.0 | vit_tiny" \
  python main.py --config-name driftscaffold method=driftscaffold \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_1000.0/seed_42 \
    model.name=vit_tiny \
    common.seed=42 common.global_epoch=40 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha1000.0_vit_tiny_driftscaffold_seed42"

# ── Alpha 1.0 ─────────────────────────────────────────────
run_experiment "brain_tumor | alpha=1.0 | vit_tiny" \
  python main.py --config-name driftscaffold method=driftscaffold \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_1.0/seed_42 \
    model.name=vit_tiny \
    common.seed=42 common.global_epoch=40 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha1.0_vit_tiny_driftscaffold_seed42"

# ── Alpha 0.3 ─────────────────────────────────────────────
run_experiment "brain_tumor | alpha=0.3 | vit_tiny" \
  python main.py --config-name driftscaffold method=driftscaffold \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_0.3/seed_42 \
    model.name=vit_tiny \
    common.seed=42 common.global_epoch=40 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha0.3_vit_tiny_driftscaffold_seed42"

# ── Alpha 0.03 ────────────────────────────────────────────
# run_experiment "brain_tumor | alpha=0.03 | vit_tiny" \
#   python main.py --config-name driftscaffold method=driftscaffold \
#     dataset.name=brain_tumor \
#     dataset.partition_dir=partitions/alpha_0.03/seed_42 \
#     model.name=vit_tiny \
#     common.seed=42 common.global_epoch=40 \
#     "hydra.run.dir=logs/runs/brain_tumor_alpha0.03_vit_tiny_driftscaffold_seed42"

# ── efficient1 ──────────────────────────────────────────

# ── Alpha 1000.0 ──────────────────────────────────────────
run_experiment "brain_tumor | alpha=1000.0 | efficient1" \
  python main.py --config-name driftscaffold method=driftscaffold \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_1000.0/seed_42 \
    model.name=efficient1 \
    common.seed=42 common.global_epoch=40 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha1000.0_efficient1_driftscaffold_seed42"

# ── Alpha 1.0 ─────────────────────────────────────────────
run_experiment "brain_tumor | alpha=1.0 | efficient1" \
  python main.py --config-name driftscaffold method=driftscaffold \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_1.0/seed_42 \
    model.name=efficient1 \
    common.seed=42 common.global_epoch=40 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha1.0_efficient1_driftscaffold_seed42"

# ── Alpha 0.3 ─────────────────────────────────────────────
run_experiment "brain_tumor | alpha=0.3 | efficient1" \
  python main.py --config-name driftscaffold method=driftscaffold \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_0.3/seed_42 \
    model.name=efficient1 \
    common.seed=42 common.global_epoch=40 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha0.3_efficient1_driftscaffold_seed42"

# ── Alpha 0.03 ────────────────────────────────────────────
# run_experiment "brain_tumor | alpha=0.03 | efficient1" \
#   python main.py --config-name driftscaffold method=driftscaffold \
#     dataset.name=brain_tumor \
#     dataset.partition_dir=partitions/alpha_0.03/seed_42 \
#     model.name=efficient1 \
#     common.seed=42 common.global_epoch=40 \
#     "hydra.run.dir=logs/runs/brain_tumor_alpha0.03_efficient1_driftscaffold_seed42"

  
# ── efficient1_gn ──────────────────────────────────────────

# ── Alpha 1000.0 ──────────────────────────────────────────
run_experiment "brain_tumor | alpha=1000.0 | efficient1_gn" \
  python main.py --config-name driftscaffold method=driftscaffold \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_1000.0/seed_42 \
    model.name=efficient1_gn \
    common.seed=42 common.global_epoch=40 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha1000.0_efficient1_gn_driftscaffold_seed42"

# ── Alpha 1.0 ─────────────────────────────────────────────
run_experiment "brain_tumor | alpha=1.0 | efficient1_gn" \
  python main.py --config-name driftscaffold method=driftscaffold \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_1.0/seed_42 \
    model.name=efficient1_gn \
    common.seed=42 common.global_epoch=40 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha1.0_efficient1_gn_driftscaffold_seed42"

# ── Alpha 0.3 ─────────────────────────────────────────────
run_experiment "brain_tumor | alpha=0.3 | efficient1_gn" \
  python main.py --config-name driftscaffold method=driftscaffold \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_0.3/seed_42 \
    model.name=efficient1_gn \
    common.seed=42 common.global_epoch=40 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha0.3_efficient1_gn_driftscaffold_seed42"

# ── Alpha 0.03 ────────────────────────────────────────────
# run_experiment "brain_tumor | alpha=0.03 | efficient1_gn" \
#   python main.py --config-name driftscaffold method=driftscaffold \
#     dataset.name=brain_tumor \
#     dataset.partition_dir=partitions/alpha_0.03/seed_42 \
#     model.name=efficient1_gn \
#     common.seed=42 common.global_epoch=40 \
#     "hydra.run.dir=logs/runs/brain_tumor_alpha0.03_efficient1_gn_driftscaffold_seed42"
  

echo ""
echo "========================================"
echo " All experiments completed!"
echo " Finished: $(date)"
echo " Log saved to: $LOG_FILE"
echo "========================================"
