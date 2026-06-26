#!/bin/bash

set -euo pipefail

LOG_FILE="shell/cifar10/fedavg/seed456_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo " FedAvg Experiments"
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
# ── res9 ──────────────────────────────────────────
# ──────────────────────────────────────────────────

# ── Alpha 1000.0 ──────────────────────────────────────────
run_experiment \
  "cifar10 | res9 | alpha=1000.0 | seed=456" \
  "logs/runs/cifar10/fedavg/res9/alpha1000.0/seed456" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1000.0/seed_456 \
    model.name=res9 \
    common.seed=456 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedavg/res9/alpha1000.0/seed456"

# ── Alpha 1.0 ─────────────────────────────────────────────
run_experiment \
  "cifar10 | res9 | alpha=1.0 | seed=456" \
  "logs/runs/cifar10/fedavg/res9/alpha1.0/seed456" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1.0/seed_456 \
    model.name=res9 \
    common.seed=456 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedavg/res9/alpha1.0/seed456"


# ── Alpha 0.3 ─────────────────────────────────────────────
run_experiment \
  "cifar10 | res9 | alpha=0.3 | seed=456" \
  "logs/runs/cifar10/fedavg/res9/alpha0.3/seed456" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.3/seed_456 \
    model.name=res9 \
    common.seed=456 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedavg/res9/alpha0.3/seed456"


# ── Alpha 0.1 ────────────────────────────────────────────
run_experiment \
  "cifar10 | res9 | alpha=0.1 | seed=456" \
  "logs/runs/cifar10/fedavg/res9/alpha0.1/seed456" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.1/seed_456 \
    model.name=res9 \
    common.seed=456 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedavg/res9/alpha0.1/seed456"

# ── Alpha 0.03 ────────────────────────────────────────────
run_experiment \
  "cifar10 | res9 | alpha=0.03 | seed=456" \
  "logs/runs/cifar10/fedavg/res9/alpha0.03/seed456" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.03/seed_456 \
    model.name=res9 \
    common.seed=456 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedavg/res9/alpha0.03/seed456"


# ──────────────────────────────────────────────────
# ── vim_tiny ──────────────────────────────────────
# ──────────────────────────────────────────────────

# ── Alpha 1000.0 ──────────────────────────────────────────
run_experiment \
  "cifar10 | vim_tiny | alpha=1000.0 | seed=456" \
  "logs/runs/cifar10/fedavg/vim_tiny/alpha1000.0/seed456" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1000.0/seed_456 \
    model.name=vim_tiny \
    common.seed=456 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedavg/vim_tiny/alpha1000.0/seed456"

# ── Alpha 1.0 ─────────────────────────────────────────────
run_experiment \
  "cifar10 | vim_tiny | alpha=1.0 | seed=456" \
  "logs/runs/cifar10/fedavg/vim_tiny/alpha1.0/seed456" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1.0/seed_456 \
    model.name=vim_tiny \
    common.seed=456 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedavg/vim_tiny/alpha1.0/seed456"


# ── Alpha 0.3 ─────────────────────────────────────────────
run_experiment \
  "cifar10 | vim_tiny | alpha=0.3 | seed=456" \
  "logs/runs/cifar10/fedavg/vim_tiny/alpha0.3/seed456" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.3/seed_456 \
    model.name=vim_tiny \
    common.seed=456 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedavg/vim_tiny/alpha0.3/seed456"


# ── Alpha 0.1 ────────────────────────────────────────────
run_experiment \
  "cifar10 | vim_tiny | alpha=0.1 | seed=456" \
  "logs/runs/cifar10/fedavg/vim_tiny/alpha0.1/seed456" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.1/seed_456 \
    model.name=vim_tiny \
    common.seed=456 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedavg/vim_tiny/alpha0.1/seed456"

# ── Alpha 0.03 ────────────────────────────────────────────
run_experiment \
  "cifar10 | vim_tiny | alpha=0.03 | seed=456" \
  "logs/runs/cifar10/fedavg/vim_tiny/alpha0.03/seed456" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.03/seed_456 \
    model.name=vim_tiny \
    common.seed=456 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedavg/vim_tiny/alpha0.03/seed456"


# ──────────────────────────────────────────────────
# ── vig_tiny ──────────────────────────────────────
# ──────────────────────────────────────────────────

# ── Alpha 1000.0 ──────────────────────────────────────────
run_experiment \
  "cifar10 | vig_tiny | alpha=1000.0 | seed=456" \
  "logs/runs/cifar10/fedavg/vig_tiny/alpha1000.0/seed456" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1000.0/seed_456 \
    model.name=vig_tiny \
    common.seed=456 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedavg/vig_tiny/alpha1000.0/seed456"

# ── Alpha 1.0 ─────────────────────────────────────────────
run_experiment \
  "cifar10 | vig_tiny | alpha=1.0 | seed=456" \
  "logs/runs/cifar10/fedavg/vig_tiny/alpha1.0/seed456" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1.0/seed_456 \
    model.name=vig_tiny \
    common.seed=456 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedavg/vig_tiny/alpha1.0/seed456"


# ── Alpha 0.3 ─────────────────────────────────────────────
run_experiment \
  "cifar10 | vig_tiny | alpha=0.3 | seed=456" \
  "logs/runs/cifar10/fedavg/vig_tiny/alpha0.3/seed456" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.3/seed_456 \
    model.name=vig_tiny \
    common.seed=456 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedavg/vig_tiny/alpha0.3/seed456"


# ── Alpha 0.1 ────────────────────────────────────────────
run_experiment \
  "cifar10 | vig_tiny | alpha=0.1 | seed=456" \
  "logs/runs/cifar10/fedavg/vig_tiny/alpha0.1/seed456" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.1/seed_456 \
    model.name=vig_tiny \
    common.seed=456 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedavg/vig_tiny/alpha0.1/seed456"

# ── Alpha 0.03 ────────────────────────────────────────────
run_experiment \
  "cifar10 | vig_tiny | alpha=0.03 | seed=456" \
  "logs/runs/cifar10/fedavg/vig_tiny/alpha0.03/seed456" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.03/seed_456 \
    model.name=vig_tiny \
    common.seed=456 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedavg/vig_tiny/alpha0.03/seed456"


# ──────────────────────────────────────────────────
# ── vit_tiny ──────────────────────────────────────
# ──────────────────────────────────────────────────

# ── Alpha 1000.0 ──────────────────────────────────────────
run_experiment \
  "cifar10 | vit_tiny | alpha=1000.0 | seed=456" \
  "logs/runs/cifar10/fedavg/vit_tiny/alpha1000.0/seed456" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1000.0/seed_456 \
    model.name=vit_tiny \
    common.seed=456 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedavg/vit_tiny/alpha1000.0/seed456"

# ── Alpha 1.0 ─────────────────────────────────────────────
run_experiment \
  "cifar10 | vit_tiny | alpha=1.0 | seed=456" \
  "logs/runs/cifar10/fedavg/vit_tiny/alpha1.0/seed456" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_1.0/seed_456 \
    model.name=vit_tiny \
    common.seed=456 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedavg/vit_tiny/alpha1.0/seed456"


# ── Alpha 0.3 ─────────────────────────────────────────────
run_experiment \
  "cifar10 | vit_tiny | alpha=0.3 | seed=456" \
  "logs/runs/cifar10/fedavg/vit_tiny/alpha0.3/seed456" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.3/seed_456 \
    model.name=vit_tiny \
    common.seed=456 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedavg/vit_tiny/alpha0.3/seed456"


# ── Alpha 0.1 ────────────────────────────────────────────
run_experiment \
  "cifar10 | vit_tiny | alpha=0.1 | seed=456" \
  "logs/runs/cifar10/fedavg/vit_tiny/alpha0.1/seed456" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.1/seed_456 \
    model.name=vit_tiny \
    common.seed=456 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedavg/vit_tiny/alpha0.1/seed456"

# ── Alpha 0.03 ────────────────────────────────────────────
run_experiment \
  "cifar10 | vit_tiny | alpha=0.03 | seed=456" \
  "logs/runs/cifar10/fedavg/vit_tiny/alpha0.03/seed456" \
  python main.py --config-name ckadriftfedavg method=ckadriftfedavg \
    dataset.name=cifar10 \
    dataset.partition_dir=partitions/alpha_0.03/seed_456 \
    model.name=vit_tiny \
    common.seed=456 \
    common.global_epoch=50 \
    mode=serial \
    "hydra.run.dir=logs/runs/cifar10/fedavg/vit_tiny/alpha0.03/seed456"


echo ""
echo "========================================"
echo " All experiments completed!"
echo " Finished: $(date)"
echo " Log saved to: $LOG_FILE"
echo "========================================"