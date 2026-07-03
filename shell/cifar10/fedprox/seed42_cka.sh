#!/bin/bash

set -euo pipefail

LOG_FILE="shell/cifar10/fedprox/seed42_cka_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo " FedProx Experiments"
echo " Started: $(date)"
echo "========================================"

run_experiment() {
  local desc="$1"
  local run_dir="$2"
  shift 2

  # Si ya existe client_cka_metrics.csv, omitir el experimento
  if [[ -f "${run_dir}/client_cka_metrics.csv" ]]; then
    echo ""
    echo "----------------------------------------"
    echo " Skipping: $desc"
    echo " Reason:   ${run_dir}/client_cka_metrics.csv already exists"
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
  "cifar10 | res9 | alpha=1000.0 | seed=42" \
  "logs/runs/cifar10/fedprox/res9/alpha1000.0/seed42" \
  python scripts/compute_cka_offline.py --run-dir logs/runs/cifar10/fedprox/res9/alpha1000.0/seed42 --device cuda --skip-existing 


# ── Alpha 1.0 ─────────────────────────────────────────────
run_experiment \
  "cifar10 | res9 | alpha=1.0 | seed=42" \
  "logs/runs/cifar10/fedprox/res9/alpha1.0/seed42" \
  python scripts/compute_cka_offline.py --run-dir logs/runs/cifar10/fedprox/res9/alpha1.0/seed42 --device cuda --skip-existing 


# ── Alpha 0.3 ─────────────────────────────────────────────
run_experiment \
  "cifar10 | res9 | alpha=0.3 | seed=42" \
  "logs/runs/cifar10/fedprox/res9/alpha0.3/seed42" \
  python scripts/compute_cka_offline.py --run-dir logs/runs/cifar10/fedprox/res9/alpha0.3/seed42 --device cuda --skip-existing 


# ── Alpha 0.1 ────────────────────────────────────────────
run_experiment \
  "cifar10 | res9 | alpha=0.1 | seed=42" \
  "logs/runs/cifar10/fedprox/res9/alpha0.1/seed42" \
  python scripts/compute_cka_offline.py --run-dir logs/runs/cifar10/fedprox/res9/alpha0.1/seed42 --device cuda --skip-existing 


# ── Alpha 0.03 ────────────────────────────────────────────
run_experiment \
  "cifar10 | res9 | alpha=0.03 | seed=42" \
  "logs/runs/cifar10/fedprox/res9/alpha0.03/seed42" \
  python scripts/compute_cka_offline.py --run-dir logs/runs/cifar10/fedprox/res9/alpha0.03/seed42 --device cuda --skip-existing 

# ──────────────────────────────────────────────────
# ── vim_tiny ──────────────────────────────────────
# ──────────────────────────────────────────────────

# ── Alpha 1000.0 ──────────────────────────────────────────
run_experiment \
  "cifar10 | vim_tiny | alpha=1000.0 | seed=42" \
  "logs/runs/cifar10/fedprox/vim_tiny/alpha1000.0/seed42" \
  python scripts/compute_cka_offline.py --run-dir logs/runs/cifar10/fedprox/vim_tiny/alpha1000.0/seed42 --device cuda --skip-existing 


# ── Alpha 1.0 ─────────────────────────────────────────────
run_experiment \
  "cifar10 | vim_tiny | alpha=1.0 | seed=42" \
  "logs/runs/cifar10/fedprox/vim_tiny/alpha1.0/seed42" \
  python scripts/compute_cka_offline.py --run-dir logs/runs/cifar10/fedprox/vim_tiny/alpha1.0/seed42 --device cuda --skip-existing 


# ── Alpha 0.3 ─────────────────────────────────────────────
run_experiment \
  "cifar10 | vim_tiny | alpha=0.3 | seed=42" \
  "logs/runs/cifar10/fedprox/vim_tiny/alpha0.3/seed42" \
  python scripts/compute_cka_offline.py --run-dir logs/runs/cifar10/fedprox/vim_tiny/alpha0.3/seed42 --device cuda --skip-existing 


# ── Alpha 0.1 ────────────────────────────────────────────
run_experiment \
  "cifar10 | vim_tiny | alpha=0.1 | seed=42" \
  "logs/runs/cifar10/fedprox/vim_tiny/alpha0.1/seed42" \
  python scripts/compute_cka_offline.py --run-dir logs/runs/cifar10/fedprox/vim_tiny/alpha0.1/seed42 --device cuda --skip-existing 


# ── Alpha 0.03 ────────────────────────────────────────────
run_experiment \
  "cifar10 | vim_tiny | alpha=0.03 | seed=42" \
  "logs/runs/cifar10/fedprox/vim_tiny/alpha0.03/seed42" \
  python scripts/compute_cka_offline.py --run-dir logs/runs/cifar10/fedprox/vim_tiny/alpha0.03/seed42 --device cuda --skip-existing 


# ──────────────────────────────────────────────────
# ── vig_tiny ──────────────────────────────────────
# ──────────────────────────────────────────────────

# ── Alpha 1000.0 ──────────────────────────────────────────
run_experiment \
  "cifar10 | vig_tiny | alpha=1000.0 | seed=42" \
  "logs/runs/cifar10/fedprox/vig_tiny/alpha1000.0/seed42" \
  python scripts/compute_cka_offline.py --run-dir logs/runs/cifar10/fedprox/vig_tiny/alpha1000.0/seed42 --device cuda --skip-existing 


# ── Alpha 1.0 ─────────────────────────────────────────────
run_experiment \
  "cifar10 | vig_tiny | alpha=1.0 | seed=42" \
  "logs/runs/cifar10/fedprox/vig_tiny/alpha1.0/seed42" \
  python scripts/compute_cka_offline.py --run-dir logs/runs/cifar10/fedprox/vig_tiny/alpha1.0/seed42 --device cuda --skip-existing 


# ── Alpha 0.3 ─────────────────────────────────────────────
run_experiment \
  "cifar10 | vig_tiny | alpha=0.3 | seed=42" \
  "logs/runs/cifar10/fedprox/vig_tiny/alpha0.3/seed42" \
  python scripts/compute_cka_offline.py --run-dir logs/runs/cifar10/fedprox/vig_tiny/alpha0.3/seed42 --device cuda --skip-existing 


# ── Alpha 0.1 ────────────────────────────────────────────
run_experiment \
  "cifar10 | vig_tiny | alpha=0.1 | seed=42" \
  "logs/runs/cifar10/fedprox/vig_tiny/alpha0.1/seed42" \
  python scripts/compute_cka_offline.py --run-dir logs/runs/cifar10/fedprox/vig_tiny/alpha0.1/seed42 --device cuda --skip-existing 


# ── Alpha 0.03 ────────────────────────────────────────────
run_experiment \
  "cifar10 | vig_tiny | alpha=0.03 | seed=42" \
  "logs/runs/cifar10/fedprox/vig_tiny/alpha0.03/seed42" \
  python scripts/compute_cka_offline.py --run-dir logs/runs/cifar10/fedprox/vig_tiny/alpha0.03/seed42 --device cuda --skip-existing 


# ──────────────────────────────────────────────────
# ── vit_tiny ──────────────────────────────────────
# ──────────────────────────────────────────────────

# ── Alpha 1000.0 ──────────────────────────────────────────
run_experiment \
  "cifar10 | vit_tiny | alpha=1000.0 | seed=42" \
  "logs/runs/cifar10/fedprox/vit_tiny/alpha1000.0/seed42" \
  python scripts/compute_cka_offline.py --run-dir logs/runs/cifar10/fedprox/vit_tiny/alpha1000.0/seed42 --device cuda --skip-existing 


# ── Alpha 1.0 ─────────────────────────────────────────────
run_experiment \
  "cifar10 | vit_tiny | alpha=1.0 | seed=42" \
  "logs/runs/cifar10/fedprox/vit_tiny/alpha1.0/seed42" \
  python scripts/compute_cka_offline.py --run-dir logs/runs/cifar10/fedprox/vit_tiny/alpha1.0/seed42 --device cuda --skip-existing 


# ── Alpha 0.3 ─────────────────────────────────────────────
run_experiment \
  "cifar10 | vit_tiny | alpha=0.3 | seed=42" \
  "logs/runs/cifar10/fedprox/vit_tiny/alpha0.3/seed42" \
  python scripts/compute_cka_offline.py --run-dir logs/runs/cifar10/fedprox/vit_tiny/alpha0.3/seed42 --device cuda --skip-existing 


# ── Alpha 0.1 ────────────────────────────────────────────
run_experiment \
  "cifar10 | vit_tiny | alpha=0.1 | seed=42" \
  "logs/runs/cifar10/fedprox/vit_tiny/alpha0.1/seed42" \
  python scripts/compute_cka_offline.py --run-dir logs/runs/cifar10/fedprox/vit_tiny/alpha0.1/seed42 --device cuda --skip-existing 


# ── Alpha 0.03 ────────────────────────────────────────────
run_experiment \
  "cifar10 | vit_tiny | alpha=0.03 | seed=42" \
  "logs/runs/cifar10/fedprox/vit_tiny/alpha0.03/seed42" \
  python scripts/compute_cka_offline.py --run-dir logs/runs/cifar10/fedprox/vit_tiny/alpha0.03/seed42 --device cuda --skip-existing 



echo ""
echo "========================================"
echo " All experiments completed!"
echo " Finished: $(date)"
echo " Log saved to: $LOG_FILE"
echo "========================================"
