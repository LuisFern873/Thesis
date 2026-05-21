#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh — Full experiment matrix for the drift study
#
# Experiment matrix (methodology Section 6 / Quick Reference):
#   2 datasets × 4 α values × 5 model variants × 2 algorithms × 3 seeds
#   = 192 total runs
#
# Note: EfficientNet-GN and EfficientNet-LN are run under FedAvg only
#       (normalization ablation). FedProx is run for EfficientNet-BN,
#       ViT-Tiny, and Vim-Tiny.
#
# Usage:
#   bash run_experiments.sh                     # run everything
#   bash run_experiments.sh --dataset cifar10   # one dataset only
#   bash run_experiments.sh --dataset brain_tumor --model efficient0
#   bash run_experiments.sh --method driftfedavg
#   bash run_experiments.sh --alpha 0.03
#   DRYRUN=1 bash run_experiments.sh            # print commands, don't run
#
# Resume: a run is skipped if logs/runs/<run_name>/drift_metrics.csv
#         already exists and has >= GLOBAL_EPOCH data rows.
# =============================================================================

# Do NOT use set -e: one failed run should not abort the whole matrix.
set -uo pipefail

# ── Experiment parameters ────────────────────────────────────────────────────
DATASETS=("cifar10" "brain_tumor")
ALPHAS=("1000.0" "1.0" "0.3" "0.03")
SEEDS=(42 123 456)
GLOBAL_EPOCH=40

# Models and which algorithms they run with.
# Each entry: "fedavg_method:fedprox_method"  (empty = not run)
declare -A MODEL_METHODS=(
    ["efficient0"]="driftfedavg:driftfedprox"
    ["efficient0_gn"]="driftfedavg:"
    ["efficient0_ln"]="driftfedavg:"
    ["vit_tiny"]="driftfedavg:driftfedprox"
    ["vim_tiny"]="driftfedavg:driftfedprox"
)

# Deterministic model order for reproducible iteration
MODEL_ORDER=("efficient0" "efficient0_gn" "efficient0_ln" "vit_tiny" "vim_tiny")

PYTHON="${PYTHON:-python}"
LOG_DIR="logs/runs"
PROGRESS_LOG="logs/run_progress.log"
DRYRUN="${DRYRUN:-0}"

# ── CLI argument parsing ─────────────────────────────────────────────────────
FILTER_DATASET=""
FILTER_METHOD=""
FILTER_MODEL=""
FILTER_ALPHA=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) FILTER_DATASET="$2"; shift 2 ;;
        --method)  FILTER_METHOD="$2";  shift 2 ;;
        --model)   FILTER_MODEL="$2";   shift 2 ;;
        --alpha)   FILTER_ALPHA="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Setup ────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR" logs/summary

# ── Helpers ──────────────────────────────────────────────────────────────────
run_name() {
    local dataset=$1 alpha=$2 model=$3 method=$4 seed=$5
    echo "${dataset}_alpha${alpha}_${model}_${method}_seed${seed}"
}

is_done() {
    local name=$1
    local csv="$LOG_DIR/$name/drift_metrics.csv"
    [[ ! -f "$csv" ]] && return 1
    # Count data rows (total lines minus header)
    local rows
    rows=$(( $(wc -l < "$csv") - 1 ))
    [[ "$rows" -ge "$GLOBAL_EPOCH" ]]
}

run_one() {
    local dataset=$1 alpha=$2 model=$3 method=$4 seed=$5
    local name
    name=$(run_name "$dataset" "$alpha" "$model" "$method" "$seed")

    if [[ "$DRYRUN" != "1" ]] && is_done "$name"; then
        echo "[SKIP] $name"
        return 0
    fi

    # Hydra override: redirect output to logs/runs/<name>/
    # Must be passed as a separate argument without quotes around the = sign.
    local out_dir="$LOG_DIR/$name"

    local cmd=(
        "$PYTHON" main.py
        --config-name "$method"
        "method=$method"
        "dataset.name=$dataset"
        "dataset.partition_dir=partitions/alpha_${alpha}/seed_${seed}"
        "model.name=$model"
        "common.seed=$seed"
        "common.global_epoch=$GLOBAL_EPOCH"
        "hydra.run.dir=$out_dir"
    )

    if [[ "$DRYRUN" == "1" ]]; then
        echo "[DRY] ${cmd[*]}"
        return 0
    fi

    echo "[RUN] $name"
    echo "$(date '+%Y-%m-%d %H:%M:%S') — Starting: $name" >> "$PROGRESS_LOG"

    if "${cmd[@]}"; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') — Done:     $name" >> "$PROGRESS_LOG"
        echo "[OK]  $name"
        return 0
    else
        local exit_code=$?
        echo "$(date '+%Y-%m-%d %H:%M:%S') — FAILED:   $name (exit $exit_code)" >> "$PROGRESS_LOG"
        echo "[FAIL] $name" >&2
        return 1
    fi
}

# ── Main loop ────────────────────────────────────────────────────────────────
total=0
skipped=0
done_count=0
failed=0

for dataset in "${DATASETS[@]}"; do
    [[ -n "$FILTER_DATASET" && "$dataset" != "$FILTER_DATASET" ]] && continue

    for alpha in "${ALPHAS[@]}"; do
        [[ -n "$FILTER_ALPHA" && "$alpha" != "$FILTER_ALPHA" ]] && continue

        for model in "${MODEL_ORDER[@]}"; do
            [[ -n "$FILTER_MODEL" && "$model" != "$FILTER_MODEL" ]] && continue

            IFS=':' read -r fedavg_method fedprox_method <<< "${MODEL_METHODS[$model]}"

            for method in "$fedavg_method" "$fedprox_method"; do
                [[ -z "$method" ]] && continue
                [[ -n "$FILTER_METHOD" && "$method" != "$FILTER_METHOD" ]] && continue

                for seed in "${SEEDS[@]}"; do
                    total=$(( total + 1 ))
                    name=$(run_name "$dataset" "$alpha" "$model" "$method" "$seed")

                    if [[ "$DRYRUN" != "1" ]] && is_done "$name"; then
                        skipped=$(( skipped + 1 ))
                        echo "[SKIP] $name"
                        continue
                    fi

                    if run_one "$dataset" "$alpha" "$model" "$method" "$seed"; then
                        done_count=$(( done_count + 1 ))
                    else
                        failed=$(( failed + 1 ))
                    fi
                done
            done
        done
    done
done

echo ""
echo "============================================================"
echo "Summary: $done_count done | $skipped skipped | $failed failed | $total total"
echo "============================================================"

[[ "$failed" -gt 0 ]] && exit 1 || exit 0
