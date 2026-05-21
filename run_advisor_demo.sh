#!/usr/bin/env bash
# =============================================================================
# run_advisor_demo.sh
#
# Full experimental pipeline for the advisor presentation.
# Scope: Brain Tumor MRI dataset, seed=42, FedAvg (driftfedavg) only.
#
# Runs 5 experiments (one per model variant × 4 α levels):
#   brain_tumor × α ∈ {1000.0, 1.0, 0.3, 0.03} × 5 models × seed=42
#   = 20 runs total
#
# After training, automatically:
#   1. Runs sanity checks on all outputs
#   2. Aggregates results into logs/summary/
#   3. Generates all figures into logs/figures/
#
# Usage:
#   bash run_advisor_demo.sh
#
# Prerequisites (run once before this script):
#   bash setup_cluster.sh
#
# Expected wall time (single GPU, A100-class):
#   ~5–8 hours for all 20 runs
#   ~15–25 min per EfficientNet run
#   ~20–35 min per ViT-Tiny run
#   ~25–45 min per Vim-Tiny run (requires mamba_ssm + CUDA)
# =============================================================================

set -uo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-.venv/bin/python}"
DATASET="brain_tumor"
SEED=42
METHOD="driftfedavg"
GLOBAL_EPOCH=40
LOG_DIR="logs/runs"
PROGRESS_LOG="logs/run_progress.log"

# α values: 1000.0 = IID baseline, then increasing heterogeneity
ALPHAS=("1000.0" "1.0" "0.3" "0.03")

# All 5 model variants in the study
MODELS=("efficient0" "efficient0_gn" "efficient0_ln" "vit_tiny" "vim_tiny")

# ── Helpers ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${BOLD}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*" >&2; }
info() { echo -e "${CYAN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

run_name() {
    echo "${DATASET}_alpha${1}_${2}_${METHOD}_seed${SEED}"
}

is_done() {
    local csv="$LOG_DIR/$(run_name "$1" "$2")/drift_metrics.csv"
    [[ ! -f "$csv" ]] && return 1
    local rows=$(( $(wc -l < "$csv") - 1 ))
    [[ "$rows" -ge "$GLOBAL_EPOCH" ]]
}

# ── Pre-flight checks ─────────────────────────────────────────────────────────
log "Pre-flight checks..."

if [[ ! -f "$PYTHON" && ! $(command -v "$PYTHON" 2>/dev/null) ]]; then
    fail "Python not found at '$PYTHON'."
    fail "Either run 'bash setup_cluster.sh' first, or set PYTHON=/path/to/python"
    exit 1
fi

if [[ ! -f "main.py" ]]; then
    fail "main.py not found. Run this script from the FL-bench root directory."
    exit 1
fi

# Verify partitions exist for all α values
for alpha in "${ALPHAS[@]}"; do
    pdir="data/${DATASET}/partitions/alpha_${alpha}/seed_${SEED}"
    if [[ ! -f "${pdir}/partition.pkl" ]]; then
        fail "Partition missing: ${pdir}/partition.pkl"
        fail "Run: $PYTHON scripts/generate_all_partitions.py --dataset $DATASET"
        exit 1
    fi
done
ok "All partitions verified."

mkdir -p "$LOG_DIR" logs/summary logs/figures

# ── Count what needs to run ───────────────────────────────────────────────────
total_runs=$(( ${#ALPHAS[@]} * ${#MODELS[@]} ))
pending=0
for alpha in "${ALPHAS[@]}"; do
    for model in "${MODELS[@]}"; do
        is_done "$alpha" "$model" || pending=$(( pending + 1 ))
    done
done

echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD} Advisor Demo — Brain Tumor MRI, FedAvg, seed=42${NC}"
echo -e "${BOLD}============================================================${NC}"
info "Dataset  : $DATASET"
info "Method   : $METHOD (FedAvg with drift instrumentation)"
info "Seed     : $SEED"
info "Rounds   : $GLOBAL_EPOCH"
info "α values : ${ALPHAS[*]}"
info "Models   : ${MODELS[*]}"
info "Total    : $total_runs runs ($pending pending, $(( total_runs - pending )) already done)"
echo ""

# ── Training loop ─────────────────────────────────────────────────────────────
log "Starting training..."
echo ""

done_count=0
failed_count=0
failed_runs=()

for alpha in "${ALPHAS[@]}"; do
    echo -e "${BOLD}── α = ${alpha} ──────────────────────────────────────────────${NC}"

    for model in "${MODELS[@]}"; do
        name=$(run_name "$alpha" "$model")

        if is_done "$alpha" "$model"; then
            ok "SKIP  $name  (already complete)"
            done_count=$(( done_count + 1 ))
            continue
        fi

        log "RUN   $name"
        echo "$(date '+%Y-%m-%d %H:%M:%S') — Starting: $name" >> "$PROGRESS_LOG"

        "$PYTHON" main.py \
            --config-name "$METHOD" \
            "method=$METHOD" \
            "dataset.name=$DATASET" \
            "dataset.partition_dir=partitions/alpha_${alpha}/seed_${SEED}" \
            "model.name=$model" \
            "common.seed=$SEED" \
            "common.global_epoch=$GLOBAL_EPOCH" \
            "hydra.run.dir=$LOG_DIR/$name"

        exit_code=$?
        if [[ $exit_code -eq 0 ]]; then
            ok "DONE  $name"
            echo "$(date '+%Y-%m-%d %H:%M:%S') — Done:   $name" >> "$PROGRESS_LOG"
            done_count=$(( done_count + 1 ))
        else
            fail "FAIL  $name  (exit $exit_code)"
            echo "$(date '+%Y-%m-%d %H:%M:%S') — FAILED: $name (exit $exit_code)" >> "$PROGRESS_LOG"
            failed_count=$(( failed_count + 1 ))
            failed_runs+=("$name")
        fi
        echo ""
    done
done

# ── Training summary ──────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD} Training complete${NC}"
echo -e "${BOLD}============================================================${NC}"
ok  "Done:   $done_count / $total_runs"
[[ $failed_count -gt 0 ]] && fail "Failed: $failed_count" || true
echo ""

if [[ $failed_count -gt 0 ]]; then
    warn "Failed runs:"
    for r in "${failed_runs[@]}"; do
        warn "  - $r"
    done
    warn "Check logs at: $LOG_DIR/<run_name>/main.log"
    echo ""
fi

# Abort post-processing if nothing succeeded
if [[ $done_count -eq 0 ]]; then
    fail "No runs completed. Cannot generate results."
    exit 1
fi

# ── Post-processing ───────────────────────────────────────────────────────────
log "Running sanity checks..."
"$PYTHON" scripts/sanity_check.py --logs-dir "$LOG_DIR" --expected-rounds "$GLOBAL_EPOCH" \
    && ok "Sanity checks passed." \
    || warn "Some sanity check warnings — review above before presenting."
echo ""

log "Aggregating results..."
"$PYTHON" scripts/aggregate_results.py \
    --logs-dir "$LOG_DIR" \
    --out-dir  "logs/summary"
ok "Results aggregated → logs/summary/"
echo ""

log "Generating figures..."
"$PYTHON" scripts/plot_results.py \
    --logs-dir "$LOG_DIR" \
    --out-dir  "logs/figures" \
    --dataset  "$DATASET" \
    --method   "$METHOD"
ok "Figures saved → logs/figures/"
echo ""

# ── Final summary ─────────────────────────────────────────────────────────────
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD} Pipeline complete — outputs ready for presentation${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""
echo -e "${BOLD}Figures produced (logs/figures/):${NC}"
ls -1 logs/figures/*.png logs/figures/*.txt 2>/dev/null | while read -r f; do
    echo "  $(basename "$f")"
done
echo ""
echo -e "${BOLD}Summary tables (logs/summary/):${NC}"
echo "  all_results.csv   — one row per completed run"
echo "  seed_agg.csv      — aggregated metrics (single seed here)"
echo ""
echo -e "${BOLD}TensorBoard:${NC}"
echo "  tensorboard --logdir $LOG_DIR --port 6006"
echo "  Then open: http://localhost:6006"
echo ""
echo -e "${BOLD}Progress log:${NC}"
echo "  cat $PROGRESS_LOG"
echo ""

[[ $failed_count -gt 0 ]] && exit 1 || exit 0
