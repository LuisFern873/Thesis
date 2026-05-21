#!/usr/bin/env bash
# =============================================================================
# setup_cluster.sh — One-time environment setup for the Khipu cluster (Ubuntu)
#
# Run this once after cloning the repository:
#   bash setup_cluster.sh
#
# What it does:
#   1. Creates a Python virtual environment (.venv)
#   2. Installs all dependencies from .env/requirements.txt
#   3. Installs Vim-Tiny (mamba_ssm) if CUDA is available
#   4. Verifies the installation
#   5. Generates all 24 data partitions (2 datasets × 4 α × 3 seeds)
#   6. Runs the unit test suite
#   7. Prints a dry-run of the experiment matrix
# =============================================================================

set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV_DIR=".venv"

echo "============================================================"
echo "FL-bench Drift Experiment — Cluster Setup"
echo "============================================================"

# ── 1. Virtual environment ───────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[1/7] Creating virtual environment at $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "[1/7] Virtual environment already exists at $VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# ── 2. Install dependencies ──────────────────────────────────────────────────
echo "[2/7] Installing dependencies from .env/requirements.txt ..."
"$VENV_PIP" install --upgrade pip --quiet
"$VENV_PIP" install -r .env/requirements.txt --quiet
echo "      Done."

# ── 3. Vim-Tiny (mamba_ssm) — CUDA only ─────────────────────────────────────
echo "[3/7] Checking CUDA availability for mamba_ssm (Vim-Tiny) ..."
CUDA_AVAILABLE=$("$VENV_PYTHON" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

if [[ "$CUDA_AVAILABLE" == "True" ]]; then
    echo "      CUDA detected. Attempting to install mamba_ssm ..."
    # mamba_ssm requires CUDA and a compatible compiler.
    # Install from the Vim submodule if present, otherwise try PyPI.
    if [[ -d "Vim/mamba-1p1p1" ]]; then
        "$VENV_PIP" install -e Vim/mamba-1p1p1 --quiet && echo "      mamba_ssm installed from Vim/mamba-1p1p1." \
            || echo "      [WARN] mamba_ssm install failed. Vim-Tiny will be unavailable."
    else
        "$VENV_PIP" install mamba-ssm --quiet && echo "      mamba_ssm installed from PyPI." \
            || echo "      [WARN] mamba_ssm install failed. Vim-Tiny will be unavailable."
    fi
else
    echo "      No CUDA detected. Skipping mamba_ssm (Vim-Tiny unavailable)."
fi

# ── 4. Verify installation ───────────────────────────────────────────────────
echo "[4/7] Verifying installation ..."
"$VENV_PYTHON" - <<'EOF'
import torch, timm, numpy, pandas, sklearn, matplotlib, seaborn
print(f"  torch      {torch.__version__}  (CUDA: {torch.cuda.is_available()})")
print(f"  timm       {timm.__version__}")
print(f"  numpy      {numpy.__version__}")
print(f"  pandas     {pandas.__version__}")
print(f"  matplotlib {matplotlib.__version__}")
try:
    import mamba_ssm
    print(f"  mamba_ssm  {mamba_ssm.__version__}  (Vim-Tiny available)")
except ImportError:
    print("  mamba_ssm  NOT installed (Vim-Tiny unavailable)")
EOF

# ── 5. Generate data partitions ──────────────────────────────────────────────
echo "[5/7] Generating data partitions ..."
"$VENV_PYTHON" scripts/generate_all_partitions.py
echo "      Verifying partitions ..."
"$VENV_PYTHON" scripts/verify_partitions.py

# ── 6. Unit tests ────────────────────────────────────────────────────────────
echo "[6/7] Running unit tests ..."
"$VENV_PYTHON" -m pytest tests/test_drift_metrics.py -q
echo "      All tests passed."

# ── 7. Dry-run experiment matrix ─────────────────────────────────────────────
echo "[7/7] Dry-run of experiment matrix (first 10 commands) ..."
PYTHON="$VENV_PYTHON" DRYRUN=1 bash run_experiments.sh 2>&1 | head -30

echo ""
echo "============================================================"
echo "Setup complete. To run experiments:"
echo ""
echo "  # Full matrix (192 runs):"
echo "  PYTHON=$VENV_PYTHON bash run_experiments.sh"
echo ""
echo "  # CIFAR-10 only:"
echo "  PYTHON=$VENV_PYTHON bash run_experiments.sh --dataset cifar10"
echo ""
echo "  # Single cell (for testing):"
echo "  PYTHON=$VENV_PYTHON bash run_experiments.sh \\"
echo "    --dataset cifar10 --model efficient0 --alpha 0.03"
echo ""
echo "  # After runs complete:"
echo "  $VENV_PYTHON scripts/sanity_check.py"
echo "  $VENV_PYTHON scripts/aggregate_results.py"
echo "  $VENV_PYTHON scripts/plot_results.py"
echo "============================================================"
