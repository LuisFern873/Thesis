# Experiment Guide — Architecture-Driven Client Drift in FL

This document is the single reference for running, monitoring, and analysing
the full experiment matrix on the Khipu cluster.

---

## 1. One-Time Setup

```bash
# Clone and enter the repo (if not already done)
git clone <repo-url> FL-bench && cd FL-bench

# Run the setup script — creates .venv, installs deps,
# generates all 24 partitions, runs unit tests
bash setup_cluster.sh
```

The setup script prints the exact commands to use afterwards.
All subsequent commands use `.venv/bin/python` as the interpreter.

---

## 2. Directory Structure

After a full run the project looks like this:

```
FL-bench/
├── data/
│   ├── cifar10/partitions/alpha_{α}/seed_{seed}/
│   │   ├── partition.pkl        # FL-bench partition object
│   │   ├── args.json            # partition config snapshot
│   │   └── all_stats.json       # per-client HD, entropy, label dist.
│   └── brain_tumor/partitions/  # same structure
│
├── logs/
│   ├── runs/
│   │   └── {dataset}_alpha{α}_{model}_{method}_seed{seed}/
│   │       ├── drift_metrics.csv   ← PRIMARY output (one row per round)
│   │       ├── metrics.csv         ← FL-bench client-side metrics
│   │       ├── metrics.png         ← FL-bench learning curve
│   │       ├── main.log            ← full training log
│   │       └── events.out.tfevents ← TensorBoard binary log
│   │
│   ├── summary/
│   │   ├── all_results.csv      ← one row per run (final-round values)
│   │   └── seed_agg.csv         ← mean ± std per cell across 3 seeds
│   │
│   ├── figures/
│   │   ├── fig1_accuracy_vs_round_{dataset}_{method}.png
│   │   ├── fig2_drift_vs_round_{dataset}_alpha{α}_{method}.png
│   │   ├── fig3_interference_vs_round_{dataset}_alpha{α}_{method}.png
│   │   ├── fig4_normalization_ablation_{dataset}_{method}.png
│   │   ├── fig5_fairness_vs_alpha_{dataset}_{method}.png
│   │   └── table1_comparison_{dataset}_{method}.txt
│   │
│   ├── model_specs.txt          ← parameter counts for all models
│   └── run_progress.log         ← timestamped start/done/fail per run
│
├── run_experiments.sh           ← main experiment runner
├── setup_cluster.sh             ← one-time cluster setup
└── scripts/
    ├── generate_all_partitions.py
    ├── verify_partitions.py
    ├── sanity_check.py
    ├── aggregate_results.py
    └── plot_results.py
```

### drift_metrics.csv schema (one row per communication round)

| Column | Description |
|---|---|
| `round` | Communication round (1-indexed) |
| `lr` | Learning rate used by clients this round |
| `global_acc` | Global test accuracy (%) |
| `global_f1` | Global macro F1 (%) |
| `global_precision` | Global macro precision (%) |
| `global_recall` | Global macro recall (%) |
| `convergence_flag` | 1 at the first round convergence is detected, else 0 |
| `drift_{group}_mean` | Mean L2 drift across clients for group ∈ {norm, feature, head} |
| `drift_{group}_std` | Std of L2 drift across clients |
| `interference_{group}` | Mean pairwise cosine similarity of pseudo-gradients |
| `fairness_gap` | max − min per-client accuracy (%) |
| `client_acc_min/max/std` | Per-client accuracy statistics |

---

## 3. Running Experiments

### Full matrix (192 runs)
```bash
PYTHON=.venv/bin/python bash run_experiments.sh
```

### One dataset at a time (recommended — run CIFAR-10 first)
```bash
PYTHON=.venv/bin/python bash run_experiments.sh --dataset cifar10
PYTHON=.venv/bin/python bash run_experiments.sh --dataset brain_tumor
```

### Filter by any dimension
```bash
# Single model
PYTHON=.venv/bin/python bash run_experiments.sh --model efficient0

# Single α level
PYTHON=.venv/bin/python bash run_experiments.sh --alpha 0.03

# Single cell (useful for testing)
PYTHON=.venv/bin/python bash run_experiments.sh \
  --dataset cifar10 --model efficient0 --alpha 0.03
```

### Dry-run (print all commands without executing)
```bash
DRYRUN=1 bash run_experiments.sh | head -40
```

### Resume after interruption
The runner automatically skips any run whose `drift_metrics.csv` already
has ≥ 40 rows. Just re-run the same command.

### Run name convention
```
{dataset}_alpha{α}_{model}_{method}_seed{seed}
# e.g.: cifar10_alpha0.03_efficient0_driftfedavg_seed42
```

---

## 4. Monitoring Progress

### TensorBoard (live during training)
```bash
tensorboard --logdir logs/runs --port 6006
# Open http://localhost:6006
```
Tags available: `train/global_acc`, `train/global_f1`, `drift/{group}_mean`,
`interference/{group}`, `fairness/gap`, `train/lr`.

### Progress log
```bash
tail -f logs/run_progress.log
```

### Count completed runs
```bash
find logs/runs -name "drift_metrics.csv" | wc -l
```

---

## 5. Post-Experiment Analysis

Run these in order after all experiments complete:

```bash
# 1. Sanity check — flags crashed/truncated/silent runs
.venv/bin/python scripts/sanity_check.py

# 2. Aggregate — produces all_results.csv and seed_agg.csv
.venv/bin/python scripts/aggregate_results.py

# 3. Generate all figures and comparison tables
.venv/bin/python scripts/plot_results.py
```

All outputs land in `logs/summary/` and `logs/figures/`.

### Generate plots for a specific subset
```bash
# One dataset only
.venv/bin/python scripts/plot_results.py --dataset cifar10

# One method only
.venv/bin/python scripts/plot_results.py --method driftfedprox

# One α level for drift/interference figures
.venv/bin/python scripts/plot_results.py --alpha 0.03
```

---

## 6. Output Files Explained

### `logs/summary/all_results.csv`
One row per completed run. Columns: all experimental dimensions
(dataset, alpha, model, method, seed) plus every metric at the final round
and the convergence round. Use this for custom analysis.

### `logs/summary/seed_agg.csv`
One row per (dataset, alpha, model, method) cell. Each metric has a
`_mean` and `_std` column aggregated over the 3 seeds. The
`high_variance_flag` column marks cells where accuracy std > 3%.
This is the primary table for the thesis.

### `logs/figures/table1_comparison_{dataset}_{method}.txt`
Human-readable comparison table matching methodology Section 5.5.
Reports Acc@40±std, convergence round, drift-norm, interference, fairness gap.

---

## 7. Reproducing a Specific Run

Every run directory contains a `.hydra/config.yaml` snapshot of the exact
configuration used. To reproduce run `X`:

```bash
# Option A: re-run via the experiment script with matching filters
PYTHON=.venv/bin/python bash run_experiments.sh \
  --dataset cifar10 --model efficient0 --alpha 0.03 --method driftfedavg

# Option B: run directly with main.py using the saved config
.venv/bin/python main.py \
  --config-name driftfedavg \
  method=driftfedavg \
  dataset.name=cifar10 \
  dataset.partition_dir=partitions/alpha_0.03/seed_42 \
  model.name=efficient0 \
  common.seed=42 \
  "hydra.run.dir=logs/runs/cifar10_alpha0.03_efficient0_driftfedavg_seed42"
```

The run will be skipped automatically if `drift_metrics.csv` already has
40 rows. Use `--force` (delete the directory first) to re-run.

---

## 8. Comparing Multiple Runs

```python
import pandas as pd

# Load aggregated results
df = pd.read_csv("logs/summary/seed_agg.csv")

# Compare EfficientNet-BN vs ViT-Tiny on CIFAR-10 at α=0.03
mask = (df.dataset == "cifar10") & (df.alpha == "0.03") & (df.method == "driftfedavg")
subset = df[mask][["model", "global_acc@final_mean", "global_acc@final_std",
                    "drift_norm_mean@final_mean", "convergence_round_mean"]]
print(subset.to_string(index=False))
```

---

## 9. Expected Experiment Duration

Rough estimates on a single GPU (A100-class):

| Model | Time per run (40 rounds, E=5) |
|---|---|
| EfficientNet-B0 (any norm) | ~15–25 min |
| ViT-Tiny | ~20–35 min |
| Vim-Tiny | ~25–45 min |

Total for 192 runs: ~60–90 GPU-hours.
With 4 parallel GPUs: ~15–22 hours wall time.

Use `mode: parallel` and `parallel.num_workers: 4` in the config to
parallelise client training within a single run (reduces per-run time by ~2×).

---

## 10. Vim-Tiny Note

Vim-Tiny requires `mamba_ssm`, which needs CUDA and a compatible C++ compiler.
If it is unavailable, the 15 Vim-Tiny runs will fail with an ImportError.
All other 177 runs are unaffected. Check availability with:

```bash
.venv/bin/python -c "from src.utils.models import MODELS; MODELS['vim_tiny'](dataset='cifar10', pretrained=False)"
```
