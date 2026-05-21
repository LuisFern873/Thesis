# Advisor Presentation — Experiment Guide

**Scope:** Brain Tumor MRI · FedAvg · seed=42 · 20 runs (5 models × 4 α values)

---

## Quick Start (everything in one command)

```bash
# From the FL-bench root directory on the Khipu cluster:
bash run_advisor_demo.sh
```

That's it. The script handles training, sanity checks, aggregation, and plot
generation automatically. When it finishes, all outputs are in `logs/`.

---

## Prerequisites

Run once before the demo script (only needed the first time):

```bash
bash setup_cluster.sh
```

This creates `.venv`, installs dependencies, and generates all partitions.

---

## What the Script Runs

20 training runs in this order:

| # | α | Heterogeneity | Model | Run name |
|---|---|---|---|---|
| 1 | 1000.0 | IID baseline | EfficientNet-BN | `brain_tumor_alpha1000.0_efficient0_driftfedavg_seed42` |
| 2 | 1000.0 | IID baseline | EfficientNet-GN | `brain_tumor_alpha1000.0_efficient0_gn_driftfedavg_seed42` |
| 3 | 1000.0 | IID baseline | EfficientNet-LN | `brain_tumor_alpha1000.0_efficient0_ln_driftfedavg_seed42` |
| 4 | 1000.0 | IID baseline | ViT-Tiny | `brain_tumor_alpha1000.0_vit_tiny_driftfedavg_seed42` |
| 5 | 1000.0 | IID baseline | Vim-Tiny | `brain_tumor_alpha1000.0_vim_tiny_driftfedavg_seed42` |
| 6–10 | 1.0 | Low het. | all 5 models | `brain_tumor_alpha1.0_..._seed42` |
| 11–15 | 0.3 | High het. | all 5 models | `brain_tumor_alpha0.3_..._seed42` |
| 16–20 | 0.03 | Extreme het. | all 5 models | `brain_tumor_alpha0.03_..._seed42` |

**Resume:** If the script is interrupted, re-run it. Any run whose
`drift_metrics.csv` already has 40 rows is automatically skipped.

---

## Running Each Experiment Individually

If you want to run a single experiment manually, use this pattern:

```bash
.venv/bin/python main.py \
    --config-name driftfedavg \
    method=driftfedavg \
    dataset.name=brain_tumor \
    dataset.partition_dir=partitions/alpha_<ALPHA>/seed_42 \
    model.name=<MODEL> \
    common.seed=42 \
    common.global_epoch=40 \
    "hydra.run.dir=logs/runs/brain_tumor_alpha<ALPHA>_<MODEL>_driftfedavg_seed42"
```

### All 20 individual commands

**α = 1000.0 (IID baseline)**
```bash
.venv/bin/python main.py --config-name driftfedavg method=driftfedavg \
  dataset.name=brain_tumor dataset.partition_dir=partitions/alpha_1000.0/seed_42 \
  model.name=efficient0 common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha1000.0_efficient0_driftfedavg_seed42"

.venv/bin/python main.py --config-name driftfedavg method=driftfedavg \
  dataset.name=brain_tumor dataset.partition_dir=partitions/alpha_1000.0/seed_42 \
  model.name=efficient0_gn common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha1000.0_efficient0_gn_driftfedavg_seed42"

.venv/bin/python main.py --config-name driftfedavg method=driftfedavg \
  dataset.name=brain_tumor dataset.partition_dir=partitions/alpha_1000.0/seed_42 \
  model.name=efficient0_ln common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha1000.0_efficient0_ln_driftfedavg_seed42"

.venv/bin/python main.py --config-name driftfedavg method=driftfedavg \
  dataset.name=brain_tumor dataset.partition_dir=partitions/alpha_1000.0/seed_42 \
  model.name=vit_tiny common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha1000.0_vit_tiny_driftfedavg_seed42"

.venv/bin/python main.py --config-name driftfedavg method=driftfedavg \
  dataset.name=brain_tumor dataset.partition_dir=partitions/alpha_1000.0/seed_42 \
  model.name=vim_tiny common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha1000.0_vim_tiny_driftfedavg_seed42"
```

**α = 1.0 (low heterogeneity)**
```bash
.venv/bin/python main.py --config-name driftfedavg method=driftfedavg \
  dataset.name=brain_tumor dataset.partition_dir=partitions/alpha_1.0/seed_42 \
  model.name=efficient0 common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha1.0_efficient0_driftfedavg_seed42"

.venv/bin/python main.py --config-name driftfedavg method=driftfedavg \
  dataset.name=brain_tumor dataset.partition_dir=partitions/alpha_1.0/seed_42 \
  model.name=efficient0_gn common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha1.0_efficient0_gn_driftfedavg_seed42"

.venv/bin/python main.py --config-name driftfedavg method=driftfedavg \
  dataset.name=brain_tumor dataset.partition_dir=partitions/alpha_1.0/seed_42 \
  model.name=efficient0_ln common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha1.0_efficient0_ln_driftfedavg_seed42"

.venv/bin/python main.py --config-name driftfedavg method=driftfedavg \
  dataset.name=brain_tumor dataset.partition_dir=partitions/alpha_1.0/seed_42 \
  model.name=vit_tiny common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha1.0_vit_tiny_driftfedavg_seed42"

.venv/bin/python main.py --config-name driftfedavg method=driftfedavg \
  dataset.name=brain_tumor dataset.partition_dir=partitions/alpha_1.0/seed_42 \
  model.name=vim_tiny common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha1.0_vim_tiny_driftfedavg_seed42"
```

**α = 0.3 (high heterogeneity)**
```bash
.venv/bin/python main.py --config-name driftfedavg method=driftfedavg \
  dataset.name=brain_tumor dataset.partition_dir=partitions/alpha_0.3/seed_42 \
  model.name=efficient0 common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha0.3_efficient0_driftfedavg_seed42"

.venv/bin/python main.py --config-name driftfedavg method=driftfedavg \
  dataset.name=brain_tumor dataset.partition_dir=partitions/alpha_0.3/seed_42 \
  model.name=efficient0_gn common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha0.3_efficient0_gn_driftfedavg_seed42"

.venv/bin/python main.py --config-name driftfedavg method=driftfedavg \
  dataset.name=brain_tumor dataset.partition_dir=partitions/alpha_0.3/seed_42 \
  model.name=efficient0_ln common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha0.3_efficient0_ln_driftfedavg_seed42"

.venv/bin/python main.py --config-name driftfedavg method=driftfedavg \
  dataset.name=brain_tumor dataset.partition_dir=partitions/alpha_0.3/seed_42 \
  model.name=vit_tiny common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha0.3_vit_tiny_driftfedavg_seed42"

.venv/bin/python main.py --config-name driftfedavg method=driftfedavg \
  dataset.name=brain_tumor dataset.partition_dir=partitions/alpha_0.3/seed_42 \
  model.name=vim_tiny common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha0.3_vim_tiny_driftfedavg_seed42"
```

**α = 0.03 (extreme heterogeneity)**
```bash
.venv/bin/python main.py --config-name driftfedavg method=driftfedavg \
  dataset.name=brain_tumor dataset.partition_dir=partitions/alpha_0.03/seed_42 \
  model.name=efficient0 common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha0.03_efficient0_driftfedavg_seed42"

.venv/bin/python main.py --config-name driftfedavg method=driftfedavg \
  dataset.name=brain_tumor dataset.partition_dir=partitions/alpha_0.03/seed_42 \
  model.name=efficient0_gn common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha0.03_efficient0_gn_driftfedavg_seed42"

.venv/bin/python main.py --config-name driftfedavg method=driftfedavg \
  dataset.name=brain_tumor dataset.partition_dir=partitions/alpha_0.03/seed_42 \
  model.name=efficient0_ln common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha0.03_efficient0_ln_driftfedavg_seed42"

.venv/bin/python main.py --config-name driftfedavg method=driftfedavg \
  dataset.name=brain_tumor dataset.partition_dir=partitions/alpha_0.03/seed_42 \
  model.name=vit_tiny common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha0.03_vit_tiny_driftfedavg_seed42"

.venv/bin/python main.py --config-name driftfedavg method=driftfedavg \
  dataset.name=brain_tumor dataset.partition_dir=partitions/alpha_0.03/seed_42 \
  model.name=vim_tiny common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha0.03_vim_tiny_driftfedavg_seed42"
```

---

## Generating Plots After Training

Run these three commands in order after training completes:

```bash
# 1. Sanity check — flags any crashed or truncated runs
.venv/bin/python scripts/sanity_check.py

# 2. Aggregate — produces summary CSVs
.venv/bin/python scripts/aggregate_results.py

# 3. Generate all figures
.venv/bin/python scripts/plot_results.py --dataset brain_tumor --method driftfedavg
```

Or run them all at once:
```bash
.venv/bin/python scripts/sanity_check.py && \
.venv/bin/python scripts/aggregate_results.py && \
.venv/bin/python scripts/plot_results.py --dataset brain_tumor --method driftfedavg
```

---

## Plots Produced

All figures are saved to `logs/figures/` as 150 DPI PNG files.

### Figure 1 — Accuracy vs. Round
**File:** `fig1_accuracy_vs_round_brain_tumor_driftfedavg.png`

One figure with 4 subplots (one per α level). Each subplot shows global test
accuracy over 40 communication rounds for all 5 model variants.

**What to look for:**
- IID (α=1000): all models should converge; differences reveal inductive bias
- α=0.03: largest spread between models; EfficientNet-BN expected to lag

---

### Figure 2 — Per-Layer Drift vs. Round
**Files:** `fig2_drift_vs_round_brain_tumor_alpha{α}_driftfedavg.png` (4 files)

One figure per α level, with 3 subplots (norm / feature / head layer groups).
Shows mean L2 drift `||θ_k - θ_global||₂` across clients over rounds.

**What to look for:**
- Norm-layer drift: EfficientNet-BN should show highest drift (BatchNorm stats diverge)
- EfficientNet-GN/LN should have norm drift closer to ViT-Tiny (normalization ablation)
- Drift should plateau after ~round 20 for low heterogeneity

---

### Figure 3 — Gradient Alignment (Interference) vs. Round
**Files:** `fig3_interference_vs_round_brain_tumor_alpha{α}_driftfedavg.png` (4 files)

Mean pairwise cosine similarity of client pseudo-gradients per layer group.
Values near +1 = clients agree; near 0 = random; near -1 = conflicting updates.

**What to look for:**
- Feature-layer interference decreases as heterogeneity increases (α → 0.03)
- ViT-Tiny may show different interference profile than CNNs due to global attention

---

### Figure 4 — Normalization Ablation
**File:** `fig4_normalization_ablation_brain_tumor_driftfedavg.png`

Two bar charts side by side:
- Left: final accuracy for EfficientNet-BN vs. GN vs. LN at each α
- Right: final norm-layer drift for the same three variants

**What to look for:**
- The gap between BN and GN/LN in drift isolates the BatchNorm contribution
- If GN/LN accuracy ≈ ViT-Tiny accuracy, normalization explains most of the gap

---

### Figure 5 — Fairness Gap vs. α
**File:** `fig5_fairness_vs_alpha_brain_tumor_driftfedavg.png`

Max-minus-min per-client accuracy plotted against α (log scale), one line per model.

**What to look for:**
- All models: fairness gap increases as α decreases (more heterogeneity = more unfair)
- EfficientNet-BN expected to have the largest fairness gap at α=0.03

---

### Table 1 — Primary Comparison Table
**File:** `table1_comparison_brain_tumor_driftfedavg.txt`

Plain-text table with columns:
`Model | α | Acc@final±std | Conv.Round | Drift-norm | Interference | Fairness`

This is the main results table for the thesis and advisor presentation.

---

## Monitoring During Training

```bash
# Live TensorBoard (open http://localhost:6006)
tensorboard --logdir logs/runs --port 6006

# Watch progress log
tail -f logs/run_progress.log

# Count completed runs
find logs/runs -name "drift_metrics.csv" -exec wc -l {} \; | \
  awk '$1 >= 41 {print "DONE:", $2}' | wc -l
```

---

## Output File Locations

```
logs/
├── runs/
│   └── brain_tumor_alpha{α}_{model}_driftfedavg_seed42/
│       ├── drift_metrics.csv   ← 40 rows, one per round
│       ├── metrics.csv         ← FL-bench client-side metrics
│       ├── main.log            ← full training log
│       └── events.out.tfevents ← TensorBoard
├── summary/
│   ├── all_results.csv         ← one row per run
│   └── seed_agg.csv            ← aggregated (single seed here)
└── figures/
    ├── fig1_accuracy_vs_round_brain_tumor_driftfedavg.png
    ├── fig2_drift_vs_round_brain_tumor_alpha1000.0_driftfedavg.png
    ├── fig2_drift_vs_round_brain_tumor_alpha1.0_driftfedavg.png
    ├── fig2_drift_vs_round_brain_tumor_alpha0.3_driftfedavg.png
    ├── fig2_drift_vs_round_brain_tumor_alpha0.03_driftfedavg.png
    ├── fig3_interference_vs_round_brain_tumor_alpha1000.0_driftfedavg.png
    ├── fig3_interference_vs_round_brain_tumor_alpha1.0_driftfedavg.png
    ├── fig3_interference_vs_round_brain_tumor_alpha0.3_driftfedavg.png
    ├── fig3_interference_vs_round_brain_tumor_alpha0.03_driftfedavg.png
    ├── fig4_normalization_ablation_brain_tumor_driftfedavg.png
    ├── fig5_fairness_vs_alpha_brain_tumor_driftfedavg.png
    └── table1_comparison_brain_tumor_driftfedavg.txt
```

---

## Note on Vim-Tiny

Vim-Tiny requires `mamba_ssm`, which needs CUDA and a C++ compiler.
If it fails to import, the 4 Vim-Tiny runs will be skipped with an error.
The remaining 16 runs are unaffected. Check availability:

```bash
.venv/bin/python -c "
from src.utils.models import MODELS
m = MODELS['vim_tiny'](dataset='brain_tumor', pretrained=False)
print('Vim-Tiny OK, params:', sum(p.numel() for p in m.parameters())/1e6, 'M')
"
```
