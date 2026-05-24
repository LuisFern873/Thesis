# Advisor Presentation — Experiment Guide

**Scope:** Brain Tumor MRI · FedAvg · seed=42 · 24 runs (6 models × 4 α values)

---

## Quick Start (everything in one command)

```bash
# From the FL-bench root directory on the Khipu cluster:
bash run_advisor_demo.sh
```

The script handles training, sanity checks, aggregation, and plot generation
automatically. When it finishes, all outputs are in `logs/`.

---

## Prerequisites

Run once before the demo script (only needed the first time):

```bash
bash setup_cluster.sh
```

---

## Model Variants and Parameter Counts

| Model key | Description | Total params | Base params | Head params (4-class) | Trainable |
|---|---|---|---|---|---|
| `efficient0` | EfficientNet-B0 (BatchNorm) | **4.013M** | 4.008M | 0.005M | 4.013M |
| `efficient0_gn` | EfficientNet-B0 (GroupNorm-32) | **4.013M** | 4.008M | 0.005M | 4.013M |
| `efficient0_ln` | EfficientNet-B0 (LayerNorm/GN-1) | **4.013M** | 4.008M | 0.005M | 4.013M |
| `efficient1` | EfficientNet-B1 (BatchNorm) | **6.518M** | 6.513M | 0.005M | 6.518M |
| `vit_tiny` | ViT-Tiny patch16-224 | **5.526M** | 5.524M | 0.002M | 5.526M |
| `vim_tiny` | Vim-Tiny (SSM) | ~7.0M | ~7.0M | 0.005M | ~7.0M |

> **EfficientNet-B1 vs B0:** B1 adds ~2.5M parameters (+62%) through wider and
> deeper MBConv blocks. Both use BatchNorm, so the B0 vs B1 comparison isolates
> the effect of **model capacity** on client drift, independent of normalization type.
> All three EfficientNet-B0 normalization variants share identical parameter counts
> because GN and LN have the same number of learnable parameters as BN per channel.

---

## What the Script Runs

24 training runs (6 models × 4 α values), in this order:

| # | α | Heterogeneity | Model |
|---|---|---|---|
| 1–6 | 1000.0 | IID baseline | all 6 models |
| 7–12 | 1.0 | Low het. | all 6 models |
| 13–18 | 0.3 | High het. | all 6 models |
| 19–24 | 0.03 | Extreme het. | all 6 models |

**Resume:** Re-run the script at any time. Runs with a complete
`drift_metrics.csv` (40 rows) are automatically skipped.

---

## Running Each Experiment Individually

Template:
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

### All 24 individual commands

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
  model.name=efficient1 common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha1000.0_efficient1_driftfedavg_seed42"

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
  model.name=efficient1 common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha1.0_efficient1_driftfedavg_seed42"

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
  model.name=efficient1 common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha0.3_efficient1_driftfedavg_seed42"

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
  model.name=efficient1 common.seed=42 common.global_epoch=40 \
  "hydra.run.dir=logs/runs/brain_tumor_alpha0.03_efficient1_driftfedavg_seed42"

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

```bash
# 1. Sanity check
.venv/bin/python scripts/sanity_check.py

# 2. Aggregate
.venv/bin/python scripts/aggregate_results.py

# 3. Generate all figures
.venv/bin/python scripts/plot_results.py --dataset brain_tumor --method driftfedavg
```

---

## Plots Produced (13 files total)

All figures saved to `logs/figures/` as 150 DPI PNG files.

### Figure 1 — Accuracy vs. Round
**File:** `fig1_accuracy_vs_round_brain_tumor_driftfedavg.png`

4 subplots (one per α). All 6 models shown with mean ± std shading.
**What to look for:** IID convergence parity; B1 vs B0 accuracy gap at α=0.03.

---

### Figure 2 — Per-Layer Drift vs. Round
**Files:** `fig2_drift_vs_round_brain_tumor_alpha{α}_driftfedavg.png` (4 files)

3 subplots per figure (norm / feature / head). Mean L2 drift across clients.
**What to look for:** EfficientNet-BN highest norm drift; B1 drift vs B0 drift
(does more capacity amplify or dampen drift?).

---

### Figure 3 — Gradient Alignment vs. Round
**Files:** `fig3_interference_vs_round_brain_tumor_alpha{α}_driftfedavg.png` (4 files)

Mean pairwise cosine similarity of pseudo-gradients per layer group.
**What to look for:** Feature-layer alignment drops as α decreases.

---

### Figure 4 — Normalization Ablation
**File:** `fig4_normalization_ablation_brain_tumor_driftfedavg.png`

Bar chart: B0-BN vs B0-GN vs B0-LN accuracy and norm-layer drift at each α.
**What to look for:** Gap between BN and GN/LN isolates BatchNorm contribution.

---

### Figure 5 — Fairness Gap vs. α
**File:** `fig5_fairness_vs_alpha_brain_tumor_driftfedavg.png`

Max-minus-min per-client accuracy vs. α (log scale), one line per model.
**What to look for:** B1 fairness gap vs B0 — does capacity help equity?

---

### Figure 6 — EfficientNet-B0 vs B1 Side-by-Side *(new)*
**File:** `fig6_b0_vs_b1_brain_tumor_driftfedavg.png`

2×2 grid comparing B0 (4.01M params) and B1 (6.52M params):
- Row 1: Accuracy vs. round at IID (α=1000) and extreme non-IID (α=0.03)
- Row 2: Norm-layer drift vs. round at the same two α levels

**Research question answered:** Does scaling from B0 → B1 (+2.5M params, +62%)
change the drift profile or convergence behaviour under non-IID FL?
Both models use BatchNorm, so this comparison isolates **model capacity** from
normalization type.

---

### Table 1 — Primary Comparison Table
**File:** `table1_comparison_brain_tumor_driftfedavg.txt`

Columns: `Model | α | Acc@final±std | Conv.Round | Drift-norm | Interference | Fairness`

---

## Output File Locations

```
logs/
├── runs/
│   └── brain_tumor_alpha{α}_{model}_driftfedavg_seed42/
│       ├── drift_metrics.csv   <- 40 rows, one per round
│       ├── metrics.csv         <- FL-bench client-side metrics
│       ├── main.log
│       └── events.out.tfevents
├── summary/
│   ├── all_results.csv
│   └── seed_agg.csv
└── figures/
    ├── fig1_accuracy_vs_round_brain_tumor_driftfedavg.png
    ├── fig2_drift_vs_round_brain_tumor_alpha{1000.0,1.0,0.3,0.03}_driftfedavg.png  (x4)
    ├── fig3_interference_vs_round_brain_tumor_alpha{...}_driftfedavg.png            (x4)
    ├── fig4_normalization_ablation_brain_tumor_driftfedavg.png
    ├── fig5_fairness_vs_alpha_brain_tumor_driftfedavg.png
    ├── fig6_b0_vs_b1_brain_tumor_driftfedavg.png
    └── table1_comparison_brain_tumor_driftfedavg.txt
```

---

## Monitoring During Training

```bash
tensorboard --logdir logs/runs --port 6006   # open http://localhost:6006
tail -f logs/run_progress.log
find logs/runs -name "drift_metrics.csv" | wc -l   # count completed runs
```

---

## Note on Vim-Tiny

Requires `mamba_ssm` (CUDA + C++ compiler). If unavailable, the 4 Vim-Tiny
runs fail; the remaining 20 are unaffected. Check:

```bash
.venv/bin/python -c "
from src.utils.models import MODELS
m = MODELS['vim_tiny'](dataset='brain_tumor', pretrained=False)
print('Vim-Tiny OK, params:', sum(p.numel() for p in m.parameters())/1e6, 'M')
"
```
