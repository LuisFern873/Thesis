# FL-bench Codebase Analysis vs. Research Proposal

## Executive Summary

The codebase is a customised fork of **FL-bench** (KarhouTam/FL-bench) with three target architectures implemented — EfficientNet-B0, DeiT-Tiny, and Vision Mamba-Tiny — running on the Brain Tumor MRI dataset. Preliminary exploratory runs have been completed (20 rounds, IID and non-IID). The setup is a viable starting point, but several critical gaps exist between the proposal requirements and the current implementation.

---

## ⚠️ Conflicts: Proposal vs. Implementation

These are explicit mismatches requiring resolution before writing the thesis.

| Dimension | Proposal Says | Codebase Shows | Severity |
|---|---|---|---|
| **Communication rounds** | 100 rounds | `global_epoch: 20` in `defaults.yaml`; all existing CSVs have only 20 steps | 🔴 Critical |
| **Multi-seed evaluation** | ≥3 seeds | Only seed 42 found in all `args.json` files; no multi-seed runner exists | 🔴 Critical |
| **Heterogeneity levels** | Study behaviour under varying α | Only α=0.5 is implemented in `non-iid/args.json`; no sweep exists | 🔴 Critical |
| **Macro F1-Score** | Primary metric alongside accuracy | `Metrics` class has `macro_precision` and `macro_recall` but **no `macro_f1` property** | 🔴 Critical |
| **Client-level fairness** | Required evaluation dimension | No per-client F1-score aggregation or fairness index (e.g., Jain's, Std-Dev across clients) is computed | 🔴 Critical |
| **EfficientNet variant** | Proposal implies a specific variant | `efficient0` (EfficientNet-B0) is used; this should be explicitly acknowledged and justified | 🟡 Moderate |
| **FedProx μ sweep** | FedProx is heterogeneity-robust baseline | μ=0.01 in `defaults.yaml` but `FedProxServer.get_hyperparams()` defaults to μ=1.0; inconsistency and no sensitivity study | 🟡 Moderate |
| **Training data augmentation** | Standard medical imaging practice | No augmentation in `get_dataset_transforms()` — only Normalize + Resize for both train and test | 🟡 Moderate |
| **Vision Mamba pretrained weights** | Pretrained initialisation assumed | `_load_pretrained_weights()` is a **no-op stub** (`pass`); Mamba always starts from scratch | 🟡 Moderate |
| **Normalisation for tumor dataset** | Dataset-specific norm statistics | `tumor` is absent from `DATA_MEAN`/`DATA_STD` dicts → fallback to Resize-only (no normalisation) | 🟡 Moderate |
| **Image size inconsistency** | DATA_SHAPE: `(3,224,224)` | `preprocess.py` defaults to 64×64 unless overridden at run-time; stored at whatever size was chosen | 🟡 Moderate |

---

## 1. Experimental Setup — Gaps and Weaknesses

### 1.1 Evaluation Metrics (🔴 High Priority)

**Problem:** `Metrics` class (src/utils/metrics.py) does not expose a macro F1-score property. The class stores raw predictions and targets, so F1 can be computed from them — it's just never surfaced.

```python
# What exists:
@property def macro_precision(self): ...  # ✅
@property def macro_recall(self): ...     # ✅
@property def accuracy(self): ...         # ✅

# What's missing:
@property def macro_f1(self): ...         # ❌ NOT IMPLEMENTED
```

**Impact:** The core primary metric of your thesis is not being computed or logged anywhere.

### 1.2 Client-Level Fairness (🔴 High Priority)

**Problem:** TensorBoard logging (`display_metrics`) pushes per-client accuracy scalars but never computes:
- Standard deviation of per-client test accuracy
- Jain's fairness index
- Worst-10% client accuracy (bottom decile)
- Per-client macro F1

These are necessary to support the "client-level fairness" evaluation dimension in the proposal.

### 1.3 Multi-Seed Reproducibility (🔴 High Priority)

**Problem:** `fix_random_seed()` is correctly implemented (covers Python, NumPy, PyTorch, and CUDA determinism) — but there is no orchestration layer to run the same configuration across 3+ seeds and aggregate results (mean ± std). All existing runs used seed=42 only.

**What's needed:** A sweep script that runs `python main.py common.seed=X` for X ∈ {42, 123, 456} and then computes aggregate statistics across the output CSVs.

### 1.4 Heterogeneity Range (🔴 High Priority)

**Problem:** The proposal's central thesis concerns performance *under varying heterogeneity*. Only α=0.5 has been tested. You need at minimum:
- α=0.1 (high heterogeneity)
- α=0.5 (medium — current)
- α=1.0 (low heterogeneity, closer to IID)

Each requires generating a separate partition and running all model × algorithm combinations.

### 1.5 Communication Rounds (🔴 High Priority)

**Problem:** `defaults.yaml` has `global_epoch: 20`. All existing CSVs in `experiments/` only contain 20-step traces. The proposal requires 100 rounds. This is not a cosmetic issue — convergence curves, stability assessments, and fairness over time all require the full 100-round budget.

### 1.6 Data Normalisation Missing for Tumor Dataset (🟡 Moderate)

**Problem:** `DATA_MEAN` and `DATA_STD` in `constants.py` do not contain a `"tumor"` key. The fallback in `get_dataset_transforms()` drops the `Normalize` transform silently and applies only `Resize((224, 224))`. This means your models receive un-normalised pixel values in [0,1] (or [0,255] raw), which significantly impacts training stability across architectures differently.

**Fix:** Compute dataset-specific mean/std from the training set and register them:
```python
# Approximate for Brain Tumor RGB MRI (compute precisely):
DATA_MEAN["tumor"] = [0.1765, 0.1765, 0.1765]
DATA_STD["tumor"]  = [0.1796, 0.1796, 0.1796]
```

### 1.7 No Training Augmentation (🟡 Moderate)

**Problem:** Both `train_data_transform` and `test_data_transform` are identical in `get_dataset_transforms()`. Medical imaging best practice includes random horizontal flip and rotation at minimum. Without augmentation, models are more likely to overfit on clients with small local datasets (especially under high non-IID).

### 1.8 Vision Mamba Pretrained Weights Are a Stub (🟡 Moderate)

**Problem:** `VisionMambaModel._load_pretrained_weights()` is a `pass` statement. If EfficientNet and DeiT are initialised with pretrained ImageNet weights (`use_torchvision_pretrained_weights: false` is currently set for both) while Mamba starts randomly, the comparison is confounded. You must either:
- Use the same initialisation strategy (all pretrained OR all random-init) across architectures
- Or explicitly justify and disclose the difference

**Note:** The defaults.yaml currently has `use_torchvision_pretrained_weights: false`, which means EfficientNet also starts from random init — this is at least consistent, but may not be what you intend given the IID experiments show "pretrained vision models" in their titles.

### 1.9 FedProx μ Inconsistency (🟡 Moderate)

**Problem:** Two conflicting defaults:
- `config/defaults.yaml`: `fedprox.mu: 0.01`
- `src/server/fedprox.py get_hyperparams()`: `default=1.0`

The config file takes priority. But for a rigorous comparison between FedAvg and FedProx, you should tune μ per-architecture or at least document the chosen value and justify it. A μ sweep {0.001, 0.01, 0.1, 1.0} would be ideal.

### 1.10 Join Ratio (Implicit Client Selection) (🟡 Moderate)

**Problem:** `join_ratio: 0.1` with `client_num=10` means only **1 client** participates per round. This is unusually low. FL research with 10 clients typically uses join_ratio ≥ 0.5 (5 clients/round). With only 1 client selected, FedAvg degenerates into sequential fine-tuning with model broadcasting, which likely inflates noise in the learning curves.

---

## 2. Concrete Improvements (Prioritised by Impact)

### Priority 1 — Must Fix Before Any Results Are Valid

| Change | File | Effort |
|---|---|---|
| Add `macro_f1` property to `Metrics` | `src/utils/metrics.py` | ~5 min |
| Add tumor normalisation stats to constants | `src/utils/constants.py` | ~15 min |
| Set `global_epoch: 100` in defaults | `config/defaults.yaml` | ~1 min |
| Set `join_ratio: 0.5` (or justify 0.1 for 10 clients) | `config/defaults.yaml` | ~1 min |

### Priority 2 — Required for Thesis Claims

| Change | File | Effort |
|---|---|---|
| Write multi-seed sweep script | `experiments/run_sweep.py` (new) | ~1 hour |
| Partition data at α ∈ {0.1, 0.5, 1.0} | `generate_data.py` CLI | ~30 min |
| Add fairness metrics logging (std-dev, worst-k%) | `src/server/fedavg.py` | ~2 hours |
| Add training augmentation (HorizontalFlip, Rotation) | `src/server/fedavg.py` → `get_dataset_transforms()` | ~15 min |

### Priority 3 — Strengthens Scientific Rigor

| Change | File | Effort |
|---|---|---|
| Implement actual Mamba pretrained weight loading | `src/utils/models.py` | ~2-4 hours |
| FedProx μ sensitivity study | `experiments/run_fedprox_sweep.py` (new) | ~1 hour |
| Compute per-client F1 and log to TensorBoard | `src/server/fedavg.py` | ~1-2 hours |
| Log Macro-F1 in TensorBoard (not just accuracy) | `src/server/fedavg.py` | ~30 min |

---

## 3. How to Run (Step-by-Step)

### Step 0: Environment Setup

```powershell
# From the FL-bench root directory
cd c:\Users\LENOVO\Desktop\Thesis\Benchmark\FL-bench

# Create and activate virtual environment (venv already exists)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r .env\requirements.txt
# (or if .env dir has requirements: pip install -r requirements.txt)

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **Note:** Vision Mamba requires the `vim` package (from the Vim repository) to be importable. Verify with `python -c "from vim.models_mamba import VisionMamba"`. If this fails, you need to install the Vim package manually from https://github.com/hustvl/Vim.

### Step 1: Preprocess the Raw Dataset (if not done)

```powershell
cd data\tumor
python preprocess.py
# When prompted:
# Image size: 224
# Ratio: 1.0
# Seed: 42
cd ..\..
```

### Step 2: Partition the Dataset

Run one partition command per heterogeneity level you want to benchmark:

```powershell
# High heterogeneity (α=0.1) — 10 clients
python generate_data.py -d tumor -a 0.1 -cn 10 --seed 42 --split sample --test_ratio 0.2

# Medium heterogeneity (α=0.5) — ALREADY DONE at data/tumor/non-iid/
# This is the current partition loaded by the system
python generate_data.py -d tumor -a 0.5 -cn 10 --seed 42 --split sample --test_ratio 0.2

# Low heterogeneity (α=1.0)
python generate_data.py -d tumor -a 1.0 -cn 10 --seed 42 --split sample --test_ratio 0.2

# IID baseline
python generate_data.py -d tumor --iid 1.0 -cn 10 --seed 42 --split sample --test_ratio 0.2
```

> **Critical:** The system loads from `data/tumor/partition.pkl` (not from named subfolders). You must copy the desired partition into the root `data/tumor/` before each run, OR modify the server to accept a partition path argument.

### Step 3: Run a Single Experiment

```powershell
# FedAvg + EfficientNet-B0 (100 rounds, 10 clients, SGD lr=0.01)
python main.py `
  method=fedavg `
  model.name=efficient0 `
  dataset.name=tumor `
  common.global_epoch=100 `
  common.join_ratio=0.5 `
  common.seed=42 `
  optimizer.name=sgd `
  optimizer.lr=0.01 `
  common.batch_size=32 `
  common.monitor=tensorboard

# FedAvg + DeiT-Tiny
python main.py `
  method=fedavg `
  model.name=deit `
  dataset.name=tumor `
  common.global_epoch=100 `
  common.join_ratio=0.5 `
  common.seed=42

# FedAvg + Vision Mamba-Tiny
python main.py `
  method=fedavg `
  model.name=mamba `
  dataset.name=tumor `
  common.global_epoch=100 `
  common.join_ratio=0.5 `
  common.seed=42

# FedProx + EfficientNet-B0
python main.py `
  method=fedprox `
  model.name=efficient0 `
  dataset.name=tumor `
  common.global_epoch=100 `
  common.join_ratio=0.5 `
  common.seed=42 `
  fedprox.mu=0.01
```

Outputs are saved to `out/<method>/<dataset>/<timestamp>/`.

### Step 4: Monitor Training

```powershell
# In a separate terminal, from the FL-bench root:
tensorboard --logdir=out
# Then open http://localhost:6006 in browser
```

### Step 5: Multi-Seed Run (Recommended Pattern Until Sweep Script Exists)

Since no sweep script exists, run manually:

```powershell
foreach ($seed in @(42, 123, 456)) {
    python main.py `
        method=fedavg `
        model.name=efficient0 `
        dataset.name=tumor `
        common.global_epoch=100 `
        common.join_ratio=0.5 `
        common.seed=$seed `
        common.monitor=null  # disable tensorboard for speed
}
```

Results are saved to CSV at `out/fedavg/tumor/<timestamp>/metrics.csv`. Aggregate these across seeds manually after all three runs.

### Step 6: Full Experiment Matrix

The minimum experiment matrix for the thesis (2 FL algorithms × 3 architectures × 3 seeds × 3 α values = **54 runs**):

| FL Method | Architecture | α values | Seeds |
|---|---|---|---|
| FedAvg | EfficientNet-B0 | 0.1, 0.5, 1.0 | 42, 123, 456 |
| FedAvg | DeiT-Tiny | 0.1, 0.5, 1.0 | 42, 123, 456 |
| FedAvg | Vision Mamba-Tiny | 0.1, 0.5, 1.0 | 42, 123, 456 |
| FedProx | EfficientNet-B0 | 0.1, 0.5, 1.0 | 42, 123, 456 |
| FedProx | DeiT-Tiny | 0.1, 0.5, 1.0 | 42, 123, 456 |
| FedProx | Vision Mamba-Tiny | 0.1, 0.5, 1.0 | 42, 123, 456 |

Each run is 100 rounds × 5 local epochs × 5 selected clients per round.

---

## 4. What's Working Well

- ✅ Dirichlet partitioning (`dirichlet()`) is correctly implemented in `generate_data.py`
- ✅ `fix_random_seed()` covers Python, NumPy, PyTorch + CUDA, and `cudnn.deterministic=True`
- ✅ FedProx proximal term is correctly implemented in `FedProxClient.fit()`
- ✅ FedAvg weighted aggregation by sample count is correct
- ✅ All three architectures are registered in `MODELS` dict and loadable
- ✅ TensorBoard per-client accuracy logging is already implemented (see `display_metrics()`)
- ✅ `save_metrics=true` ensures CSVs are always saved for offline analysis
- ✅ `DecoupledModel` base class enables clean integration of all architectures

