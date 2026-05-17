# Rigorous Experiment Methodology & Implementation Checklist
## Architecture-Driven Client Drift in Federated Learning
### Luis Fernando Méndez Lázaro — UTEC 2026

---

## Section 1 — Experimental Setup

### 1.1 Datasets

#### Dataset A: Brain Tumor MRI
- **Source:** Kaggle Brain Tumor MRI Dataset (Msoud Nickparvar, 2021)
  - URL: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- **Classes:** 4 — glioma, meningioma, pituitary, no-tumor
- **Total images:** ~7,023 (train: 5,712 / test: 1,311)
- **Image size:** Resize to **224 × 224** (required by ViT-Tiny patch tokenizer)
- **Channels:** RGB (convert grayscale images to 3-channel via `transforms.Grayscale(3)` if needed)
- **Preprocessing pipeline (applied identically to all models):**
  ```python
  train_transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomRotation(degrees=15),
      transforms.ColorJitter(brightness=0.2, contrast=0.2),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])  # ImageNet stats
  ])
  test_transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
  ])
  ```
- **Rationale for 224×224:** ViT-Tiny uses 16×16 patches, requiring input size divisible by 16. EfficientNet-B0 and Vim-tiny are both compatible with 224×224.

#### Dataset B: CIFAR-10
- **Source:** Built-in via `torchvision.datasets.CIFAR10`
- **Classes:** 10 — airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Total images:** 60,000 (train: 50,000 / test: 10,000)
- **Image size:** **32 × 32** native; upsample to **224 × 224** for ViT-Tiny and Vim-tiny
- **Preprocessing pipeline:**
  ```python
  # For EfficientNet-B0 (native 32×32 with resize)
  # For ViT-Tiny and Vim-tiny (requires 224×224)
  train_transform = transforms.Compose([
      transforms.Resize((224, 224)),        # only needed for ViT/SSM
      transforms.RandomCrop(224, padding=4) if using_224 else transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                           std=[0.2470, 0.2435, 0.2616])  # CIFAR-10 stats
  ])
  ```
  > **Implementation note:** Use a single unified input size of 224×224 for all three models to eliminate image resolution as a confounding variable. This is the recommended approach for cross-architecture comparisons.

### 1.2 Data Partitioning (Non-IID)

#### Strategy: Dirichlet Label Distribution Skew
Use the Dirichlet distribution `Dir(α)` to partition training data across 10 clients. This is the standard non-IID benchmark protocol.

**Formal procedure:**
```
For each class c in {0, ..., C-1}:
    Draw proportions p_c ~ Dir(α)   # vector of length K (num_clients)
    Assign floor(p_c[k] * N_c) samples to client k
    where N_c = total samples of class c
```

**α values to test (Dirichlet Label Distribution Skew):**
| α | Heterogeneity level | Hellinger Distance (HD)* | Description |
|---|---|---|---|
| 1000 | **IID (Baseline)** | ~0.0 | Uniform distribution; completely IID |
| 1.0 | **Low** | ~0.5 | Mild skew; approaching uniform |
| 0.3 | **High** | ~0.75 | High skew; highly non-IID partition |
| 0.03 | **Extreme** | ~0.9 | Extreme skew; most clients see 1-2 classes |

*> Note: The mapping between α and HD is based on Jimenez et al. (2025).*

**Partitioning constraints:**
- Each client must receive a **minimum of 10 samples** per assigned class. If a client receives fewer, redistribute using a minimum-guarantee rebalancing step.
- The **test set is not partitioned**: all clients and the server evaluate on the same global test set to measure generalization.
- Partition indices are **saved to disk** as `data/partitions/{dataset}_alpha{α}_seed{seed}.pkl` to guarantee identical data splits across all runs.

**FL-bench command:**
```bash
# Brain Tumor MRI (requires custom dataset integration)
python generate_data.py -d brain_tumor -a 1000 -cn 10 --seed 42
python generate_data.py -d brain_tumor -a 1.0 -cn 10 --seed 42
python generate_data.py -d brain_tumor -a 0.3 -cn 10 --seed 42
python generate_data.py -d brain_tumor -a 0.03 -cn 10 --seed 42

# CIFAR-10
python generate_data.py -d cifar10 -a 1000 -cn 10 --seed 42
python generate_data.py -d cifar10 -a 1.0 -cn 10 --seed 42
python generate_data.py -d cifar10 -a 0.3 -cn 10 --seed 42
python generate_data.py -d cifar10 -a 0.03 -cn 10 --seed 42
```

### 1.3 Heterogeneity Quantification
For each generated partition, compute and log the **actual** heterogeneity level to verify it matches the intended `α`:
```python
def compute_partition_stats(partition_dict, num_classes):
    """Log label distribution per client and Earth Mover's Distance from uniform."""
    stats = {}
    for client_id, indices in partition_dict.items():
        labels = [dataset[i][1] for i in indices]
        dist = np.bincount(labels, minlength=num_classes) / len(labels)
        stats[client_id] = {
            "n_samples": len(indices),
            "label_distribution": dist.tolist(),
            "dominant_class": int(np.argmax(dist)),
            "entropy": float(-np.sum(dist * np.log(dist + 1e-9)))
        }
    return stats
```
Save these stats to `logs/partition_stats/{dataset}_alpha{α}_seed{seed}.json`.

---

## Section 2 — Model Architectures

### 2.1 Architecture Specifications

| Property | EfficientNet-B0 | ViT-Tiny | Vim-tiny |
|---|---|---|---|
| Family | CNN | ViT | SSM |
| Parameters | ~5.3M | ~5.7M | ~7M |
| Input size | 224 × 224 | 224 × 224 | 224 × 224 |
| Normalization | BatchNorm | LayerNorm | LayerNorm |
| Feature type | Local spatial filters | Global self-attention | Bidirectional state scan |
| timm identifier | — | `vit_tiny_patch16_224` | — |
| Pretrained init | ImageNet (torchvision) | ImageNet (timm) | ImageNet (mamba_ssm) |

> **On pretrained weights:** All three models are initialized from ImageNet pretrained weights. This is standard practice in federated learning on small-to-medium datasets and prevents random-init variance from confounding the drift analysis. The classifier head is re-initialized randomly to match the target number of classes.

### 2.2 Model Registration in FL-bench (`src/utils/models.py`)

```python
import timm
import torch.nn as nn
from torchvision.models import efficientnet_b0

def replace_batchnorm(model, norm_type="group", num_groups=32):
    """
    Recursively replace all BatchNorm2d layers with GroupNorm or LayerNorm.
    norm_type: "group" | "layer"
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            if norm_type == "group":
                new_norm = nn.GroupNorm(
                    num_groups=min(num_groups, num_features),
                    num_channels=num_features
                )
            elif norm_type == "layer":
                new_norm = nn.GroupNorm(1, num_features)  # GN with 1 group == LN for 2D
            setattr(model, name, new_norm)
        else:
            replace_batchnorm(module, norm_type, num_groups)
    return model


def get_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "efficientnet_b0":
        model = efficientnet_b0(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "efficientnet_b0_gn":
        model = efficientnet_b0(weights="IMAGENET1K_V1")
        model = replace_batchnorm(model, norm_type="group", num_groups=32)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "efficientnet_b0_ln":
        model = efficientnet_b0(weights="IMAGENET1K_V1")
        model = replace_batchnorm(model, norm_type="layer")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "vit_tiny":
        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=True,
            num_classes=num_classes
        )

    elif model_name == "vim_tiny":
        # Requires mamba_ssm; import conditionally
        from mamba_ssm.models.vim import VisionMamba
        model = VisionMamba(
            patch_size=16, embed_dim=192, depth=12,
            num_classes=num_classes, img_size=224
        )
        # Load pretrained weights if available; otherwise use random init
        # and document this as a limitation

    return model
```

### 2.3 Layer Taxonomy for Drift Analysis
Define a consistent grouping of layer types, used by all drift metrics:
```python
LAYER_GROUPS = {
    "norm":    (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm),
    "feature": (nn.Conv2d, nn.Linear),   # excl. classifier head
    "head":    None,                      # the final classifier layer only
}

def classify_layer(name: str, module: nn.Module) -> str:
    """Return the layer group for a named module."""
    if "classifier" in name or "head" in name:
        return "head"
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        return "norm"
    elif isinstance(module, (nn.Conv2d, nn.Linear)):
        return "feature"
    return "other"  # pooling, activation, etc. — excluded from drift metrics
```

---

## Section 3 — Federated Learning Protocol

### 3.1 FL Algorithm Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Algorithm | FedAvg, FedProx | FedAvg is the baseline; FedProx adds proximal regularization |
| Number of clients (K) | 10 | Cross-silo simulation; all clients participate every round |
| Client participation ratio | 1.0 (100%) | Full participation eliminates sampling noise from drift analysis |
| Communication rounds (T) | 100 | Sufficient to observe convergence plateau and round-wise drift evolution |
| Local epochs (E) | 5 | Standard non-IID setting; enough local steps to accumulate drift |
| Local batch size (B) | 32 | Fixed across all clients and models |
| FedProx μ | 0.01 | Start with 0.01; ablate with 0.001 and 0.1 in secondary experiments |

### 3.2 Optimizer and Learning Rate

```yaml
# config/drift_fedavg.yaml  (and drift_fedprox.yaml)
optimizer:
  name: SGD
  lr: 0.01
  momentum: 0.9
  weight_decay: 1.0e-4

lr_scheduler:
  name: CosineAnnealingLR
  T_max: 100          # matches total rounds
  eta_min: 1.0e-5
```

> **Critical:** The learning rate scheduler steps once per communication round (not per local epoch), applied to the **global model** before broadcasting. This ensures all clients start each round from the same LR regardless of local convergence.

### 3.3 Data Loading per Client
```python
# Each client uses its local partition with:
DataLoader(
    client_dataset,
    batch_size=32,
    shuffle=True,          # shuffle within local epoch
    num_workers=2,
    pin_memory=True,
    drop_last=True         # avoid batch-size-1 issues with BatchNorm
)
```
> `drop_last=True` is **mandatory** for EfficientNet-BN to avoid a batch of size 1 at the end of a local epoch, which causes BatchNorm to fail.

### 3.4 FL Round Protocol (Pseudocode)
```
Initialize global model θ_0 with pretrained weights
For round t = 1 to T:
    Server broadcasts θ_{t-1} to all K clients
    For each client k in parallel:
        θ_k ← local_train(θ_{t-1}, D_k, E=5, FedAvg or FedProx)
        Compute gradients: g_k = θ_{t-1} - θ_k   ← save for cosine similarity
        Compute drift: d_k = ||θ_k - θ_{t-1}||_2  ← save per layer type
    Server collects {θ_k} and {g_k}
    LOG drift metrics (Section 4) before aggregation
    θ_t ← aggregate({θ_k})  ← weighted average by dataset size
    LOG performance metrics: global accuracy, per-client accuracy
```

---

## Section 4 — Drift Measurement & Mechanistic Analysis

### 4.1 Metric 1: Per-Layer Client Drift (L2 Distance)

**Definition:**
```
drift_k^l(t) = ||θ_k^l(t) - θ_global^l(t-1)||_2
```
where `l` denotes a layer group, `k` a client index, and `t` a communication round.

**Implementation (`src/utils/drift_metrics.py`):**
```python
import torch
import torch.nn as nn
from typing import Dict, List

def compute_layer_drift(
    local_state: Dict[str, torch.Tensor],
    global_state: Dict[str, torch.Tensor],
    layer_taxonomy: Dict[str, list]  # output of classify_layer for all params
) -> Dict[str, float]:
    """
    Returns mean L2 drift per layer group for a single client.
    layer_taxonomy: {param_name: group_label}
    """
    group_diffs = {"norm": [], "feature": [], "head": []}
    for name, local_param in local_state.items():
        group = layer_taxonomy.get(name, "other")
        if group == "other":
            continue
        diff = (local_param.float() - global_state[name].float())
        group_diffs[group].append(diff.norm(p=2).item())
    return {g: float(np.mean(v)) if v else 0.0 for g, v in group_diffs.items()}
```

**Aggregation across clients:**
```python
def aggregate_drift(per_client_drifts: List[Dict]) -> Dict[str, Dict]:
    """Returns mean and std of drift across clients, per layer group."""
    groups = ["norm", "feature", "head"]
    return {
        g: {
            "mean": float(np.mean([d[g] for d in per_client_drifts])),
            "std":  float(np.std([d[g] for d in per_client_drifts]))
        }
        for g in groups
    }
```

### 4.2 Metric 2: Aggregation Interference (Gradient Cosine Similarity)

**Definition:**
```
interference^l(t) = (1 / (K*(K-1))) * Σ_{i≠j} cos(g_i^l(t), g_j^l(t))
```
where `g_k^l(t) = θ_{t-1}^l - θ_k^l(t)` is the pseudo-gradient of client `k` at round `t`.

**Implementation:**
```python
def compute_gradient_alignment(
    client_grads: List[Dict[str, torch.Tensor]],
    layer_taxonomy: Dict[str, str]
) -> Dict[str, float]:
    """
    client_grads: list of {param_name: grad_tensor} per client
    Returns mean pairwise cosine similarity per layer group.
    """
    K = len(client_grads)
    groups = ["norm", "feature", "head"]
    group_vecs = {g: [] for g in groups}

    for grads in client_grads:
        group_flat = {g: [] for g in groups}
        for name, grad in grads.items():
            g_label = layer_taxonomy.get(name, "other")
            if g_label != "other":
                group_flat[g_label].append(grad.float().flatten())
        for g in groups:
            if group_flat[g]:
                group_vecs[g].append(torch.cat(group_flat[g]))

    results = {}
    for g in groups:
        vecs = group_vecs[g]
        if len(vecs) < 2:
            results[g] = float("nan")
            continue
        sims = []
        for i in range(K):
            for j in range(i + 1, K):
                cos = torch.nn.functional.cosine_similarity(
                    vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)
                ).item()
                sims.append(cos)
        results[g] = float(np.mean(sims))
    return results
```

### 4.3 When and Where to Log

| Measurement | Timing | Location in FL-bench |
|---|---|---|
| Per-client, per-layer L2 drift | **Before** `aggregate()` every round | `DriftFedAvgServer.aggregate_client_updates()` |
| Pairwise cosine similarity | **Before** `aggregate()` every round | Same hook |
| Global test accuracy | **After** `aggregate()` every round | Existing FL-bench eval loop |
| Per-client test accuracy | **After** `aggregate()` every round | Override `evaluate()` in server |
| Partition entropy per client | Once at experiment start | `generate_data.py` post-step |
| Normalization layer running stats (μ, σ) for BN models | Every round for EfficientNet-BN only | Additional hook in `fit()` on client |

### 4.4 Expected Observations to Validate Against

These are the mechanistic predictions your results should be compared against:

| Prediction | What to check in results |
|---|---|
| EfficientNet-BN has highest drift in "norm" layer group | `drift["norm"]` for EfficientNet-BN > ViT-Tiny > Vim at α=0.1 |
| Drift stabilises for all models after round ~20 | Plot drift vs. round; look for plateau before round 25 at α=1.0 |
| EfficientNet-BN drift oscillates more (higher std) than ViT-Tiny | Compare `drift_std["norm"]` across clients at α=0.1 |
| ViT-Tiny shows higher initial interference (rounds 1–10) but converges faster | `interference["feature"]` for ViT-Tiny starts lower cos_sim, crosses EfficientNet by round 15 |
| Replacing BN → GN/LN narrows the EfficientNet drift gap to ViT-Tiny | `drift["norm"]` for EfficientNet-GN ≈ ViT-Tiny at same α |

---

## Section 5 — Evaluation Metrics

### 5.1 Performance Metrics

```python
# Computed on the global test set after each round
metrics = {
    "accuracy":  correct / total,
    "precision": sklearn.metrics.precision_score(y_true, y_pred, average="macro"),
    "recall":    sklearn.metrics.recall_score(y_true, y_pred, average="macro"),
    "f1":        sklearn.metrics.f1_score(y_true, y_pred, average="macro"),
}
```

### 5.2 Convergence Speed

Define convergence as the **first round** at which global accuracy crosses a fixed threshold and stays above it for 5 consecutive rounds:
```python
THRESHOLD_CIFAR10      = 0.70   # 70% accuracy
THRESHOLD_BRAIN_TUMOR  = 0.80   # 80% accuracy (4-class, easier)

def convergence_round(accuracy_history: List[float], threshold: float) -> int:
    for t in range(len(accuracy_history) - 4):
        if all(a >= threshold for a in accuracy_history[t:t+5]):
            return t
    return -1  # did not converge within 100 rounds
```

### 5.3 Fairness Metric

Use the **worst-client accuracy gap**:
```python
def fairness_gap(per_client_accuracies: List[float]) -> float:
    return max(per_client_accuracies) - min(per_client_accuracies)
```
Also report the **standard deviation** across clients as a secondary fairness indicator.

### 5.4 Results Aggregation Across Seeds

All metrics are aggregated over 3 seeds:
```python
final_result = {
    "acc_mean":   np.mean([r["accuracy@round100"] for r in seed_results]),
    "acc_std":    np.std( [r["accuracy@round100"] for r in seed_results]),
    "conv_mean":  np.mean([r["convergence_round"]  for r in seed_results]),
    "fair_mean":  np.mean([r["fairness_gap@round100"] for r in seed_results]),
}
```

Report as `mean ± std` in all tables. Flag any run where accuracy std > 0.03 as a high-variance result requiring inspection.

### 5.5 Primary Comparison Table Structure

| Model | α | Dataset | Acc@100 ± std | Conv. Round | Drift-norm (mean) | Interference (mean) | Fairness Gap |
|---|---|---|---|---|---|---|---|
| EfficientNet-BN | 0.1 | CIFAR-10 | | | | | |
| EfficientNet-GN | 0.1 | CIFAR-10 | | | | | |
| EfficientNet-LN | 0.1 | CIFAR-10 | | | | | |
| ViT-Tiny | 0.1 | CIFAR-10 | | | | | |
| Vim-tiny | 0.1 | CIFAR-10 | | | | | |
| ... | | | | | | | |

---

## Section 6 — Ablation Studies

### Ablation 1: Normalization Isolation
**Question:** How much of EfficientNet-BN's drift comes from BatchNorm vs. convolutional filters?

| Variant | Change from baseline |
|---|---|
| EfficientNet-BN | Default (baseline) |
| EfficientNet-GN | Replace all BN with GroupNorm(32) |
| EfficientNet-LN | Replace all BN with GroupNorm(1) ≡ LayerNorm |

Run all three variants under **identical** FL settings (same α, same seed, same round count). The gap between EfficientNet-BN and EfficientNet-GN/LN in both accuracy and drift isolates the normalization contribution.

**Controlled variables:** same optimizer, LR, local epochs, data partition.
**Outcome variable:** `drift["norm"]`, `accuracy@100`, `convergence_round`.

### Ablation 2: Heterogeneity Sensitivity
**Question:** At what α does each architecture's performance degrade non-linearly?

Plot accuracy vs. α for all five models on both datasets. Look for the α value at which each model crosses below the convergence threshold. This identifies the **critical heterogeneity threshold** per architecture.

### Ablation 3: FL Algorithm Effect
**Question:** Does FedProx's proximal term differentially benefit architectures with higher drift?

For each architecture × dataset × α combination, run both FedAvg and FedProx and report:
- `Δ_acc = acc_fedprox - acc_fedavg`
- `Δ_drift = drift_fedprox - drift_fedavg`

If FedProx helps EfficientNet-BN more than ViT-Tiny, this confirms that drift magnitude (not inductive bias) is the primary variable FedProx is correcting.

### Ablation 4: Local Epochs Sensitivity (Secondary)
**Question:** Does increasing local epochs (E) amplify drift differently across architectures?

Run E ∈ {1, 5, 10} for one cell only (α=0.1, CIFAR-10, FedAvg, seed=42). This controls the drift accumulation rate and helps separate convergence speed from drift magnitude.

---

## Section 7 — Reproducibility

### 7.1 Random Seeds
```python
SEEDS = [42, 123, 456]

def set_global_seed(seed: int):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   # disable for reproducibility
```
Set seed at the start of **each run**, before data partitioning, model initialization, and training loop. The seed governs: data partition (via `generate_data.py --seed`), weight initialization of the classifier head, client sampling order (even at ratio=1.0, for logging consistency), and local training shuffle.

### 7.2 Number of Runs and Statistical Significance
- **Primary experiments:** 3 seeds per cell
- **Minimum to report:** Do not report a cell with fewer than 3 completed seeds
- **Statistical test:** Use a paired Wilcoxon signed-rank test (non-parametric) to compare two architectures across the 3 seeds per configuration. Use α=0.05 significance level. Given the small sample (n=3 seeds), treat p-values as indicative, not definitive — document this limitation explicitly.

### 7.3 Logging Strategy

```
logs/
├── partition_stats/
│   └── cifar10_alpha0.1_seed42.json
├── runs/
│   └── cifar10_alpha0.1_efficientnet_b0_fedavg_seed42/
│       ├── config.yaml          # full config snapshot (auto-saved)
│       ├── metrics.csv          # one row per round: round, acc, drift_norm, drift_feature, etc.
│       ├── events.out.tfevents  # tensorboard binary log
│       └── checkpoints/
│           ├── round_050.pt
│           └── round_100.pt     # final global model
└── summary/
    └── all_results.csv          # aggregated across seeds, one row per cell
```

**`metrics.csv` schema (written every round):**
```
round, global_acc, global_f1, convergence_flag,
drift_norm_mean, drift_norm_std,
drift_feature_mean, drift_feature_std,
drift_head_mean, drift_head_std,
interference_norm, interference_feature, interference_head,
fairness_gap, client_acc_min, client_acc_max, client_acc_std
```

### 7.4 Environment Pinning
```bash
# After setting up the environment
pip freeze > requirements_locked.txt
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# Log this output in a file: environment.txt
nvidia-smi >> environment.txt
uname -a >> environment.txt
```
Commit both `requirements_locked.txt` and `environment.txt` to your repository.

---

## Section 8 — Implementation Checklist

Work through this list in order. Do not proceed to the next item until the current one is verified.

---

### PHASE 0 — Environment Setup

- [ ] **0.1** Clone FL-bench: `git clone https://github.com/KarhouTam/FL-bench.git && cd FL-bench`
- [ ] **0.2** Create and activate virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
- [ ] **0.3** Install base dependencies: `pip install -r .env/requirements.txt`
- [ ] **0.4** Install additional packages: `pip install timm mamba-ssm tensorboard scikit-learn`
- [ ] **0.5** Verify GPU availability: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
- [ ] **0.6** Pin environment: `pip freeze > requirements_locked.txt` and log CUDA/torch versions to `environment.txt`
- [ ] **0.7** Create `CLAUDE.md` in repo root with full project context (as defined in Section 3 of the Claude Code guide)
- [ ] **0.8** Verify Claude Code installed and authenticated: `claude --version && claude doctor`

---

### PHASE 1 — Data Preparation

- [ ] **1.1** Download Brain Tumor MRI dataset from Kaggle into `data/brain_tumor/`; verify 4 class folders exist
- [ ] **1.2** Integrate Brain Tumor MRI into FL-bench: subclass `BaseDataset` in `data/utils/datasets.py`, add to `DATASETS` dict
- [ ] **1.3** Verify CIFAR-10 auto-download works: `python -c "from torchvision.datasets import CIFAR10; CIFAR10('data/', download=True)"`
- [ ] **1.4** Write `compute_partition_stats()` function in `data/utils/partition_utils.py`
- [ ] **1.5** Generate all CIFAR-10 partitions (4 α values × 3 seeds = 12 files):
  ```bash
  for ALPHA in 1000 1.0 0.3 0.03; do
    for SEED in 42 123 456; do
      python generate_data.py -d cifar10 -a $ALPHA -cn 10 --seed $SEED
    done
  done
  ```
- [ ] **1.6** Generate all Brain Tumor MRI partitions (same loop with `-d brain_tumor`)
- [ ] **1.7** For each partition, run `compute_partition_stats()` and save JSON to `logs/partition_stats/`
- [ ] **1.8** Manually inspect 2–3 partition stat files and verify that α=0.1 produces higher entropy spread than α=1.0
- [ ] **1.9** Verify `drop_last=True` is set in all client DataLoaders (critical for BatchNorm)

---

### PHASE 2 — Model Implementation

- [ ] **2.1** Add `replace_batchnorm()` utility function to `src/utils/models.py`
- [ ] **2.2** Add `EfficientNet-B0` (BN), `EfficientNet-GN`, `EfficientNet-LN` to model registry
- [ ] **2.3** Add `ViT-Tiny` via `timm.create_model("vit_tiny_patch16_224", pretrained=True)` to registry, registered as `"vit_tiny"`
- [ ] **2.4** Add `Vim-tiny` to registry (with try/except for import and clear error message if mamba_ssm unavailable)
- [ ] **2.5** Implement `classify_layer()` taxonomy function in `src/utils/drift_metrics.py`
- [ ] **2.6** For each model, run a single forward pass and verify output shape:
  ```python
  # For all 5 model variants:
  model = get_model("efficientnet_b0", num_classes=4)
  x = torch.randn(2, 3, 224, 224)
  assert model(x).shape == (2, 4), "Output shape mismatch"
  ```
- [ ] **2.7** Verify `replace_batchnorm()` has replaced **all** BN layers: `assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())`
- [ ] **2.8** Count parameters for all 5 models and log to `logs/model_specs.txt`; confirm they are within 2× of each other
- [ ] **2.9** Verify `base` and `classifier` split exists for each model (required by FL-bench workflow)

---

### PHASE 3 — Drift Metrics Implementation

- [ ] **3.1** Create `src/utils/drift_metrics.py` with `compute_layer_drift()` function
- [ ] **3.2** Create `compute_gradient_alignment()` function in same file
- [ ] **3.3** Create `aggregate_drift()` and `aggregate_interference()` helper functions
- [ ] **3.4** Write unit tests for both functions in `tests/test_drift_metrics.py`:
  - Test that identical local and global state returns drift=0.0
  - Test that anti-parallel gradients return cosine_similarity=-1.0
  - Test that random states return non-zero drift
  ```bash
  python -m pytest tests/test_drift_metrics.py -v
  ```
- [ ] **3.5** Verify `classify_layer()` correctly categorises layers for all 5 model variants: print layer taxonomy for each model and manually spot-check 5 layers per model

---

### PHASE 4 — FL Server Instrumentation

- [ ] **4.1** Create `src/server/drift_fedavg.py` subclassing `FedAvgServer`
- [ ] **4.2** Override `aggregate_client_updates()` to: (a) compute per-client drift, (b) compute pairwise gradient alignment, (c) log all metrics to `metrics.csv` and TensorBoard
- [ ] **4.3** Override `evaluate()` to log per-client accuracy and fairness gap
- [ ] **4.4** Create `src/server/drift_fedprox.py` subclassing `FedProxServer` with the same drift instrumentation
- [ ] **4.5** Create `src/client/drift_client.py` that computes `g_k = θ_{t-1} - θ_k` after local training and passes it to the server package
- [ ] **4.6** Verify the `package()` method on both server and client correctly serialises/deserialises gradient tensors
- [ ] **4.7** Run a **smoke test**: 3 rounds, 2 clients, EfficientNet-BN, CIFAR-10, α=0.1, seed=42. Confirm `metrics.csv` is written with correct columns and non-zero drift values
- [ ] **4.8** Check TensorBoard output: `tensorboard --logdir=logs/runs/` and verify all scalar tags appear

---

### PHASE 5 — Configuration Files

- [ ] **5.1** Create `config/drift_fedavg.yaml` with all parameters from Section 3 of this document
- [ ] **5.2** Create `config/drift_fedprox.yaml` with μ=0.01
- [ ] **5.3** Set `common.monitor: tensorboard` in both config files
- [ ] **5.4** Set `common.save_path` to `logs/runs/` in both configs
- [ ] **5.5** Verify CLI override works: `python main.py --config-name drift_fedavg method=drift_fedavg common.model=vit_tiny` runs without error

---

### PHASE 6 — Experiment Runner

- [ ] **6.1** Create `run_experiments.sh` iterating over all cells:
  - 2 datasets × 4 α values × 5 model variants × 2 algorithms × 3 seeds = **192 runs**
- [ ] **6.2** Add DRYRUN mode: `DRYRUN=1 bash run_experiments.sh 2>&1 | head -50` and verify all 192 commands are printed correctly
- [ ] **6.3** Add skip logic: if `logs/runs/{run_name}/metrics.csv` already exists and has 100 rows, skip the run (enables resuming after interruption)
- [ ] **6.4** Add a progress log: `echo "$(date) — Starting: $run_name" >> logs/run_progress.log`
- [ ] **6.5** Run the first 5 cells manually (not via the script) to verify end-to-end correctness before launching the full matrix

---

### PHASE 7 — Full Experiment Execution

- [ ] **7.1** Run CIFAR-10 experiments first (faster iteration): `bash run_experiments.sh --dataset cifar10`
- [ ] **7.2** After CIFAR-10 completes, verify all 96 run directories exist and each `metrics.csv` has exactly 100 rows
- [ ] **7.3** Run a quick sanity check on results: `python scripts/sanity_check.py` (to be written — see 8.2)
- [ ] **7.4** Run Brain Tumor MRI experiments: `bash run_experiments.sh --dataset brain_tumor`
- [ ] **7.5** After all 192 runs complete, verify no `metrics.csv` files are missing or truncated

---

### PHASE 8 — Analysis & Visualisation

- [ ] **8.1** Write `scripts/aggregate_results.py` that reads all `metrics.csv` files and produces `logs/summary/all_results.csv`
- [ ] **8.2** Write `scripts/sanity_check.py` that flags runs where: final accuracy < 0.1 (likely crashed), drift_norm = 0.0 for all rounds (metric not logging), or accuracy std across seeds > 0.05
- [ ] **8.3** Produce **Figure 1:** Accuracy vs. round for all 5 models × 2 datasets × α=1.0, 0.3, 0.03 (6 subplots × 2 datasets = 12 panels). Use mean ± std shading.
- [ ] **8.4** Produce **Figure 2:** Per-layer drift (norm, feature, head) vs. round for all 5 models at α=0.03 on both datasets. 3 subplots (one per layer group).
- [ ] **8.5** Produce **Figure 3:** Gradient cosine similarity vs. round for all 3 base architectures at α=0.03. 3 subplots (one per layer group).
- [ ] **8.6** Produce **Figure 4 (Ablation 1):** Bar chart of drift@round100 and accuracy@round100 for EfficientNet-BN vs. GN vs. LN at each α level.
- [ ] **8.7** Produce **Figure 5:** Fairness gap (max-min per-client accuracy) vs. α, one line per model.
- [ ] **8.8** Compute statistical significance (Wilcoxon test) for EfficientNet-BN vs. ViT-Tiny accuracy at α=0.03 across 3 seeds. Document p-value.
- [ ] **8.9** Fill in the primary comparison table from Section 5.5 with all results.
- [ ] **8.10** Write a findings summary document `results/findings.md` with one paragraph per research sub-question, directly referencing the figures and table.

---

### PHASE 9 — Final Checks Before Writing

- [ ] **9.1** Confirm all 180 `metrics.csv` files are in the repository (or linked via data version control)
- [ ] **9.2** Confirm `requirements_locked.txt` and `environment.txt` are committed
- [ ] **9.3** Confirm all 5 model variants pass the forward-pass shape test (re-run after any model changes)
- [ ] **9.4** Confirm that the three EfficientNet normalization variants produce **different** drift profiles (if BN, GN, and LN all show identical drift, the ablation instrumentation is broken)
- [ ] **9.5** Run `DRYRUN=1 bash run_experiments.sh` one final time and confirm the 192 expected run names match the 192 `metrics.csv` files in `logs/runs/`

---

## Quick Reference: Experiment Matrix

| # | Dataset | α / Setup | Model | Algorithm | Seeds | Total runs |
|---|---|---|---|---|---|---|
| 1–3 | CIFAR-10 | α=0.03 | EfficientNet-BN | FedAvg | 42,123,456 | 3 |
| 4–6 | CIFAR-10 | α=0.03 | EfficientNet-BN | FedProx | 42,123,456 | 3 |
| 7–9 | CIFAR-10 | α=0.03 | EfficientNet-GN | FedAvg | 42,123,456 | 3 |
| 10–12 | CIFAR-10 | α=0.03 | EfficientNet-LN | FedAvg | 42,123,456 | 3 |
| 13–15 | CIFAR-10 | α=0.03 | ViT-Tiny | FedAvg | 42,123,456 | 3 |
| 16–18 | CIFAR-10 | α=0.03 | ViT-Tiny | FedProx | 42,123,456 | 3 |
| 19–21 | CIFAR-10 | α=0.03 | Vim-tiny | FedAvg | 42,123,456 | 3 |
| 22–24 | CIFAR-10 | α=0.03 | Vim-tiny | FedProx | 42,123,456 | 3 |
| ... | *(repeat for α=0.3, 1.0, 1000)* | | | | | 72 more |
| ... | *(repeat applicable settings for Brain Tumor MRI)* | | | | | 96 more |
| **Total** | | | | | | **192** |

> Note: EfficientNet-GN and EfficientNet-LN are run under FedAvg only for the normalization ablation (Ablation 1). If resources allow, run FedProx for these variants too. The Brain Tumor MRI smoke test (item 4.7) should use `num_classes=4`.

---

*Document version: v1.0 — Luis Fernando Méndez Lázaro — UTEC 2026*