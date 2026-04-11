# Reframing Strategy 1: Connecting Architectural Analysis to Federated Learning

## Why Your Advisor's Concern Is Valid

As originally framed, "analyze weight divergence per layer type" is a **general deep learning analysis technique**. You could do the same thing in centralized training. If you just measure which layers change most during training, you're doing model interpretability — not federated learning research.

The question your advisor is really asking is:

> *"What phenomena are you studying that **only exist because of federation**, and how does architecture interact with those phenomena?"*

If the answer is "none," then your architectural analysis belongs in a centralized training paper, not an FL thesis.

---

## The FL-Specific Phenomena You Must Target

Federation introduces dynamics that **do not exist in centralized training**. Your architectural analysis must be anchored to these:

### 1. Client Drift

In FL, each client performs E local epochs of SGD before sending updates back. During these local steps, each client's model **drifts away** from the global model in a direction determined by its local (non-IID) data. This is called *client drift* (Karimireddy et al., 2020 — SCAFFOLD paper).

**The FL-specific question:**
> *"Do different architectures experience different magnitudes and patterns of client drift under the same non-IID partition? If so, which architectural properties explain this?"*

**Why this is FL-specific:** Client drift doesn't exist in centralized training. It emerges from the combination of (local training) + (heterogeneous data) + (periodic aggregation). You're studying how architecture *modulates* this FL-specific phenomenon.

**Concrete measurement:**
```
drift_k^(t) = || w_k^(t, E) - w_global^(t) ||₂

where:
  w_k^(t, E)     = client k's weights after E local epochs in round t
  w_global^(t)    = global model at the start of round t
```

Decompose this per layer type and compare across architectures.

---

### 2. Aggregation Interference (Destructive Averaging)

When FedAvg averages weights from clients with different data distributions, it assumes the weight updates are roughly compatible. But in non-IID settings, clients may learn **contradictory features** — and averaging them can **destroy** useful information.

**The FL-specific question:**
> *"When client models are aggregated, which architectural components suffer the most from destructive averaging, and does this differ by architecture type?"*

**Why this is FL-specific:** In centralized training, there is no aggregation step. Destructive averaging is a direct consequence of the federated aggregation protocol.

**Concrete measurement:**
```
# Performance drop from aggregation
Δ_agg^(t) = Acc(w_global^(t+1)) - (1/K) Σ Acc(w_k^(t, E))

# i.e., global model accuracy AFTER aggregation vs. average of
# client accuracies BEFORE aggregation
```

If Δ_agg is consistently more negative for one architecture, aggregation is more destructive for that architecture.

You can also measure this at the **layer level** by doing partial aggregation experiments: aggregate only certain layers and keep others local, then measure performance.

---

### 3. Non-IID Sensitivity of Normalization Statistics

BatchNorm maintains running mean and variance statistics. In FL, these statistics are computed **locally on each client's non-IID data**, making them inconsistent across clients and with the global model.

**The FL-specific question:**
> *"How much of EfficientNet's performance degradation under low α is attributable to BatchNorm statistic divergence (an FL-specific artifact) rather than the convolutional architecture itself?"*

**Why this is FL-specific:** In centralized training, all data passes through the same BN statistics. The divergence only occurs because data is partitioned across clients.

**Concrete measurement:**
```
BN_divergence^(t) = (1/K) Σ_k [ || μ_k - μ_global ||₂ + || σ²_k - σ²_global ||₂ ]
```

**Control experiment:** Replace BN with GroupNorm in EfficientNet and re-run. The delta tells you exactly how much of the FL performance gap is due to normalization vs. architecture.

---

### 4. Convergence Stability Under Periodic Aggregation

FL's "train locally → aggregate → redistribute" cycle creates a fundamentally different optimization trajectory than centralized SGD. The global model follows a zigzag path — it moves toward each client's optimum during local training, then gets pulled back during aggregation.

**The FL-specific question:**
> *"Which architectures converge more smoothly under periodic aggregation, and does the strength of inductive bias predict convergence stability in FL?"*

**Concrete measurement:**
```
# Round-to-round accuracy volatility
volatility = std([ Acc^(t+1) - Acc^(t) for t in rounds ])

# Convergence efficiency (rounds to reach X% of final accuracy)
rounds_to_90pct = min { t : Acc^(t) ≥ 0.9 * Acc^(T) }
```

Compare these across architectures and α values.

---

### 5. Client-Level Fairness Under Aggregation

In FL, we care about **all clients benefiting** from federation — not just global average performance. The global model might achieve high average accuracy while performing terribly on minority clients.

**The FL-specific question:**
> *"Do certain architectures produce more equitable performance across clients under non-IID conditions, and does this interact with the aggregation algorithm (FedAvg vs. FedProx)?"*

**Concrete measurement:**
```
# Fairness metric: variance of per-client accuracy
fairness_gap = max_k(Acc_k) - min_k(Acc_k)

# Or Jain's fairness index
J = (Σ Acc_k)² / (K * Σ Acc_k²)
```

---

## The Reframed Research Narrative

### Before (weak framing — your advisor's concern)

> "We analyze how different architectures behave in federated learning by measuring weight divergence per layer."

❌ This could be done in centralized training. It's model analysis, not FL research.

### After (strong framing — FL-grounded)

> "We investigate how architectural inductive biases modulate **client drift**, **aggregation interference**, and **convergence stability** — phenomena that are unique to the federated optimization setting. Through per-layer decomposition of client drift magnitude, we identify which architectural components are most sensitive to data heterogeneity during local training and most susceptible to destructive averaging during aggregation. This analysis yields architecture-specific insights for improving federated aggregation in medical imaging deployments."

✅ Every analytical dimension is anchored to a phenomenon that **only exists because of federation**.

---

## Revised Research Questions

Replace your current RQs with ones that are explicitly FL-grounded:

| # | Research Question | FL Phenomenon |
|---|---|---|
| RQ1 | How does the magnitude and distribution of **client drift** vary across CNN, ViT, and SSM architectures under increasing data heterogeneity (α ∈ {0.1, 0.5, 1.0})? | Client drift |
| RQ2 | Which architectural components (convolutional layers, attention heads, SSM recurrence parameters) are most susceptible to **destructive averaging** during federated aggregation? | Aggregation interference |
| RQ3 | To what extent does EfficientNet's FL performance degradation under high heterogeneity stem from **BatchNorm statistic divergence** rather than inherent architectural limitations? | Non-IID normalization |
| RQ4 | Do architectures with weaker inductive biases (DeiT, Vision Mamba) produce less **equitable** performance across clients compared to EfficientNet? | Client fairness |
| RQ5 | Can architecture-specific insights from RQ1–RQ4 inform a **selective aggregation strategy** that outperforms uniform FedAvg/FedProx? | Aggregation design |

> [!IMPORTANT]
> Notice how **every RQ references a federated-specific concept** (client drift, aggregation, BatchNorm divergence, client fairness, aggregation strategy). This is the connection your advisor is looking for.

---

## How This Maps to Your 13-Week Timeline

| Weeks | Phase | Activities |
|---|---|---|
| 1–2 | **Instrumentation** | Add logging for client drift, BN statistics, per-client metrics to your FL pipeline. Set up GroupNorm variant of EfficientNet. |
| 3–5 | **Baseline Experiments** | Run full experiment matrix (3 arch × 2 algo × 3 α × 3 seeds). Collect all metrics. |
| 6–7 | **Control Experiment** | Run EfficientNet-GroupNorm variant. Compare against EfficientNet-BN to isolate normalization effect. |
| 7–8 | **Analysis** | Compute client drift decomposition, aggregation interference, fairness metrics. Generate visualizations. |
| 9 | **Synthesis** | Build decision framework/guidelines from findings. |
| 10–12 | **Writing** | Draft all thesis chapters. |
| 13 | **Defense Prep** | Practice presentation, prepare for Q&A. |

> [!WARNING]
> This timeline is tight. Weeks 3–5 depend heavily on your compute infrastructure. If experiments take longer than expected, the buffer comes from the analysis phase (which can overlap with late experiments) and writing (which you should start incrementally from week 6).

---

## Anticipated Advisor Follow-Up

Your advisor may next ask: **"What is your contribution — analysis or method?"**

Be prepared with a clear answer. Based on your timeline, I recommend:

**Primary contribution:** An empirical analysis that explains *how and why* architectural inductive biases interact with FL-specific phenomena (client drift, aggregation interference) under data heterogeneity.

**Secondary contribution:** A practical decision framework for selecting vision architectures in federated medical imaging deployments.

**Optional tertiary contribution** (if time permits): A selective aggregation strategy informed by the layer-wise analysis (e.g., "aggregate attention layers globally but keep patch embeddings personalized").

This positions your thesis as an **analytical contribution with practical implications**, which is a well-accepted thesis format.
