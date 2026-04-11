# Strengthening Your Thesis Beyond a Comparative Study

## The Core Problem

A comparative study answers: **"Which model performs better?"**
A thesis must answer: **"Why does it perform better, and what can we do about it?"**

Your advisor's concern is that running three architectures × two FL algorithms × three α values and reporting accuracy/F1 tables is a **replication exercise**, not a **contribution to knowledge**. The tables are necessary but not sufficient. You need to move from *description* to *explanation* and ideally to *prescription*.

Below are seven concrete strategies, ordered from most impactful to most incremental. **You do not need to adopt all of them** — pick 1–2 that align with your timeline and technical comfort, and execute them well.

---

## Strategy 1: Explain *Why* — Architectural Decomposition Analysis

**The idea:** Don't just report that Architecture X outperforms Y under α=0.1. Explain *which architectural property* causes this. This transforms your thesis from a benchmark into an empirical analysis paper.

### Concrete steps

1. **Measure weight divergence per layer type.** After each FL round, compute the L2 distance between each client's local model and the global model, *grouped by layer type*:
   - CNN: convolutional layers vs. batch norm vs. classifier head
   - DeiT: patch embedding vs. self-attention (Q/K/V) vs. MLP blocks vs. [CLS] token
   - Vision Mamba: SSM recurrence parameters (A, B, C, Δ) vs. linear projections vs. classifier

2. **Measure gradient divergence across clients.** Compute the cosine similarity between client gradients per round. Plot how this divergence evolves and differs by architecture. CNN local inductive biases (translation equivariance) may constrain gradient directions more than ViT/Mamba, leading to lower divergence.

3. **Analyze feature representation drift.** Use CKA (Centered Kernel Alignment) to compare intermediate representations between clients. Which architecture maintains more similar internal representations under non-IID splits?

### Example research question
> *"Do the global attention mechanisms in DeiT lead to greater weight divergence under non-IID conditions compared to the locality-constrained convolutions in EfficientNet, and does this explain their relative convergence behavior?"*

### Why this is a contribution
You would be providing **mechanistic evidence** for why certain architectures are more or less suited to FL. This is generalizable beyond your specific dataset.

> [!TIP]
> This is probably the **highest-impact, most feasible** extension. It requires no new models or algorithms — just additional logging and analysis of experiments you're already running.

---

## Strategy 2: Propose an Architecture-Aware Aggregation Strategy

**The idea:** If your analysis from Strategy 1 reveals that certain layer types diverge more under non-IID conditions, propose a **layer-wise or module-wise aggregation** strategy that treats different architectural components differently.

### Concrete steps

1. Based on your divergence analysis, identify which layers/modules are most affected by data heterogeneity.
2. Propose a modified aggregation scheme. For example:
   - **Selective aggregation:** Only aggregate the classifier head globally; keep feature extractors personalized.
   - **Weighted layer-wise aggregation:** Apply stronger regularization (higher μ in FedProx) to layers with high divergence.
   - **Hybrid freezing:** Freeze early layers (which may capture universal features) and only federate later layers.

3. Compare your architecture-aware aggregation against vanilla FedAvg/FedProx.

### Example formulation

```
For DeiT: w_global^(t+1) = Σ (n_k/n) * w_k^(t) for attention layers
          w_local^(t+1) = w_k^(t) for patch embedding (personalized)

For EfficientNet: w_global^(t+1) = Σ (n_k/n) * w_k^(t) for all layers
                  (full aggregation, since conv layers already have strong inductive bias)
```

### Why this is a contribution
You move from *"Architecture X is better"* to *"Here is a concrete method that exploits architectural differences to improve FL performance."* This is a **prescriptive** contribution.

> [!WARNING]
> This is more ambitious and requires careful experimental validation. Only pursue this if your timeline allows for an additional round of experiments after your baseline comparisons.

---

## Strategy 3: Communication Efficiency Analysis

**The idea:** In real federated deployments (hospitals, clinics), communication bandwidth is a critical constraint. Different architectures have very different parameter counts and structures, which affects how efficiently they can be communicated and compressed.

### Concrete steps

1. **Measure communication cost per round.** Total bytes transmitted = 2 × model_size × num_clients per round (upload + download). Track cumulative communication cost to reach a target accuracy.

2. **Apply gradient/model compression.** Test techniques like:
   - Top-k sparsification (only send the k largest parameter updates)
   - Quantization (reduce parameter precision from FP32 to INT8)
   - Low-rank factorization of weight updates

3. **Analyze which architecture degrades least under compression.** Hypothesis: CNN weights may be more compressible due to spatial locality patterns, while ViT attention matrices may be more sensitive to quantization.

4. **Plot Pareto frontiers:** accuracy vs. total communication cost for each architecture.

### Example research question
> *"Given a fixed communication budget, which vision architecture achieves the highest diagnostic accuracy in a federated medical imaging setting?"*

### Why this is a contribution
This adds a **practical, deployment-relevant dimension** that pure accuracy comparisons miss. It speaks directly to real-world FL constraints in healthcare settings where bandwidth between hospitals may be limited.

---

## Strategy 4: Privacy Leakage Analysis Across Architectures

**The idea:** One underexplored question is whether architectural choice affects the **privacy risk** in FL. Gradient inversion attacks reconstruct training data from shared gradients. Different architectures may leak differently.

### Concrete steps

1. Implement a gradient inversion attack (e.g., the method from Geiping et al., 2020 — "Inverting Gradients") against each architecture.
2. Measure reconstruction quality (PSNR, SSIM, LPIPS) of recovered images from shared gradients.
3. Analyze whether ViT's global attention patterns leak more spatial information than CNN's local receptive fields.
4. Test whether Vision Mamba's sequential state-space processing offers any inherent privacy advantage.

### Example research question
> *"Does the choice of vision backbone architecture in federated learning affect the vulnerability of medical images to gradient inversion attacks?"*

### Why this is a contribution
This is a **novel intersection** — FL privacy × architecture choice × medical imaging — with very little prior work. Even negative results (all architectures leak equally) would be informative.

> [!CAUTION]
> This is a substantial extension that could become its own paper. Only pursue if you have ≥3 months of runway remaining and are comfortable with adversarial ML techniques.

---

## Strategy 5: Cross-Architecture Federated Knowledge Distillation

**The idea:** Instead of assuming all clients use the same architecture, explore **heterogeneous FL** where different clients use different architectures (e.g., a resource-constrained clinic uses EfficientNet, while a research hospital uses DeiT).

### Concrete steps

1. Assign different architectures to different clients (e.g., 3 clients with EfficientNet, 3 with DeiT, 3 with Vision Mamba).
2. Since model weights are incompatible, use **federated knowledge distillation**:
   - Share a public unlabeled dataset (or use a generative model to create one).
   - Each client produces soft predictions (logits) on the public dataset.
   - Aggregate soft predictions to create a consensus, then each client distills from it.
3. Compare against homogeneous FL baselines.

### Relevant methods to build on
- FedMD (Li & Wang, 2019)
- FedDF (Lin et al., 2020)
- KT-pFL (Zhang et al., 2021)

### Why this is a contribution
This is a **realistic deployment scenario** — hospitals don't all use the same hardware or models. It also opens up the question: *"Does architectural diversity in the federation help or hurt?"*

---

## Strategy 6: Develop a Predictive Framework

**The idea:** Abstract your empirical findings into a **framework or taxonomy** that predicts FL performance from architectural properties, without needing to re-run all experiments.

### Concrete steps

1. Define measurable architectural properties:
   - **Inductive bias strength:** locality (CNN) > sequential (Mamba) > global (ViT)
   - **Parameter density:** parameters per layer, distribution of parameters across layer types
   - **Receptive field progression:** how quickly each architecture integrates global context
   - **Normalization type:** BatchNorm (problematic in FL due to non-IID stats) vs. LayerNorm

2. Correlate these properties with your observed FL metrics (convergence speed, final accuracy, client fairness).
3. Present a **decision matrix** or **flowchart** for practitioners: *"Given your heterogeneity level and compute constraints, use Architecture X with Algorithm Y."*

### Example output

| Heterogeneity | Compute Budget | Recommended Architecture | FL Algorithm |
|---|---|---|---|
| Low (α ≥ 1.0) | Low | EfficientNet-B0 | FedAvg |
| Low (α ≥ 1.0) | High | DeiT-Small | FedAvg |
| High (α ≤ 0.1) | Low | EfficientNet-B0 | FedProx |
| High (α ≤ 0.1) | High | Vision Mamba | FedProx + selective agg. |

### Why this is a contribution
This transforms empirical results into **actionable guidelines** for the FL + medical imaging community. It's what elevates a benchmark paper into a reference paper.

> [!TIP]
> This strategy is **complementary to all others** and requires no additional experiments — just deeper analysis and synthesis of your existing results. Consider this a minimum addition regardless of which other strategy you choose.

---

## Strategy 7: BatchNorm vs. LayerNorm Investigation

**The idea:** A known but underexplored issue in FL is that **Batch Normalization performs poorly under non-IID conditions** because local batch statistics diverge across clients. EfficientNet uses BatchNorm; DeiT and Mamba use LayerNorm. This confound could explain a significant portion of your results.

### Concrete steps

1. **Control for normalization type.** Replace BatchNorm in EfficientNet with GroupNorm or LayerNorm and re-run experiments. This isolates the impact of normalization from the impact of the convolutional architecture itself.
2. **Track BN running statistics divergence** across clients as a function of α.
3. **Test FedBN** (Li et al., 2021) — a method that keeps BatchNorm statistics local while aggregating other parameters.

### Why this is a contribution
If you find that EfficientNet's FL degradation under low α is primarily due to BatchNorm (and disappears with GroupNorm), this is a **critical finding** that reframes the architecture comparison. Without this control, your benchmark has a serious confound.

> [!IMPORTANT]
> This is not optional — it is a **methodological necessity**. If your thesis compares architectures and one uses BatchNorm while the others use LayerNorm, any reviewer will immediately flag this as a confound. At minimum, you must discuss it. Ideally, you control for it experimentally.

---

## Recommended Combination

For a strong thesis within a feasible timeline, I recommend combining:

| Priority | Strategy | Effort | Impact |
|---|---|---|---|
| **Must do** | **Strategy 7** (BatchNorm control) | Low | Fixes a critical confound |
| **Must do** | **Strategy 1** (Divergence analysis) | Medium | Transforms benchmark into analysis |
| **Must do** | **Strategy 6** (Predictive framework) | Low | Synthesizes findings into guidelines |
| **Should do** | **Strategy 3** (Communication efficiency) | Medium | Adds practical dimension |
| **Nice to have** | **Strategy 2** (Architecture-aware aggregation) | High | Novel method contribution |

This combination gives you:
1. A **methodologically sound** comparison (Strategy 7)
2. An **explanatory** contribution — *why* architectures differ (Strategy 1)
3. A **practical** contribution — what to use when (Strategy 6)
4. A **deployment-relevant** dimension (Strategy 3)
5. Optionally, a **methodological** contribution (Strategy 2)

---

## How This Changes Your Thesis Structure

Your thesis narrative would shift from:

> ❌ "We compared three architectures and found X was best."

To:

> ✅ "We identify how architectural inductive biases interact with data heterogeneity in federated medical imaging. Through layer-wise divergence analysis, we show that [finding]. Based on this insight, we propose [recommendation/method] and validate it across [conditions]. Our framework provides practitioners with actionable guidance for selecting vision architectures in federated healthcare deployments."

---

## Open Questions for You

1. **Timeline:** How many months do you have before your thesis defense? This determines how ambitious you can be.
2. **Compute budget:** Do you have GPU access sufficient to run additional experiment variants (e.g., EfficientNet with GroupNorm)?
3. **Advisor preference:** Does your advisor lean toward a *methods contribution* (propose a new algorithm) or an *analytical contribution* (deep empirical analysis with insights)?
4. **Have you started running experiments yet?** If not, it's easy to add logging for divergence metrics from the start. If yes, you may need to re-run with additional instrumentation.
