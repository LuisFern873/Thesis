Here's a structured summary of the most relevant related work for your **Architectural Decomposition Analysis** section, organized by the three concrete steps in your feedback.

---

## 1. Weight Divergence Per Layer Type

**[L-DAWA](https://consensus.app/papers/details/84d03db878455624b0b91747c7aa9771/)** (Rehman et al., ICCV 2023) is directly foundational here. It measures angular divergence between client and global model weights *at the layer level* and proposes a layer-wise aggregation weighting scheme under SSL+FL. This demonstrates that different layers diverge at different rates under non-IID data — precisely the mechanistic insight your analysis aims to provide, but framed as an aggregation fix rather than an explanatory study.

**[FedLap](https://consensus.app/papers/details/f5f8337fc4cd5e16b75725ea55a4ca15/)** (Charteros et al., INFOCOM 2023) applies per-layer regularization by measuring each layer's divergence from the global model individually. Crucially, it shows that in CNN-style DNNs, some layers drift dramatically more than others — setting the stage for your hypothesis that CNN convolutional blocks (locality-constrained) may behave very differently from ViT attention blocks (globally-attending) in this regard.

**[pFedLA](https://consensus.app/papers/details/ab6c086718635ba7bf1f9997cf72326d/)** (Ma et al., CVPR 2022, 225 citations) uses a hyper-network to learn layer-wise aggregation weights, directly showing that "layer-level impacts to the aggregation process" matter significantly — and that treating all layers equally leads to "lagged model convergence."

**[Weight Divergence Divide-and-Conquer](https://consensus.app/papers/details/57e6871b987054d5a3d1eb956daef1a2/)** (Chandran et al., 2021) proposes cosine-distance weight divergence as a metric to find the natural split point between generic early layers and class-specific deep layers in CNNs. This is directly usable as a methodology citation for how you compute L2/cosine-based layer divergence.

---

## 2. Gradient Divergence Across Clients

**[FedSim](https://consensus.app/papers/details/ed73e5dd64b552d6b7218befaa782a4a/)** (Palihawadana et al., 2021, 68 citations) uses inter-client gradient similarity (cosine similarity) as the primary signal for clustering clients, confirming that gradient direction alignment is a meaningful and privacy-preserving proxy for data distribution overlap. This validates your proposed metric.

**[FedPNS](https://consensus.app/papers/details/9ad62bf51c20573fb2afcab38b7a1d47/)** (Wu et al., 2021, 161 citations) directly ties the relationship between local and global gradient alignment to convergence speed, identifying adverse local updates through gradient direction checking. Your analysis of how CNN inductive biases constrain gradient directions more than ViT/Mamba connects directly to this line of work.

**[FedConv](https://consensus.app/papers/details/321d2c336f4f50199a8d2c445403f3be/)** (Xu et al., 2023) is perhaps your most important reference for the architectural comparison angle. It systematically investigates which *architectural components* (activation functions, normalization layers) explain why ViT outperforms CNN under FL heterogeneity, concluding that with targeted modifications, CNNs can match ViTs. Your work would extend this by adding Vision Mamba and providing *gradient/weight divergence evidence* rather than just accuracy numbers.

---

## 3. Feature Representation Drift via CKA

**[FedCKA](https://consensus.app/papers/details/a84a7d9744545f7487eee8f9ee3a11d0/)** (Son et al., 2021) is a must-cite. It directly applies CKA to compare representations between clients in heterogeneous FL and finds that only *certain important layers* require regularization. This validates your CKA methodology and implies that the answer to "which architecture maintains more similar internal representations" likely involves architecture-specific layer importance patterns.

**[Gao et al.](https://consensus.app/papers/details/5d41a77e6aec5424b044f1347e768a5d/)** (2023, 4 citations) uses CKA representation similarity across layers to explain *why Transformers outperform ResNets* in large-scale heterogeneous FL. This is extremely close to your research question — use it as a direct precedent and frame your contribution as extending it to include Mamba and to provide the mechanistic layer-type breakdown.

**[Reliability of CKA](https://consensus.app/papers/details/65dc57bd0c6e5a828219dcb3f778e4af/)** (Davari et al., 2022, 50 citations) is an important methodological caveat to cite: CKA is sensitive to outliers and certain transformations. Acknowledging this in your methodology shows rigor.

**[DCFL](https://consensus.app/papers/details/f1e7d3d8327b5af389a2a482f6ec0315/)** (Wang et al., IJCNN 2024) uses CKA weight similarity for client grouping, further demonstrating that CKA captures meaningful distributional similarity in the FL context.

---

## 4. Architecture-Specific FL Behavior (CNN vs. ViT vs. Mamba)

**[Li (2025)](https://consensus.app/papers/details/0947c242bd485a0f8fd5033e9b89be31/)** directly compares ResNet50 vs. ViT under FedAvg with non-IID data, finding ViT's global attention more adaptable to heterogeneous distributions. Your work would contribute the *mechanistic explanation* for findings like these.

**[Tackling heterogeneity with ViTs](https://consensus.app/papers/details/928615c353055f9982120d8d46c255b7/)** (Darzi et al., 2024) shows that ViT's multi-head attention mechanism specifically can serve as an alignment objective in medical FL, suggesting the attention blocks are the architecturally distinguishing factor.

**[Vision Mamba](https://consensus.app/papers/details/07d10e189eae5afa8b9b3fff2b9c5024/)** (Zhu et al., 2024, 1218 citations) is the primary citation for the Vim architecture. Note that **[MambaOut](https://consensus.app/papers/details/dcfc67e48b3c5cb28d979b95f76c8c90/)** (Yu et al., CVPR 2025) raises a critical point — Mamba's SSM recurrence may not confer advantages for classification tasks (which lack autoregressive characteristics), suggesting its SSM parameters (A, B, C, Δ) may behave differently from attention weights under FL divergence analysis. This is a nuance worth addressing in your discussion.

---

## Summary Table for Your Paper

| Analysis Step | Key Papers to Cite |
|---|---|
| Layer-wise weight divergence | L-DAWA, FedLap, pFedLA, Divide-and-Conquer |
| Gradient cosine similarity | FedSim, FedPNS, FedConv |
| CKA representation drift | FedCKA, Gao et al. (2023), CKA Reliability |
| CNN vs. ViT vs. Mamba in FL | Li (2025), FedConv, Vision Mamba, MambaOut |

The strongest positioning for your paper is that **FedConv** identifies *which* architectural components matter but doesn't extend to Mamba or provide gradient/CKA-level evidence, and **Gao et al.** applies CKA to Transformers vs. ResNets but not at the intra-architecture layer-type granularity your feedback proposes. Your contribution fills both gaps simultaneously.