Here's a summary of recent research on **client drift in federated learning**, organized by theme:

---

### 🔍 What is Client Drift?

Client drift is the well-known phenomenon arising from statistical mismatch in non-IID (non-independent and identically distributed) data across clients, where locally trained models diverge from one another and from the global optimum — leading to slower convergence, degraded global accuracy, and unstable training dynamics.

---

### 📄 Key Recent Papers

**1. FedBSS — Sample-level Client Drift Mitigation (AAAI 2025)**
*Xu et al., Jan 2025 · arXiv:2501.11360*

This paper identifies that client drift can be viewed as a cumulative manifestation of biases present in all local samples, where the bias between samples differs and dynamically changes as FL training progresses. The authors propose **FedBSS**, which mitigates the heterogeneity issue at the sample level — a perspective orthogonal to existing update-calibration methods.

---

**2. DRDM — Distributionally Robust Federated Learning with Client Drift Minimization (May 2025)**
*Krouka et al. · arXiv:2505.15371*

This work introduces **DRDM**, which combines a distributionally robust optimization (DRO) framework with dynamic regularization to mitigate client drift. It frames training as a min-max optimization problem aimed at maximizing performance for the worst-case client, promoting both robustness and fairness, while significantly reducing communication rounds.

---

**3. FIELDING — Clustered FL with Adaptation to Data Drifts (Nov 2024 / updated Feb 2026)**
*Li et al. · arXiv:2411.01580*

**FIELDING** is a clustered FL framework for handling diverse types of data drift with low overhead. It detects drift at individual clients and performs selective re-clustering to balance cluster quality and model performance, while remaining robust to malicious clients and varying levels of heterogeneity. Experiments show it improves final model accuracy by 1.9–5.9% and achieves target accuracy 1.16x–2.23x faster than state-of-the-art methods.

---

**4. Adaptive Self-Distillation (ASD) for Minimizing Client Drift (updated Dec 2025)**
*Yashwanth et al. · arXiv:2305.19600*

This paper proposes a novel regularization technique based on **adaptive self-distillation (ASD)** for training models on the client side, targeting the client-drift problem where every client drifts to its own local optimum under non-IID label distributions — resulting in slower convergence and poor aggregated model performance.

---

**5. Systematic Survey: FL Under Concept Drift (Nov 2025)**
*MDPI Electronics*

This survey specifically addresses continuous adaptation within federated systems when faced with non-stationary data, introducing and analyzing **Federated Drift-Aware Learning (FDAL)** — a structured taxonomy of approaches that bridges the gap between FL and concept drift, an area it identifies as relatively underexplored. It follows PRISMA guidelines and covers both local and global manifestations of drift.

---

### 🧩 Common Mitigation Strategies in the Literature

Existing approaches to mitigate client drift broadly fall into: (1) methods using the difference between local and global updates across rounds to compensate local updates (e.g., SCAFFOLD, FedDC); (2) regularization on the local loss function to push local updates toward the global model (e.g., FedProx); and (3) selective aggregation strategies on the server side.

---

Would you like deeper dives into any specific paper, or a comparison of mitigation strategies like SCAFFOLD vs. FedProx vs. FedBSS?