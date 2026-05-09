import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any

def classify_layer(name: str, module: nn.Module) -> str:
    """Return the layer group for a named module."""
    if "classifier" in name or "head" in name:
        return "head"
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        return "norm"
    elif isinstance(module, (nn.Conv2d, nn.Linear)):
        return "feature"
    return "other"  # pooling, activation, etc. — excluded from drift metrics

def compute_layer_drift(
    local_state: Dict[str, torch.Tensor],
    global_state: Dict[str, torch.Tensor],
    layer_taxonomy: Dict[str, str]
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
        if name not in global_state:
            continue
        diff = (local_param.float() - global_state[name].float())
        group_diffs[group].append(diff.norm(p=2).item())
    
    return {g: float(np.mean(v)) if v else 0.0 for g, v in group_diffs.items()}

def aggregate_drift(per_client_drifts: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Returns mean and std of drift across clients, per layer group."""
    groups = ["norm", "feature", "head"]
    results = {}
    for g in groups:
        drifts = [d[g] for d in per_client_drifts]
        results[g] = {
            "mean": float(np.mean(drifts)),
            "std":  float(np.std(drifts))
        }
    return results

def compute_gradient_alignment(
    client_grads: List[Dict[str, torch.Tensor]],
    layer_taxonomy: Dict[str, str]
) -> Dict[str, float]:
    """
    client_grads: list of {param_name: grad_tensor} per client
    Returns mean pairwise cosine similarity per layer group.
    """
    K = len(client_grads)
    if K < 2:
        return {"norm": 0.0, "feature": 0.0, "head": 0.0}
        
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
            results[g] = 0.0
            continue
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                cos = torch.nn.functional.cosine_similarity(
                    vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)
                ).item()
                sims.append(cos)
        results[g] = float(np.mean(sims)) if sims else 0.0
    return results
