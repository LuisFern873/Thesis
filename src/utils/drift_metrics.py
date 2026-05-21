import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any

def classify_layer(name: str, module: nn.Module) -> str:
    """Return the layer group for a named module."""
    if "classifier" in name or "head" in name:
        return "head"
        
    class_name = module.__class__.__name__.lower()
    
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)) or "norm" in class_name or "norm" in name:
        return "norm"
    elif isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
        return "feature"
    return "other"  # pooling, activation, etc. — excluded from drift metrics

def compute_layer_drift(
    local_state: Dict[str, torch.Tensor],
    global_state: Dict[str, torch.Tensor],
    layer_taxonomy: Dict[str, str]
) -> Dict[str, float]:
    """
    Returns the true L2 drift (Euclidean distance of the concatenated vector) 
    per layer group for a single client, resolving device and precision mismatches.
    layer_taxonomy: {param_name: group_label}
    """
    group_sq_sums = {"norm": 0.0, "feature": 0.0, "head": 0.0}
    group_has_params = {"norm": False, "feature": False, "head": False}
    
    for name, local_param in local_state.items():
        group = layer_taxonomy.get(name, "other")
        if group == "other" or name not in global_state:
            continue
        
        # Cast to float, detach, and move to CPU to guarantee safety across devices
        local_val = local_param.detach().cpu().float()
        global_val = global_state[name].detach().cpu().float()
        
        diff = local_val - global_val
        group_sq_sums[group] += diff.square().sum().item()
        group_has_params[group] = True
        
    return {
        g: float(np.sqrt(group_sq_sums[g])) if group_has_params[g] else 0.0 
        for g in group_sq_sums
    }

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
    Returns mean pairwise cosine similarity per layer group, ensuring consistent 
    parameter alignment, device safety, and division-by-zero protection.
    client_grads: list of {param_name: grad_tensor} per client
    """
    K = len(client_grads)
    if K < 2:
        return {"norm": 0.0, "feature": 0.0, "head": 0.0}
        
    groups = ["norm", "feature", "head"]
    group_vecs = {g: [] for g in groups}

    # Establish a reference key ordering from the first client to guarantee alignment
    ref_keys = list(client_grads[0].keys())

    for grads in client_grads:
        group_flat = {g: [] for g in groups}
        for name in ref_keys:
            if name not in grads:
                continue
            grad = grads[name]
            g_label = layer_taxonomy.get(name, "other")
            if g_label != "other":
                # Move to CPU, detach, and convert to float for alignment safety
                group_flat[g_label].append(grad.detach().cpu().float().flatten())
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
                norm_i = vecs[i].norm(p=2)
                norm_j = vecs[j].norm(p=2)
                # Safe-guard against division by zero if updates are zero-vectors
                if norm_i == 0.0 or norm_j == 0.0:
                    sims.append(0.0)
                    continue
                cos = torch.nn.functional.cosine_similarity(
                    vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)
                ).item()
                sims.append(cos)
        results[g] = float(np.mean(sims)) if sims else 0.0
    return results
