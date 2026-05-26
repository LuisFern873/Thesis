import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# BN buffer guard
# ---------------------------------------------------------------------------
# BatchNorm layers store three non-learnable buffers in state_dict() that must
# NEVER appear in drift or cosine-similarity computations:
#
#   running_mean        — EMA of batch means, tracks local data distribution
#   running_var         — EMA of batch variances, tracks local data distribution
#   num_batches_tracked — monotonically increasing forward-pass counter
#
# These buffers are driven by the local data distribution, not by the loss
# landscape.  Including them in metric vectors produces:
#   • constant cosine similarity (data distribution is fixed across rounds)
#   • large, flat L2 drift (steady-state divergence from global running stats)
#
# The fix: filter them out of every parameter vector used for metrics.
# They continue to flow through FedAvg aggregation via state_dict() unchanged.
# ---------------------------------------------------------------------------
BN_BUFFER_KEYWORDS: tuple[str, ...] = (
    "running_mean",
    "running_var",
    "num_batches_tracked",
)


def is_bn_buffer(key: str) -> bool:
    """Return True if *key* names a BatchNorm running-statistics buffer."""
    return any(kw in key for kw in BN_BUFFER_KEYWORDS)


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

    BN buffers (running_mean, running_var, num_batches_tracked) are explicitly
    excluded even if they appear in local_state or global_state, because they
    reflect local data statistics rather than gradient-driven optimisation and
    would produce artificially large, flat drift values.
    """
    group_sq_sums = {"norm": 0.0, "feature": 0.0, "head": 0.0}
    group_has_params = {"norm": False, "feature": False, "head": False}
    
    for name, local_param in local_state.items():
        # Explicitly exclude BN running-statistics buffers from metric vectors.
        if is_bn_buffer(name):
            continue

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

def get_norm_layer_metric_keys(
    model: nn.Module,
    layer_taxonomy: Dict[str, str] = None,
) -> List[str]:
    """Return the parameter keys that would be used for the *norm* group in
    drift / cosine-similarity computations.

    This is a test-support helper: it returns only keys from
    ``model.named_parameters()`` that are classified as ``"norm"`` and are
    not BN running-statistics buffers.  It is intentionally conservative —
    if a key passes this filter it will appear in the metric vectors.

    Args:
        model: The model to inspect.
        layer_taxonomy: Optional pre-built taxonomy dict.  If None, one is
            constructed on the fly from ``model.named_parameters()``.

    Returns:
        Sorted list of parameter key strings in the ``norm`` group.
    """
    if layer_taxonomy is None:
        layer_taxonomy = {}
        for name, _ in model.named_parameters():
            parts = name.split(".")
            module_name = ".".join(parts[:-1])
            try:
                module = model.get_submodule(module_name) if module_name else model
            except AttributeError:
                module = model
            layer_taxonomy[name] = classify_layer(name, module)

    return sorted(
        name
        for name, group in layer_taxonomy.items()
        if group == "norm" and not is_bn_buffer(name)
    )


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

    BN buffers (running_mean, running_var, num_batches_tracked) are explicitly
    excluded even if they appear in client_grads, because they reflect local
    data statistics rather than gradient-driven optimisation and would produce
    artificially constant cosine similarity values.
    """
    K = len(client_grads)
    if K < 2:
        return {"norm": 0.0, "feature": 0.0, "head": 0.0}
        
    groups = ["norm", "feature", "head"]
    group_vecs = {g: [] for g in groups}

    # Establish a reference key ordering from the first client to guarantee alignment.
    # Exclude BN buffers from the reference set so they never enter the vectors.
    ref_keys = [k for k in client_grads[0].keys() if not is_bn_buffer(k)]

    for grads in client_grads:
        group_flat = {g: [] for g in groups}
        for name in ref_keys:
            if name not in grads:
                continue
            # Redundant guard: skip BN buffers even if they slipped past ref_keys.
            if is_bn_buffer(name):
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
