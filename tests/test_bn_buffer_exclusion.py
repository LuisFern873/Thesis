"""
tests/test_bn_buffer_exclusion.py
==================================
Validates that BN running-statistics buffers are excluded from drift and
cosine-similarity metric computations while remaining present in the model's
state_dict (so FedAvg aggregation is unaffected).

Run with:
    python -m pytest tests/test_bn_buffer_exclusion.py -v
"""

import sys
from pathlib import Path

# Allow imports from the repo root regardless of where pytest is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from src.utils.drift_metrics import (
    BN_BUFFER_KEYWORDS,
    classify_layer,
    compute_gradient_alignment,
    compute_layer_drift,
    get_norm_layer_metric_keys,
    is_bn_buffer,
)


# ---------------------------------------------------------------------------
# Minimal model fixtures
# ---------------------------------------------------------------------------

class _TinyBNModel(nn.Module):
    """Two-layer model with BatchNorm — mimics EfficientNet's norm layers."""

    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(8, 4)

    def forward(self, x):
        x = self.base(x).mean(dim=[2, 3])
        return self.classifier(x)


class _TinyLNModel(nn.Module):
    """Two-layer model with LayerNorm — mimics ViT's norm layers."""

    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(16, 8),
            nn.LayerNorm(8),
        )
        self.classifier = nn.Linear(8, 4)

    def forward(self, x):
        return self.classifier(self.base(x))


def _build_taxonomy(model: nn.Module) -> dict:
    taxonomy = {}
    for name, _ in model.named_parameters():
        parts = name.split(".")
        module_name = ".".join(parts[:-1])
        try:
            module = model.get_submodule(module_name) if module_name else model
        except AttributeError:
            module = model
        taxonomy[name] = classify_layer(name, module)
    return taxonomy


# ---------------------------------------------------------------------------
# Task 4 — Unit tests
# ---------------------------------------------------------------------------

def test_is_bn_buffer_detects_all_keywords():
    """is_bn_buffer must flag every BN buffer keyword."""
    assert is_bn_buffer("base.1.running_mean")
    assert is_bn_buffer("base.1.running_var")
    assert is_bn_buffer("base.1.num_batches_tracked")
    assert not is_bn_buffer("base.1.weight")
    assert not is_bn_buffer("base.1.bias")
    assert not is_bn_buffer("classifier.weight")


def test_norm_metric_group_excludes_bn_buffers():
    """Metric computation must use parameters only — BN buffers must not appear."""
    model = _TinyBNModel()
    taxonomy = _build_taxonomy(model)
    norm_metric_keys = get_norm_layer_metric_keys(model, taxonomy)

    forbidden = list(BN_BUFFER_KEYWORDS)
    for key in norm_metric_keys:
        assert not any(f in key for f in forbidden), (
            f"Buffer '{key}' found in norm metric group. "
            f"Buffers must be excluded from drift/interference computation. "
            f"They are still aggregated in FedAvg via state_dict."
        )


def test_aggregation_includes_bn_buffers():
    """Aggregation must still include BN buffers — do not exclude from state_dict."""
    model = _TinyBNModel()
    state_keys = set(model.state_dict().keys())
    assert any("running_mean" in k for k in state_keys), (
        "running_mean not found in state_dict — aggregation may have been incorrectly modified."
    )
    assert any("running_var" in k for k in state_keys), (
        "running_var not found in state_dict — aggregation may have been incorrectly modified."
    )
    assert any("num_batches_tracked" in k for k in state_keys), (
        "num_batches_tracked not found in state_dict — aggregation may have been incorrectly modified."
    )


def test_compute_layer_drift_excludes_bn_buffers():
    """compute_layer_drift must ignore BN buffer keys even if injected into states."""
    model = _TinyBNModel()
    taxonomy = _build_taxonomy(model)

    # Build states from named_parameters (normal path)
    params = {k: v.clone() for k, v in model.named_parameters()}

    # Inject synthetic BN buffers to simulate a state_dict-based state
    injected_local = dict(params)
    injected_global = dict(params)
    injected_local["base.1.running_mean"] = torch.zeros(8) + 999.0
    injected_local["base.1.running_var"] = torch.ones(8) * 999.0
    injected_local["base.1.num_batches_tracked"] = torch.tensor(999)
    injected_global["base.1.running_mean"] = torch.zeros(8)
    injected_global["base.1.running_var"] = torch.ones(8)
    injected_global["base.1.num_batches_tracked"] = torch.tensor(0)

    # Drift computed without injected buffers (baseline)
    drift_clean = compute_layer_drift(params, params, taxonomy)

    # Drift computed with injected buffers — must be identical to clean
    drift_injected = compute_layer_drift(injected_local, injected_global, taxonomy)

    # The injected buffers have huge values (999) but must not affect norm drift
    # because is_bn_buffer() filters them out.
    assert drift_injected["norm"] == drift_clean["norm"], (
        f"BN buffers contaminated norm drift: "
        f"clean={drift_clean['norm']:.4f}, injected={drift_injected['norm']:.4f}"
    )


def test_compute_gradient_alignment_excludes_bn_buffers():
    """compute_gradient_alignment must ignore BN buffer keys even if injected."""
    model = _TinyBNModel()
    taxonomy = _build_taxonomy(model)

    params = {k: v.clone() for k, v in model.named_parameters()}

    # Two identical pseudo-gradient dicts (cosine sim should be 1.0 for all groups)
    grads_clean = [dict(params), dict(params)]

    # Inject divergent BN buffers into one client's grad dict
    grads_injected = [dict(params), dict(params)]
    grads_injected[0]["base.1.running_mean"] = torch.zeros(8)
    grads_injected[1]["base.1.running_mean"] = torch.ones(8) * 100.0  # very different

    result_clean = compute_gradient_alignment(grads_clean, taxonomy)
    result_injected = compute_gradient_alignment(grads_injected, taxonomy)

    # Injected buffers must not change the norm cosine similarity
    assert abs(result_injected["norm"] - result_clean["norm"]) < 1e-6, (
        f"BN buffers contaminated norm cosine similarity: "
        f"clean={result_clean['norm']:.6f}, injected={result_injected['norm']:.6f}"
    )


def test_ln_model_norm_keys_have_no_bn_buffers():
    """LayerNorm models have no BN buffers — norm keys should be weight/bias only."""
    model = _TinyLNModel()
    taxonomy = _build_taxonomy(model)
    norm_keys = get_norm_layer_metric_keys(model, taxonomy)

    # LayerNorm has weight and bias — both are parameters, no buffers
    assert len(norm_keys) > 0, "Expected at least one norm-group key for LayerNorm model"
    for key in norm_keys:
        assert not is_bn_buffer(key), f"Unexpected buffer key in LN model: {key}"


def test_param_taxonomy_built_from_named_parameters_only():
    """The taxonomy must be built from named_parameters(), not state_dict().

    This ensures BN buffers are never assigned a group label and therefore
    can never appear in metric vectors via the taxonomy lookup path.
    """
    model = _TinyBNModel()
    taxonomy = _build_taxonomy(model)

    param_keys = set(k for k, _ in model.named_parameters())
    state_keys = set(model.state_dict().keys())
    buffer_keys = state_keys - param_keys  # keys that are buffers only

    for buf_key in buffer_keys:
        assert buf_key not in taxonomy, (
            f"Buffer key '{buf_key}' found in param_taxonomy. "
            f"Taxonomy must be built from named_parameters() only."
        )
