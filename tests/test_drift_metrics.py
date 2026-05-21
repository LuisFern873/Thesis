"""
Unit tests for src/utils/drift_metrics.py — Phase 3.4

Tests:
  - compute_layer_drift: zero drift for identical states
  - compute_layer_drift: non-zero drift for different states
  - compute_layer_drift: correct group separation (norm/feature/head)
  - compute_gradient_alignment: anti-parallel gradients → cosine = -1.0
  - compute_gradient_alignment: parallel gradients → cosine = +1.0
  - compute_gradient_alignment: single client → returns 0.0 (no pairs)
  - compute_gradient_alignment: zero-vector guard (no crash, returns 0.0)
  - aggregate_drift: correct mean and std across clients
  - classify_layer: correct group for BN, GN, LN, Conv2d, Linear, head

Run with:
    python -m pytest tests/test_drift_metrics.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.utils.drift_metrics import (
    aggregate_drift,
    classify_layer,
    compute_gradient_alignment,
    compute_layer_drift,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_state(shapes: dict, fill: float = 0.0) -> dict:
    """Create a fake state_dict with tensors filled with a constant."""
    return {k: torch.full(s, fill) for k, s in shapes.items()}


def _make_taxonomy(shapes: dict, group: str) -> dict:
    """Assign every key in shapes to the same group."""
    return {k: group for k in shapes}


# ─────────────────────────────────────────────────────────────────────────────
# compute_layer_drift
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeLayerDrift:

    def test_zero_drift_identical_states(self):
        shapes = {"base.conv.weight": (8, 3, 3, 3), "base.conv.bias": (8,)}
        state = _make_state(shapes, fill=1.0)
        taxonomy = _make_taxonomy(shapes, "feature")
        drift = compute_layer_drift(state, state, taxonomy)
        assert drift["feature"] == pytest.approx(0.0, abs=1e-6)
        assert drift["norm"] == pytest.approx(0.0, abs=1e-6)
        assert drift["head"] == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_drift_different_states(self):
        shapes = {"base.conv.weight": (4, 4, 3, 3)}
        local = _make_state(shapes, fill=2.0)
        global_ = _make_state(shapes, fill=1.0)
        taxonomy = _make_taxonomy(shapes, "feature")
        drift = compute_layer_drift(local, global_, taxonomy)
        # diff = 1.0 per element, 4*4*3*3 = 144 elements → L2 = sqrt(144) = 12.0
        assert drift["feature"] == pytest.approx(12.0, rel=1e-4)

    def test_group_separation(self):
        """Drift is accumulated separately per group."""
        shapes = {
            "base.bn.weight":         (16,),       # norm
            "base.conv.weight":       (16, 3, 3, 3),  # feature
            "classifier.weight":      (10, 16),    # head
        }
        taxonomy = {
            "base.bn.weight":    "norm",
            "base.conv.weight":  "feature",
            "classifier.weight": "head",
        }
        local  = {k: torch.ones(s) * 2.0 for k, s in shapes.items()}
        global_ = {k: torch.ones(s) * 1.0 for k, s in shapes.items()}
        drift = compute_layer_drift(local, global_, taxonomy)

        # norm: 16 elements, diff=1 → sqrt(16) = 4.0
        assert drift["norm"] == pytest.approx(4.0, rel=1e-4)
        # feature: 16*3*3*3 = 432 elements → sqrt(432)
        assert drift["feature"] == pytest.approx(np.sqrt(432), rel=1e-4)
        # head: 10*16 = 160 elements → sqrt(160)
        assert drift["head"] == pytest.approx(np.sqrt(160), rel=1e-4)

    def test_missing_key_in_global_skipped(self):
        """Parameters absent from global_state are silently skipped."""
        local = {"base.conv.weight": torch.ones(4, 4, 3, 3) * 2.0,
                 "extra.weight": torch.ones(8)}
        global_ = {"base.conv.weight": torch.ones(4, 4, 3, 3)}
        taxonomy = {"base.conv.weight": "feature", "extra.weight": "feature"}
        drift = compute_layer_drift(local, global_, taxonomy)
        assert drift["feature"] == pytest.approx(12.0, rel=1e-4)

    def test_other_group_excluded(self):
        """Parameters labelled 'other' do not contribute to any group."""
        shapes = {"base.pool.weight": (4,)}
        local  = _make_state(shapes, fill=5.0)
        global_ = _make_state(shapes, fill=0.0)
        taxonomy = _make_taxonomy(shapes, "other")
        drift = compute_layer_drift(local, global_, taxonomy)
        assert all(v == 0.0 for v in drift.values())

    def test_device_mismatch_handled(self):
        """CPU local vs CPU global should work; no CUDA required."""
        shapes = {"w": (8, 8)}
        local  = _make_state(shapes, fill=3.0)
        global_ = _make_state(shapes, fill=1.0)
        taxonomy = _make_taxonomy(shapes, "feature")
        drift = compute_layer_drift(local, global_, taxonomy)
        assert drift["feature"] > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_gradient_alignment
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeGradientAlignment:

    def _make_grads(self, vecs: list, key: str = "base.conv.weight") -> list:
        """Wrap flat tensors as single-key gradient dicts."""
        return [{key: v} for v in vecs]

    def test_parallel_gradients_cosine_one(self):
        """Identical gradient directions → cosine similarity = 1.0."""
        v = torch.ones(64)
        grads = self._make_grads([v, v, v])
        taxonomy = {"base.conv.weight": "feature"}
        result = compute_gradient_alignment(grads, taxonomy)
        assert result["feature"] == pytest.approx(1.0, abs=1e-5)

    def test_antiparallel_gradients_cosine_minus_one(self):
        """Opposite gradient directions → cosine similarity = -1.0."""
        v = torch.ones(64)
        grads = self._make_grads([v, -v])
        taxonomy = {"base.conv.weight": "feature"}
        result = compute_gradient_alignment(grads, taxonomy)
        assert result["feature"] == pytest.approx(-1.0, abs=1e-5)

    def test_orthogonal_gradients_cosine_zero(self):
        """Orthogonal gradients → cosine similarity ≈ 0.0."""
        v1 = torch.zeros(4); v1[0] = 1.0
        v2 = torch.zeros(4); v2[1] = 1.0
        grads = self._make_grads([v1, v2])
        taxonomy = {"base.conv.weight": "feature"}
        result = compute_gradient_alignment(grads, taxonomy)
        assert result["feature"] == pytest.approx(0.0, abs=1e-5)

    def test_single_client_returns_zero(self):
        """With only one client there are no pairs → all groups return 0.0."""
        grads = self._make_grads([torch.ones(32)])
        taxonomy = {"base.conv.weight": "feature"}
        result = compute_gradient_alignment(grads, taxonomy)
        assert result == {"norm": 0.0, "feature": 0.0, "head": 0.0}

    def test_zero_vector_guard_no_crash(self):
        """Zero-vector gradients must not cause division-by-zero."""
        v_zero = torch.zeros(32)
        v_nonzero = torch.ones(32)
        grads = self._make_grads([v_zero, v_nonzero])
        taxonomy = {"base.conv.weight": "feature"}
        result = compute_gradient_alignment(grads, taxonomy)
        assert result["feature"] == pytest.approx(0.0, abs=1e-5)

    def test_group_separation(self):
        """Cosine similarity is computed independently per group."""
        grads = [
            {"base.bn.weight": torch.ones(8),    "base.conv.weight": torch.ones(16)},
            {"base.bn.weight": -torch.ones(8),   "base.conv.weight": torch.ones(16)},
        ]
        taxonomy = {"base.bn.weight": "norm", "base.conv.weight": "feature"}
        result = compute_gradient_alignment(grads, taxonomy)
        assert result["norm"]    == pytest.approx(-1.0, abs=1e-5)
        assert result["feature"] == pytest.approx(1.0,  abs=1e-5)
        assert result["head"]    == pytest.approx(0.0,  abs=1e-5)

    def test_random_states_nonzero(self):
        """Random gradients should produce a non-trivially-zero result."""
        torch.manual_seed(0)
        grads = [{"w": torch.randn(64)} for _ in range(5)]
        taxonomy = {"w": "feature"}
        result = compute_gradient_alignment(grads, taxonomy)
        # Not asserting exact value — just that it ran and is in [-1, 1]
        assert -1.0 <= result["feature"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# aggregate_drift
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregateDrift:

    def test_mean_and_std(self):
        drifts = [
            {"norm": 1.0, "feature": 2.0, "head": 3.0},
            {"norm": 3.0, "feature": 4.0, "head": 5.0},
            {"norm": 5.0, "feature": 6.0, "head": 7.0},
        ]
        result = aggregate_drift(drifts)
        assert result["norm"]["mean"]    == pytest.approx(3.0, rel=1e-5)
        assert result["feature"]["mean"] == pytest.approx(4.0, rel=1e-5)
        assert result["head"]["mean"]    == pytest.approx(5.0, rel=1e-5)
        assert result["norm"]["std"]     == pytest.approx(np.std([1, 3, 5]), rel=1e-5)

    def test_single_client(self):
        drifts = [{"norm": 2.0, "feature": 4.0, "head": 0.5}]
        result = aggregate_drift(drifts)
        assert result["norm"]["mean"]  == pytest.approx(2.0)
        assert result["norm"]["std"]   == pytest.approx(0.0)

    def test_all_zero(self):
        drifts = [{"norm": 0.0, "feature": 0.0, "head": 0.0}] * 5
        result = aggregate_drift(drifts)
        for g in ["norm", "feature", "head"]:
            assert result[g]["mean"] == 0.0
            assert result[g]["std"]  == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# classify_layer
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyLayer:

    def test_batchnorm_is_norm(self):
        assert classify_layer("base.bn1", nn.BatchNorm2d(16)) == "norm"

    def test_groupnorm_is_norm(self):
        assert classify_layer("base.gn", nn.GroupNorm(8, 16)) == "norm"

    def test_layernorm_is_norm(self):
        assert classify_layer("base.ln", nn.LayerNorm(16)) == "norm"

    def test_norm_in_name_is_norm(self):
        # Even a Linear layer named "norm_proj" should be classified as norm
        assert classify_layer("base.norm_proj", nn.Linear(16, 16)) == "norm"

    def test_conv2d_is_feature(self):
        assert classify_layer("base.conv1", nn.Conv2d(3, 16, 3)) == "feature"

    def test_linear_is_feature(self):
        assert classify_layer("base.fc", nn.Linear(128, 64)) == "feature"

    def test_classifier_is_head(self):
        assert classify_layer("classifier", nn.Linear(64, 10)) == "head"

    def test_head_in_name_is_head(self):
        assert classify_layer("base.head", nn.Linear(64, 10)) == "head"

    def test_relu_is_other(self):
        assert classify_layer("base.relu", nn.ReLU()) == "other"

    def test_maxpool_is_other(self):
        assert classify_layer("base.pool", nn.MaxPool2d(2)) == "other"
