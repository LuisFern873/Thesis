"""
tests/test_cka_drift_unit.py
============================
Smoke and example unit tests for src/utils/cka_drift.py.

Covers:
  - SIMTORCH_AVAILABLE is a bool
  - CSV header columns match the schema exactly (8 columns in order)
  - extract_alpha returns "unknown" for a string with no alpha_ pattern
  - get_layer_spec("efficient0") returns a list of length 10
  - get_layer_spec("vit_tiny") returns length 15
  - get_layer_spec("vim_tiny") returns length 14
  - build_probe_loader returns a DataLoader with shuffle=False and drop_last=False
  - Unknown model name fallback returns ["base", "classifier"] and logs a warning
    containing the model name

Requirements: 1.4, 2.2, 2.3, 2.4, 2.5, 4.2, 4.3

Run with:
    python -m pytest tests/test_cka_drift_unit.py -v
"""

import sys
from pathlib import Path

# Allow imports from the repo root regardless of where pytest is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.utils.cka_drift import (
    SIMTORCH_AVAILABLE,
    extract_alpha,
    get_layer_spec,
    build_probe_loader,
)
from src.server.ckadriftfedavg import CKADriftFedAvgServer


# ---------------------------------------------------------------------------
# Requirement 1.4 — SIMTORCH_AVAILABLE is a bool
# ---------------------------------------------------------------------------

class TestSimtorchAvailableFlag:
    def test_simtorch_available_is_bool(self):
        """SIMTORCH_AVAILABLE must be a plain Python bool (not int, not None)."""
        assert isinstance(SIMTORCH_AVAILABLE, bool), (
            f"Expected SIMTORCH_AVAILABLE to be bool, got {type(SIMTORCH_AVAILABLE)}"
        )


# ---------------------------------------------------------------------------
# Requirement 6.1 — CSV header columns match the schema exactly
# ---------------------------------------------------------------------------

class TestCsvHeaders:
    """The _CKA_CSV_HEADERS class attribute must match the schema exactly."""

    EXPECTED_HEADERS = [
        "round",
        "client_id",
        "model_name",
        "dataset",
        "alpha",
        "layer_index",
        "layer_name",
        "cka_diagonal",
    ]

    def test_csv_header_count(self):
        """There must be exactly 8 columns."""
        assert len(CKADriftFedAvgServer._CKA_CSV_HEADERS) == 8, (
            f"Expected 8 CSV headers, got {len(CKADriftFedAvgServer._CKA_CSV_HEADERS)}: "
            f"{CKADriftFedAvgServer._CKA_CSV_HEADERS}"
        )

    def test_csv_header_names_and_order(self):
        """Column names and their order must match the schema exactly."""
        assert CKADriftFedAvgServer._CKA_CSV_HEADERS == self.EXPECTED_HEADERS, (
            f"CSV headers mismatch.\n"
            f"  Expected: {self.EXPECTED_HEADERS}\n"
            f"  Got:      {CKADriftFedAvgServer._CKA_CSV_HEADERS}"
        )


# ---------------------------------------------------------------------------
# Requirements 6.5, 11.3 — extract_alpha
# ---------------------------------------------------------------------------

class TestExtractAlpha:
    def test_returns_unknown_for_no_alpha_pattern(self):
        """extract_alpha must return the string "unknown" when no alpha_ pattern exists."""
        result = extract_alpha("partitions/seed_42/run_1")
        assert result == "unknown", (
            f'Expected "unknown", got {result!r}'
        )

    def test_returns_unknown_for_empty_string(self):
        """extract_alpha must return "unknown" for an empty string."""
        assert extract_alpha("") == "unknown"

    def test_returns_unknown_for_unrelated_string(self):
        """extract_alpha must return "unknown" when the string has no alpha_ prefix."""
        assert extract_alpha("beta_0.1/gamma_0.3") == "unknown"

    def test_returns_float_for_valid_pattern(self):
        """Sanity check: extract_alpha returns a float when the pattern is present."""
        result = extract_alpha("partitions/alpha_0.1/seed_42")
        assert isinstance(result, float)
        assert result == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Requirements 2.2, 2.3, 2.4, 2.5 — get_layer_spec lengths
# ---------------------------------------------------------------------------

class TestGetLayerSpecLengths:
    """get_layer_spec must return lists of the correct length for each architecture."""

    def test_efficient0_returns_10_layers(self):
        """efficient0 Layer_Spec must have exactly 10 entries."""
        spec = get_layer_spec("efficient0")
        assert isinstance(spec, list), f"Expected list, got {type(spec)}"
        assert len(spec) == 10, (
            f"Expected 10 layers for efficient0, got {len(spec)}: {spec}"
        )

    def test_efficient0_gn_returns_10_layers(self):
        """efficient0_gn Layer_Spec must have exactly 10 entries."""
        spec = get_layer_spec("efficient0_gn")
        assert len(spec) == 10, (
            f"Expected 10 layers for efficient0_gn, got {len(spec)}"
        )

    def test_efficient0_ln_returns_10_layers(self):
        """efficient0_ln Layer_Spec must have exactly 10 entries."""
        spec = get_layer_spec("efficient0_ln")
        assert len(spec) == 10, (
            f"Expected 10 layers for efficient0_ln, got {len(spec)}"
        )

    def test_vit_tiny_returns_15_layers(self):
        """vit_tiny Layer_Spec must have exactly 15 entries."""
        spec = get_layer_spec("vit_tiny")
        assert isinstance(spec, list), f"Expected list, got {type(spec)}"
        assert len(spec) == 15, (
            f"Expected 15 layers for vit_tiny, got {len(spec)}: {spec}"
        )

    def test_vim_tiny_returns_14_layers(self):
        """vim_tiny Layer_Spec must have exactly 14 entries."""
        spec = get_layer_spec("vim_tiny")
        assert isinstance(spec, list), f"Expected list, got {type(spec)}"
        assert len(spec) == 14, (
            f"Expected 14 layers for vim_tiny, got {len(spec)}: {spec}"
        )

    def test_all_spec_entries_are_strings(self):
        """Every entry in every Layer_Spec must be a non-empty string."""
        for model_name in ("efficient0", "efficient0_gn", "efficient0_ln",
                           "vit_tiny", "vim_tiny"):
            spec = get_layer_spec(model_name)
            for i, entry in enumerate(spec):
                assert isinstance(entry, str) and entry, (
                    f"Layer spec entry [{i}] for {model_name!r} is not a "
                    f"non-empty string: {entry!r}"
                )


# ---------------------------------------------------------------------------
# Requirements 4.2, 4.3 — build_probe_loader DataLoader properties
# ---------------------------------------------------------------------------

class _SyntheticDataset(Dataset):
    """Minimal synthetic dataset: returns (3-channel 32×32 tensor, label)."""

    def __init__(self, n: int = 20):
        self._n = n
        # Simulate a data_transform attribute (will be overridden by build_probe_loader)
        self.data_transform = None

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        # Return a float tensor so Normalize can be applied
        x = torch.rand(3, 32, 32)
        if self.data_transform is not None:
            x = self.data_transform(x)
        return x, idx % 4


class TestBuildProbeLoader:
    """build_probe_loader must return a DataLoader with shuffle=False and drop_last=False."""

    def test_returns_dataloader(self):
        """build_probe_loader must return a torch DataLoader instance."""
        dataset = _SyntheticDataset(n=20)
        loader = build_probe_loader(
            testset=dataset,
            batch_size=4,
            dataset_name="cifar10",
        )
        assert isinstance(loader, DataLoader), (
            f"Expected DataLoader, got {type(loader)}"
        )

    def test_shuffle_is_false(self):
        """The returned DataLoader must have shuffle=False (Requirement 4.2)."""
        dataset = _SyntheticDataset(n=20)
        loader = build_probe_loader(
            testset=dataset,
            batch_size=4,
            dataset_name="cifar10",
        )
        # DataLoader stores the sampler; shuffle=False means a SequentialSampler
        from torch.utils.data import SequentialSampler
        assert isinstance(loader.sampler, SequentialSampler), (
            f"Expected SequentialSampler (shuffle=False), got {type(loader.sampler)}"
        )

    def test_drop_last_is_false(self):
        """The returned DataLoader must have drop_last=False (Requirement 4.3)."""
        dataset = _SyntheticDataset(n=20)
        loader = build_probe_loader(
            testset=dataset,
            batch_size=4,
            dataset_name="cifar10",
        )
        assert loader.drop_last is False, (
            f"Expected drop_last=False, got {loader.drop_last}"
        )

    def test_batch_size_respected(self):
        """The DataLoader must use the supplied batch_size."""
        dataset = _SyntheticDataset(n=20)
        loader = build_probe_loader(
            testset=dataset,
            batch_size=5,
            dataset_name="cifar10",
        )
        assert loader.batch_size == 5

    def test_all_samples_accessible(self):
        """With drop_last=False all samples must be reachable."""
        n = 17  # intentionally not divisible by batch_size
        dataset = _SyntheticDataset(n=n)
        loader = build_probe_loader(
            testset=dataset,
            batch_size=4,
            dataset_name="cifar10",
        )
        total = sum(x.shape[0] for x, _ in loader)
        assert total == n, (
            f"Expected {n} samples total, got {total}"
        )


# ---------------------------------------------------------------------------
# Requirement 2.5 — Unknown model name fallback + warning
# ---------------------------------------------------------------------------

class TestUnknownModelNameFallback:
    """get_layer_spec must fall back to ["base", "classifier"] for unknown names
    and log a warning that includes the model name literally."""

    def test_fallback_returns_base_classifier(self):
        """Unknown model name must return exactly ["base", "classifier"]."""
        result = get_layer_spec("totally_unknown_model_xyz")
        assert result == ["base", "classifier"], (
            f'Expected ["base", "classifier"], got {result!r}'
        )

    def test_fallback_logs_warning_with_model_name(self, caplog):
        """The warning message must contain the unknown model name literally."""
        model_name = "nonexistent_arch_abc"
        with caplog.at_level(logging.WARNING, logger="src.utils.cka_drift"):
            get_layer_spec(model_name)

        # At least one warning record must mention the model name
        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any(model_name in msg for msg in warning_messages), (
            f"Expected a warning containing {model_name!r}, "
            f"but got: {warning_messages}"
        )

    def test_fallback_warning_different_names(self, caplog):
        """The warning must include the specific model name, not a generic message."""
        model_name = "another_unknown_model_999"
        with caplog.at_level(logging.WARNING, logger="src.utils.cka_drift"):
            get_layer_spec(model_name)

        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any(model_name in msg for msg in warning_messages), (
            f"Warning must contain the model name {model_name!r} literally. "
            f"Got: {warning_messages}"
        )
