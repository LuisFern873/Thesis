"""
tests/test_islice_fix.py
========================
Verifica que compute_cka_diagonal convierte islice a list antes de pasarlo
a CKA.compute(), resolviendo el error:
  TypeError: object of type 'itertools.islice' has no len()

El test mockea SimilarityModel y CKA para aislar exactamente el comportamiento
del iterador — sin necesidad de GPU ni de datos reales.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from itertools import islice

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.cka_drift import compute_cka_diagonal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loader(n_batches: int = 30, batch_size: int = 4) -> DataLoader:
    """Synthetic DataLoader con n_batches * batch_size muestras."""
    x = torch.rand(n_batches * batch_size, 3, 8, 8)
    y = torch.zeros(n_batches * batch_size, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4))


def _fake_cka_class(captured: dict):
    """
    Devuelve una clase CKA falsa que:
    - En compute(data_iter), guarda el argumento recibido en captured['data_iter']
      e intenta llamar len() sobre él (igual que simtorch real).
    - Devuelve una matriz identidad 2x2 como resultado.
    """
    class FakeCKA:
        def __init__(self, sim_a, sim_b, device=None):
            pass

        def compute(self, data_iter):
            # Simula exactamente lo que hace simtorch: llama len()
            captured['len_called'] = True
            captured['len_value'] = len(data_iter)   # <-- esto fallaba con islice
            captured['type'] = type(data_iter).__name__
            return np.eye(2)

        def plot_similarity(self, **kwargs):
            pass

    return FakeCKA


def _fake_similarity_model_class():
    class FakeSimilarityModel:
        def __init__(self, model, layers_to_include=None, device=None):
            self.model_activations = {"layer1": None, "layer2": None}
    return FakeSimilarityModel


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIsliceFix:
    """Verifica que compute_cka_diagonal pasa una lista (no islice) a CKA.compute()."""

    def test_probe_batches_positive_passes_list(self):
        """Con probe_batches=5, CKA.compute() debe recibir una list, no un islice."""
        captured = {}
        loader = _make_loader(n_batches=30, batch_size=4)
        model = _make_model()

        with patch('src.utils.cka_drift.SimilarityModel', _fake_similarity_model_class()), \
             patch('src.utils.cka_drift.CKA', _fake_cka_class(captured)):

            result, _ = compute_cka_diagonal(
                global_model=model,
                client_model=model,
                layer_spec=["0"],
                probe_loader=loader,
                probe_batches=5,
            )

        assert result is not None, "compute_cka_diagonal devolvió None — falló internamente"
        assert captured.get('len_called'), "CKA.compute() no fue llamado"
        assert captured['type'] == 'list', (
            f"Se esperaba 'list' pero CKA.compute() recibió '{captured['type']}'. "
            "El fix list(islice(...)) no está aplicado."
        )
        assert captured['len_value'] == 5, (
            f"Se esperaban 5 batches, se pasaron {captured['len_value']}"
        )

    def test_probe_batches_unlimited_passes_list(self):
        """Con probe_batches=-1, CKA.compute() debe recibir una list con todos los batches."""
        captured = {}
        n_batches = 10
        loader = _make_loader(n_batches=n_batches, batch_size=4)
        model = _make_model()

        with patch('src.utils.cka_drift.SimilarityModel', _fake_similarity_model_class()), \
             patch('src.utils.cka_drift.CKA', _fake_cka_class(captured)):

            result, _ = compute_cka_diagonal(
                global_model=model,
                client_model=model,
                layer_spec=["0"],
                probe_loader=loader,
                probe_batches=-1,
            )

        assert result is not None
        assert captured['type'] == 'list', (
            f"Con probe_batches=-1 se esperaba 'list', se recibió '{captured['type']}'"
        )
        assert captured['len_value'] == n_batches, (
            f"Se esperaban {n_batches} batches, se pasaron {captured['len_value']}"
        )

    def test_probe_batches_zero_passes_list(self):
        """Con probe_batches=0, CKA.compute() debe recibir una list con todos los batches."""
        captured = {}
        n_batches = 8
        loader = _make_loader(n_batches=n_batches, batch_size=4)
        model = _make_model()

        with patch('src.utils.cka_drift.SimilarityModel', _fake_similarity_model_class()), \
             patch('src.utils.cka_drift.CKA', _fake_cka_class(captured)):

            result, _ = compute_cka_diagonal(
                global_model=model,
                client_model=model,
                layer_spec=["0"],
                probe_loader=loader,
                probe_batches=0,
            )

        assert result is not None
        assert captured['type'] == 'list'
        assert captured['len_value'] == n_batches

    def test_batch_limit_is_respected(self):
        """probe_batches=3 sobre un loader de 20 batches debe pasar exactamente 3."""
        captured = {}
        loader = _make_loader(n_batches=20, batch_size=4)
        model = _make_model()

        with patch('src.utils.cka_drift.SimilarityModel', _fake_similarity_model_class()), \
             patch('src.utils.cka_drift.CKA', _fake_cka_class(captured)):

            compute_cka_diagonal(
                global_model=model,
                client_model=model,
                layer_spec=["0"],
                probe_loader=loader,
                probe_batches=3,
            )

        assert captured['len_value'] == 3, (
            f"Se esperaban 3 batches limitados, se pasaron {captured['len_value']}"
        )

    def test_returns_diagonal_array(self):
        """compute_cka_diagonal debe devolver un np.ndarray 1-D (la diagonal)."""
        captured = {}
        loader = _make_loader(n_batches=5, batch_size=4)
        model = _make_model()

        with patch('src.utils.cka_drift.SimilarityModel', _fake_similarity_model_class()), \
             patch('src.utils.cka_drift.CKA', _fake_cka_class(captured)):

            result, _ = compute_cka_diagonal(
                global_model=model,
                client_model=model,
                layer_spec=["0"],
                probe_loader=loader,
                probe_batches=5,
            )

        # La FakeCKA devuelve np.eye(2), cuya diagonal es [1., 1.]
        assert isinstance(result, np.ndarray), f"Se esperaba ndarray, se obtuvo {type(result)}"
        assert result.ndim == 1, f"Se esperaba array 1-D, se obtuvo shape {result.shape}"
        np.testing.assert_array_almost_equal(result, [1.0, 1.0])

    def test_original_islice_would_fail(self):
        """Confirma que islice SIN list() lanza TypeError al llamar len() — el bug original."""
        loader = _make_loader(n_batches=10, batch_size=4)
        sliced = islice(loader, 5)
        with pytest.raises(TypeError, match="no len"):
            len(sliced)
