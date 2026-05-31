"""CKA Representation Drift Utility Module.

Provides helper functions for measuring representation-space drift between
client and global models using Centered Kernel Alignment (CKA) via the
simtorch library.

This module degrades gracefully when simtorch is unavailable: all public
names are still importable, but SIMTORCH_AVAILABLE will be False and
SimilarityModel / CKA will be None.
"""

import itertools
import logging
import re
import sys
from itertools import islice
from pathlib import Path
from typing import Optional, Union, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# ---------------------------------------------------------------------------
# simtorch import guard (Requirement 1.1, 1.2, 1.4)
# ---------------------------------------------------------------------------
# Prepend the simtorch directory (sibling of the repo root) to sys.path so
# that the import works regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "simtorch"))

try:
    from simtorch.model.sim_model import SimilarityModel
    from simtorch.similarity.cka import CKA

    SIMTORCH_AVAILABLE: bool = True
except (ImportError, ModuleNotFoundError):
    SIMTORCH_AVAILABLE = False
    SimilarityModel = None  # type: ignore[assignment,misc]
    CKA = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture layer map (Requirement 2.1, 2.2, 2.3, 2.4, 2.7)
# ---------------------------------------------------------------------------
# Maps each model name key to a Layer_Spec list of submodule name substrings.
# Matching is performed using str.__contains__ against the full dotted path
# returned by model.named_modules().
#
# DESIGN PRINCIPLE — one semantic unit per entry
# -----------------------------------------------
# Each entry should correspond to one semantically meaningful processing
# stage (e.g. a full transformer block, a MBConv stage, a patch embedding).
# This keeps the CKA diagonal comparable across architectures and avoids
# double-counting caused by substring pollution (see vim_tiny note below).
#
# SUBSTRING POLLUTION WARNING
# ---------------------------
# SimilarityModel uses str.__contains__ to match module paths.  A spec entry
# like "base.layers.1" is a substring of "base.layers.10", "base.layers.11",
# "base.layers.1.mixer", and "base.layers.1.drop_path".  This causes a single
# logical block to produce multiple CKA rows, inflating the diagonal and
# making cross-architecture comparison unreliable.
#
# The fix applied here is to use the most specific unambiguous prefix for
# each entry.  For vim_tiny this means using "base.layers.0.mixer" (the SSM
# core of each block) rather than "base.layers.0" (the full block container
# which also matches sub-modules).  See the vim_tiny entry for full rationale.
ARCHITECTURE_LAYER_MAP: dict[str, list[str]] = {
    "efficient0": [
        "base.features.0", "base.features.1", "base.features.2",
        "base.features.3", "base.features.4", "base.features.5",
        "base.features.6", "base.features.7", "base.features.8",
        "classifier",
    ],  # 10 layers
    "efficient0_gn": [
        "base.features.0", "base.features.1", "base.features.2",
        "base.features.3", "base.features.4", "base.features.5",
        "base.features.6", "base.features.7", "base.features.8",
        "classifier",
    ],  # 10 layers — same spec, different norm layers inside
    "efficient0_ln": [
        "base.features.0", "base.features.1", "base.features.2",
        "base.features.3", "base.features.4", "base.features.5",
        "base.features.6", "base.features.7", "base.features.8",
        "classifier",
    ],  # 10 layers
    "efficient1_gn": [
        "base.features.0", "base.features.1", "base.features.2",
        "base.features.3", "base.features.4", "base.features.5",
        "base.features.6", "base.features.7", "base.features.8",
        "classifier",
    ],  # 10 layers
    "efficient1_ln": [
        "base.features.0", "base.features.1", "base.features.2",
        "base.features.3", "base.features.4", "base.features.5",
        "base.features.6", "base.features.7", "base.features.8",
        "classifier",
    ],  # 10 layers
    "vit_tiny": [
        "base.patch_embed",
        "base.blocks.0",  "base.blocks.1",  "base.blocks.2",
        "base.blocks.3",  "base.blocks.4",  "base.blocks.5",
        "base.blocks.6",  "base.blocks.7",  "base.blocks.8",
        "base.blocks.9",  "base.blocks.10", "base.blocks.11",
        "base.norm",
        "classifier",
    ],  # 15 layers
    # -----------------------------------------------------------------------
    # vim_tiny — corrected spec (was: 15 entries with 3 bugs)
    # -----------------------------------------------------------------------
    # Bug 1 — substring pollution (critical):
    #   The old spec used "base.layers.N" as entries.  Because SimilarityModel
    #   matches via str.__contains__, "base.layers.1" also matches
    #   "base.layers.10", "base.layers.11", "base.layers.1.mixer", and
    #   "base.layers.1.drop_path".  The actual run produced 39 rows per client
    #   instead of the intended 15: each block contributed 3 rows (container +
    #   mixer + drop_path) and patch_embed contributed 3 rows (container +
    #   proj + norm).
    #
    # Bug 2 — phantom entry (base.norm_f):
    #   "base.norm_f" was listed but does not exist in the VisionMamba
    #   checkpoint used here (confirmed: 0 matches in the CSV).  The module
    #   is named differently depending on the Vim repo version.  Removing it
    #   avoids a silent zero-match entry that would produce a meaningless CKA
    #   value of 1.0 (identical empty activations).
    #
    # Bug 3 — asymmetric layer.0 (minor):
    #   base.layers.0 has no drop_path sub-module (stochastic depth is only
    #   applied from layer 1 onward), so the old spec produced 2 rows for
    #   layer 0 and 3 rows for layers 1–11.
    #
    # Fix — use ".mixer" suffix for each block:
    #   Each Mamba block's SSM core is named "base.layers.N.mixer".  This
    #   string is unique (not a prefix of any other module path), so it
    #   produces exactly one CKA row per block.  It is also the functional
    #   analogue of "base.blocks.N.attn" in ViT-Tiny: both capture the
    #   sequence-mixing operation that is the architectural differentiator.
    #
    # Structural correspondence with vit_tiny (14 entries each):
    #   vit_tiny  base.patch_embed   ↔  vim_tiny  base.patch_embed
    #   vit_tiny  base.blocks.N      ↔  vim_tiny  base.layers.N.mixer  (N=0..11)
    #   vit_tiny  base.norm          ↔  (absent in this Vim version — omitted)
    #   vit_tiny  classifier         ↔  vim_tiny  classifier
    "vim_tiny": [
        "base.patch_embed",
        "base.layers.0.mixer", 
        "base.layers.0.norm",
        "base.layers.1.mixer",
        "base.layers.1.norm",  
        "base.layers.2.mixer",
        "base.layers.2.norm",
        "base.layers.3.mixer",
        "base.layers.3.norm", 
        "base.layers.4.mixer",
        "base.layers.4.norm",  
        "base.layers.5.mixer",
        "base.layers.5.norm",
        "base.layers.6.mixer",
        "base.layers.6.norm", 
        "base.layers.7.mixer",
        "base.layers.7.norm",  
        "base.layers.8.mixer",
        "base.layers.8.norm",
        "base.layers.9.mixer",
        "base.layers.9.norm", 
        "base.layers.10.mixer",
        "base.layers.10.norm", 
        "base.layers.11.mixer",
        "base.layers.11.norm",
        "classifier",
    ],  # 14 layers — 1 patch_embed + 12 SSM blocks (mixer only) + 1 head
}


# ---------------------------------------------------------------------------
# Layer spec helpers (Requirement 2.5, 2.6)
# ---------------------------------------------------------------------------

def count_matching_submodules(model: nn.Module, spec: list[str]) -> int:
    """Count the number of named-module paths that contain any spec substring.

    Matching uses ``str.__contains__``: a path matches if *any* element of
    *spec* is a substring of the full dotted path returned by
    ``model.named_modules()``.

    Args:
        model: The PyTorch module whose submodule paths are inspected.
        spec:  A list of substrings to match against each path.

    Returns:
        The number of submodule paths (including the root empty string) that
        contain at least one substring from *spec*.
    """
    count = 0
    for path, _ in model.named_modules():
        if any(substring in path for substring in spec):
            count += 1
    return count


def get_layer_spec(
    model_name: str,
    model: Optional[nn.Module] = None,
) -> list[str]:
    """Resolve the Layer_Spec for *model_name* from ``ARCHITECTURE_LAYER_MAP``.

    If *model_name* is not present in the map, a warning is logged that
    includes the model name literally and the default spec
    ``["base", "classifier"]`` is returned.

    When *model* is provided the resolved spec is validated against the
    model's actual submodule paths via :func:`count_matching_submodules`.
    If no submodule path matches any spec substring a ``ValueError`` is
    raised identifying the model name and the offending spec.

    Args:
        model_name: Key to look up in ``ARCHITECTURE_LAYER_MAP``.
        model:      Optional model instance used for zero-match validation.

    Returns:
        The resolved ``Layer_Spec`` list of submodule name substrings.

    Raises:
        ValueError: When *model* is provided and the resolved spec matches
                    zero submodule paths in the model.
    """
    if model_name in ARCHITECTURE_LAYER_MAP:
        spec = ARCHITECTURE_LAYER_MAP[model_name]
    else:
        logger.warning(
            "Model name %r not found in ARCHITECTURE_LAYER_MAP; "
            "falling back to default spec ['base', 'classifier'].",
            model_name,
        )
        spec = ["base", "classifier"]

    if model is not None:
        match_count = count_matching_submodules(model, spec)
        if match_count == 0:
            raise ValueError(
                f"Layer spec for model '{model_name}' matched zero submodule "
                f"paths in the provided model. Spec was: {spec!r}. "
                "Check that the model architecture matches the expected "
                "submodule naming convention."
            )

    return spec


# ---------------------------------------------------------------------------
# Alpha extraction helper (Requirement 6.5, 11.3)
# ---------------------------------------------------------------------------

def extract_alpha(partition_dir: str) -> Union[float, str]:
    """Extract alpha value from partition_dir string.

    Uses regex alpha_(\\d+\\.?\\d*) to find the first match.
    Returns float on success, "unknown" on failure.

    Args:
        partition_dir: A string (typically a directory path) that may contain
                       a substring of the form ``alpha_<number>``, e.g.
                       ``"partitions/alpha_0.1/seed_42"``.

    Returns:
        The extracted alpha as a ``float`` when the pattern is found and the
        matched substring can be converted; the string ``"unknown"`` otherwise.
    """
    match = re.search(r"alpha_(\d+\.?\d*)", partition_dir)
    if match:
        return float(match.group(1))
    return "unknown"


# ---------------------------------------------------------------------------
# Probe DataLoader construction (Requirement 4.1, 4.2, 4.3)
# ---------------------------------------------------------------------------

def build_probe_loader(
    testset,
    batch_size: int,
    dataset_name: str,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a deterministic probe :class:`~torch.utils.data.DataLoader`.

    Applies the test-time transform pipeline ``Resize(224) → Normalize`` to
    *testset* (no random augmentations) and returns a :class:`DataLoader`
    with ``shuffle=False`` and ``drop_last=False``.

    The function creates a shallow copy of *testset* and overrides only its
    ``data_transform`` attribute so that the original dataset object is not
    mutated.  The normalisation statistics are looked up from
    :data:`src.utils.constants.DATA_MEAN` and
    :data:`src.utils.constants.DATA_STD` using *dataset_name* as the key.

    Args:
        testset:      A :class:`~data.utils.datasets.BaseDataset` instance
                      (or any ``Dataset`` with a ``data_transform`` attribute).
        batch_size:   Number of samples per batch.
        dataset_name: Key used to look up ``DATA_MEAN`` / ``DATA_STD``.
        num_workers:  Number of worker processes for data loading (default 0).
        pin_memory:   Whether to pin memory for faster GPU transfer (default
                      ``False``).

    Returns:
        A :class:`~torch.utils.data.DataLoader` configured for deterministic
        activation extraction.

    Requirements: 4.1, 4.2, 4.3
    """
    from src.utils.constants import DATA_MEAN, DATA_STD  # local import to avoid circularity
    import copy

    mean = DATA_MEAN[dataset_name]
    std = DATA_STD[dataset_name]

    probe_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Shallow-copy the dataset so we don't mutate the caller's testset.
    probe_set = copy.copy(testset)
    probe_set.data_transform = probe_transform

    return DataLoader(
        probe_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# ---------------------------------------------------------------------------
# CKA diagonal computation (Requirements 3.3, 3.4, 3.5, 3.7, 4.4, 4.5)
# ---------------------------------------------------------------------------

def compute_cka_diagonal(
    global_model: nn.Module,
    client_model: nn.Module,
    layer_spec: list[str],
    probe_loader: DataLoader,
    probe_batches: int,
    heatmap_save_path: Optional[Path] = None,
    heatmap_title: str = "",
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Compute the CKA diagonal between *global_model* and *client_model*.

    Both models are wrapped in :class:`SimilarityModel` using *layer_spec*
    as the ``layers_to_include`` argument.  A :class:`CKA` object is then
    instantiated and ``compute()`` is called on the probe data.

    Batch limiting:
    - ``probe_batches >= 1``: only the first *probe_batches* batches of
      *probe_loader* are consumed (via ``itertools.islice``).
    - ``probe_batches == -1`` or ``probe_batches == 0``: the full
      *probe_loader* is consumed without truncation.

    After ``CKA.compute()`` returns the N×N similarity matrix, the diagonal
    is extracted with ``numpy.diag(matrix)`` and returned as a 1-D array of
    length N.

    If *heatmap_save_path* is provided, ``cka.plot_similarity()`` is called
    with ``savefig=True``, ``save_path=heatmap_save_path``, and
    ``title=heatmap_title`` to persist a heatmap PNG.

    Any :class:`Exception` (excluding :class:`KeyboardInterrupt` and
    :class:`SystemExit`) is caught, a warning is logged, and ``None`` is
    returned so that the caller can continue processing remaining clients.

    Args:
        global_model:      The reference (global) model in ``eval()`` mode.
        client_model:      The client's locally-trained model in ``eval()`` mode.
        layer_spec:        List of submodule name substrings passed as
                           ``layers_to_include`` to :class:`SimilarityModel`.
        probe_loader:      A deterministic :class:`DataLoader` used to collect
                           activations for the CKA estimate.
        probe_batches:     Maximum number of batches to consume.  Pass -1 or 0
                           to consume the entire loader.
        heatmap_save_path: Optional path at which to save the CKA heatmap PNG.
        heatmap_title:     Title string forwarded to ``plot_similarity()``.

    Returns:
        A tuple containing:
        - A 1-D :class:`numpy.ndarray` of length N (the CKA diagonal), or ``None`` if failed.
        - A list of layer names in the exact execution order, or ``None`` if failed.
    """
    try:
        with torch.no_grad():
            try:
                device = next(client_model.parameters()).device
                if device.type == "cpu" and torch.cuda.is_available():
                    device = torch.device("cuda:0")
            except StopIteration:
                device = "cpu" if not torch.cuda.is_available() else torch.device("cuda:0")

            # Wrap both models in SimilarityModel (Requirement 3.3)
            sim_global = SimilarityModel(global_model, layers_to_include=layer_spec, device=device)
            sim_client = SimilarityModel(client_model, layers_to_include=layer_spec, device=device)

            # Instantiate CKA and select the data iterator (Requirements 3.4, 4.4, 4.5)
            cka = CKA(sim_global, sim_client, device=device)
            if probe_batches >= 1:
                # Convert to list so simtorch can call len() on it
                data_iter = list(islice(probe_loader, probe_batches))
            else:
                # probe_batches == -1 or 0 → consume the full loader
                data_iter = list(probe_loader)  # type: ignore[assignment]

            matrix = cka.compute(data_iter)

            # Extract the diagonal (Requirement 3.5)
            diagonal: np.ndarray = np.diag(matrix)
            layer_names = list(sim_client.model_activations.keys())

            # Optionally save the heatmap PNG (Requirement 7.4)
            if heatmap_save_path is not None:
                cka.plot_similarity(
                    savefig=True,
                    save_path=heatmap_save_path,
                    title=heatmap_title,
                )
                plt.close("all")

            return diagonal, layer_names

    except (KeyboardInterrupt, SystemExit):
        # Re-raise signals that should terminate the process
        raise
    except Exception as exc:  # noqa: BLE001
        # Requirement 3.7: catch all other exceptions, log a warning, return None
        logger.warning(
            "CKA computation failed: %s: %s",
            type(exc).__name__,
            exc,
            exc_info=True
        )
        return None, None
