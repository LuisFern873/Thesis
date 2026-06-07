"""CKADriftFedAvgServer — DriftFedAvg with CKA representation-drift measurement.

Extends :class:`DriftFedAvgServer` with per-round, per-layer CKA (Centered
Kernel Alignment) measurement between the global model and a sample of
client models.  CKA computation is a **pure analysis step**: it runs after
clients complete local training and before aggregation, using a shared probe
dataloader.  It does not modify training dynamics, model weights, or
optimizer states.

Separation of concerns
-----------------------
Training stores CKA values in two ways:

1. ``cka_metrics.csv`` — one row per (round, client, layer) with the scalar
   diagonal value.  Used by ``scripts/aggregate_results.py`` and
   ``scripts/plot_results.py`` for lightweight analysis without loading
   large arrays.

2. ``cka_matrices/round_<N>_client_<ID>.npz`` — the full N×N CKA similarity
   matrix persisted with ``numpy.savez_compressed``.  Keys: ``matrix``
   (shape N×N float32) and ``layer_names`` (1-D object array of strings).
   These files are the inputs for ``scripts/plot_cka_heatmaps.py``, which
   generates heatmap PNGs offline with configurable titles, colour maps, and
   layout — without any need to re-run training.

Requirements addressed: 1.3, 5.1, 6.1, 7.6, 9.1
"""

import os
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import torch

from src.server.driftfedavg import DriftFedAvgServer
from src.utils.cka_drift import (
    SIMTORCH_AVAILABLE,
    build_probe_loader,
    compute_cka_diagonal,
    extract_alpha,
    get_layer_spec,
)


class CKADriftFedAvgServer(DriftFedAvgServer):
    """FedAvg server with L2 drift, gradient alignment, and CKA measurement.

    Inherits all drift instrumentation from :class:`DriftFedAvgServer` and
    adds CKA-based representation-drift measurement on top.

    Class attributes
    ----------------
    algorithm_name : str
        Identifier used by the dynamic import mechanism in ``main.py``.
    """

    algorithm_name = "CKADriftFedAvg"

    # ------------------------------------------------------------------
    # CSV header for cka_metrics.csv (Requirement 6.1)
    # ------------------------------------------------------------------
    _CKA_CSV_HEADERS = [
        "round",
        "client_id",
        "model_name",
        "dataset",
        "alpha",
        "layer_index",
        "layer_name",
        "cka_diagonal",
    ]

    def __init__(self, args, **kwargs):
        """Initialise the server, validate CKA config, and set up outputs.

        Steps (in order):
        1. Delegate to ``DriftFedAvgServer.__init__`` (which in turn calls
           ``FedAvgServer.__init__``).
        2. Validate the ``cka`` config section via :meth:`_validate_cka_config`.
        3. Check ``SIMTORCH_AVAILABLE``; disable CKA with a warning if False.
        4. Create the ``cka_matrices/`` output subdirectory.
        5. Initialise ``cka_metrics.csv`` via :meth:`_init_cka_csv`.

        Args:
            args: Hydra ``DictConfig`` object.  Must contain a ``cka`` section
                  with ``interval``, ``client_sample``, and ``probe_batches``
                  integer fields.
            **kwargs: Forwarded to ``DriftFedAvgServer.__init__``.

        Raises:
            ValueError: If any ``cka`` config field is outside its valid range.
        """
        # 1. Parent initialisation (sets self.output_dir, self.model,
        #    self.args, self.logger, self.testset, etc.)
        super().__init__(args, **kwargs)

        # 2. Validate cka config params (Requirement 5.1)
        self._validate_cka_config()

        # 3. Check simtorch availability (Requirement 1.3)
        if not SIMTORCH_AVAILABLE:
            self.logger.log(
                "WARNING: simtorch not found — CKA measurement is disabled "
                "for this run.  Install simtorch to enable CKA instrumentation."
            )
            self._cka_enabled = False
        else:
            self._cka_enabled = True

        # 4. Create cka_matrices output directory (stores full N×N matrices as
        #    .npz files for offline heatmap generation via plot_cka_heatmaps.py)
        self.cka_matrices_dir = self.output_dir / "cka_matrices"
        os.makedirs(self.cka_matrices_dir, exist_ok=True)

        # 5. Initialise cka_metrics.csv (Requirement 6.1)
        self._init_cka_csv()

    # ------------------------------------------------------------------
    # Config validation (Requirement 5.1)
    # ------------------------------------------------------------------

    def _validate_cka_config(self) -> None:
        """Validate the ``cka`` section of the Hydra config.

        Accepts either ``cka.rounds`` (list of 1-based round indices) or the
        legacy ``cka.interval`` (integer).  If both are present, ``rounds``
        takes precedence and a warning is logged.

        Raises:
            ValueError: For each field that is outside its valid range:
                - ``cka.rounds`` entries must all be >= 1
                - ``cka.interval`` must be >= 1 (legacy fallback)
                - ``cka.client_sample`` must be >= 1
                - ``cka.probe_batches`` must be >= 1, or -1/0 for unlimited
        """
        cka_cfg = self.args.cka
        client_sample = cka_cfg.client_sample
        probe_batches = cka_cfg.probe_batches

        # Determine scheduling mode: explicit rounds list vs. legacy interval
        has_rounds = hasattr(cka_cfg, "rounds") and cka_cfg.rounds is not None
        has_interval = hasattr(cka_cfg, "interval") and cka_cfg.interval is not None

        if has_rounds:
            if has_interval:
                self.logger.log(
                    "WARNING: Both cka.rounds and cka.interval are set. "
                    "cka.rounds takes precedence; cka.interval will be ignored."
                )
            rounds = list(cka_cfg.rounds)
            if not rounds:
                raise ValueError("cka.rounds must not be empty.")
            invalid = [r for r in rounds if r < 1]
            if invalid:
                raise ValueError(
                    f"All cka.rounds entries must be >= 1, got invalid: {invalid}"
                )
            self._cka_rounds_set = set(rounds)
            self._cka_use_rounds_list = True
        elif has_interval:
            interval = cka_cfg.interval
            if interval < 1:
                raise ValueError(
                    f"cka.interval must be >= 1, got {interval}"
                )
            self._cka_rounds_set = None
            self._cka_use_rounds_list = False
        else:
            raise ValueError(
                "cka config must contain either 'rounds' (list) or 'interval' (int)."
            )

        if client_sample < 1:
            raise ValueError(
                f"cka.client_sample must be >= 1, got {client_sample}"
            )
        if probe_batches < 1 and probe_batches not in (-1, 0):
            raise ValueError(
                f"cka.probe_batches must be >= 1 or -1/0 for unlimited, got {probe_batches}"
            )

    # ------------------------------------------------------------------
    # CSV initialisation (stub — implemented in task 2.2)
    # ------------------------------------------------------------------

    def _init_cka_csv(self) -> None:
        """Create ``cka_metrics.csv`` in ``self.output_dir`` with the header row.

        On failure, logs a warning and sets ``self._cka_csv_init_failed = True``
        so that the first append operation can retry.

        Requirements: 6.1, 6.2
        """
        self._cka_csv_init_failed = False
        self._cka_csv_disabled = False
        csv_path = self.output_dir / "cka_metrics.csv"
        try:
            with open(csv_path, "w", newline="") as f:
                f.write(",".join(self._CKA_CSV_HEADERS) + "\n")
        except Exception as e:
            self.logger.log(
                f"WARNING: Failed to initialise cka_metrics.csv: {e}. "
                "Will retry on first write."
            )
            self._cka_csv_init_failed = True

    # ------------------------------------------------------------------
    # CSV write helper (stub — implemented in task 2.2)
    # ------------------------------------------------------------------

    def _write_cka_rows(
        self,
        round_idx: int,
        client_id: int,
        diagonal: np.ndarray,
        layer_names: list,
    ) -> None:
        """Append one row per layer to ``cka_metrics.csv``.

        Opens the file in append mode (``"a"``) and writes all rows for the
        given client in a single open-write-close cycle.  On the first write
        failure, logs an error and sets ``self._cka_csv_disabled = True``; on
        subsequent failures, silently skips.

        Args:
            round_idx:   1-based communication round index.
            client_id:   Client identifier from ``selected_clients``.
            diagonal:    CKA diagonal vector (length N, one value per layer).
            layer_names: Ordered list of layer name strings corresponding to
                         each position in *diagonal*.

        Requirements: 6.2, 6.3, 6.4, 6.6, 6.7
        """
        # Guard: if CSV is permanently disabled, skip silently
        if self._cka_csv_disabled:
            return

        csv_path = self.output_dir / "cka_metrics.csv"

        # If init failed earlier, retry creating the file with the header now
        if self._cka_csv_init_failed:
            try:
                with open(csv_path, "w", newline="") as f:
                    f.write(",".join(self._CKA_CSV_HEADERS) + "\n")
                self._cka_csv_init_failed = False
            except Exception as e:
                self.logger.log(
                    f"ERROR: Retry of cka_metrics.csv creation failed: {e}. "
                    "All subsequent CKA CSV writes will be skipped."
                )
                self._cka_csv_disabled = True
                return

        alpha = extract_alpha(self.args.dataset.partition_dir)
        model_name = self.args.model.name
        dataset_name = self.args.dataset.name

        try:
            with open(csv_path, "a", newline="") as f:
                for layer_index, (layer_name, value) in enumerate(
                    zip(layer_names, diagonal)
                ):
                    row = (
                        f"{round_idx},"
                        f"{client_id},"
                        f"{model_name},"
                        f"{dataset_name},"
                        f"{alpha},"
                        f"{layer_index},"
                        f"{layer_name},"
                        f"{value:.6f}\n"
                    )
                    f.write(row)
        except Exception as e:
            self.logger.log(
                f"ERROR: Failed to write CKA rows to cka_metrics.csv: {e}. "
                "All subsequent CKA CSV writes will be skipped."
            )
            self._cka_csv_disabled = True

    # ------------------------------------------------------------------
    # TensorBoard logging (stub — implemented in task 2.3)
    # ------------------------------------------------------------------

    def _log_cka_tensorboard(
        self,
        round_idx: int,
        all_diagonals: list,
    ) -> None:
        """Log CKA scalars to TensorBoard.

        Logs:
        - ``cka/mean_diagonal``: mean of all diagonal values across sampled
          clients and layers.
        - ``cka/layer_{i}``: per-layer mean across sampled clients for each
          layer index *i*.

        Guards on ``hasattr(self, "tensorboard")`` and
        ``self.args.common.monitor == "tensorboard"``.

        Args:
            round_idx:     1-based communication round index (used as the
                           TensorBoard step).
            all_diagonals: List of 1-D numpy arrays, one per sampled client.

        Requirements: 7.1, 7.2, 7.3, 7.5
        """
        if not (
            hasattr(self, "tensorboard")
            and getattr(self.args.common, "monitor", None) == "tensorboard"
        ):
            return

        tb = self.tensorboard

        # Requirement 7.1 — mean across all clients and all layers
        mean_diagonal = float(np.mean(np.concatenate(all_diagonals)))
        tb.add_scalar("cka/mean_diagonal", mean_diagonal, round_idx)

        # Requirement 7.2 — per-layer mean across sampled clients
        num_layers = len(all_diagonals[0])
        for i in range(num_layers):
            layer_mean = float(np.mean([d[i] for d in all_diagonals]))
            tb.add_scalar(f"cka/layer_{i}", layer_mean, round_idx)

    # ------------------------------------------------------------------
    # Core CKA measurement loop (stub — implemented in task 2.4)
    # ------------------------------------------------------------------

    def _run_cka_round(
        self,
        client_packages: "OrderedDict[int, Dict[str, Any]]",
    ) -> None:
        """Execute the CKA measurement loop for one scheduled round.

        High-level steps:
        1. Build ``probe_loader`` from ``self.testset``; log warning and return
           on failure.
        2. Construct ``Global_Model_Copy`` (deepcopy + load
           ``public_model_params`` + eval).
        3. Sample ``min(cka.client_sample, len(selected_clients))`` clients
           without replacement from ``self.selected_clients``.
        4. For each sampled client:
           a. Construct ``Client_Model_Copy`` (deepcopy + load client params +
              eval).
           b. Resolve ``layer_spec`` via :func:`get_layer_spec`.
           c. Call :func:`compute_cka_diagonal`; if diagonal is not ``None``:
              - call :meth:`_write_cka_rows` to append to ``cka_metrics.csv``.
              - persist the full N×N matrix as
                ``cka_matrices/round_<N>_client_<ID>.npz`` (keys: ``matrix``,
                ``layer_names``) for offline heatmap generation.
              - append diagonal to ``all_diagonals``.
           d. Delete ``Client_Model_Copy`` and call
              ``torch.cuda.empty_cache()`` if CUDA.
        5. Delete ``Global_Model_Copy`` and call ``torch.cuda.empty_cache()``
           if CUDA.
        6. If ``all_diagonals`` is non-empty, call
           :meth:`_log_cka_tensorboard`.

        No heatmap PNGs are generated during training.  Use
        ``scripts/plot_cka_heatmaps.py`` after training to render heatmaps
        from the stored ``.npz`` files.

        Args:
            client_packages: Mapping of client ID → package dict returned by
                             the trainer.  Used to access
                             ``regular_model_params`` for each client.

        Requirements: 3.1, 3.2, 3.3, 3.6, 5.3, 5.4, 7.4, 8.1, 8.2, 8.3,
                      8.4, 10.1, 10.2, 10.3, 10.4
        """
        import random as _random

        round_idx = self.current_epoch + 1  # 1-based round index
        is_cuda = self.device.type == "cuda"

        # ------------------------------------------------------------------
        # Step 1: Build probe loader (Requirement 4.1–4.3)
        # ------------------------------------------------------------------
        # Use a dedicated probe_batch_size (smaller than training batch_size)
        # to keep activation tensors manageable.  Falls back to
        # common.batch_size if probe_batch_size is not set in config.
        cka_cfg = self.args.cka
        if hasattr(cka_cfg, "probe_batch_size") and cka_cfg.probe_batch_size:
            batch_size = int(cka_cfg.probe_batch_size)
        else:
            batch_size = self.args.common.batch_size
        dataset_name = self.args.dataset.name
        num_workers = self.args.common.dataloader_num_workers
        # pin_memory=True accelerates host→GPU transfer for probe batches.
        pin_memory = is_cuda

        try:
            probe_loader = build_probe_loader(
                self.testset,
                batch_size=batch_size,
                dataset_name=dataset_name,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        except Exception as e:
            self.logger.log(
                f"WARNING: Cannot build probe loader in round {round_idx} "
                f"— skipping CKA: {e}"
            )
            return

        # ------------------------------------------------------------------
        # Step 2: Offload training model to CPU and flush GPU cache.
        # (Requirements 3.1, 8.1, 8.3)
        # ------------------------------------------------------------------
        # self.model (~6.5 GB for res9, ~4 GB for efficient0) stays on GPU
        # during training but is not needed while CKA runs.  Moving it to
        # CPU frees that VRAM so that the two CKA model copies + intermediate
        # activations can live on GPU, keeping forward passes fast.
        # The model is moved back to self.device unconditionally in the
        # finally block below.
        if is_cuda:
            self.model.cpu()
            torch.cuda.empty_cache()

        try:
            Global_Model_Copy = deepcopy(self.model).to(self.device)
            Global_Model_Copy.load_state_dict(self.public_model_params, strict=False)
            Global_Model_Copy.eval()

            # ------------------------------------------------------------------
            # Step 3: Sample clients (Requirements 5.3, 10.4)
            # ------------------------------------------------------------------
            k = min(self.args.cka.client_sample, len(self.selected_clients))
            sampled_clients = _random.sample(self.selected_clients, k)

            # ------------------------------------------------------------------
            # Step 4: Per-client CKA loop (Requirements 3.2, 3.3, 8.2, 10.1)
            # ------------------------------------------------------------------
            all_diagonals: list = []
            probe_batches = self.args.cka.probe_batches

            for client_id in sampled_clients:
                # 4a. Construct Client_Model_Copy (Requirement 3.2, 8.1, 8.3)
                Client_Model_Copy = deepcopy(self.model).to(self.device)
                client_params = client_packages[client_id]["regular_model_params"]
                Client_Model_Copy.load_state_dict(client_params, strict=False)
                Client_Model_Copy.eval()

                # 4b. Resolve layer_spec (Requirement 3.3)
                layer_spec = get_layer_spec(self.args.model.name, Global_Model_Copy)

                # 4c. Compute CKA diagonal + full matrix (Requirements 3.3–3.5, 3.7, 10.2)
                diagonal, matrix, layer_names = compute_cka_diagonal(
                    global_model=Global_Model_Copy,
                    client_model=Client_Model_Copy,
                    layer_spec=layer_spec,
                    probe_loader=probe_loader,
                    probe_batches=probe_batches,
                )

                if diagonal is not None:
                    # 4d. Write diagonal rows to cka_metrics.csv (Requirement 6.4)
                    self._write_cka_rows(
                        round_idx=round_idx,
                        client_id=client_id,
                        diagonal=diagonal,
                        layer_names=layer_names,
                    )

                    # 4e. Persist full N×N matrix for offline heatmap generation.
                    npz_path = (
                        self.cka_matrices_dir
                        / f"round_{round_idx:04d}_client_{client_id:03d}.npz"
                    )
                    try:
                        np.savez_compressed(
                            npz_path,
                            matrix=matrix.astype(np.float32),
                            layer_names=np.array(layer_names, dtype=object),
                        )
                    except Exception as e:
                        self.logger.log(
                            f"WARNING: Failed to save CKA matrix for round {round_idx} "
                            f"client {client_id}: {e}"
                        )

                    all_diagonals.append(diagonal)

                # 4f. Clean up client copy (Requirements 8.2, 10.3)
                del Client_Model_Copy
                if is_cuda:
                    torch.cuda.empty_cache()

            # ------------------------------------------------------------------
            # Step 5: Clean up global copy (Requirement 8.2)
            # ------------------------------------------------------------------
            del Global_Model_Copy
            if is_cuda:
                torch.cuda.empty_cache()

            # ------------------------------------------------------------------
            # Step 6: TensorBoard logging (Requirements 7.1, 7.2, 7.3)
            # ------------------------------------------------------------------
            if all_diagonals:
                self._log_cka_tensorboard(round_idx, all_diagonals)

        finally:
            # Always restore the training model to its original device so
            # subsequent training rounds are unaffected.
            if is_cuda:
                self.model.to(self.device)

    # ------------------------------------------------------------------
    # Scheduling helper
    # ------------------------------------------------------------------

    def _is_cka_round(self, round_idx: int) -> bool:
        """Return True if CKA should be measured at *round_idx*.

        Uses ``cka.rounds`` (explicit list) when available, otherwise falls
        back to the legacy ``cka.interval`` modulo check.

        Args:
            round_idx: 1-based communication round index.
        """
        if self._cka_use_rounds_list:
            return round_idx in self._cka_rounds_set
        return round_idx % self.args.cka.interval == 0

    # ------------------------------------------------------------------
    # aggregate_client_updates override (stub — implemented in task 2.5)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def aggregate_client_updates(
        self,
        client_packages: "OrderedDict[int, Dict[str, Any]]",
    ) -> None:
        """Run CKA measurement (if scheduled), then delegate to parent.

        If ``self._cka_enabled`` is ``True`` and
        ``(self.current_epoch + 1) % self.args.cka.interval == 0``, calls
        :meth:`_run_cka_round` before delegating to
        ``super().aggregate_client_updates()``.

        The ``super()`` call performs L2 drift + gradient alignment
        computation followed by FedAvg weighted aggregation.

        Args:
            client_packages: Mapping of client ID → package dict.

        Requirements: 3.1, 5.2, 5.4, 5.5, 6.8
        """
        # CKA measurement block — runs before aggregation (Requirement 5.4)
        round_idx = self.current_epoch + 1  # 1-based
        if self._cka_enabled and self._is_cka_round(round_idx):
            self._run_cka_round(client_packages)

        # Delegate to DriftFedAvgServer: L2 drift + gradient alignment + FedAvg
        super().aggregate_client_updates(client_packages)
