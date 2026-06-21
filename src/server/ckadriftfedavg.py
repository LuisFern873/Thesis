"""CKADriftFedAvgServer — DriftFedAvg with checkpoint saving for offline CKA.

Extends :class:`DriftFedAvgServer` with checkpoint saving at the rounds
specified in ``cka.rounds``.  CKA computation is completely decoupled from
training: this server only saves the global and per-client model state dicts
to disk.  The offline script ``scripts/compute_cka_offline.py`` reads those
checkpoints and computes all CKA metrics after training finishes, with no
impact on training throughput and no GPU memory pressure.

Directory layout written by this server
-----------------------------------------
::

    <output_dir>/
    ├── checkpoints/
    │   ├── run_metadata.json          # all config needed by the offline script
    │   ├── training_state.pt          # resume checkpoint (overwritten each round)
    │   ├── round_0001/
    │   │   ├── global.pt              # global model state dict (CPU tensors)
    │   │   ├── client_000.pt
    │   │   ├── client_001.pt
    │   │   └── ...
    │   ├── round_0002/
    │   └── ...
    ├── drift_metrics.csv              # L2 drift + gradient alignment (from DriftFedAvg)
    └── ...  (standard FL-bench outputs)

The offline script then writes:
    <output_dir>/
    ├── cka_metrics.csv
    └── cka_matrices/
        ├── round_0001_client_000.npz
        └── ...

Resume training
---------------
Pass ``common.resume_checkpoint=<path_to_training_state.pt>`` (or the
directory containing it) on the CLI to resume an interrupted run::

    python main.py --config-name ckadriftfedavg method=ckadriftfedavg \\
        ... \\
        "hydra.run.dir=logs/runs/<same_run_name>" \\
        common.resume_checkpoint=logs/runs/<same_run_name>/checkpoints/training_state.pt
"""

import json
import os
import time
from collections import OrderedDict
from typing import Any, Dict

import torch
from rich.progress import track

from src.server.driftfedavg import DriftFedAvgServer


class CKADriftFedAvgServer(DriftFedAvgServer):
    """FedAvg server with L2 drift, gradient alignment, and offline CKA checkpointing.

    Inherits all drift instrumentation from :class:`DriftFedAvgServer` and
    adds checkpoint saving at rounds listed in ``cka.rounds`` so that CKA
    can be computed offline without any training-time overhead.
    """

    algorithm_name = "CKADriftFedAvg"

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        self._validate_cka_config()

        # Create checkpoints directory
        self.checkpoints_dir = self.output_dir / "checkpoints"
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Write run_metadata.json so the offline script has everything it
        # needs without requiring access to the Hydra config.
        self._write_run_metadata()

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    def _validate_cka_config(self) -> None:
        """Validate the ``cka`` section and set ``_cka_rounds_set``."""
        cka_cfg = self.args.cka
        has_rounds = hasattr(cka_cfg, "rounds") and cka_cfg.rounds is not None
        has_interval = hasattr(cka_cfg, "interval") and cka_cfg.interval is not None

        if has_rounds:
            if has_interval:
                self.logger.log(
                    "WARNING: Both cka.rounds and cka.interval are set. "
                    "cka.rounds takes precedence."
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
                raise ValueError(f"cka.interval must be >= 1, got {interval}")
            self._cka_rounds_set = None
            self._cka_use_rounds_list = False
        else:
            raise ValueError(
                "cka config must contain either 'rounds' (list) or 'interval' (int)."
            )

        client_sample = cka_cfg.client_sample
        probe_batches = cka_cfg.probe_batches
        if client_sample < 1:
            raise ValueError(f"cka.client_sample must be >= 1, got {client_sample}")
        if probe_batches < 1 and probe_batches not in (-1, 0):
            raise ValueError(
                f"cka.probe_batches must be >= 1 or -1/0 for unlimited, got {probe_batches}"
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _write_run_metadata(self) -> None:
        """Write ``checkpoints/run_metadata.json`` for the offline CKA script.

        Contains every piece of information the offline script needs to
        reconstruct models and build the probe dataloader without access
        to the live Hydra config object.
        """
        from src.utils.cka_drift import extract_alpha

        cka_cfg = self.args.cka
        if self._cka_use_rounds_list:
            cka_rounds = sorted(self._cka_rounds_set)
        else:
            # Expand interval into explicit round list for clarity
            T = self.args.common.global_epoch
            cka_rounds = list(range(self.args.cka.interval, T + 1, self.args.cka.interval))

        probe_batch_size = (
            int(cka_cfg.probe_batch_size)
            if hasattr(cka_cfg, "probe_batch_size") and cka_cfg.probe_batch_size
            else int(self.args.common.batch_size)
        )

        metadata = {
            # Run identity
            "output_dir": str(self.output_dir),
            "model_name": self.args.model.name,
            "dataset_name": self.args.dataset.name,
            "partition_dir": str(self.args.dataset.partition_dir),
            "alpha": str(extract_alpha(str(self.args.dataset.partition_dir))),
            "seed": int(self.args.common.seed),
            # CKA schedule
            "cka_rounds": cka_rounds,
            "client_sample": int(cka_cfg.client_sample),
            "probe_batches": int(cka_cfg.probe_batches),
            "probe_batch_size": probe_batch_size,
            # Hints for model reconstruction
            "use_pretrained": bool(self.args.model.use_torchvision_pretrained_weights),
            "num_clients": int(self.client_num),
            "global_epoch": int(self.args.common.global_epoch),
        }

        metadata_path = self.checkpoints_dir / "run_metadata.json"
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.log(f"WARNING: Failed to write run_metadata.json: {e}")

    # ------------------------------------------------------------------
    # Scheduling helper
    # ------------------------------------------------------------------

    def _is_cka_round(self, round_idx: int) -> bool:
        """Return True if checkpoints should be saved at *round_idx*."""
        if self._cka_use_rounds_list:
            return round_idx in self._cka_rounds_set
        return round_idx % self.args.cka.interval == 0

    # ------------------------------------------------------------------
    # Checkpoint saving
    # ------------------------------------------------------------------

    def _save_cka_checkpoint(
        self,
        round_idx: int,
        client_packages: "OrderedDict[int, Dict[str, Any]]",
    ) -> None:
        """Save global + per-client state dicts for round *round_idx*.

        Files written::

            checkpoints/round_{round_idx:04d}/global.pt
            checkpoints/round_{round_idx:04d}/client_{client_id:03d}.pt

        All tensors are moved to CPU before saving so that the offline
        script can load them without a GPU.

        Only clients listed in ``self.selected_clients`` are saved (the
        same set that was used in this round).  The offline script will
        respect ``cka.client_sample`` to subsample further if desired.

        Args:
            round_idx:       1-based communication round index.
            client_packages: Mapping of client ID → package dict.
        """
        round_dir = self.checkpoints_dir / f"round_{round_idx:04d}"
        try:
            os.makedirs(round_dir, exist_ok=True)
        except Exception as e:
            self.logger.log(
                f"WARNING: Cannot create checkpoint dir {round_dir}: {e} "
                f"— skipping checkpoint for round {round_idx}"
            )
            return

        # Global model state dict (before aggregation — pre-aggregation
        # global params are what clients received this round)
        global_path = round_dir / "global.pt"
        try:
            global_state = {
                k: v.detach().cpu() for k, v in self.public_model_params.items()
            }
            torch.save(global_state, global_path)
        except Exception as e:
            self.logger.log(
                f"WARNING: Failed to save global checkpoint for round {round_idx}: {e}"
            )

        # Per-client state dicts
        for client_id, package in client_packages.items():
            client_path = round_dir / f"client_{client_id:03d}.pt"
            try:
                client_state = {
                    k: v.detach().cpu()
                    for k, v in package["regular_model_params"].items()
                }
                torch.save(client_state, client_path)
            except Exception as e:
                self.logger.log(
                    f"WARNING: Failed to save client {client_id} checkpoint "
                    f"for round {round_idx}: {e}"
                )

        self.logger.log(
            f"  [CKA] Saved checkpoints for round {round_idx} "
            f"({len(client_packages)} clients) → {round_dir}"
        )

    # ------------------------------------------------------------------
    # Resume-aware training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Training loop with per-round resume checkpointing.

        Behaviour
        ---------
        * **Fresh run**: identical to :meth:`FedAvgServer.train` but also
          writes ``checkpoints/training_state.pt`` at the end of every round.
        * **Resumed run**: detects ``common.resume_checkpoint`` in the config
          (a path to ``training_state.pt`` or its parent directory), restores
          all server state, skips already-completed rounds, and continues from
          where training stopped.

        The resume checkpoint is overwritten atomically after each round, so
        only *one* ``training_state.pt`` file is kept on disk at any time —
        space usage is O(model_size), not O(model_size × rounds).
        """
        # ---- resolve resume path ----------------------------------------
        resume_path = getattr(self.args.common, "resume_checkpoint", None)
        start_round = 0  # 0-based index of the first round to execute

        if resume_path and str(resume_path).lower() not in ("", "null", "none"):
            from pathlib import Path as _Path

            rp = _Path(resume_path)
            # Accept either the directory or the .pt file itself
            if rp.is_dir():
                rp = rp / "training_state.pt"
            if rp.is_file():
                start_round = self.load_training_checkpoint(rp)
            else:
                self.logger.log(
                    f"  [Resume] WARNING: resume_checkpoint '{resume_path}' not found "
                    f"— starting from scratch."
                )

        # ---- main round loop --------------------------------------------
        avg_round_time = 0
        for E in self.train_progress_bar:
            # Skip rounds already completed in a previous run
            if E < start_round:
                continue

            self.current_epoch = E
            self.verbose = (
                (self.current_epoch + 1) % self.args.common.verbose_gap == 0
            )

            self.logger.log(
                "-" * 28,
                f"ROUND {E + 1}/{self.args.common.global_epoch} START",
                "-" * 28,
            )
            self.selected_clients = self.client_sample_stream[E]
            self.logger.log(f"Selected clients: {self.selected_clients}")

            begin = time.time()
            self.train_one_round()
            end = time.time()
            round_duration = end - begin
            avg_round_time = (
                avg_round_time * (E - start_round) + round_duration
            ) / (E - start_round + 1)
            self.logger.log(
                f"ROUND {E + 1} FINISHED in {round_duration:.2f}s "
                f"(Avg: {avg_round_time:.2f}s)"
            )

            if (
                self.args.common.test.server.interval > 0
                and (E + 1) % self.args.common.test.server.interval == 0
            ):
                self.test_global_model()
            if (
                self.args.common.test.client.interval > 0
                and (E + 1) % self.args.common.test.client.interval == 0
            ):
                self.test_client_models()

            self.display_metrics()

            # Persist resume checkpoint (overwrites previous one atomically)
            self.save_training_checkpoint(self.checkpoints_dir)

        self.logger.log(
            f"{self.algorithm_name}'s average time taken by each global epoch: "
            f"{int(avg_round_time // 60)} min {(avg_round_time % 60):.2f} sec."
        )

    # ------------------------------------------------------------------
    # aggregate_client_updates override
    # ------------------------------------------------------------------

    @torch.no_grad()
    def aggregate_client_updates(
        self,
        client_packages: "OrderedDict[int, Dict[str, Any]]",
    ) -> None:
        """Save CKA checkpoints (if this is a scheduled round), then aggregate.

        Checkpoint saving happens **before** aggregation so that both the
        pre-aggregation global model (what clients trained from) and the
        post-local-training client models are captured together — exactly
        the pair needed for CKA measurement.

        Args:
            client_packages: Mapping of client ID → package dict.
        """
        round_idx = self.current_epoch + 1  # 1-based
        if self._is_cka_round(round_idx):
            self._save_cka_checkpoint(round_idx, client_packages)

        # L2 drift + gradient alignment + FedAvg weighted aggregation
        super().aggregate_client_updates(client_packages)
