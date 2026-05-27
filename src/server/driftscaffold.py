"""
DriftSCAFFOLDServer — SCAFFOLD with per-round drift and interference instrumentation.

Inherits all measurement logic from DriftFedAvgServer and layers SCAFFOLD's
variance-reduction mechanism on top:
  - Server and per-client control variates (c_global, c_local) correct the
    client-drift bias in local gradients.
  - Aggregation uses SCAFFOLD's y_delta update rule with a global_lr scaling
    factor instead of plain FedAvg weighted averaging.
  - Drift metrics are still computed from the weight-space displacement
    θ_{t-1} → θ_k (before aggregation), so the measurements remain
    comparable across methods.

SCAFFOLD-specific design notes
--------------------------------
1. return_diff = True  — clients return y_delta = θ_k − θ_{t-1} (the local
   parameter *increment*), not the full θ_k.  To recover the local weights
   for drift computation we reconstruct:
       θ_k = θ_{t-1} + y_delta_k
   This is done inside aggregate_client_updates before calling the parent.

2. Control variate update — c_global is updated server-side after aggregation
   exactly as in the base SCAFFOLDServer.  c_local is updated client-side
   inside SCAFFOLDClient.train() and returned via c_delta.

3. LR scheduler — same server-side CosineAnnealingLR as DriftFedAvgServer
   (null client-side scheduler, server injects lr into args each round).

4. CSV schema — identical to DriftFedAvgServer (drift_metrics.csv), so all
   three drift methods (FedAvg / FedProx / SCAFFOLD) produce directly
   comparable output files.

Config key: ``driftscaffold.global_lr`` (set by main.py's parse_args from
``get_hyperparams``; accessible as ``args.driftscaffold.global_lr``).
"""

from argparse import ArgumentParser
from copy import deepcopy
from collections import OrderedDict
from typing import Any, Dict, List

import torch

from src.client.driftscaffold import DriftSCAFFOLDClient
from src.server.driftfedavg import DriftFedAvgServer
from src.utils.drift_metrics import (
    aggregate_drift,
    compute_gradient_alignment,
    compute_layer_drift,
)


class DriftSCAFFOLDServer(DriftFedAvgServer):
    algorithm_name = "DriftSCAFFOLD"
    client_cls = DriftSCAFFOLDClient
    return_diff = True  # clients return y_delta, not full params

    @staticmethod
    def get_hyperparams(args_list=None):
        parser = ArgumentParser()
        parser.add_argument("--global_lr", type=float, default=1.0)
        return parser.parse_args(args_list)

    def __init__(self, args):
        super().__init__(args)

        # Control variates — mirrors SCAFFOLDServer.__init__
        self.c_global: List[torch.Tensor] = [
            torch.zeros_like(param)
            for param in self.public_model_params.values()
        ]
        self.c_local: List[List[torch.Tensor]] = [
            deepcopy(self.c_global) for _ in self.train_clients
        ]

    # ──────────────────────────────────────────────────────────────────────
    # Server → client payload: inject control variates
    # ──────────────────────────────────────────────────────────────────────
    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["c_global"] = self.c_global
        server_package["c_local"] = self.c_local[client_id]
        return server_package

    # ──────────────────────────────────────────────────────────────────────
    # Aggregation: drift measurement + SCAFFOLD update + c_global refresh
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def aggregate_client_updates(
        self, client_packages: OrderedDict[int, Dict[str, Any]]
    ):
        global_state = self.public_model_params  # θ_{t-1}

        per_client_drifts: List[Dict[str, float]] = []
        client_grads: List[Dict[str, torch.Tensor]] = []

        # ── Drift measurement ─────────────────────────────────────────────
        # y_delta = θ_k − θ_{t-1}  →  θ_k = θ_{t-1} + y_delta
        # Pseudo-gradient for interference: g_k = θ_{t-1} − θ_k = −y_delta
        param_names = list(global_state.keys())
        for package in client_packages.values():
            y_delta: List[torch.Tensor] = package["y_delta"]

            # Reconstruct local weights from the increment
            local_state: Dict[str, torch.Tensor] = {
                name: global_state[name].detach().cpu().float()
                      + y_delta[i].detach().cpu().float()
                for i, name in enumerate(param_names)
            }

            drift = compute_layer_drift(local_state, global_state, self.param_taxonomy)
            per_client_drifts.append(drift)

            # Pseudo-gradient: g_k = θ_{t-1} − θ_k  (= −y_delta)
            grads: Dict[str, torch.Tensor] = {
                name: -y_delta[i].detach().cpu().float()
                for i, name in enumerate(param_names)
            }
            client_grads.append(grads)

        self._last_drift_stats = aggregate_drift(per_client_drifts)
        self._last_interference = compute_gradient_alignment(
            client_grads, self.param_taxonomy
        )

        # ── SCAFFOLD parameter update ─────────────────────────────────────
        # θ_{t} = θ_{t-1} + global_lr * mean(y_delta_k)
        y_delta_list = [pkg["y_delta"] for pkg in client_packages.values()]
        c_delta_list = [pkg["c_delta"] for pkg in client_packages.values()]
        weights = torch.ones(len(y_delta_list)) / len(y_delta_list)

        for param, y_deltas in zip(
            self.public_model_params.values(), zip(*y_delta_list)
        ):
            param.data += self.args.driftscaffold.global_lr * torch.sum(
                torch.stack(y_deltas, dim=-1) * weights, dim=-1
            )

        # ── Update global control variate ─────────────────────────────────
        for c_global, c_deltas in zip(self.c_global, zip(*c_delta_list)):
            c_global.data += torch.stack(c_deltas, dim=-1).sum(dim=-1) / self.client_num
