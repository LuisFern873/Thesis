"""
DriftFedAvgServer — FedAvg with per-round drift and interference instrumentation.

Implements the measurement protocol from methodology Sections 3–5:
  - Per-layer L2 client drift (norm / feature / head groups)
  - Pairwise gradient cosine similarity (aggregation interference)
  - Per-client accuracy, fairness gap, convergence flag
  - Server-side CosineAnnealingLR applied once per communication round
  - Full metrics.csv written every round per the Section 7.3 schema

LR scheduler design decision
------------------------------
The methodology requires the LR scheduler to step once per *communication
round*, not per local epoch.  FL-bench's client-side scheduler steps once per
local epoch inside fit(), which would cause 5× more steps than intended for
E=5.  To avoid this without modifying the shared FedAvgClient, we:
  1. Set lr_scheduler.name: null in the config (no client-side scheduler).
  2. Implement server-side CosineAnnealingLR that updates args.optimizer.lr
     before broadcasting to clients each round.
This guarantees all clients start every round from the same, globally-decayed LR.
"""

import csv
import math
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import torch

from src.server.fedavg import FedAvgServer
from src.utils.drift_metrics import (
    aggregate_drift,
    classify_layer,
    compute_gradient_alignment,
    compute_layer_drift,
)
from src.utils.metrics import Metrics

# Convergence thresholds per methodology Section 5.2
CONVERGENCE_THRESHOLDS = {
    "cifar10": 70.0,
    "brain_tumor": 80.0,
}
CONVERGENCE_WINDOW = 5  # must stay above threshold for this many consecutive rounds


class DriftFedAvgServer(FedAvgServer):
    algorithm_name = "DriftFedAvg"

    @staticmethod
    def get_hyperparams(args_list=None):
        """No method-specific hyperparameters for DriftFedAvg.
        Defined so main.py's parent-class walk completes without NameError
        when DriftFedProxServer (which inherits from this class) is loaded.
        """
        from argparse import ArgumentParser, Namespace
        return ArgumentParser().parse_args(args_list or [])

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        # ── Layer taxonomy: param_name → group label ──────────────────────
        self.param_taxonomy: Dict[str, str] = {}
        for name, _ in self.model.named_parameters():
            parts = name.split(".")
            module_name = ".".join(parts[:-1])
            try:
                module = self.model.get_submodule(module_name) if module_name else self.model
            except AttributeError:
                module = self.model
            self.param_taxonomy[name] = classify_layer(name, module)

        # ── Server-side cosine LR schedule ────────────────────────────────
        # Reads lr_init and T_max from config; steps once per round.
        self._lr_init: float = float(args.optimizer.lr)
        self._T_max: int = int(args.common.global_epoch)
        self._eta_min: float = float(
            getattr(getattr(args, "lr_scheduler", None), "eta_min", 1e-5)
            if hasattr(args, "lr_scheduler") else 1e-5
        )

        # ── Convergence tracking ──────────────────────────────────────────
        self._acc_history: List[float] = []
        self._convergence_round: int = -1
        self._threshold: float = CONVERGENCE_THRESHOLDS.get(
            args.dataset.name, 70.0
        )

        # ── CSV initialisation ────────────────────────────────────────────
        # Named drift_metrics.csv to avoid collision with FL-bench's own
        # metrics.csv (written by save_metrics_stats in FedAvgServer).
        self.metrics_csv_path = self.output_dir / "drift_metrics.csv"
        self._init_metrics_csv()

    # ──────────────────────────────────────────────────────────────────────
    # CSV schema (methodology Section 7.3)
    # ──────────────────────────────────────────────────────────────────────
    _CSV_HEADERS = [
        "round",
        "lr",
        "global_acc",
        "global_f1",
        "global_precision",
        "global_recall",
        "convergence_flag",
        # Raw L2 drift (scale-dependent; use for absolute magnitude comparisons)
        "drift_norm_mean",    "drift_norm_std",
        "drift_feature_mean", "drift_feature_std",
        "drift_head_mean",    "drift_head_std",
        # Normalised (RMS) drift = raw / sqrt(N); use for cross-group / cross-arch comparisons
        "drift_norm_norm_mean",    "drift_norm_norm_std",
        "drift_feature_norm_mean", "drift_feature_norm_std",
        "drift_head_norm_mean",    "drift_head_norm_std",
        "interference_norm", "interference_feature", "interference_head",
        "fairness_gap",
        "client_acc_min",
        "client_acc_max",
        "client_acc_std",
    ]

    def _init_metrics_csv(self):
        with open(self.metrics_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(self._CSV_HEADERS)

    # ──────────────────────────────────────────────────────────────────────
    # Server-side cosine LR decay
    # ──────────────────────────────────────────────────────────────────────
    def _cosine_lr(self, t: int) -> float:
        """Return the cosine-annealed LR for round t (0-indexed)."""
        return self._eta_min + 0.5 * (self._lr_init - self._eta_min) * (
            1 + math.cos(math.pi * t / self._T_max)
        )

    def _update_client_lr(self):
        """Inject the current round's LR into args so clients pick it up."""
        new_lr = self._cosine_lr(self.current_epoch)
        self.args.optimizer.lr = new_lr

    # ──────────────────────────────────────────────────────────────────────
    # Override train_one_round to apply LR before client dispatch
    # ──────────────────────────────────────────────────────────────────────
    def train_one_round(self):
        self._update_client_lr()
        client_packages = self.trainer.train()
        self.aggregate_client_updates(client_packages)

    # ──────────────────────────────────────────────────────────────────────
    # Core override: compute drift + interference BEFORE aggregation
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def aggregate_client_updates(
        self, client_packages: OrderedDict[int, Dict[str, Any]]
    ):
        global_state = self.public_model_params  # θ_{t-1}

        per_client_drifts: List[Dict[str, float]] = []
        client_grads: List[Dict[str, torch.Tensor]] = []

        for package in client_packages.values():
            local_state = package["regular_model_params"]

            # L2 drift per layer group
            drift = compute_layer_drift(local_state, global_state, self.param_taxonomy)
            per_client_drifts.append(drift)

            # Pseudo-gradient: g_k = θ_{t-1} − θ_k
            grads = {
                name: global_state[name].detach().cpu().float()
                       - local_state[name].detach().cpu().float()
                for name in global_state
                if name in local_state
            }
            client_grads.append(grads)

        # Aggregate drift and interference
        self._last_drift_stats = aggregate_drift(per_client_drifts)
        self._last_interference = compute_gradient_alignment(
            client_grads, self.param_taxonomy
        )

        # Standard FedAvg weighted aggregation
        super().aggregate_client_updates(client_packages)

    # ──────────────────────────────────────────────────────────────────────
    # Override display_metrics to write CSV + TensorBoard after eval
    # ──────────────────────────────────────────────────────────────────────
    def display_metrics(self):
        super().display_metrics()

        # Guard: drift stats may not exist on the very first call if
        # aggregate_client_updates hasn't run yet (shouldn't happen, but safe).
        if not hasattr(self, "_last_drift_stats"):
            return

        round_idx = self.current_epoch + 1

        # ── Global accuracy / F1 / precision / recall ─────────────────────
        global_acc = global_f1 = global_prec = global_rec = 0.0
        if (
            round_idx in self.test_results
            and "centralized" in self.test_results[round_idx]
        ):
            m: Metrics = self.test_results[round_idx]["centralized"]["after"]["test"]
            if m.size > 0:
                global_acc  = m.accuracy
                global_f1   = m.macro_f1
                global_prec = m.macro_precision
                global_rec  = m.macro_recall

        # ── Convergence flag ──────────────────────────────────────────────
        self._acc_history.append(global_acc)
        convergence_flag = 0
        if self._convergence_round == -1:
            window = self._acc_history[-CONVERGENCE_WINDOW:]
            if (
                len(window) == CONVERGENCE_WINDOW
                and all(a >= self._threshold for a in window)
            ):
                self._convergence_round = round_idx - CONVERGENCE_WINDOW + 1
                convergence_flag = 1

        # ── Per-client fairness metrics ───────────────────────────────────
        client_accs: List[float] = []
        for cid in self.selected_clients:
            if self.current_epoch in self.client_metrics.get(cid, {}):
                m_c = self.client_metrics[cid][self.current_epoch]["after"]["test"]
                if m_c.size > 0:
                    client_accs.append(m_c.accuracy)

        if client_accs:
            acc_arr = np.array(client_accs)
            fairness_gap  = float(acc_arr.max() - acc_arr.min())
            client_acc_min = float(acc_arr.min())
            client_acc_max = float(acc_arr.max())
            client_acc_std = float(acc_arr.std())
        else:
            fairness_gap = client_acc_min = client_acc_max = client_acc_std = 0.0

        ds = self._last_drift_stats
        ig = self._last_interference
        current_lr = self._cosine_lr(self.current_epoch)

        # ── Write CSV row ─────────────────────────────────────────────────
        row = [
            round_idx,
            round(current_lr, 8),
            round(global_acc,  4),
            round(global_f1,   4),
            round(global_prec, 4),
            round(global_rec,  4),
            convergence_flag,
            # Raw L2 drift
            round(ds["norm"]["mean"],    6), round(ds["norm"]["std"],    6),
            round(ds["feature"]["mean"], 6), round(ds["feature"]["std"], 6),
            round(ds["head"]["mean"],    6), round(ds["head"]["std"],    6),
            # Normalised (RMS) drift
            round(ds["norm"]["norm_mean"],    6), round(ds["norm"]["norm_std"],    6),
            round(ds["feature"]["norm_mean"], 6), round(ds["feature"]["norm_std"], 6),
            round(ds["head"]["norm_mean"],    6), round(ds["head"]["norm_std"],    6),
            round(ig["norm"],    6),
            round(ig["feature"], 6),
            round(ig["head"],    6),
            round(fairness_gap,    4),
            round(client_acc_min,  4),
            round(client_acc_max,  4),
            round(client_acc_std,  4),
        ]
        with open(self.metrics_csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        # ── TensorBoard ───────────────────────────────────────────────────
        if hasattr(self, "tensorboard"):
            tb = self.tensorboard
            tb.add_scalar("train/lr",              current_lr,    round_idx)
            tb.add_scalar("train/global_acc",      global_acc,    round_idx)
            tb.add_scalar("train/global_f1",       global_f1,     round_idx)
            tb.add_scalar("train/global_precision",global_prec,   round_idx)
            tb.add_scalar("train/global_recall",   global_rec,    round_idx)
            tb.add_scalar("fairness/gap",          fairness_gap,  round_idx)
            tb.add_scalar("fairness/acc_std",      client_acc_std,round_idx)
            tb.add_scalar("fairness/acc_min",      client_acc_min,round_idx)
            for group in ["norm", "feature", "head"]:
                tb.add_scalar(f"drift/{group}_mean",      ds[group]["mean"],      round_idx)
                tb.add_scalar(f"drift/{group}_std",       ds[group]["std"],       round_idx)
                tb.add_scalar(f"drift/{group}_norm_mean", ds[group]["norm_mean"], round_idx)
                tb.add_scalar(f"drift/{group}_norm_std",  ds[group]["norm_std"],  round_idx)
                tb.add_scalar(f"interference/{group}",    ig[group],              round_idx)
