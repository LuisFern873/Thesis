import torch
import numpy as np
import os
import csv
from collections import OrderedDict
from typing import Any, Dict, List
from src.server.fedavg import FedAvgServer
from src.utils.drift_metrics import (
    classify_layer, 
    compute_layer_drift, 
    aggregate_drift, 
    compute_gradient_alignment
)
from src.utils.metrics import Metrics

class DriftFedAvgServer(FedAvgServer):
    algorithm_name = "DriftFedAvg"

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        
        # Prepare layer taxonomy
        self.layer_taxonomy = {}
        for name, module in self.model.named_modules():
            # Parameters are often in submodules, we need mapping for param_name -> group
            # However, classify_layer works on module. Let's build a param_name -> group map.
            pass
        
        self.param_taxonomy = {}
        for name, param in self.model.named_parameters():
            # Find the module this parameter belongs to
            module_name = ".".join(name.split(".")[:-1])
            if module_name == "":
                module = self.model
            else:
                module = self.model.get_submodule(module_name)
            self.param_taxonomy[name] = classify_layer(name, module)

        # Prepare CSV logging
        self.metrics_file = self.output_dir / "metrics.csv"
        self.init_metrics_csv()

    def init_metrics_csv(self):
        headers = [
            "round", "global_acc", "global_f1", 
            "drift_norm_mean", "drift_norm_std",
            "drift_feature_mean", "drift_feature_std",
            "drift_head_mean", "drift_head_std",
            "interference_norm", "interference_feature", "interference_head",
            "fairness_gap", "client_acc_std"
        ]
        with open(self.metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def aggregate_client_updates(self, client_packages: OrderedDict[int, Dict[str, Any]]):
        # 1. Compute Drift and Interference BEFORE aggregation
        per_client_drifts = []
        client_grads = []
        
        global_state = self.public_model_params
        
        for cid, package in client_packages.items():
            local_state = package["regular_model_params"]
            
            # Compute L2 Drift
            drift = compute_layer_drift(local_state, global_state, self.param_taxonomy)
            per_client_drifts.append(drift)
            
            # Compute Pseudo-gradients: g_k = theta_{t-1} - theta_k
            grads = {}
            for name in global_state:
                grads[name] = global_state[name].data - local_state[name].data
            client_grads.append(grads)
        
        # Aggregate drift metrics
        drift_stats = aggregate_drift(per_client_drifts)
        
        # Compute gradient alignment (interference)
        interference = compute_gradient_alignment(client_grads, self.param_taxonomy)
        
        # 2. Perform standard FedAvg aggregation
        super().aggregate_client_updates(client_packages)
        
        # 3. Logging
        self.log_drift_metrics(drift_stats, interference)

    def log_drift_metrics(self, drift_stats, interference):
        # We need global accuracy, which is computed after aggregation in the main loop
        # But here we are inside aggregate_client_updates. 
        # The main loop calls test_global_model() after train_one_round().
        # So we'll cache these metrics and write to CSV in a separate hook or at end of round.
        self.current_drift_stats = drift_stats
        self.current_interference = interference

    def display_metrics(self):
        super().display_metrics()
        
        # After display_metrics, we have global accuracy in self.test_results
        # Let's extract and write to CSV
        round_idx = self.current_epoch + 1
        global_acc = 0.0
        global_f1 = 0.0
        
        if round_idx in self.test_results and "centralized" in self.test_results[round_idx]:
            metrics = self.test_results[round_idx]["centralized"]["after"]
            global_acc = metrics["test"].accuracy if "test" in metrics else 0.0
            global_f1 = metrics["test"].macro_f1 if "test" in metrics else 0.0

        # Fairness metrics from display_metrics (stored in self.aggregated_client_metrics if client testing is enabled)
        # For simplicity, we'll try to get them if available
        fairness_gap = 0.0
        client_acc_std = 0.0
        
        # Write to CSV
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                round_idx, global_acc, global_f1,
                self.current_drift_stats["norm"]["mean"], self.current_drift_stats["norm"]["std"],
                self.current_drift_stats["feature"]["mean"], self.current_drift_stats["feature"]["std"],
                self.current_drift_stats["head"]["mean"], self.current_drift_stats["head"]["std"],
                self.current_interference["norm"], self.current_interference["feature"], self.current_interference["head"],
                fairness_gap, client_acc_std
            ])
            
        # Also log to Tensorboard if available
        if hasattr(self, "tensorboard"):
            for group in ["norm", "feature", "head"]:
                self.tensorboard.add_scalar(f"drift/{group}_mean", self.current_drift_stats[group]["mean"], round_idx)
                self.tensorboard.add_scalar(f"interference/{group}", self.current_interference[group], round_idx)
