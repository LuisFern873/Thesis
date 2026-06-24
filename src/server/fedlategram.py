# src/server/fedlategram.py
"""
FedLateGram: Federated Learning with Late-Layer Gram Matrix Regularization.

Inherits from CKADriftFedAvgServer to get:
  - drift_metrics.csv (L2 drift + gradient alignment per layer group)
  - CKA checkpoints (global + client state dicts at scheduled rounds)
  - resume training support
  - server-side cosine LR schedule

Adds on top:
  - Late-layer detection (last `late_fraction` of backbone + classifier)
  - Global gram matrix reference (proxy dataset or avg of client grams)
  - Selective FedAvg aggregation (only late layers after warm-up)
  - Gram penalty broadcast to clients via package()

MRO:
  FedLateGramServer
    → CKADriftFedAvgServer   (CKA checkpoints + resume train loop)
      → DriftFedAvgServer    (drift CSV + cosine LR + gradient alignment)
        → FedAvgServer       (base FL loop)
"""
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Any, Dict

import torch
from omegaconf import DictConfig

from src.client.fedlategram import FedLateGramClient
from src.server.ckadriftfedavg import CKADriftFedAvgServer


class FedLateGramServer(CKADriftFedAvgServer):

    algorithm_name = "FedLateGram"
    client_cls = FedLateGramClient
    return_diff = False

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument(
            "--lam", type=float, default=0.1,
            help="Gram penalty weight λ",
        )
        parser.add_argument(
            "--tau_cka", type=float, default=0.85,
            help="CKA threshold to detect late layers (informational; fallback used)",
        )
        parser.add_argument(
            "--late_fraction", type=float, default=0.4,
            help="Fraction of base submodules treated as 'late' (fallback)",
        )
        parser.add_argument(
            "--freeze_strategy", type=str, default="warm_then_freeze",
            choices=["full_freeze", "slow_update", "warm_then_freeze"],
            help="How to handle early layers after warm-up",
        )
        parser.add_argument(
            "--T_warm", type=int, default=10,
            help="Warm-up rounds before activating gram penalty / freezing",
        )
        parser.add_argument(
            "--alpha_early_lr", type=float, default=0.05,
            help="Gradient scale factor for early layers (slow_update only)",
        )
        parser.add_argument(
            "--freq_early", type=int, default=5,
            help="Aggregate early layers every N rounds (slow_update only)",
        )
        parser.add_argument(
            "--gram_ref", type=str, default="proxy",
            choices=["proxy", "avg_local"],
            help="Source for global gram reference: proxy dataset or avg of client grams",
        )
        parser.add_argument(
            "--proxy_size", type=int, default=256,
            help="Number of samples for proxy gram reference (gram_ref=proxy)",
        )
        return parser.parse_args(args_list)

    def __init__(self, args: DictConfig):
        # CKADriftFedAvgServer.__init__ → DriftFedAvgServer.__init__ → FedAvgServer.__init__
        # This sets up: drift CSV, cosine LR, layer taxonomy, CKA checkpoint dir,
        # run_metadata.json, and all FL-bench base infrastructure.
        super().__init__(args)

        # Late-layer names (relative to self.model)
        self.late_layer_names: list[str] = self._detect_late_layers()
        self.logger.log(
            f"[FedLateGram] Late layers ({len(self.late_layer_names)}): "
            f"{self.late_layer_names}"
        )

        # Global gram matrices, updated each round after warm-up
        self.global_grams: dict[str, torch.Tensor] = {}

        # Proxy dataloader built lazily (after warm-up) to avoid consuming
        # random numbers during __init__ and displacing client_sample_stream.
        self._proxy_loader = None

        # Warm-up state
        self._warming_up = (
            self.args.fedlategram.freeze_strategy == "warm_then_freeze"
        )

    # ------------------------------------------------------------------
    # Layer detection
    # ------------------------------------------------------------------

    def _detect_late_layers(self) -> list[str]:
        """Return submodule names considered 'late': last `late_fraction`
        of top-level children of self.model.base, plus 'classifier'."""
        base_modules = [
            name
            for name, _ in self.model.base.named_modules()
            if name and "." not in name
        ]
        if not base_modules:
            return ["base", "classifier"]

        fraction = self.args.fedlategram.late_fraction
        cutoff = max(0, int(len(base_modules) * (1 - fraction)))
        return [f"base.{n}" for n in base_modules[cutoff:]] + ["classifier"]

    def _is_late_param(self, param_name: str) -> bool:
        return any(param_name.startswith(p) for p in self.late_layer_names)

    # ------------------------------------------------------------------
    # Proxy dataloader
    # ------------------------------------------------------------------

    def _build_proxy_loader(self):
        from torch.utils.data import DataLoader, Subset
        proxy_size = self.args.fedlategram.proxy_size
        indices = list(range(min(proxy_size, len(self.dataset))))
        return DataLoader(
            Subset(self.dataset, indices),
            batch_size=min(64, proxy_size),
            shuffle=False,
            num_workers=self.args.common.dataloader_num_workers,
            pin_memory=self.args.common.use_cuda,
        )

    # ------------------------------------------------------------------
    # Global gram computation (server side, proxy dataset)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_global_grams(self) -> dict[str, torch.Tensor]:
        """Compute (D,D) gram matrices from global model on proxy dataset."""
        self.model.eval()
        self.dataset.eval()

        target_names = [n for n in self.late_layer_names if n != "classifier"]
        activations: dict[str, list[torch.Tensor]] = {n: [] for n in target_names}
        handles = []

        for full_name in target_names:
            module = self.model
            try:
                for part in full_name.split("."):
                    module = getattr(module, part)
            except AttributeError:
                continue

            def make_hook(n):
                def hook(mod, inp, out):
                    t = out.detach()
                    if t.dim() == 4:
                        t = t.mean(dim=(2, 3))   # (N,C,H,W) → (N,C)
                    elif t.dim() > 2:
                        t = t.flatten(start_dim=1)
                    activations[n].append(t.cpu())
                return hook

            handles.append(module.register_forward_hook(make_hook(full_name)))

        for batch in self._proxy_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            self.model(x.to(self.device))

        for h in handles:
            h.remove()

        grams = {}
        for name, acts in activations.items():
            if acts:
                Phi = torch.cat(acts, dim=0)          # (N, D)
                grams[name] = (Phi.T @ Phi) / Phi.shape[0]  # (D, D)

        self.model.train()
        return grams

    # ------------------------------------------------------------------
    # train_one_round — merges DriftFedAvg LR injection + FedLateGram logic
    # ------------------------------------------------------------------

    def train_one_round(self):
        # 1. Cosine LR decay (from DriftFedAvgServer)
        self._update_client_lr()

        # 2. Warm-up flag management
        if self.args.fedlategram.freeze_strategy == "warm_then_freeze":
            was_warming = self._warming_up
            self._warming_up = self.current_epoch < self.args.fedlategram.T_warm
            if was_warming and not self._warming_up:
                self.logger.log(
                    f"[FedLateGram] Warm-up complete at round {self.current_epoch + 1}. "
                    "Activating gram penalty and early-layer freezing."
                )
        else:
            self._warming_up = False

        # 3. Build proxy loader and refresh global grams (after warm-up only)
        if self.args.fedlategram.gram_ref == "proxy" and not self._warming_up:
            if self._proxy_loader is None:
                self._proxy_loader = self._build_proxy_loader()
            self.global_grams = self._compute_global_grams()

        # 4. Client training
        client_packages = self.trainer.train()

        # 5. avg_local gram reference
        if self.args.fedlategram.gram_ref == "avg_local" and not self._warming_up:
            self._aggregate_local_grams(client_packages)

        # 6. Log gram losses
        gram_losses = [pkg.get("loss_gram_mean", 0.0) for pkg in client_packages.values()]
        task_losses = [pkg.get("loss_task_mean", 0.0) for pkg in client_packages.values()]
        if any(g > 0 for g in gram_losses):
            self.logger.log(
                f"[FedLateGram Round {self.current_epoch + 1}] "
                f"avg task: {sum(task_losses)/len(task_losses):.4f} | "
                f"avg gram: {sum(gram_losses)/len(gram_losses):.4f}"
            )

        # 7. Aggregate (triggers CKA checkpoint + drift metrics via super chain)
        self.aggregate_client_updates(client_packages)

    # ------------------------------------------------------------------
    # aggregate_client_updates — CKA checkpoint → drift metrics → FLG selective agg
    #
    # Call order through MRO:
    #   FedLateGramServer.aggregate_client_updates
    #     → calls CKADriftFedAvgServer.aggregate_client_updates
    #         which saves CKA checkpoint if scheduled, then calls
    #       → DriftFedAvgServer.aggregate_client_updates
    #           which computes drift + gradient alignment, then calls
    #         → FedAvgServer.aggregate_client_updates  (standard FedAvg)
    #
    # We intercept AFTER drift metrics are computed but REPLACE the
    # FedAvg weighted average with FedLateGram's selective aggregation.
    # This is achieved by calling super() up through CKA+Drift layers
    # (giving them their checkpoint/CSV work) but then replacing the
    # FedAvg aggregation with our selective version.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def aggregate_client_updates(
        self, client_packages: OrderedDict[int, Dict[str, Any]]
    ):
        # Let CKADriftFedAvg do: CKA checkpoint save + drift/interference
        # computation + _last_drift_stats / _last_interference population.
        # We call the CKA layer directly so both CKA saving and drift CSV
        # writing happen normally.
        #
        # But we must NOT let FedAvgServer.aggregate_client_updates run,
        # because we need selective aggregation instead of full FedAvg.
        # Solution: call CKA + Drift work manually, then do FLG aggregation.

        # ── CKA checkpoint (CKADriftFedAvgServer logic) ──────────────────
        round_idx = self.current_epoch + 1
        if self._is_cka_round(round_idx):
            self._save_cka_checkpoint(round_idx, client_packages)

        # ── Drift metrics (DriftFedAvgServer logic) ───────────────────────
        from src.utils.drift_metrics import (
            aggregate_drift,
            compute_gradient_alignment,
            compute_layer_drift,
        )
        global_state = self.public_model_params
        per_client_drifts = []
        client_grads = []
        for package in client_packages.values():
            local_state = package["regular_model_params"]
            per_client_drifts.append(
                compute_layer_drift(local_state, global_state, self.param_taxonomy)
            )
            client_grads.append({
                name: global_state[name].detach().cpu().float()
                      - local_state[name].detach().cpu().float()
                for name in global_state if name in local_state
            })
        self._last_drift_stats = aggregate_drift(per_client_drifts)
        self._last_interference = compute_gradient_alignment(
            client_grads, self.param_taxonomy
        )

        # ── FedLateGram selective aggregation ────────────────────────────
        if self._warming_up:
            # Warm-up: standard FedAvg on all parameters
            # Call FedAvgServer directly to avoid re-running CKA/drift
            from src.server.fedavg import FedAvgServer
            FedAvgServer.aggregate_client_updates(self, client_packages)
            return

        client_weights = [pkg["weight"] for pkg in client_packages.values()]
        total_weight = sum(client_weights)
        weights = torch.tensor(
            [w / total_weight for w in client_weights], dtype=torch.float
        )

        for name, global_param in self.public_model_params.items():
            if self._is_late_param(name):
                stacked = torch.stack(
                    [pkg["regular_model_params"][name] for pkg in client_packages.values()],
                    dim=-1,
                )
                global_param.data = torch.sum(stacked * weights, dim=-1)

            elif self.args.fedlategram.freeze_strategy == "slow_update":
                if self.current_epoch % self.args.fedlategram.freq_early == 0:
                    stacked = torch.stack(
                        [pkg["regular_model_params"][name] for pkg in client_packages.values()],
                        dim=-1,
                    )
                    global_param.data = torch.sum(stacked * weights, dim=-1)
                # else: keep current global value

            # full_freeze: keep current global value (no-op)

        self.model.load_state_dict(self.public_model_params, strict=False)

    # ------------------------------------------------------------------
    # package — broadcast late layer names + global grams to clients
    # ------------------------------------------------------------------

    def package(self, client_id: int) -> dict:
        pkg = super().package(client_id)
        pkg["late_layer_names"] = self.late_layer_names
        pkg["global_grams"] = {k: v.clone().cpu() for k, v in self.global_grams.items()}
        pkg["lam"] = self.args.fedlategram.lam
        pkg["freeze_strategy"] = self.args.fedlategram.freeze_strategy
        pkg["warming_up"] = self._warming_up
        pkg["alpha_early_lr"] = self.args.fedlategram.alpha_early_lr
        return pkg

    # ------------------------------------------------------------------
    # avg_local gram aggregation helper
    # ------------------------------------------------------------------

    def _aggregate_local_grams(
        self, client_packages: OrderedDict[int, Dict[str, Any]]
    ):
        """Weighted average of clients' local gram matrices."""
        client_weights = [pkg["weight"] for pkg in client_packages.values()]
        total = sum(client_weights)
        self.global_grams = {}
        for pkg, w in zip(client_packages.values(), client_weights):
            for name, G in pkg.get("local_grams", {}).items():
                if name not in self.global_grams:
                    self.global_grams[name] = torch.zeros_like(G)
                self.global_grams[name] += (w / total) * G.cpu()

    # ------------------------------------------------------------------
    # Resume checkpoint — extend base with FedLateGram-specific state
    # ------------------------------------------------------------------

    def save_training_checkpoint(self, checkpoint_dir) -> None:
        """Override to also persist FedLateGram state (grams, warm-up flag)."""
        import os
        import torch as _torch
        os.makedirs(checkpoint_dir, exist_ok=True)

        state = {
            "current_epoch": self.current_epoch,
            "public_model_params": {
                k: v.detach().cpu() for k, v in self.public_model_params.items()
            },
            "clients_personal_model_params": self.clients_personal_model_params,
            "client_optimizer_states": self.client_optimizer_states,
            "client_lr_scheduler_states": self.client_lr_scheduler_states,
            "client_sample_stream": self.client_sample_stream,
            "aggregated_client_metrics": self.aggregated_client_metrics,
            # FedLateGram extras
            "global_grams": {k: v.cpu() for k, v in self.global_grams.items()},
            "_warming_up": self._warming_up,
        }
        dest = checkpoint_dir / "training_state.pt"
        tmp  = checkpoint_dir / "training_state.pt.tmp"
        _torch.save(state, tmp)
        os.replace(tmp, dest)
        self.logger.log(
            f"  [FedLateGram] Saved checkpoint after round "
            f"{self.current_epoch + 1} → {dest}"
        )

    def load_training_checkpoint(self, checkpoint_path) -> int:
        """Override to also restore FedLateGram state."""
        self.logger.log(
            f"  [FedLateGram] Loading checkpoint from {checkpoint_path}"
        )
        import torch as _torch
        state = _torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.current_epoch = state["current_epoch"]
        for k, v in state["public_model_params"].items():
            self.public_model_params[k].data.copy_(v)
        self.model.load_state_dict(self.public_model_params, strict=False)

        self.clients_personal_model_params = state["clients_personal_model_params"]
        self.client_optimizer_states       = state["client_optimizer_states"]
        self.client_lr_scheduler_states    = state["client_lr_scheduler_states"]
        self.client_sample_stream          = state["client_sample_stream"]
        self.aggregated_client_metrics     = state["aggregated_client_metrics"]

        # FedLateGram extras
        self.global_grams  = state.get("global_grams", {})
        self._warming_up   = state.get("_warming_up", self._warming_up)

        resume_from = self.current_epoch + 1
        self.logger.log(
            f"  [FedLateGram] Restored. Resuming from round "
            f"{resume_from + 1}/{self.args.common.global_epoch}."
        )
        return resume_from
