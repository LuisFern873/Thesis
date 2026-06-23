# src/server/fedlategram.py
"""
FedLateGram: Federated Learning with Late-Layer Gram Matrix Regularization.

Differentiates early vs late layers based on CKA analysis findings:
- Early layers (CKA ≈ 1.0): frozen or slowed after warm-up
- Late layers (CKA collapses under heterogeneity): aggregated normally + gram penalty
"""
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict

import torch
from omegaconf import DictConfig

from src.client.fedlategram import FedLateGramClient
from src.server.fedavg import FedAvgServer


class FedLateGramServer(FedAvgServer):

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
            "--T_warm", type=int, default=20,
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
        super().__init__(args)

        # Detect which parameter names belong to "late" layers
        self.late_layer_names: list[str] = self._detect_late_layers()
        self.logger.log(
            f"[FedLateGram] Late layers ({len(self.late_layer_names)}): "
            f"{self.late_layer_names}"
        )

        # Global gram matrices updated each round (keyed by submodule name)
        self.global_grams: dict[str, torch.Tensor] = {}

        # Proxy dataloader for gram_ref=proxy
        self._proxy_loader = None
        if self.args.fedlategram.gram_ref == "proxy":
            self._proxy_loader = self._build_proxy_loader()

        # Warm-up state
        self._warming_up = (
            self.args.fedlategram.freeze_strategy == "warm_then_freeze"
        )

    # ------------------------------------------------------------------
    # Layer detection
    # ------------------------------------------------------------------

    def _detect_late_layers(self) -> list[str]:
        """
        Return the submodule names (relative to self.model) that are
        considered 'late'.  Uses the last `late_fraction` of direct
        named submodules inside self.model.base, plus 'classifier'.
        """
        # Collect immediate named submodules of base (depth-1 names only)
        base_modules = [
            name
            for name, _ in self.model.base.named_modules()
            if name and "." not in name  # top-level children only
        ]

        if not base_modules:
            # Fallback: all-base treated as late
            return ["base", "classifier"]

        fraction = self.args.fedlategram.late_fraction
        cutoff = max(0, int(len(base_modules) * (1 - fraction)))
        late_base = [f"base.{n}" for n in base_modules[cutoff:]]
        return late_base + ["classifier"]

    def _is_late_param(self, param_name: str) -> bool:
        """True if a named parameter belongs to a late layer."""
        return any(param_name.startswith(prefix) for prefix in self.late_layer_names)

    # ------------------------------------------------------------------
    # Proxy dataloader
    # ------------------------------------------------------------------

    def _build_proxy_loader(self):
        from torch.utils.data import DataLoader, Subset

        proxy_size = self.args.fedlategram.proxy_size
        indices = list(range(min(proxy_size, len(self.dataset))))
        subset = Subset(self.dataset, indices)
        return DataLoader(
            subset,
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
        """
        Compute gram matrices from the current global model using the
        proxy dataset.  Only late submodules (excluding 'classifier')
        are hooked because we care about feature-space gram similarity.
        """
        self.model.eval()
        self.dataset.eval()

        # Only hook submodules that are inside self.model (not 'classifier' itself —
        # classifier activations = logits which are less informative for style).
        target_module_names = [n for n in self.late_layer_names if n != "classifier"]

        activations: dict[str, list[torch.Tensor]] = {n: [] for n in target_module_names}
        handles = []

        for full_name in target_module_names:
            module = self.model
            try:
                for part in full_name.split("."):
                    module = getattr(module, part)
            except AttributeError:
                continue  # skip if submodule path doesn't exist

            def make_hook(n):
                def hook(mod, inp, out):
                    activations[n].append(out.detach().flatten(start_dim=1).cpu())
                return hook

            handles.append(module.register_forward_hook(make_hook(full_name)))

        for batch in self._proxy_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device)
            self.model(x)

        for h in handles:
            h.remove()

        grams: dict[str, torch.Tensor] = {}
        for name, acts in activations.items():
            if acts:
                Phi = torch.cat(acts, dim=0)          # (N, D)
                grams[name] = (Phi @ Phi.T) / Phi.shape[0]  # (N, N)

        self.model.train()
        return grams

    # ------------------------------------------------------------------
    # FL-bench overrides
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Training loop with resume-checkpoint support.

        Writes ``<output_dir>/checkpoints/training_state.pt`` after every
        round (atomic overwrite).  Pass
        ``common.resume_checkpoint=<path>`` on the CLI to resume.
        Also persists ``global_grams`` so the gram reference is restored
        correctly after a resume.
        """
        from pathlib import Path as _Path

        self._checkpoints_dir = self.output_dir / "checkpoints"
        import os
        os.makedirs(self._checkpoints_dir, exist_ok=True)

        # ---- resolve resume path ----------------------------------------
        resume_path = getattr(self.args.common, "resume_checkpoint", None)
        start_round = 0

        if resume_path and str(resume_path).lower() not in ("", "null", "none"):
            rp = _Path(resume_path)
            if rp.is_dir():
                rp = rp / "training_state.pt"
            if rp.is_file():
                start_round = self._load_flg_checkpoint(rp)
            else:
                self.logger.log(
                    f"[FedLateGram] WARNING: resume_checkpoint '{resume_path}' "
                    "not found — starting from scratch."
                )

        # ---- main round loop (mirrors FedAvgServer.train) ---------------
        import time
        avg_round_time = 0
        for E in self.train_progress_bar:
            if E < start_round:
                continue

            self.current_epoch = E
            self.verbose = (self.current_epoch + 1) % self.args.common.verbose_gap == 0

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

            # Persist resume checkpoint (atomic overwrite)
            self._save_flg_checkpoint(self._checkpoints_dir)

        self.logger.log(
            f"{self.algorithm_name}'s average time taken by each global epoch: "
            f"{int(avg_round_time // 60)} min {(avg_round_time % 60):.2f} sec."
        )

    def _save_flg_checkpoint(self, checkpoint_dir) -> None:
        """Save training state including FedLateGram-specific fields."""
        import os
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
            # FedLateGram-specific state
            "global_grams": {k: v.cpu() for k, v in self.global_grams.items()},
            "_warming_up": self._warming_up,
        }
        dest = checkpoint_dir / "training_state.pt"
        tmp = checkpoint_dir / "training_state.pt.tmp"
        torch.save(state, tmp)
        import os as _os
        _os.replace(tmp, dest)
        self.logger.log(
            f"  [FedLateGram] Saved checkpoint after round "
            f"{self.current_epoch + 1} → {dest}"
        )

    def _load_flg_checkpoint(self, checkpoint_path) -> int:
        """Restore server state from checkpoint. Returns the next round index."""
        self.logger.log(f"  [FedLateGram] Loading checkpoint from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.current_epoch = state["current_epoch"]
        for k, v in state["public_model_params"].items():
            self.public_model_params[k].data.copy_(v)
        self.model.load_state_dict(self.public_model_params, strict=False)

        self.clients_personal_model_params = state["clients_personal_model_params"]
        self.client_optimizer_states = state["client_optimizer_states"]
        self.client_lr_scheduler_states = state["client_lr_scheduler_states"]
        self.client_sample_stream = state["client_sample_stream"]
        self.aggregated_client_metrics = state["aggregated_client_metrics"]

        # Restore FedLateGram-specific state
        self.global_grams = state.get("global_grams", {})
        self._warming_up = state.get("_warming_up", self._warming_up)

        resume_from = self.current_epoch + 1
        self.logger.log(
            f"  [FedLateGram] Restored. Resuming from round "
            f"{resume_from + 1}/{self.args.common.global_epoch}."
        )
        return resume_from

    def train_one_round(self):
        # Update warm-up flag
        if self.args.fedlategram.freeze_strategy == "warm_then_freeze":
            was_warming = self._warming_up
            self._warming_up = self.current_epoch < self.args.fedlategram.T_warm
            if was_warming and not self._warming_up:
                self.logger.log(
                    f"[FedLateGram] Warm-up complete at round {self.current_epoch}. "
                    "Activating gram penalty and early-layer freezing."
                )
        else:
            self._warming_up = False

        # Refresh global gram reference before clients train
        if self.args.fedlategram.gram_ref == "proxy" and not self._warming_up:
            self.global_grams = self._compute_global_grams()

        client_packages = self.trainer.train()

        # Build gram reference from local grams if avg_local strategy
        if self.args.fedlategram.gram_ref == "avg_local" and not self._warming_up:
            self._aggregate_local_grams(client_packages)

        # Log per-round gram loss
        gram_losses = [
            pkg.get("loss_gram_mean", 0.0) for pkg in client_packages.values()
        ]
        task_losses = [
            pkg.get("loss_task_mean", 0.0) for pkg in client_packages.values()
        ]
        if any(g > 0 for g in gram_losses):
            avg_gram = sum(gram_losses) / len(gram_losses)
            avg_task = sum(task_losses) / len(task_losses)
            self.logger.log(
                f"[FedLateGram Round {self.current_epoch + 1}] "
                f"avg task loss: {avg_task:.4f} | avg gram loss: {avg_gram:.4f}"
            )

        self.aggregate_client_updates(client_packages)

    def package(self, client_id: int) -> dict:
        pkg = super().package(client_id)
        pkg["late_layer_names"] = self.late_layer_names
        # Move grams to CPU for safe serialization (Ray workers, etc.)
        pkg["global_grams"] = {
            k: v.clone().cpu() for k, v in self.global_grams.items()
        }
        pkg["lam"] = self.args.fedlategram.lam
        pkg["freeze_strategy"] = self.args.fedlategram.freeze_strategy
        pkg["warming_up"] = self._warming_up
        pkg["alpha_early_lr"] = self.args.fedlategram.alpha_early_lr
        return pkg

    @torch.no_grad()
    def aggregate_client_updates(
        self, client_packages: OrderedDict[int, Dict[str, Any]]
    ):
        """
        Selective FedAvg aggregation:
        - During warm-up: standard FedAvg on all parameters.
        - After warm-up:
            - Late layers: always aggregated.
            - Early layers: handled per freeze_strategy.
        """
        if self._warming_up:
            # Standard FedAvg on everything during warm-up
            super().aggregate_client_updates(client_packages)
            return

        client_weights = [pkg["weight"] for pkg in client_packages.values()]
        total_weight = sum(client_weights)
        weights = torch.tensor(
            [w / total_weight for w in client_weights], dtype=torch.float
        )

        for name, global_param in self.public_model_params.items():
            if self._is_late_param(name):
                # Always aggregate late layers
                stacked = torch.stack(
                    [
                        pkg["regular_model_params"][name]
                        for pkg in client_packages.values()
                    ],
                    dim=-1,
                )
                global_param.data = torch.sum(stacked * weights, dim=-1)

            elif self.args.fedlategram.freeze_strategy == "slow_update":
                # Aggregate early layers only every freq_early rounds
                if self.current_epoch % self.args.fedlategram.freq_early == 0:
                    stacked = torch.stack(
                        [
                            pkg["regular_model_params"][name]
                            for pkg in client_packages.values()
                        ],
                        dim=-1,
                    )
                    global_param.data = torch.sum(stacked * weights, dim=-1)
                # else: keep current global value unchanged

            # full_freeze: retain current global value unconditionally (no-op)

        self.model.load_state_dict(self.public_model_params, strict=False)

    # ------------------------------------------------------------------
    # Local gram aggregation (avg_local strategy)
    # ------------------------------------------------------------------

    def _aggregate_local_grams(
        self, client_packages: OrderedDict[int, Dict[str, Any]]
    ):
        """Compute weighted average of clients' local gram matrices."""
        client_weights = [pkg["weight"] for pkg in client_packages.values()]
        total = sum(client_weights)
        self.global_grams = {}
        for pkg, w in zip(client_packages.values(), client_weights):
            for name, G in pkg.get("local_grams", {}).items():
                if name not in self.global_grams:
                    self.global_grams[name] = torch.zeros_like(G)
                self.global_grams[name] += (w / total) * G.cpu()
