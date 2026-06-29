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
            choices=["full_freeze", "slow_update", "warm_then_freeze", "warm_then_slow_update"],
            help=(
                "How to handle early layers after warm-up. "
                "'warm_then_slow_update': like warm_then_freeze but applies slow_update "
                "(scaled gradients + infrequent aggregation) instead of freezing."
            ),
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
        parser.add_argument(
            "--mu_clf", type=float, default=0.0,
            help=(
                "FedProx penalty weight on the classifier head. "
                "Adds μ/2 · ‖W_clf^k − W_clf^global‖² to the client loss. "
                "Orthogonal to gram penalty: operates in parameter space, "
                "not activation space, so it does not interfere with CE. "
                "Set to 0.0 (default) to disable."
            ),
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

        # Per-layer lambda weights (populated each round when lam_adaptive=True).
        # Empty dict signals the client to fall back to scalar lam.
        self.layer_lambdas: dict[str, float] = {}

        # Proxy dataloader built lazily (after warm-up) to avoid consuming
        # random numbers during __init__ and displacing client_sample_stream.
        self._proxy_loader = None

        # Warm-up state — both warm_then_* strategies start in warm-up mode
        self._warming_up = self.args.fedlategram.freeze_strategy in (
            "warm_then_freeze", "warm_then_slow_update"
        )

        # Snapshot of global classifier weights sent to clients each round
        # for the FedProx-on-classifier term (mu_clf > 0 only).
        # Stored on CPU; clients move to their device as needed.
        self._global_clf_params: dict[str, torch.Tensor] = (
            self._snapshot_classifier() if self.args.fedlategram.mu_clf > 0.0 else {}
        )

    # ------------------------------------------------------------------
    # Layer detection
    # ------------------------------------------------------------------

    def _detect_late_layers(self) -> list[str]:
        """Return submodule names considered 'late': last `late_fraction`
        of meaningful sub-modules of self.model.base, plus 'classifier'.

        Granularity strategy — handles both flat CNNs and hierarchical
        transformer/GNN architectures:

        1. Start from depth-1 children of base (no '.' in name).
        2. Expand any child that is an nn.Sequential (or similar container)
           with >= MIN_BLOCK_SIZE direct children. This replaces e.g. 'blocks'
           with 'blocks.0', 'blocks.1', …, 'blocks.11' so that each transformer
           or GNN block counts as one unit for the late_fraction cutoff.
        3. Apply cutoff on the resulting flat list.
        """
        MIN_BLOCK_SIZE = 4  # expand a container if it has >= this many direct children

        depth1 = [
            name for name, _ in self.model.base.named_modules()
            if name and "." not in name
        ]
        if not depth1:
            return ["base", "classifier"]

        expanded: list[str] = []
        for top_name in depth1:
            top_module = getattr(self.model.base, top_name)
            direct_children = [
                n for n, _ in top_module.named_modules()
                if n and "." not in n
            ]
            if len(direct_children) >= MIN_BLOCK_SIZE:
                # Expand: replace 'blocks' with 'blocks.0' … 'blocks.N'
                expanded.extend(f"{top_name}.{c}" for c in direct_children)
            else:
                expanded.append(top_name)

        fraction = self.args.fedlategram.late_fraction
        cutoff = max(0, int(len(expanded) * (1 - fraction)))
        late = [f"base.{n}" for n in expanded[cutoff:]] + ["classifier"]

        self.logger.log(
            f"[FedLateGram] _detect_late_layers: {len(expanded)} units "
            f"(expanded from {len(depth1)} depth-1 children), "
            f"cutoff={cutoff} → {len(late)} late prefixes"
        )
        return late

    def _is_late_param(self, param_name: str) -> bool:
        return any(param_name.startswith(p) for p in self.late_layer_names)

    def _snapshot_classifier(self) -> dict[str, torch.Tensor]:
        """Return a CPU copy of the global classifier parameters."""
        return {
            name: param.detach().cpu().clone()
            for name, param in self.model.classifier.named_parameters()
        }

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
        self.model.to(self.device)
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
                    # Vim/Mamba layers return (hidden_states, residual) tuples;
                    # take the first element (hidden states).
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    if not isinstance(out, torch.Tensor):
                        return
                    t = out.detach()
                    if t.dim() == 4:
                        t = t.mean(dim=(2, 3))   # (N,C,H,W) → (N,C)
                    elif t.dim() == 3:
                        t = t.mean(dim=1)        # (N,L,D) → (N,D) for sequence models
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
                Phi = torch.cat(acts, dim=0)               # (N, D)
                G = (Phi.T @ Phi) / Phi.shape[0]           # (D, D)
                G = G / G.norm(p="fro").clamp(min=1e-8)    # unit-normalise
                grams[name] = G

        self.model.train()
        return grams

    # ------------------------------------------------------------------
    # train_one_round — merges DriftFedAvg LR injection + FedLateGram logic
    # ------------------------------------------------------------------

    def train_one_round(self):
        # 1. Cosine LR decay — identical to DriftFedAvgServer.train_one_round.
        #    Must run BEFORE trainer.train() so clients receive the decayed LR.
        self._update_client_lr()

        # 2. Warm-up flag management.
        #    warm_then_freeze / warm_then_slow_update: warming_up=True for
        #    rounds 0..(T_warm-1); after that, switch to full_freeze or
        #    slow_update respectively.
        #    Any other strategy activates gram penalty from round 0.
        if self.args.fedlategram.freeze_strategy in ("warm_then_freeze", "warm_then_slow_update"):
            was_warming = self._warming_up
            self._warming_up = self.current_epoch < self.args.fedlategram.T_warm
            if was_warming and not self._warming_up:
                active_strategy = (
                    "full_freeze"
                    if self.args.fedlategram.freeze_strategy == "warm_then_freeze"
                    else "slow_update"
                )
                self.logger.log(
                    f"[FedLateGram] Warm-up complete at round {self.current_epoch + 1}. "
                    f"Activating gram penalty and '{active_strategy}' for early layers."
                )
        else:
            self._warming_up = False

        # 3. Build proxy loader and refresh global grams (after warm-up only).
        #    Lazy build avoids displacing the RNG stream during __init__.
        if self.args.fedlategram.gram_ref == "proxy" and not self._warming_up:
            if self._proxy_loader is None:
                self._proxy_loader = self._build_proxy_loader()
            self.global_grams = self._compute_global_grams()

        # 4. Refresh global classifier snapshot for FedProx-on-classifier term.
        #    Done every round (warm-up or not) so clients always anchor to the
        #    current global classifier, matching standard FedProx semantics.
        if self.args.fedlategram.mu_clf > 0.0:
            self._global_clf_params = self._snapshot_classifier()

        # 5. Client training — clients receive warming_up flag via package()
        #    and behave identically to FedAvgClient during warm-up.
        client_packages = self.trainer.train()

        # 6. avg_local gram reference update (no-op during warm-up)
        if self.args.fedlategram.gram_ref == "avg_local" and not self._warming_up:
            self._aggregate_local_grams(client_packages)

        # 7. Adaptive per-layer lambdas — computed after grams are updated.
        if (
            self.args.fedlategram.get("lam_adaptive", True)
            and not self._warming_up
        ):
            self.layer_lambdas = self._compute_layer_lambdas(client_packages)
            self.logger.log(
                f"[Round {self.current_epoch}] layer_lambdas: "
                + ", ".join(
                    f"{n.split('.')[-1]}={v:.4f}"
                    for n, v in self.layer_lambdas.items()
                )
            )
        else:
            self.layer_lambdas = {}   # empty → client falls back to scalar lam

        # 8. Log gram + prox losses (only meaningful after warm-up)
        gram_losses = [pkg.get("loss_gram_mean", 0.0) for pkg in client_packages.values()]
        task_losses = [pkg.get("loss_task_mean", 0.0) for pkg in client_packages.values()]
        prox_losses = [pkg.get("loss_prox_mean", 0.0) for pkg in client_packages.values()]
        if any(g > 0 for g in gram_losses) or any(p > 0 for p in prox_losses):
            self.logger.log(
                f"[FedLateGram Round {self.current_epoch + 1}] "
                f"avg task: {sum(task_losses)/len(task_losses):.4f} | "
                f"avg gram: {sum(gram_losses)/len(gram_losses):.4f} | "
                f"avg prox_clf: {sum(prox_losses)/len(prox_losses):.4f}"
            )

        # 9. Aggregate — goes through FedLateGramServer.aggregate_client_updates
        #    which handles CKA checkpoint + drift metrics + selective FedAvg.
        self.aggregate_client_updates(client_packages)

    # ------------------------------------------------------------------
    # aggregate_client_updates — CKA checkpoint → drift metrics → FLG selective agg
    #
    # Design: the MRO chain (CKADriftFedAvg → DriftFedAvg → FedAvg) handles
    # checkpointing and drift CSV correctly, but its terminal step always runs
    # FedAvgServer.aggregate_client_updates (full aggregation on all params).
    # We need to intercept AFTER drift metrics are populated but BEFORE the
    # FedAvg weighted average so we can apply selective aggregation instead.
    #
    # Strategy:
    #   1. Run CKA checkpoint saving (CKADriftFedAvgServer logic) directly.
    #   2. Run drift + interference computation (DriftFedAvgServer logic)
    #      directly — this populates _last_drift_stats / _last_interference
    #      so display_metrics() can write the drift CSV every round,
    #      INCLUDING during warm-up (consistent with the FedAvg baseline).
    #   3. During warm-up: run full FedAvg aggregation (identical to baseline).
    #      After warm-up: run FedLateGram selective aggregation.
    #
    # Importing at module level is avoided for drift_metrics to keep the
    # import inside the method that actually uses it (matches base class style).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def aggregate_client_updates(
        self, client_packages: OrderedDict[int, Dict[str, Any]]
    ):
        from src.utils.drift_metrics import (
            aggregate_drift,
            compute_gradient_alignment,
            compute_layer_drift,
        )
        from src.server.fedavg import FedAvgServer

        # ── Step 1: CKA checkpoint (CKADriftFedAvgServer logic) ──────────
        round_idx = self.current_epoch + 1
        if self._is_cka_round(round_idx):
            self._save_cka_checkpoint(round_idx, client_packages)

        # ── Step 2: Drift metrics (DriftFedAvgServer logic) ───────────────
        # Computed every round — warm-up or not — so drift_metrics.csv is
        # complete and directly comparable to the FedAvg/FedProx baselines.
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

        # ── Step 3a: Warm-up — full FedAvg on ALL parameters ─────────────
        # Bypasses only the FedAvg layer of the MRO (CKA + drift already done
        # above). Behaviour is bit-for-bit identical to the FedAvg baseline
        # because: same LR (cosine from _update_client_lr), same optimizer,
        # same join_ratio=1.0, same seed, and no gram penalty in fit().
        if self._warming_up:
            FedAvgServer.aggregate_client_updates(self, client_packages)
            return

        # ── Step 3b: Post-warm-up — FedLateGram selective aggregation ────
        client_weights = [pkg["weight"] for pkg in client_packages.values()]
        total_weight = sum(client_weights)
        weights = torch.tensor(
            [w / total_weight for w in client_weights], dtype=torch.float
        )

        for name, global_param in self.public_model_params.items():
            if self._is_late_param(name):
                # Late layers: always aggregate across all clients (FedAvg)
                stacked = torch.stack(
                    [pkg["regular_model_params"][name] for pkg in client_packages.values()],
                    dim=-1,
                )
                global_param.data = torch.sum(stacked * weights, dim=-1)

            elif self.args.fedlategram.freeze_strategy in ("slow_update", "warm_then_slow_update"):
                # Early layers: aggregate only every freq_early rounds
                if self.current_epoch % self.args.fedlategram.freq_early == 0:
                    stacked = torch.stack(
                        [pkg["regular_model_params"][name] for pkg in client_packages.values()],
                        dim=-1,
                    )
                    global_param.data = torch.sum(stacked * weights, dim=-1)
                # else: retain current global value (no update)

            # full_freeze strategy: early layers keep current global value (no-op)

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
        pkg["layer_lambdas"] = dict(self.layer_lambdas)   # plain float dict, safe to serialize
        pkg["mu_clf"] = self.args.fedlategram.mu_clf
        # global_clf_params is {} when mu_clf == 0 (no overhead)
        pkg["global_clf_params"] = {k: v.clone() for k, v in self._global_clf_params.items()}
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
    # Adaptive per-layer lambda computation
    # ------------------------------------------------------------------

    def _compute_layer_lambdas(
        self, client_packages: OrderedDict
    ) -> dict[str, float]:
        """
        Compute per-layer λ proportional to mean gram divergence across clients.
        Returns a dict {layer_name: lambda_value} for gram-tracked late layers only
        (i.e. late_layer_names excluding 'classifier').

        Invariant: Σ λ_ℓ ≈ λ_0 (scalar lam), so the hyperparameter remains
        interpretable without retuning.

        The classifier is intentionally excluded:
        - It is in late_layer_names solely for selective FedAvg aggregation.
        - It has no gram matrix (gram penalty on logits is redundant with
          cross-entropy and conceptually inconsistent with representation
          regularisation).
        - Assigning it a fixed lam0 weight (previous behaviour) broke the
          Σ λ_ℓ ≈ λ_0 invariant by adding an unnormalised constant on top of
          the normalised gram-layer weights.
        """
        lam0 = self.args.fedlategram.lam
        weights = [pkg["weight"] for pkg in client_packages.values()]
        total_w = sum(weights) + 1e-8

        # Only consider layers that actually have gram matrices.
        gram_names = [n for n in self.late_layer_names if n != "classifier"]

        divergences: dict[str, float] = {}
        for name in gram_names:
            if name not in self.global_grams:
                divergences[name] = 0.0
                continue

            G_global = self.global_grams[name].cpu()
            layer_divergence = 0.0
            for pkg, w in zip(client_packages.values(), weights):
                G_k = pkg.get("local_grams", {}).get(name)
                if G_k is None:
                    continue
                layer_divergence += (w / total_w) * (G_k.cpu() - G_global).norm(p="fro").item()
            divergences[name] = layer_divergence

        total_d = sum(divergences.values()) + 1e-8

        # Normalise so Σ λ_ℓ = λ_0 across gram-tracked layers only.
        return {
            name: float(lam0 * divergences[name] / total_d)
            for name in gram_names
        }

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
            "layer_lambdas": dict(self.layer_lambdas),
            "_global_clf_params": {k: v.cpu() for k, v in self._global_clf_params.items()},
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
        self.layer_lambdas = state.get("layer_lambdas", {})
        self._global_clf_params = state.get("_global_clf_params", {})

        resume_from = self.current_epoch + 1
        self.logger.log(
            f"  [FedLateGram] Restored. Resuming from round "
            f"{resume_from + 1}/{self.args.common.global_epoch}."
        )
        return resume_from
