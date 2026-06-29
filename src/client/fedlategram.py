# src/client/fedlategram.py
"""
FedLateGram client: local training with gram-matrix regularization on late layers.

Key behaviors:
- During warm-up (warming_up=True): identical to FedAvgClient.fit().
- After warm-up:
    - Captures late-layer activations during the SINGLE task-loss forward pass
      (no second forward — hooks are registered before the forward and removed
      immediately after, so no extra compute).
    - Computes gram penalty vs. global reference, normalized by D² to keep
      loss magnitude O(1) regardless of feature dimension.
    - Zeroes / scales gradients of early-layer parameters per freeze_strategy.
- Sends per-round loss statistics (task + gram) back to the server.
"""
from typing import Any

import torch
from tqdm import tqdm

from src.client.fedavg import FedAvgClient


class FedLateGramClient(FedAvgClient):

    # ------------------------------------------------------------------
    # Parameter loading
    # ------------------------------------------------------------------

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.late_layer_names: list[str] = package["late_layer_names"]
        self.global_grams: dict[str, torch.Tensor] = package["global_grams"]
        self.lam: float = package["lam"]
        self.freeze_strategy: str = package["freeze_strategy"]
        self.warming_up: bool = package["warming_up"]
        self.alpha_early_lr: float = package["alpha_early_lr"]
        self.layer_lambdas: dict[str, float] = package.get("layer_lambdas", {})

    def _is_late_param(self, name: str) -> bool:
        return any(name.startswith(prefix) for prefix in self.late_layer_names)

    # ------------------------------------------------------------------
    # Hook helpers
    # ------------------------------------------------------------------

    def _register_gram_hooks(self) -> tuple[dict, list]:
        """
        Register forward hooks on all late submodules, including classifier.
        Returns (activations_dict, handles_list).
        Caller is responsible for removing handles after the forward.
        """
        target_names = list(self.late_layer_names)
        activations: dict[str, torch.Tensor] = {}
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
                    t = out
                    # Global average pool spatial dims to keep D = C_out
                    # This prevents O(C²H²W²) gram matrices.
                    if t.dim() == 4:
                        t = t.mean(dim=(2, 3))      # (N, C, H, W) → (N, C)
                    elif t.dim() == 3:
                        t = t.mean(dim=1)           # (N, L, D) → (N, D) for SSM/transformer
                    elif t.dim() > 2:
                        t = t.flatten(start_dim=1)  # (N, ...) → (N, D)
                    activations[n] = t              # keep grad graph intact
                return hook

            handles.append(module.register_forward_hook(make_hook(full_name)))

        return activations, handles

    def _compute_grams(
        self, activations: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Compute normalised (D, D) gram matrices from captured activations.

        Each gram G = (Phi.T @ Phi) / N is then divided by its own Frobenius
        norm (clamped to 1e-8) so that every gram lives on the unit sphere.
        This keeps the gram-loss magnitude O(1) regardless of activation scale
        or feature dimension, preventing NaN explosions at round 1 when the
        model is far from convergence.
        """
        grams = {}
        for name, Phi in activations.items():
            # Phi: (N, D)  →  gram: (D, D), normalised by N
            G = (Phi.T @ Phi) / Phi.shape[0]
            # Unit-normalise so ‖G‖_F = 1  (safe against zero-norm)
            G = G / G.norm(p="fro").clamp(min=1e-8)
            grams[name] = G
        return grams

    def _gram_loss(
        self,
        grams_local: dict[str, torch.Tensor],
        grams_global: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Frobenius-norm penalty between unit-normalised gram matrices.

        When layer_lambdas is populated (adaptive mode), each layer uses its
        own λ_ℓ. Otherwise falls back to the scalar self.lam for all layers:

            loss = Σ_ℓ  λ_ℓ · ‖Ĝ_ℓ^local − Ĝ_ℓ^global‖_F²

        Because both grams are unit-normalised, the max possible value per
        layer is 4.0, keeping the total loss comfortably O(1) regardless of
        feature dimension or activation magnitude.
        """
        loss = torch.tensor(0.0, device=self.device)
        for name in grams_local:
            if name not in grams_global:
                continue
            # Use per-layer λ if available; fall back to scalar lam.
            lam = self.layer_lambdas.get(name, self.lam) if self.layer_lambdas else self.lam
            G_loc = grams_local[name]
            G_ref = grams_global[name].to(self.device).detach()
            # Unit-normalise the reference gram to match the local norm
            G_ref = G_ref / G_ref.norm(p="fro").clamp(min=1e-8)
            diff = G_loc - G_ref
            loss = loss + lam * torch.norm(diff, p="fro") ** 2
        return loss

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(self):
        # During warm-up, delegate entirely to FedAvgClient.fit() so that
        # the executed bytecode is bit-for-bit identical to the FedAvg
        # baseline (no tqdm wrapper, no extra state, same RNG progression).
        if self.warming_up:
            super().fit()
            # Zero-fill accumulators so package() sends consistent fields.
            self._steps = 1
            self._loss_task_accum = 0.0
            self._loss_gram_accum = 0.0
            self._last_local_grams = {}
            return

        self.model.train()
        self.dataset.train()

        loss_task_accum = 0.0
        loss_gram_accum = 0.0
        steps = 0
        last_local_grams: dict[str, torch.Tensor] = {}

        active_gram = self.lam > 0.0 and bool(self.global_grams)

        for _ in range(self.local_epoch):
            pbar = tqdm(
                self.trainloader,
                desc=f"      Client {self.client_id} (FedLateGram)",
                leave=False,
                disable=self.args.mode == "parallel",
            )
            for x, y in pbar:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)

                loss_gram = torch.tensor(0.0, device=self.device)

                if active_gram:
                    # Register hooks BEFORE the forward so activations are
                    # captured during the task-loss forward (single pass).
                    activations, handles = self._register_gram_hooks()

                # ---- Single forward pass ----
                logit = self.model(x)
                loss_task = self.criterion(logit, y)

                if active_gram:
                    # Remove hooks immediately after the forward
                    for h in handles:
                        h.remove()

                    if activations:
                        grams_local = self._compute_grams(activations)
                        loss_gram = self._gram_loss(grams_local, self.global_grams)
                        last_local_grams = {
                            k: v.detach().cpu() for k, v in grams_local.items()
                        }

                loss = loss_task + loss_gram

                self.optimizer.zero_grad()
                loss.backward()

                # ---- Gradient modification for early layers ----
                if self.freeze_strategy in ("full_freeze", "warm_then_freeze"):
                    for pname, param in self.model.named_parameters():
                        if not self._is_late_param(pname) and param.grad is not None:
                            param.grad = None
                elif self.freeze_strategy in ("slow_update", "warm_then_slow_update"):
                    for pname, param in self.model.named_parameters():
                        if not self._is_late_param(pname) and param.grad is not None:
                            param.grad.mul_(self.alpha_early_lr)

                self.optimizer.step()

                loss_task_accum += loss_task.item()
                loss_gram_accum += loss_gram.item()
                steps += 1

                pbar.set_postfix(
                    task=f"{loss_task.item():.4f}",
                    gram=f"{loss_gram.item():.6f}",
                )

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self._steps = max(steps, 1)
        self._loss_task_accum = loss_task_accum
        self._loss_gram_accum = loss_gram_accum
        self._last_local_grams = last_local_grams

    # ------------------------------------------------------------------
    # Package override: include loss stats and local grams
    # ------------------------------------------------------------------

    def package(self) -> dict:
        pkg = super().package()
        steps = getattr(self, "_steps", 1)
        pkg["loss_task_mean"] = getattr(self, "_loss_task_accum", 0.0) / steps
        pkg["loss_gram_mean"] = getattr(self, "_loss_gram_accum", 0.0) / steps
        pkg["local_grams"] = getattr(self, "_last_local_grams", {})
        return pkg
