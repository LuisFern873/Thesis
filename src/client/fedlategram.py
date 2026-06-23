# src/client/fedlategram.py
"""
FedLateGram client: local training with gram-matrix regularization on late layers.

Key behaviors:
- During warm-up (warming_up=True): identical to FedAvgClient.fit().
- After warm-up:
    - Computes gram penalty on late-layer activations vs. global reference.
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

    def _is_late_param(self, name: str) -> bool:
        return any(name.startswith(prefix) for prefix in self.late_layer_names)

    # ------------------------------------------------------------------
    # Gram computation (inline, hooks registered and removed per batch)
    # ------------------------------------------------------------------

    def _compute_gram_inline(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Register forward hooks on late submodules, run a second forward
        pass on `x`, and return gram matrices for each hooked layer.

        Hooks are always removed after the forward, even on exception.
        """
        target_names = [n for n in self.late_layer_names if n != "classifier"]
        activations: dict[str, torch.Tensor] = {}
        handles = []

        for full_name in target_names:
            module = self.model
            try:
                for part in full_name.split("."):
                    module = getattr(module, part)
            except AttributeError:
                continue  # submodule not present in this architecture

            def make_hook(n):
                def hook(mod, inp, out):
                    # Keep computation graph for backward through gram loss
                    activations[n] = out.flatten(start_dim=1)
                return hook

            handles.append(module.register_forward_hook(make_hook(full_name)))

        try:
            self.model(x)   # second forward — activations are captured
        finally:
            for h in handles:
                h.remove()

        grams: dict[str, torch.Tensor] = {}
        for name, Phi in activations.items():
            grams[name] = (Phi @ Phi.T) / Phi.shape[0]   # (B, B)

        return grams

    def _gram_loss(
        self,
        grams_local: dict[str, torch.Tensor],
        grams_global: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Frobenius-norm penalty: Σ_ℓ ‖G_ℓ^local − G_ℓ^global‖_F²"""
        loss = torch.tensor(0.0, device=self.device)
        for name in grams_local:
            if name in grams_global:
                G_ref = grams_global[name].to(self.device).detach()
                diff = grams_local[name] - G_ref
                loss = loss + torch.norm(diff, p="fro") ** 2
        return loss

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(self):
        self.model.train()
        self.dataset.train()

        # Accumulators for logging
        loss_task_accum = 0.0
        loss_gram_accum = 0.0
        steps = 0

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

                # ---- Task loss (standard forward) ----
                logit = self.model(x)
                loss_task = self.criterion(logit, y)
                loss = loss_task

                # ---- Gram penalty (active after warm-up) ----
                loss_gram = torch.tensor(0.0, device=self.device)
                if (
                    not self.warming_up
                    and self.lam > 0.0
                    and self.global_grams
                ):
                    grams_local = self._compute_gram_inline(x)
                    if grams_local:
                        loss_gram = self._gram_loss(grams_local, self.global_grams)
                        loss = loss + self.lam * loss_gram

                self.optimizer.zero_grad()
                loss.backward()

                # ---- Gradient modification for early layers ----
                if not self.warming_up:
                    if self.freeze_strategy == "full_freeze":
                        # Zero out gradients for early params entirely
                        for name, param in self.model.named_parameters():
                            if not self._is_late_param(name) and param.grad is not None:
                                param.grad = None
                    elif self.freeze_strategy == "slow_update":
                        # Scale down gradients for early params
                        for name, param in self.model.named_parameters():
                            if (
                                not self._is_late_param(name)
                                and param.grad is not None
                            ):
                                param.grad.mul_(self.alpha_early_lr)

                self.optimizer.step()

                # Accumulate for logging
                loss_task_accum += loss_task.item()
                loss_gram_accum += loss_gram.item()
                steps += 1

                pbar.set_postfix(
                    task=f"{loss_task.item():.4f}",
                    gram=f"{loss_gram.item():.4f}",
                )

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Store for package()
        self._steps = max(steps, 1)
        self._loss_task_accum = loss_task_accum
        self._loss_gram_accum = loss_gram_accum

        # Cache local grams from last batch for avg_local strategy
        # (only relevant when gram_ref=avg_local; server ignores otherwise)
        if not self.warming_up and self.lam > 0.0 and steps > 0:
            try:
                last_x = x  # noqa: F821 — x is defined inside the loop
                with torch.no_grad():
                    self._last_local_grams = self._compute_gram_inline(last_x)
            except Exception:
                self._last_local_grams = {}
        else:
            self._last_local_grams = {}

    # ------------------------------------------------------------------
    # Package override: include loss stats and local grams
    # ------------------------------------------------------------------

    def package(self) -> dict:
        pkg = super().package()

        steps = getattr(self, "_steps", 1)
        pkg["loss_task_mean"] = getattr(self, "_loss_task_accum", 0.0) / steps
        pkg["loss_gram_mean"] = getattr(self, "_loss_gram_accum", 0.0) / steps
        pkg["local_grams"] = {
            k: v.detach().cpu()
            for k, v in getattr(self, "_last_local_grams", {}).items()
        }

        return pkg
