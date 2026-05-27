"""
DriftSCAFFOLDClient — SCAFFOLD client with two stability fixes for SSM architectures.

Fix 1 — Gradient clipping before control variate correction
------------------------------------------------------------
Vim-Tiny's SSM layers (dt_proj, x_proj, A_log) produce gradients that are
orders of magnitude larger than standard Conv/Linear layers.  SCAFFOLD's
correction term  g ← g + (c − c_i)  amplifies these by coef = 1/(E·lr) = 20
each round, causing exponential weight divergence and NaN by round ~11.

The fix: clip the *combined* gradient (loss gradient + variate correction) to
a maximum L2 norm of MAX_GRAD_NORM before the optimizer step.  This is
standard practice for SSM/transformer training (e.g. Mamba uses clip=1.0).

Fix 2 — Empty-dataloader guard
-------------------------------
With α=0.03 and 10 clients some clients receive 0 training samples after the
train/test split.  SCAFFOLDClient.get_data_batch() re-raises StopIteration
when the dataloader is empty on the very first call, crashing the experiment.

The fix: detect an empty trainloader in train() before calling train_with_eval,
and return zero y_delta / c_delta so server aggregation is unaffected.

Design notes
------------
- Only fit() and get_data_batch() are overridden; train(), set_parameters(),
  and package() are inherited unchanged from SCAFFOLDClient so the full
  SCAFFOLD protocol (y_delta, c_delta, c_local update) is preserved exactly.
- The empty-client branch in train() is the only place we deviate from the
  parent: we skip train_with_eval() and synthesise zero deltas instead.
"""

from typing import Any

import torch

from src.client.scaffold import SCAFFOLDClient


class DriftSCAFFOLDClient(SCAFFOLDClient):
    """SCAFFOLD client with gradient clipping and empty-dataloader safety."""

    # Maximum L2 norm for the combined gradient vector.
    # 1.0 is the standard value used in Mamba / ViM training.
    MAX_GRAD_NORM: float = 1.0

    # ──────────────────────────────────────────────────────────────────────
    # Override train() only to handle the empty-client edge case.
    # For clients that have data, the full parent logic runs unchanged.
    # ──────────────────────────────────────────────────────────────────────
    def train(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)

        # Empty client: skip training, return zero deltas so aggregation is
        # unaffected.  c_local is left unchanged (no update for this client).
        if len(self.trainloader.dataset) == 0:
            with torch.no_grad():
                self.y_delta = [
                    torch.zeros_like(p)
                    for p in server_package["regular_model_params"].values()
                ]
                self.c_delta = [torch.zeros_like(c) for c in self.c_local]
            return self.package()

        # Normal path: identical to SCAFFOLDClient.train() from here on.
        # We call train_with_eval() (which calls our overridden fit()) then
        # compute y_delta, c_plus, c_delta exactly as the parent does.
        self.train_with_eval()

        with torch.no_grad():
            self.y_delta = []
            c_plus = []
            self.c_delta = []

            model_params = self.model.state_dict()
            for key in server_package["regular_model_params"].keys():
                x, y_i = server_package["regular_model_params"][key], model_params[key]
                self.y_delta.append(y_i.cpu() - x)

            coef = 1 / (self.local_epoch * self.args.optimizer.lr)
            for c, c_i, y_del in zip(self.c_global, self.c_local, self.y_delta):
                c_plus.append(c_i - c - coef * y_del)

            for c_p, c_l in zip(c_plus, self.c_local):
                self.c_delta.append(c_p - c_l)

            self.c_local = c_plus

        return self.package()

    # ──────────────────────────────────────────────────────────────────────
    # Override fit() to add gradient clipping after variate correction.
    # Structure is identical to SCAFFOLDClient.fit(); only one line is added.
    # ──────────────────────────────────────────────────────────────────────
    def fit(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.args.common.local_epoch):
            x, y = self.get_data_batch()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            self.optimizer.zero_grad()
            loss.backward()

            # SCAFFOLD control variate correction (identical to parent)
            for param, c, c_i in zip(
                self.model.parameters(), self.c_global, self.c_local
            ):
                if param.requires_grad:
                    param.grad.data += (c - c_i).to(self.device)

            # Clip combined gradient to prevent SSM parameter explosion
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.MAX_GRAD_NORM
            )

            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    # ──────────────────────────────────────────────────────────────────────
    # Override get_data_batch() to handle truly empty dataloaders.
    # The parent's fallback re-raises StopIteration on an empty loader.
    # ──────────────────────────────────────────────────────────────────────
    def get_data_batch(self):
        try:
            x, y = next(self.iter_trainloader)
            if len(x) <= 1:
                x, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            x, y = next(self.iter_trainloader)   # propagates if truly empty
        return x.to(self.device), y.to(self.device)
