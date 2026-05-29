from src.client.fedavg import FedAvgClient


class DriftFedProxClient(FedAvgClient):
    """FedProx client for the drift study.

    Identical to FedProxClient but reads the proximal coefficient from
    ``self.args.driftfedprox.mu`` instead of ``self.args.fedprox.mu``,
    because main.py stores DriftFedProxServer's hyperparams under the
    ``driftfedprox`` config key (derived from the server class name).
    """

    def fit(self):
        self.model.train()
        self.dataset.train()
        global_params = [w_t.detach().clone() for w_t in self.model.parameters()]
        mu = self.args.driftfedprox.mu
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                for w, w_t in zip(self.model.parameters(), global_params):
                    if w.requires_grad:
                        w.grad.data += mu * (w.data - w_t.data)
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
