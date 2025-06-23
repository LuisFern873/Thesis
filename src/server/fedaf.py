import torch
from collections import OrderedDict
from typing import Any, Dict
from argparse import ArgumentParser, Namespace
from omegaconf import DictConfig
from src.client.fedaf import FedAFClient
from src.server.fedavg import FedAvgServer


class FedAFServer(FedAvgServer):
    algorithm_name: str = "FEDAF"
    client_cls = FedAFClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--lambda_loc", type=float, default=0.5)
        return parser.parse_args(args_list)


    def __init__(self, args: DictConfig):
        super().__init__(args)
        self.clients_prev_model_params = {i: {} for i in self.train_clients}


    """
        Global Model training Loss

        - Collect condensed data from all clients
        - Collect softlabels

        - Cross Entropy Loss (CE Loss)
        - Local-Global Knowledge Matching (LGKM) loss
    """
    @torch.no_grad()
    def global_model_training(
        self, client_packages: OrderedDict[int, Dict[str, Any]]
    ):
        client_weights = [package["weight"] for package in client_packages.values()]
        weights = torch.tensor(client_weights) / sum(client_weights)
        if self.return_diff:  # inputs are model params diff
            for name, global_param in self.public_model_params.items():
                diffs = torch.stack(
                    [
                        package["model_params_diff"][name]
                        for package in client_packages.values()
                    ],
                    dim=-1,
                )
                aggregated = torch.sum(diffs * weights, dim=-1)
                self.public_model_params[name].data -= aggregated
        else:
            for name, global_param in self.public_model_params.items():
                client_params = torch.stack(
                    [
                        package["regular_model_params"][name]
                        for package in client_packages.values()
                    ],
                    dim=-1,
                )
                aggregated = torch.sum(client_params * weights, dim=-1)

                global_param.data = aggregated
        self.model.load_state_dict(self.public_model_params, strict=False)


    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at
        server side) in each communication round."""
        client_packages = self.trainer.train()
        self.global_model_training(client_packages)

    # For Model Re-sampling?
    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["prev_model_params"] = self.clients_prev_model_params[client_id]
        return server_package
