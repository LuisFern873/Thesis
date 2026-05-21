"""
DriftFedProxServer — FedProx with the same drift/interference instrumentation
as DriftFedAvgServer.

Inherits all measurement logic from DriftFedAvgServer and swaps the client
class to FedProxClient, which adds the proximal penalty term μ‖w − w_t‖²
to each client's local loss.

The proximal coefficient μ is read from args.fedprox.mu (default 0.01).
"""

from src.client.fedprox import FedProxClient
from src.server.driftfedavg import DriftFedAvgServer


class DriftFedProxServer(DriftFedAvgServer):
    algorithm_name = "DriftFedProx"
    client_cls = FedProxClient

    @staticmethod
    def get_hyperparams(args_list=None):
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument("--mu", type=float, default=0.01)
        return parser.parse_args(args_list)
