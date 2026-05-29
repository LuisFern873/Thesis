"""
DriftFedProxServer — FedProx with the same drift/interference instrumentation
as DriftFedAvgServer.

Inherits all measurement logic from DriftFedAvgServer and swaps the client
class to DriftFedProxClient, which adds the proximal penalty term μ‖w − w_t‖²
to each client's local loss.

The proximal coefficient μ is read from args.driftfedprox.mu (default 0.01).
main.py stores hyperparams under the key matching the server class name
(``driftfedprox``), so the client must read from that same key.
"""

from src.client.driftfedprox import DriftFedProxClient
from src.server.driftfedavg import DriftFedAvgServer


class DriftFedProxServer(DriftFedAvgServer):
    algorithm_name = "DriftFedProx"
    client_cls = DriftFedProxClient

    @staticmethod
    def get_hyperparams(args_list=None):
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument("--mu", type=float, default=0.01)
        return parser.parse_args(args_list)
