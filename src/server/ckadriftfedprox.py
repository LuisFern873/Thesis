"""CKADriftFedProxServer — FedProx with CKA representation-drift measurement.

This module implements :class:`CKADriftFedProxServer`, which combines:

- **FedProx proximal penalty** (from :class:`DriftFedProxServer` /
  :class:`DriftFedProxClient`) — adds μ‖w − w_t‖² to each client's local loss.
- **L2 drift + gradient alignment** (from :class:`DriftFedAvgServer`) — measures
  per-layer weight drift and gradient cosine similarity each round.
- **CKA representation-drift measurement** (from :class:`CKADriftFedAvgServer`) —
  measures per-layer CKA between the global model and sampled client models.

MRO (Python C3 linearisation)
------------------------------
``CKADriftFedProxServer(DriftFedProxServer, CKADriftFedAvgServer)`` resolves to:

    CKADriftFedProxServer
    → DriftFedProxServer          (adds FedProx client + get_hyperparams --mu)
    → CKADriftFedAvgServer        (adds CKA measurement + aggregate override)
    → DriftFedAvgServer           (adds L2 drift + gradient alignment)
    → FedAvgServer                (base FedAvg aggregation)

Why no method overrides are needed
------------------------------------
- ``aggregate_client_updates``: resolved from ``CKADriftFedAvgServer`` — runs
  the CKA block then calls ``super().aggregate_client_updates()`` which chains
  through ``DriftFedAvgServer`` (drift metrics) and finally ``FedAvgServer``
  (weighted average).  This is the correct order.
- ``client_cls``: resolved from ``DriftFedProxServer`` — uses
  :class:`DriftFedProxClient` which applies the proximal penalty.
- ``get_hyperparams``: resolved from ``DriftFedProxServer`` — adds ``--mu``.
- All CKA helpers (``_run_cka_round``, ``_init_cka_csv``, etc.): resolved from
  ``CKADriftFedAvgServer``.

Requirements addressed: 9.2
"""

from src.client.driftfedprox import DriftFedProxClient
from src.server.ckadriftfedavg import CKADriftFedAvgServer
from src.server.driftfedprox import DriftFedProxServer


class CKADriftFedProxServer(DriftFedProxServer, CKADriftFedAvgServer):
    """FedProx server with L2 drift, gradient alignment, and CKA measurement.

    Inherits all CKA instrumentation from :class:`CKADriftFedAvgServer` via
    MRO and uses :class:`DriftFedProxClient` (with proximal penalty) as the
    client class.  No additional method overrides are required.

    Class attributes
    ----------------
    algorithm_name : str
        Identifier used by the dynamic import mechanism in ``main.py``.
        Lowercased to ``"ckadriftfedproxserver"`` for discovery.
    """

    algorithm_name = "CKADriftFedProx"
