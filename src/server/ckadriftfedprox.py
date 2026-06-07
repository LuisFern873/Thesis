"""CKADriftFedProxServer — FedProx with checkpoint saving for offline CKA.

Combines:
- **FedProx proximal penalty** (from :class:`DriftFedProxServer`) — adds μ‖w − w_t‖² to local loss.
- **L2 drift + gradient alignment** (from :class:`DriftFedAvgServer`).
- **CKA checkpoint saving** (from :class:`CKADriftFedAvgServer`) — saves global
  and client state dicts at scheduled rounds for offline CKA computation via
  ``scripts/compute_cka_offline.py``.

MRO (Python C3 linearisation)
------------------------------
``CKADriftFedProxServer(DriftFedProxServer, CKADriftFedAvgServer)`` resolves to:

    CKADriftFedProxServer
    → DriftFedProxServer      (FedProx client + get_hyperparams --mu)
    → CKADriftFedAvgServer    (checkpoint saving + aggregate override)
    → DriftFedAvgServer       (L2 drift + gradient alignment)
    → FedAvgServer            (base FedAvg aggregation)
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
