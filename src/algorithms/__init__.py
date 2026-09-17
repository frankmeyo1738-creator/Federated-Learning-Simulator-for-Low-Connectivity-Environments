"""
src.algorithms — Federated aggregation algorithms.

Lazy re-exports to avoid a circular import with src.core:
    src.algorithms.__init__  ->  src.algorithms.fedavg  ->  src.core.client
        ->  src.core.__init__  ->  src.core.server  ->  src.algorithms (cycle!)

Direct leaf-module imports (e.g. from src.algorithms.fedavg import fedavg_aggregate)
always work because they bypass this __init__.py.
"""


def __getattr__(name):
    if name == "fedavg_aggregate":
        from src.algorithms.fedavg import fedavg_aggregate
        return fedavg_aggregate
    if name == "fedprox_aggregate":
        from src.algorithms.fedprox import fedprox_aggregate
        return fedprox_aggregate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
