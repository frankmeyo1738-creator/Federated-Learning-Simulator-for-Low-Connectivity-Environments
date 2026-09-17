"""
src.core — Core federated learning components.

Lazy re-exports to avoid a circular import with src.algorithms:
    src.core.__init__  ->  src.core.server  ->  src.algorithms.fedavg
        ->  src.core.client  ->  src.core.__init__ (cycle!)

Direct leaf-module imports (e.g. from src.core.server import FLServer)
always work because they bypass this __init__.py.
"""


def __getattr__(name):
    if name == "FLServer":
        from src.core.server import FLServer
        return FLServer
    if name == "FLClient":
        from src.core.client import FLClient
        return FLClient
    if name == "ModelUpdate":
        from src.core.client import ModelUpdate
        return ModelUpdate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
