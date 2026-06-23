"""
Root conftest.py — Federated Learning Network Simulator
=========================================================
Resolves circular import between src.core and src.algorithms by
pre-importing leaf modules in dependency order before any test
module triggers the package __init__.py files.

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""

# Pre-import leaf modules to break the circular dependency chain:
#   src.algorithms.__init__ → src.algorithms.fedavg → src.core.client
#   → src.core.__init__ → src.core.server → src.algorithms.fedavg (cycle!)
#
# By importing the leaf modules first (client, then algorithms),
# Python's module cache prevents the __init__.py from re-triggering
# the cycle.
import src.config                   # noqa: F401  (no deps)
import src.core.client              # noqa: F401  (depends on src.config)
import src.algorithms.fedavg        # noqa: F401  (depends on src.core.client)
import src.algorithms.fedprox       # noqa: F401  (depends on src.core.client)
