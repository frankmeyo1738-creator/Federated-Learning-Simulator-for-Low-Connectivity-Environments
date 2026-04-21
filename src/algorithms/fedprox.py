"""
FedProx — Federated Proximal
==============================
Li et al., "Federated Optimization in Heterogeneous Networks"
(MLSys 2020).

Aggregation rule: identical to FedAvg (weighted average).
The proximal term (μ/2 · ||w - w_global||²) is applied during
local training on the client side — see ``FLClient.train()``.

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""

from typing import Dict, List

import torch

from src.core.client import ModelUpdate


def fedprox_aggregate(updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
    """Perform FedProx aggregation (same as FedAvg at aggregation time).

    The FedProx proximal term is applied during local training,
    not during aggregation. Therefore, the aggregation step is
    identical to standard Federated Averaging.

    Parameters
    ----------
    updates : List[ModelUpdate]
        List of model updates from participating clients.

    Returns
    -------
    Dict[str, torch.Tensor]
        Aggregated model state dict.

    See Also
    --------
    src.algorithms.fedavg.fedavg_aggregate : Shared aggregation logic.
    src.core.client.FLClient.train : Where the proximal term is applied.
    """
    if not updates:
        raise ValueError("Cannot aggregate with zero updates")

    # Total number of samples across all clients
    total_samples = sum(u.num_samples for u in updates)

    # Initialise aggregated weights with zeros
    agg_weights: Dict[str, torch.Tensor] = {}
    for key in updates[0].weights:
        agg_weights[key] = torch.zeros_like(updates[0].weights[key], dtype=torch.float32)

    # Weighted sum (identical to FedAvg)
    for update in updates:
        weight = update.num_samples / total_samples
        for key in agg_weights:
            agg_weights[key] += weight * update.weights[key].float()

    return agg_weights
