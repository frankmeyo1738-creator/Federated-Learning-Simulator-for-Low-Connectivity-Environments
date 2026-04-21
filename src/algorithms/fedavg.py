"""
FedAvg — Federated Averaging
==============================
McMahan et al., "Communication-Efficient Learning of Deep Networks
from Decentralized Data" (AISTATS 2017).

Aggregation rule: weighted average of client model weights,
weighted by the number of local training samples.

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""

import copy
from typing import Dict, List

import torch

from src.core.client import ModelUpdate


def fedavg_aggregate(updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
    """Perform Federated Averaging aggregation.

    Parameters
    ----------
    updates : List[ModelUpdate]
        List of model updates from participating clients.

    Returns
    -------
    Dict[str, torch.Tensor]
        Aggregated model state dict.

    Notes
    -----
    Each client's contribution is weighted by its number of training
    samples:  w_new = Σ (n_k / N) · w_k
    where n_k is client k's sample count and N is the total.
    """
    if not updates:
        raise ValueError("Cannot aggregate with zero updates")

    # Total number of samples across all clients
    total_samples = sum(u.num_samples for u in updates)

    # Initialise aggregated weights with zeros (same structure as first update)
    agg_weights: Dict[str, torch.Tensor] = {}
    for key in updates[0].weights:
        agg_weights[key] = torch.zeros_like(updates[0].weights[key], dtype=torch.float32)

    # Weighted sum
    for update in updates:
        weight = update.num_samples / total_samples
        for key in agg_weights:
            agg_weights[key] += weight * update.weights[key].float()

    return agg_weights
