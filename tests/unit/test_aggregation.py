"""
Unit Tests — Aggregation Algorithms
======================================
Tests for ``fedavg_aggregate()`` and ``fedprox_aggregate()`` in
``src/algorithms/``.

Verifies that the weighted-average math is correct for various
client sample-count distributions, and confirms that FedProx
aggregation is identical to FedAvg (the proximal term is applied
during client training, not at aggregation time).

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""

import pytest
import torch

from src.algorithms.fedavg import fedavg_aggregate
from src.algorithms.fedprox import fedprox_aggregate
from src.core.client import ModelUpdate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_update(
    client_id: int,
    weight_value: float,
    num_samples: int,
) -> ModelUpdate:
    """Create a minimal ModelUpdate with a single-parameter 'model'.

    Uses a single key ``'w'`` mapped to a scalar tensor so the
    weighted-average arithmetic can be verified by hand.
    """
    return ModelUpdate(
        client_id=client_id,
        weights={"w": torch.tensor(weight_value)},
        num_samples=num_samples,
        loss=0.1,
        round_num=1,
        training_time_seconds=0.5,
        model_size_bytes=1024,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFedAvgAggregation:
    """Tests for the Federated Averaging aggregation function."""

    def test_fedavg_weighted_average_correct(self):
        """Verify weighted average with two clients of equal weight values
        but different sample counts.

        Client A: 100 samples, weight=1.0
        Client B: 300 samples, weight=1.0
        Expected: (100*1.0 + 300*1.0) / 400 = 1.0

        Both clients have the same weight value, so sample-count
        weighting should not change the result.
        """
        updates = [
            _make_update(client_id=0, weight_value=1.0, num_samples=100),
            _make_update(client_id=1, weight_value=1.0, num_samples=300),
        ]

        result = fedavg_aggregate(updates)

        assert torch.isclose(result["w"], torch.tensor(1.0)), (
            f"Expected 1.0 but got {result['w'].item()}"
        )

    def test_fedavg_single_client(self):
        """Verify that a single client's update passes through unchanged.

        With only one client, the aggregated weight must exactly equal
        the client's weight regardless of sample count.
        """
        updates = [_make_update(client_id=0, weight_value=3.14, num_samples=50)]

        result = fedavg_aggregate(updates)

        assert torch.isclose(result["w"], torch.tensor(3.14)), (
            f"Expected 3.14 but got {result['w'].item()}"
        )

    def test_fedavg_empty_raises_error(self):
        """Verify that passing an empty list of updates raises ValueError.

        Aggregation with zero updates is undefined — the function must
        fail explicitly rather than returning an empty or zeroed model.
        """
        with pytest.raises(ValueError, match="Cannot aggregate with zero updates"):
            fedavg_aggregate([])

    def test_fedavg_unequal_sample_weights(self):
        """Verify weighted average with three clients of different weights
        and different sample counts.

        Client 0: 100 samples, weight=0.0
        Client 1: 200 samples, weight=0.5
        Client 2: 700 samples, weight=1.0
        Expected: (100*0.0 + 200*0.5 + 700*1.0) / 1000 = 0.8

        This is the core correctness test for the FedAvg formula:
        w_new = Σ (n_k / N) · w_k
        """
        updates = [
            _make_update(client_id=0, weight_value=0.0, num_samples=100),
            _make_update(client_id=1, weight_value=0.5, num_samples=200),
            _make_update(client_id=2, weight_value=1.0, num_samples=700),
        ]

        result = fedavg_aggregate(updates)

        assert torch.isclose(result["w"], torch.tensor(0.8)), (
            f"Expected 0.8 but got {result['w'].item()}"
        )


class TestFedProxAggregation:
    """Tests for the FedProx aggregation function."""

    def test_fedprox_matches_fedavg_aggregation(self):
        """Verify that fedprox_aggregate and fedavg_aggregate return
        identical results for the same inputs.

        FedProx only differs from FedAvg in the client-side training
        (proximal term). The server-side aggregation is mathematically
        identical. This test ensures no accidental divergence was
        introduced between the two implementations.
        """
        updates = [
            _make_update(client_id=0, weight_value=2.0, num_samples=150),
            _make_update(client_id=1, weight_value=4.0, num_samples=350),
        ]

        fedavg_result = fedavg_aggregate(updates)
        fedprox_result = fedprox_aggregate(updates)

        assert torch.isclose(fedavg_result["w"], fedprox_result["w"]), (
            f"FedAvg={fedavg_result['w'].item()}, "
            f"FedProx={fedprox_result['w'].item()} — should be identical"
        )
