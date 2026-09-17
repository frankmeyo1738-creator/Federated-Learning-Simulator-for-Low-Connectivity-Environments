"""
Integration Tests — FedProx Equivalence to FedAvg at μ = 0
===========================================================
Validates that FedProx with proximal term μ = 0.0 produces mathematically
and empirically identical model parameter updates to FedAvg across local training
and global aggregation.

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""

import copy
import random
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.fedavg import fedavg_aggregate
from src.algorithms.fedprox import fedprox_aggregate
from src.config import ExperimentConfig, NetworkProfile
from src.core.client import FLClient, ModelUpdate
from src.core.server import FLServer


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

class SimpleNet(nn.Module):
    """Small linear model for deterministic numerical testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def _create_synthetic_data(num_samples: int = 60, seed: int = 42):
    """Generate reproducible synthetic classification data."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(num_samples, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def _make_config(algorithm: str, mu: float, output_dir: str, seed: int = 42) -> ExperimentConfig:
    """Create a standardized configuration for FedAvg or FedProx."""
    profile = NetworkProfile(
        name="ideal_profile",
        latency_min_ms=0.0,
        latency_max_ms=0.0,
        bandwidth_mbps=100.0,
        packet_loss_rate=0.0,
        dropout_probability=0.0,
        dropout_duration_rounds=0,
    )

    return ExperimentConfig(
        experiment_name=f"test_{algorithm}_mu_{mu}",
        num_clients=2,
        num_rounds=2,
        fraction_fit=1.0,
        algorithm=algorithm,
        mu=mu,
        dataset="mnist",
        network_profile=profile,
        output_dir=output_dir,
        local_epochs=2,
        batch_size=8,
        learning_rate=0.05,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestFedProxMuZero:
    """Tests establishing the exact equivalence of FedProx (μ=0) and FedAvg."""

    @pytest.mark.asyncio
    async def test_client_local_training_identical_at_mu_zero(self, tmp_path):
        """FLClient.train() with fedprox and mu=0.0 produces identical weights to fedavg."""
        dataset = _create_synthetic_data(num_samples=32, seed=42)

        cfg_fedavg = _make_config("fedavg", mu=0.0, output_dir=str(tmp_path / "avg"))
        cfg_fedprox_0 = _make_config("fedprox", mu=0.0, output_dir=str(tmp_path / "prox0"))

        client_fedavg = FLClient(client_id=0, dataset_partition=dataset, config=cfg_fedavg)
        client_fedprox = FLClient(client_id=0, dataset_partition=dataset, config=cfg_fedprox_0)

        # Common global model
        torch.manual_seed(100)
        global_model = SimpleNet()

        # Seed before running client training to ensure identical DataLoader shuffle order
        torch.manual_seed(999)
        random.seed(999)
        np.random.seed(999)
        update_fedavg = await client_fedavg.train(global_model, round_num=1)

        torch.manual_seed(999)
        random.seed(999)
        np.random.seed(999)
        update_fedprox = await client_fedprox.train(global_model, round_num=1)

        # Verify loss and sample counts match
        assert update_fedavg.loss == pytest.approx(update_fedprox.loss, rel=1e-7)
        assert update_fedavg.num_samples == update_fedprox.num_samples

        # Verify all parameter tensors are bit-for-bit identical
        for key in update_fedavg.weights:
            assert torch.equal(update_fedavg.weights[key], update_fedprox.weights[key]), (
                f"Weight mismatch in parameter '{key}' between FedAvg and FedProx (mu=0)!"
            )

    def test_aggregation_identical_at_mu_zero(self):
        """fedprox_aggregate and fedavg_aggregate produce identical global weights."""
        t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        t2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

        updates = [
            ModelUpdate(0, {"w": t1}, num_samples=20, loss=0.2, round_num=1, training_time_seconds=0.1, model_size_bytes=100),
            ModelUpdate(1, {"w": t2}, num_samples=80, loss=0.1, round_num=1, training_time_seconds=0.1, model_size_bytes=100),
        ]

        agg_fedavg = fedavg_aggregate(updates)
        agg_fedprox = fedprox_aggregate(updates)

        for k in agg_fedavg:
            assert torch.equal(agg_fedavg[k], agg_fedprox[k])

    def test_end_to_end_simulation_identical_at_mu_zero(self, tmp_path):
        """A complete multi-round simulation of FedAvg matches FedProx (μ=0) exactly."""
        seed = 42

        # 1. Run FedAvg
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        data_avg = _create_synthetic_data(num_samples=64, seed=seed)
        cfg_avg = _make_config("fedavg", mu=0.0, output_dir=str(tmp_path / "sim_avg"), seed=seed)
        clients_avg = [
            FLClient(0, torch.utils.data.Subset(data_avg, list(range(0, 32))), cfg_avg),
            FLClient(1, torch.utils.data.Subset(data_avg, list(range(32, 64))), cfg_avg),
        ]
        torch.manual_seed(seed)
        model_avg = SimpleNet()
        server_avg = FLServer(cfg_avg, model_avg, clients_avg)
        metrics_avg = server_avg.run()

        # 2. Run FedProx with mu=0.0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        data_prox = _create_synthetic_data(num_samples=64, seed=seed)
        cfg_prox = _make_config("fedprox", mu=0.0, output_dir=str(tmp_path / "sim_prox"), seed=seed)
        clients_prox = [
            FLClient(0, torch.utils.data.Subset(data_prox, list(range(0, 32))), cfg_prox),
            FLClient(1, torch.utils.data.Subset(data_prox, list(range(32, 64))), cfg_prox),
        ]
        torch.manual_seed(seed)
        model_prox = SimpleNet()
        server_prox = FLServer(cfg_prox, model_prox, clients_prox)
        metrics_prox = server_prox.run()

        # Verify rounds and metrics
        assert len(metrics_avg) == len(metrics_prox)
        for r in range(len(metrics_avg)):
            assert metrics_avg[r].global_loss == pytest.approx(metrics_prox[r].global_loss, rel=1e-7)
            assert metrics_avg[r].participating_clients == metrics_prox[r].participating_clients

        # Verify final model weights match exactly
        sd_avg = server_avg.global_model.state_dict()
        sd_prox = server_prox.global_model.state_dict()
        for k in sd_avg:
            assert torch.equal(sd_avg[k], sd_prox[k]), f"Final model mismatch at key '{k}'!"

    @pytest.mark.asyncio
    async def test_fedprox_positive_mu_diverges_from_fedavg(self, tmp_path):
        """Sanity check: setting mu > 0 (e.g. mu=1.0) DOES modify weights relative to FedAvg."""
        dataset = _create_synthetic_data(num_samples=32, seed=42)

        cfg_fedavg = _make_config("fedavg", mu=0.0, output_dir=str(tmp_path / "avg_div"))
        cfg_fedprox_1 = _make_config("fedprox", mu=1.0, output_dir=str(tmp_path / "prox_div"))

        client_fedavg = FLClient(client_id=0, dataset_partition=dataset, config=cfg_fedavg)
        client_fedprox = FLClient(client_id=0, dataset_partition=dataset, config=cfg_fedprox_1)

        torch.manual_seed(100)
        global_model = SimpleNet()

        torch.manual_seed(999)
        random.seed(999)
        np.random.seed(999)
        update_fedavg = await client_fedavg.train(global_model, round_num=1)

        torch.manual_seed(999)
        random.seed(999)
        np.random.seed(999)
        update_fedprox = await client_fedprox.train(global_model, round_num=1)

        # Check that with mu=1.0, at least one parameter tensor differs
        has_difference = False
        for key in update_fedavg.weights:
            if not torch.equal(update_fedavg.weights[key], update_fedprox.weights[key]):
                has_difference = True
                break

        assert has_difference, "FedProx with mu=1.0 did not diverge from FedAvg!"
