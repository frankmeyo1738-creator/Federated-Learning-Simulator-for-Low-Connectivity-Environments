"""
Integration Tests — Deterministic Simulation Seeding
=====================================================
Tests that setting the random seed ensures byte-identical reproducibility
across complete simulation runs (client selection, network impairment decisions,
local training updates, and aggregated global model weights).

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

from src.config import ExperimentConfig, NetworkProfile
from src.core.client import FLClient
from src.core.server import FLServer


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    """Minimal linear model for fast integration testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def _create_synthetic_data(num_samples: int = 120, seed: int = 42):
    """Generate reproducible synthetic classification data."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(num_samples, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return dataset


def _seed_all(seed: int) -> None:
    """Seed all pseudo-random number generators identically to src/main.py."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_server_and_clients(seed: int, tmp_path, profile_name: str = "test_profile"):
    """Instantiate a fully wired FLServer and FLClient cohort."""
    _seed_all(seed)

    profile = NetworkProfile(
        name=profile_name,
        latency_min_ms=20.0,
        latency_max_ms=50.0,
        bandwidth_mbps=5.0,
        packet_loss_rate=0.05,
        dropout_probability=0.1,
        dropout_duration_rounds=1,
    )

    config = ExperimentConfig(
        experiment_name=f"test_seed_{seed}",
        num_clients=4,
        num_rounds=3,
        fraction_fit=0.75,
        algorithm="fedavg",
        mu=0.0,
        dataset="mnist",
        network_profile=profile,
        output_dir=str(tmp_path),
        local_epochs=1,
        batch_size=8,
        learning_rate=0.05,
        seed=seed,
    )

    full_dataset = _create_synthetic_data(num_samples=160, seed=seed)
    # Split into 4 partitions of 40 samples each
    partition_size = 40
    clients = []
    for cid in range(config.num_clients):
        subset = torch.utils.data.Subset(
            full_dataset,
            list(range(cid * partition_size, (cid + 1) * partition_size))
        )
        clients.append(FLClient(cid, subset, config))

    # Shared initial model
    torch.manual_seed(seed)
    model = TinyModel()
    test_loader = DataLoader(_create_synthetic_data(num_samples=40, seed=seed + 1), batch_size=8)

    server = FLServer(config, model, clients, test_loader=test_loader)
    return server


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestDeterministicSimulation:
    """End-to-end integration test of simulation determinism."""

    def test_same_seed_produces_identical_weights_and_metrics(self, tmp_path):
        """Two full runs with seed=42 must produce bit-identical results."""
        out_dir_1 = tmp_path / "run_1"
        out_dir_2 = tmp_path / "run_2"

        server_1 = _build_server_and_clients(seed=42, tmp_path=out_dir_1)
        metrics_1 = server_1.run()

        server_2 = _build_server_and_clients(seed=42, tmp_path=out_dir_2)
        metrics_2 = server_2.run()

        assert len(metrics_1) == len(metrics_2) == 3

        for r in range(len(metrics_1)):
            m1 = metrics_1[r]
            m2 = metrics_2[r]

            assert m1.round_num == m2.round_num
            assert m1.participating_clients == m2.participating_clients, (
                f"Round {r+1} participating client count mismatch: {m1.participating_clients} vs {m2.participating_clients}"
            )
            assert m1.dropped_updates == m2.dropped_updates, (
                f"Round {r+1} dropped updates mismatch: {m1.dropped_updates} vs {m2.dropped_updates}"
            )
            assert m1.global_accuracy == pytest.approx(m2.global_accuracy, rel=1e-7)
            assert m1.global_loss == pytest.approx(m2.global_loss, rel=1e-7)
            assert m1.total_bytes_transmitted == m2.total_bytes_transmitted

        # Verify final model parameters match exactly
        state_dict_1 = server_1.global_model.state_dict()
        state_dict_2 = server_2.global_model.state_dict()

        for key in state_dict_1:
            assert torch.equal(state_dict_1[key], state_dict_2[key]), (
                f"Parameter tensor '{key}' differs between identical-seed runs!"
            )

    def test_different_seeds_produce_different_metrics(self, tmp_path):
        """Two runs with different seeds (42 vs 99) must produce differing trajectories."""
        out_dir_1 = tmp_path / "seed_42"
        out_dir_2 = tmp_path / "seed_99"

        server_1 = _build_server_and_clients(seed=42, tmp_path=out_dir_1)
        metrics_1 = server_1.run()

        server_2 = _build_server_and_clients(seed=99, tmp_path=out_dir_2)
        metrics_2 = server_2.run()

        # Check that final model parameters differ
        state_dict_1 = server_1.global_model.state_dict()
        state_dict_2 = server_2.global_model.state_dict()

        differ = False
        for key in state_dict_1:
            if not torch.equal(state_dict_1[key], state_dict_2[key]):
                differ = True
                break

        assert differ, "Runs with different seeds produced identical weights unexpectedly!"
