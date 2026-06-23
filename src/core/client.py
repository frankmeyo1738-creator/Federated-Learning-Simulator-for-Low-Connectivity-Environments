"""
Layer 2 — Federated Learning Client
=====================================
Represents a single FL participant. Holds a local data partition,
trains on the global model, and returns a ModelUpdate.

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Model update container
# ---------------------------------------------------------------------------

@dataclass
class ModelUpdate:
    """Encapsulates everything the server needs from one round of local training."""
    client_id: int
    weights: Dict[str, torch.Tensor]
    num_samples: int
    loss: float
    round_num: int
    # Metadata for analytics
    training_time_seconds: float = 0.0
    model_size_bytes: int = 0


# ---------------------------------------------------------------------------
# FL Client
# ---------------------------------------------------------------------------

class FLClient:
    """Federated Learning client that trains locally on its data partition.

    Parameters
    ----------
    client_id : int
        Unique identifier for this client.
    dataset_partition : Subset
        The local data partition assigned to this client.
    config : ExperimentConfig
        Global experiment configuration.
    """

    def __init__(
        self,
        client_id: int,
        dataset_partition: Subset,
        config: ExperimentConfig,
    ) -> None:
        self.client_id = client_id
        self.dataset_partition = dataset_partition
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def train(self, global_model: nn.Module, round_num: int) -> ModelUpdate:
        """Run local training starting from the global model weights.

        Parameters
        ----------
        global_model : nn.Module
            Current global model broadcast by the server.
        round_num : int
            Current FL round number (used for logging / metadata).

        Returns
        -------
        ModelUpdate
            Contains updated weights, sample count, loss, and metadata.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        start_time = time.time()

        # Deep-copy global model so we don't mutate the server's copy
        model = copy.deepcopy(global_model).to(device)
        model.train()

        # Prepare data loader
        loader = DataLoader(
            self.dataset_partition,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        optimizer = optim.SGD(
            model.parameters(),
            lr=self.config.learning_rate,
        )
        criterion = nn.CrossEntropyLoss()

        # Keep reference to global weights for FedProx proximal term
        global_weights = {
            name: param.clone().detach()
            for name, param in global_model.named_parameters()
        }

        total_loss = 0.0
        num_batches = 0

        for _epoch in range(self.config.local_epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, labels)

                # FedProx: add proximal term  μ/2 · ||w - w_global||²
                if self.config.algorithm == "fedprox" and self.config.mu > 0:
                    proximal_term = 0.0
                    for name, param in model.named_parameters():
                        proximal_term += ((param - global_weights[name].to(device)) ** 2).sum()
                    loss += (self.config.mu / 2.0) * proximal_term

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - start_time

        # Calculate model size in bytes
        model_size = sum(
            p.nelement() * p.element_size()
            for p in model.parameters()
        )

        cpu_weights = {k: v.cpu() for k, v in model.state_dict().items()}

        return ModelUpdate(
            client_id=self.client_id,
            weights=cpu_weights,
            num_samples=len(self.dataset_partition),
            loss=avg_loss,
            round_num=round_num,
            training_time_seconds=elapsed,
            model_size_bytes=model_size,
        )

    def get_data_size(self) -> int:
        """Return the number of samples in this client's partition."""
        return len(self.dataset_partition)
