"""
Layer 2 — Federated Learning Server
=====================================
Orchestrates the FL training loop: selects clients per round,
distributes the global model, collects model updates (through the
Network Impairment Layer), aggregates, and records metrics.

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""

import asyncio
import logging
import math
import random
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.config import ExperimentConfig
from src.core.client import FLClient, ModelUpdate
from src.network.impairment import NetworkImpairmentEngine
from src.analytics.tracker import AnalyticsTracker, RoundMetrics
from src.algorithms.fedavg import fedavg_aggregate
from src.algorithms.fedprox import fedprox_aggregate

logger = logging.getLogger(__name__)


class FLServer:
    """Central FL server that coordinates the federated training process.

    Parameters
    ----------
    config : ExperimentConfig
        Validated experiment configuration.
    model : nn.Module
        The global model architecture (uninitialised weights are fine).
    clients : List[FLClient]
        Pre-initialised FL clients with their data partitions.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        clients: List[FLClient],
    ) -> None:
        self.config = config
        self.global_model = model
        self.clients = clients

        # Initialise adjacent-layer components
        self.impairment_engine = NetworkImpairmentEngine(config.network_profile)
        self.analytics = AnalyticsTracker(config.output_dir, config.experiment_name)

        # Reproducibility
        if config.seed is not None:
            random.seed(config.seed)
            torch.manual_seed(config.seed)

    def run(self) -> List[RoundMetrics]:
        """Execute the full FL training loop synchronously.

        Returns
        -------
        List[RoundMetrics]
            Per-round metrics for the entire experiment.
        """
        all_metrics: List[RoundMetrics] = []

        logger.info(
            "Starting FL experiment '%s' — %d rounds, %d clients, algorithm=%s",
            self.config.experiment_name,
            self.config.num_rounds,
            self.config.num_clients,
            self.config.algorithm,
        )

        for round_num in range(1, self.config.num_rounds + 1):
            logger.info("━━━ Round %d / %d ━━━", round_num, self.config.num_rounds)
            round_start = time.time()

            # 1. Select clients for this round
            selected_ids = self.select_clients(round_num)
            logger.info("Selected %d clients: %s", len(selected_ids), selected_ids)

            # 2. Train selected clients and collect updates via impairment layer
            updates, dropped = asyncio.get_event_loop().run_until_complete(
                self._collect_updates(selected_ids, round_num)
            )

            logger.info(
                "Received %d updates, %d dropped", len(updates), dropped
            )

            # 3. Aggregate (only if we received updates)
            if updates:
                self.global_model = self.aggregate(updates)

            # 4. Evaluate global model (placeholder — implement per dataset)
            accuracy, loss = self._evaluate_global_model()

            # 5. Compute round-level network metrics
            total_bytes = sum(u.model_size_bytes for u in updates)
            avg_latency = self.impairment_engine.get_avg_latency()

            round_duration = time.time() - round_start

            # 6. Record metrics
            metrics = RoundMetrics(
                round_num=round_num,
                global_accuracy=accuracy,
                global_loss=loss,
                participating_clients=len(updates),
                dropped_updates=dropped,
                avg_latency_ms=avg_latency,
                total_bytes_transmitted=total_bytes,
                round_duration_seconds=round_duration,
            )
            self.analytics.record_round(metrics)
            all_metrics.append(metrics)

            logger.info(
                "Round %d complete — accuracy=%.4f, loss=%.4f, duration=%.2fs",
                round_num, accuracy, loss, round_duration,
            )

        # Export results at the end
        self.analytics.export_csv()
        self.analytics.plot_accuracy_curve()
        self.analytics.plot_communication_cost()
        self.analytics.plot_client_participation()
        summary = self.analytics.generate_summary()

        logger.info("Experiment '%s' complete.", self.config.experiment_name)
        logger.info("Summary: %s", summary)

        return all_metrics

    # ------------------------------------------------------------------
    # Client selection
    # ------------------------------------------------------------------

    def select_clients(self, round_num: int) -> List[int]:
        """Select a random subset of clients for the given round.

        Parameters
        ----------
        round_num : int
            Current round number (reserved for future stratified sampling).

        Returns
        -------
        List[int]
            Client IDs chosen to participate this round.
        """
        num_selected = max(1, int(self.config.num_clients * self.config.fraction_fit))
        all_ids = list(range(self.config.num_clients))
        return random.sample(all_ids, min(num_selected, len(all_ids)))

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self, updates: List[ModelUpdate]) -> nn.Module:
        """Aggregate model updates using the configured algorithm.

        Parameters
        ----------
        updates : List[ModelUpdate]
            Successfully received client model updates.

        Returns
        -------
        nn.Module
            Updated global model.
        """
        if self.config.algorithm == "fedavg":
            new_weights = fedavg_aggregate(updates)
        elif self.config.algorithm == "fedprox":
            new_weights = fedprox_aggregate(updates)
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

        self.global_model.load_state_dict(new_weights)
        return self.global_model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _collect_updates(
        self,
        selected_ids: List[int],
        round_num: int,
    ) -> tuple[List[ModelUpdate], int]:
        """Train selected clients and transmit updates through impairment layer.

        Returns
        -------
        tuple[List[ModelUpdate], int]
            (received_updates, dropped_count)
        """
        received: List[ModelUpdate] = []
        dropped = 0

        # Run client training concurrently
        tasks = []
        for cid in selected_ids:
            client = self.clients[cid]
            tasks.append(self._train_and_transmit(client, round_num))

        results = await asyncio.gather(*tasks)

        for result in results:
            if result is None:
                dropped += 1
            else:
                received.append(result)

        return received, dropped

    async def _train_and_transmit(
        self,
        client: FLClient,
        round_num: int,
    ) -> Optional[ModelUpdate]:
        """Train one client and transmit update through Network Impairment Engine."""
        # Local training
        update = await client.train(self.global_model, round_num)

        # Transmit through impairment layer (may return None if dropped)
        transmitted = await self.impairment_engine.transmit(
            update, client.client_id, round_num
        )
        return transmitted

    def _evaluate_global_model(self) -> tuple[float, float]:
        """Evaluate the global model on a test set.

        Returns
        -------
        tuple[float, float]
            (accuracy, loss) — currently returns placeholders.

        TODO: Implement proper evaluation with a held-out test set.
        """
        # Placeholder — will be implemented when data loading is connected
        return 0.0, 0.0
