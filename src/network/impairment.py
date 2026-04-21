"""
Layer 3 — Network Impairment Engine
=====================================
Intercepts all model update transmissions and applies configured
impairment conditions (latency, packet loss, bandwidth throttling,
client dropout). This is the core research contribution.

Key implementation detail:
    Latency uses Gaussian distribution — random.gauss(mu, sigma) where
    mu is the midpoint of min/max range and sigma is 15% of the range.
    This produces realistic variance rather than uniform random.

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""

import asyncio
import logging
import random
from collections import defaultdict
from enum import Enum
from typing import Dict, Optional, Set, Tuple

from src.config import NetworkProfile
from src.core.client import ModelUpdate

logger = logging.getLogger(__name__)


class NetworkCondition(Enum):
    """Represents the current state of the simulated network for a client."""
    NORMAL = "normal"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"


class NetworkImpairmentEngine:
    """Simulates realistic network impairments on model update transmissions.

    Called exclusively by ``FLServer`` during the aggregation phase.
    Returns ``None`` for dropped updates — the server handles missing
    updates gracefully.

    Parameters
    ----------
    profile : NetworkProfile
        Network condition profile (e.g. rural_zambia, severe_disruption).
    """

    def __init__(self, profile: NetworkProfile) -> None:
        self.profile = profile

        # Latency distribution parameters (Gaussian)
        self._latency_mu = (profile.latency_min_ms + profile.latency_max_ms) / 2.0
        latency_range = profile.latency_max_ms - profile.latency_min_ms
        self._latency_sigma = latency_range * 0.15  # 15% of range

        # Dropout tracking: client_id → round when dropout ends
        self._dropout_end: Dict[int, int] = {}

        # Metrics tracking
        self._latency_samples: list[float] = []
        self._total_transmissions = 0
        self._total_drops = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def transmit(
        self,
        update: ModelUpdate,
        client_id: int,
        round_num: int,
    ) -> Optional[ModelUpdate]:
        """Simulate transmission of a model update through the impaired network.

        Parameters
        ----------
        update : ModelUpdate
            The model update to transmit.
        client_id : int
            ID of the sending client.
        round_num : int
            Current FL round number.

        Returns
        -------
        Optional[ModelUpdate]
            The update if successfully transmitted, or ``None`` if dropped.
        """
        self._total_transmissions += 1

        # 1. Check client dropout (multi-round disconnection)
        if self._apply_dropout(client_id, round_num):
            logger.debug(
                "Client %d DROPPED OUT (round %d) — disconnected until round %d",
                client_id, round_num, self._dropout_end.get(client_id, -1),
            )
            self._total_drops += 1
            return None

        # 2. Check packet loss (single-update drop)
        if self._apply_packet_loss():
            logger.debug(
                "Client %d update LOST (packet loss) at round %d",
                client_id, round_num,
            )
            self._total_drops += 1
            return None

        # 3. Apply bandwidth throttling delay
        bw_delay = self._apply_bandwidth_throttle(update)

        # 4. Apply latency
        latency = await self._apply_latency()

        total_delay_ms = latency + bw_delay
        self._latency_samples.append(total_delay_ms)

        logger.debug(
            "Client %d transmitted — latency=%.1fms, bw_delay=%.1fms, total=%.1fms",
            client_id, latency, bw_delay, total_delay_ms,
        )

        return update

    def get_avg_latency(self) -> float:
        """Return the average latency across all successful transmissions."""
        if not self._latency_samples:
            return 0.0
        return sum(self._latency_samples) / len(self._latency_samples)

    def get_drop_rate(self) -> float:
        """Return the observed drop rate so far."""
        if self._total_transmissions == 0:
            return 0.0
        return self._total_drops / self._total_transmissions

    def reset_metrics(self) -> None:
        """Reset per-experiment metrics."""
        self._latency_samples.clear()
        self._total_transmissions = 0
        self._total_drops = 0

    # ------------------------------------------------------------------
    # Impairment implementations
    # ------------------------------------------------------------------

    async def _apply_latency(self) -> float:
        """Simulate network latency using Gaussian distribution.

        Returns the sampled latency in milliseconds and sleeps for the
        equivalent duration to simulate real-time delay.

        Returns
        -------
        float
            Sampled latency in milliseconds (clamped to min/max bounds).
        """
        latency_ms = random.gauss(self._latency_mu, self._latency_sigma)

        # Clamp to configured bounds
        latency_ms = max(self.profile.latency_min_ms, latency_ms)
        latency_ms = min(self.profile.latency_max_ms, latency_ms)

        # Simulate the delay (convert ms → seconds)
        await asyncio.sleep(latency_ms / 1000.0)

        return latency_ms

    def _apply_packet_loss(self) -> bool:
        """Determine if this transmission should be dropped due to packet loss.

        Returns
        -------
        bool
            ``True`` if the packet is lost (should be dropped), ``False`` otherwise.
        """
        return random.random() < self.profile.packet_loss_rate

    def _apply_bandwidth_throttle(self, update: ModelUpdate) -> float:
        """Calculate additional delay caused by bandwidth constraints.

        Parameters
        ----------
        update : ModelUpdate
            The update being transmitted (used to determine payload size).

        Returns
        -------
        float
            Additional delay in milliseconds due to bandwidth throttling.
        """
        if self.profile.bandwidth_mbps <= 0:
            return 0.0

        # Convert model size from bytes → megabits
        size_megabits = (update.model_size_bytes * 8) / (1024 * 1024)

        # Time = size / bandwidth  (seconds → milliseconds)
        delay_seconds = size_megabits / self.profile.bandwidth_mbps
        return delay_seconds * 1000.0

    def _apply_dropout(self, client_id: int, round_num: int) -> bool:
        """Determine if a client has dropped out for this round.

        Dropout is a multi-round event: once triggered, the client remains
        offline for ``dropout_duration_rounds`` consecutive rounds.

        Parameters
        ----------
        client_id : int
            The client attempting to transmit.
        round_num : int
            Current round number.

        Returns
        -------
        bool
            ``True`` if the client is currently in a dropout period.
        """
        # Check if client is already in a dropout period
        if client_id in self._dropout_end:
            if round_num < self._dropout_end[client_id]:
                return True  # Still offline
            else:
                # Dropout period ended — remove tracking
                del self._dropout_end[client_id]

        # Roll for new dropout
        if random.random() < self.profile.dropout_probability:
            self._dropout_end[client_id] = (
                round_num + self.profile.dropout_duration_rounds
            )
            return True

        return False
