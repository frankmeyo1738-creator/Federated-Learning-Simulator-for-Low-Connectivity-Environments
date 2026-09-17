"""
Integration Tests — Dropout State Persistence
===============================================
Tests multi-round client disconnection state tracking in NetworkImpairmentEngine.
Verifies that once triggered, a client remains offline for exactly
`dropout_duration_rounds` consecutive rounds before becoming eligible to transmit again.

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""

import pytest
import torch

from src.config import ExperimentConfig, NetworkProfile
from src.core.client import FLClient, ModelUpdate
from src.network.impairment import NetworkImpairmentEngine


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def _dummy_update(client_id: int, round_num: int) -> ModelUpdate:
    """Create a lightweight model update for transmission testing."""
    return ModelUpdate(
        client_id=client_id,
        weights={"w": torch.ones(2)},
        num_samples=10,
        loss=0.5,
        round_num=round_num,
        training_time_seconds=0.1,
        model_size_bytes=1024,
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestDropoutPersistence:
    """Tests multi-round persistent dropout dynamics."""

    @pytest.mark.asyncio
    async def test_dropout_multi_round_persistence_lifecycle(self):
        """A client that drops out at round R with duration D must remain offline
        through rounds R, ..., R+D-1, and be eligible again at round R+D.
        """
        # Configure 3 rounds of dropout duration
        # packet_loss=0 to isolate dropout behavior; zero latency for instant testing
        profile = NetworkProfile(
            name="dropout_test",
            latency_min_ms=0.0,
            latency_max_ms=0.0,
            bandwidth_mbps=100.0,
            packet_loss_rate=0.0,
            dropout_probability=1.0,  # Force dropout on first roll
            dropout_duration_rounds=3,
        )

        engine = NetworkImpairmentEngine(profile)
        client_id = 1

        # Round 1: Dropout triggers (dropout_end = 1 + 3 = 4)
        up1 = _dummy_update(client_id, round_num=1)
        res1 = await engine.transmit(up1, client_id, round_num=1)
        assert res1 is None, "Round 1: Client should have dropped out."
        assert engine._dropout_end.get(client_id) == 4

        # Round 2: Still within dropout window (2 < 4)
        up2 = _dummy_update(client_id, round_num=2)
        res2 = await engine.transmit(up2, client_id, round_num=2)
        assert res2 is None, "Round 2: Client should still be offline (round 2 < 4)."

        # Round 3: Still within dropout window (3 < 4)
        up3 = _dummy_update(client_id, round_num=3)
        res3 = await engine.transmit(up3, client_id, round_num=3)
        assert res3 is None, "Round 3: Client should still be offline (round 3 < 4)."

        # Round 4: Dropout has expired (4 >= 4)
        # Turn off dropout_probability so a new dropout isn't triggered immediately
        profile.dropout_probability = 0.0
        up4 = _dummy_update(client_id, round_num=4)
        res4 = await engine.transmit(up4, client_id, round_num=4)

        assert res4 is not None, "Round 4: Client should have recovered and successfully transmitted."
        assert client_id not in engine._dropout_end, "Client should be removed from _dropout_end once recovered."

    @pytest.mark.asyncio
    async def test_multiple_clients_independent_dropout_tracking(self):
        """Independent clients have distinct dropout timelines."""
        profile = NetworkProfile(
            name="multi_client_dropout",
            latency_min_ms=0.0,
            latency_max_ms=0.0,
            bandwidth_mbps=100.0,
            packet_loss_rate=0.0,
            dropout_probability=0.0,  # Controlled manually
            dropout_duration_rounds=2,
        )
        engine = NetworkImpairmentEngine(profile)

        # Manually trigger dropout for Client 0 at Round 1 (offline rounds 1, 2; back at 3)
        profile.dropout_probability = 1.0
        res0_r1 = await engine.transmit(_dummy_update(0, 1), client_id=0, round_num=1)
        assert res0_r1 is None
        assert engine._dropout_end[0] == 3

        # Client 1 does not drop at Round 1
        profile.dropout_probability = 0.0
        res1_r1 = await engine.transmit(_dummy_update(1, 1), client_id=1, round_num=1)
        assert res1_r1 is not None

        # At Round 2, trigger dropout for Client 1 (offline rounds 2, 3; back at 4)
        profile.dropout_probability = 1.0
        res1_r2 = await engine.transmit(_dummy_update(1, 2), client_id=1, round_num=2)
        assert res1_r2 is None
        assert engine._dropout_end[1] == 4

        # Client 0 is also still offline at Round 2
        res0_r2 = await engine.transmit(_dummy_update(0, 2), client_id=0, round_num=2)
        assert res0_r2 is None

        # At Round 3, no new dropouts
        profile.dropout_probability = 0.0
        # Client 0 should now be recovered (3 >= 3)
        res0_r3 = await engine.transmit(_dummy_update(0, 3), client_id=0, round_num=3)
        assert res0_r3 is not None

        # Client 1 should STILL be offline (3 < 4)
        res1_r3 = await engine.transmit(_dummy_update(1, 3), client_id=1, round_num=3)
        assert res1_r3 is None

        # At Round 4, Client 1 recovers (4 >= 4)
        res1_r4 = await engine.transmit(_dummy_update(1, 4), client_id=1, round_num=4)
        assert res1_r4 is not None

    @pytest.mark.asyncio
    async def test_zero_duration_dropout_does_not_persist(self):
        """When dropout_duration_rounds=0, dropout is transient (1 round only)."""
        profile = NetworkProfile(
            name="transient_dropout",
            latency_min_ms=0.0,
            latency_max_ms=0.0,
            bandwidth_mbps=100.0,
            packet_loss_rate=0.0,
            dropout_probability=1.0,
            dropout_duration_rounds=0,
        )
        engine = NetworkImpairmentEngine(profile)

        # Round 1: drops out, end round is 1 + 0 = 1
        res1 = await engine.transmit(_dummy_update(0, 1), client_id=0, round_num=1)
        assert res1 is None
        assert engine._dropout_end[0] == 1

        # Round 2: 2 >= 1, so dropout immediately ends
        profile.dropout_probability = 0.0
        res2 = await engine.transmit(_dummy_update(0, 2), client_id=0, round_num=2)
        assert res2 is not None
        assert 0 not in engine._dropout_end
