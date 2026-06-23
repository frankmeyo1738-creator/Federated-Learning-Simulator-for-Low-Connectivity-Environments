"""
Unit Tests — Network Impairment Engine
========================================
Tests for ``NetworkImpairmentEngine`` in ``src/network/impairment.py``.

Validates that dropout, packet loss, and latency behave correctly at
boundary conditions (0% and 100%) and that measured latency falls
within expected bounds for Gaussian-distributed delays.

All tests use ``asyncio.run()`` to invoke the async ``transmit()`` method.

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""

import asyncio

import pytest
import torch

from src.config import NetworkProfile
from src.core.client import ModelUpdate
from src.network.impairment import NetworkImpairmentEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_update():
    """Create a minimal ModelUpdate for transmission tests.

    The weight values are irrelevant — we only care about whether
    the update is returned (transmitted) or None (dropped).
    """
    return ModelUpdate(
        client_id=0,
        weights={"w": torch.tensor(1.0)},
        num_samples=100,
        loss=0.1,
        round_num=1,
        training_time_seconds=0.5,
        model_size_bytes=1024,
    )


def _make_profile(**overrides) -> NetworkProfile:
    """Create a NetworkProfile with safe defaults, overriding as needed.

    Defaults: no latency, no packet loss, no dropout, 10 Mbps bandwidth.
    """
    defaults = {
        "name": "test_profile",
        "latency_min_ms": 0.0,
        "latency_max_ms": 0.0,
        "bandwidth_mbps": 10.0,
        "packet_loss_rate": 0.0,
        "dropout_probability": 0.0,
        "dropout_duration_rounds": 0,
    }
    defaults.update(overrides)
    return NetworkProfile(**defaults)


# ---------------------------------------------------------------------------
# Dropout Tests
# ---------------------------------------------------------------------------

class TestDropout:
    """Tests for client dropout behaviour."""

    def test_baseline_no_dropout(self, dummy_update):
        """Verify that with dropout_probability=0.0, every single
        transmission succeeds (returns non-None).

        This is the control case — if any transmission is dropped at
        0% probability, the random number generation is broken.
        """
        profile = _make_profile(dropout_probability=0.0)
        engine = NetworkImpairmentEngine(profile)

        results = []
        for i in range(100):
            result = asyncio.run(
                engine.transmit(dummy_update, client_id=i % 10, round_num=i)
            )
            results.append(result)

        assert all(r is not None for r in results), (
            "0% dropout should never drop any updates"
        )

    def test_certain_dropout(self, dummy_update):
        """Verify that with dropout_probability=1.0, every transmission
        is dropped (returns None).

        This is the extreme case — the engine must guarantee no
        transmissions succeed when dropout is certain.
        """
        profile = _make_profile(
            dropout_probability=1.0, dropout_duration_rounds=1
        )
        engine = NetworkImpairmentEngine(profile)

        results = []
        for i in range(10):
            result = asyncio.run(
                engine.transmit(dummy_update, client_id=i, round_num=1)
            )
            results.append(result)

        assert all(r is None for r in results), (
            "100% dropout should drop all updates"
        )


# ---------------------------------------------------------------------------
# Packet Loss Tests
# ---------------------------------------------------------------------------

class TestPacketLoss:
    """Tests for packet-loss behaviour."""

    def test_packet_loss_zero(self, dummy_update):
        """Verify that with packet_loss_rate=0.0, all transmissions
        succeed.

        Zero packet loss should guarantee every update is delivered,
        regardless of how many transmissions are attempted.
        """
        profile = _make_profile(packet_loss_rate=0.0)
        engine = NetworkImpairmentEngine(profile)

        results = []
        for i in range(100):
            result = asyncio.run(
                engine.transmit(dummy_update, client_id=i % 10, round_num=i)
            )
            results.append(result)

        assert all(r is not None for r in results), (
            "0% packet loss should never drop updates"
        )

    def test_certain_packet_loss(self, dummy_update):
        """Verify that with packet_loss_rate=1.0, all transmissions
        are dropped.

        At 100% packet loss, every update should be lost to simulate
        complete network failure.
        """
        profile = _make_profile(packet_loss_rate=1.0)
        engine = NetworkImpairmentEngine(profile)

        results = []
        for i in range(10):
            result = asyncio.run(
                engine.transmit(dummy_update, client_id=i, round_num=1)
            )
            results.append(result)

        assert all(r is None for r in results), (
            "100% packet loss should drop all updates"
        )


# ---------------------------------------------------------------------------
# Latency Tests
# ---------------------------------------------------------------------------

class TestLatency:
    """Tests for latency simulation."""

    def test_latency_within_bounds(self, dummy_update):
        """Verify that average latency falls within generous bounds when
        latency_min=100ms and latency_max=200ms.

        The engine uses a Gaussian distribution (mu=150, sigma=15).
        Individual samples are clamped to [100, 200], so the average
        should comfortably fall between 50 and 400 ms (generous bounds
        to account for bandwidth delay added on top of base latency).
        """
        profile = _make_profile(
            latency_min_ms=100.0,
            latency_max_ms=200.0,
            bandwidth_mbps=10.0,
        )
        engine = NetworkImpairmentEngine(profile)

        for i in range(20):
            asyncio.run(
                engine.transmit(dummy_update, client_id=i % 5, round_num=i)
            )

        avg_latency = engine.get_avg_latency()
        assert 50.0 < avg_latency < 400.0, (
            f"Average latency {avg_latency:.1f}ms is outside expected "
            f"bounds [50, 400] for min=100, max=200"
        )


# ---------------------------------------------------------------------------
# Baseline Profile (Integration-style Smoke Test)
# ---------------------------------------------------------------------------

class TestBaselineProfile:
    """End-to-end smoke test with a 'perfect' network profile."""

    def test_baseline_profile_passes_all(self, dummy_update):
        """Verify that a baseline profile with zero loss, zero dropout,
        and zero latency achieves a 100% pass rate.

        This is the 'ideal network' control test — it validates that
        the engine does not introduce any spurious drops when all
        impairment parameters are disabled.
        """
        profile = _make_profile(
            name="baseline",
            latency_min_ms=0.0,
            latency_max_ms=0.0,
            bandwidth_mbps=100.0,
            packet_loss_rate=0.0,
            dropout_probability=0.0,
            dropout_duration_rounds=0,
        )
        engine = NetworkImpairmentEngine(profile)

        passed = 0
        total = 100
        for i in range(total):
            result = asyncio.run(
                engine.transmit(dummy_update, client_id=i % 10, round_num=i)
            )
            if result is not None:
                passed += 1

        assert passed == total, (
            f"Baseline profile should have 100% pass rate, got {passed}/{total}"
        )
        assert engine.get_drop_rate() == 0.0
