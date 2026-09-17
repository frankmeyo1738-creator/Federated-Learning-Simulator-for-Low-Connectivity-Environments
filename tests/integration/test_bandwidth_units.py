"""
Integration Tests — Bandwidth Unit Conversion and Accounting
==============================================================
Validates that bandwidth throttling calculations and communication cost tracking
strictly adhere to binary mebibytes (1 MiB = 1,048,576 bytes = 1024² B) rather
than decimal megabytes (10⁶ B).

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""

import pytest
import torch

from src.analytics.tracker import AnalyticsTracker, RoundMetrics
from src.config import NetworkProfile
from src.core.client import ModelUpdate
from src.network.impairment import NetworkImpairmentEngine


class TestBandwidthUnitConversion:
    """Verifies standard binary byte-to-megabit and byte-to-mebibyte conversions."""

    def test_analytics_tracker_uses_genuine_binary_mebibytes(self, tmp_path):
        """AnalyticsTracker.generate_summary() must divide total bytes by 1024 * 1024 (1048576)."""
        tracker = AnalyticsTracker(output_dir=str(tmp_path), experiment_name="bandwidth_test")

        # Record 2 rounds with exactly 1 MiB (1,048,576 bytes) each
        bytes_per_round = 1024 * 1024  # 1 MiB
        for r in range(1, 3):
            tracker.record_round(
                RoundMetrics(
                    round_num=r,
                    global_accuracy=0.95,
                    global_loss=0.15,
                    participating_clients=5,
                    dropped_updates=0,
                    avg_latency_ms=100.0,
                    total_bytes_transmitted=bytes_per_round,
                    round_duration_seconds=1.0,
                )
            )

        summary = tracker.generate_summary()

        expected_total_bytes = 2 * 1024 * 1024  # 2,097,152 bytes
        assert summary["total_bytes_transmitted"] == expected_total_bytes

        # Must equal exactly 2.0 MiB
        assert summary["total_bytes_transmitted_mb"] == pytest.approx(2.0, rel=1e-7)

        # Must NOT match decimal calculation (2,097,152 / 1,000,000 = 2.097152 MB)
        decimal_mb = expected_total_bytes / 1e6
        assert abs(summary["total_bytes_transmitted_mb"] - decimal_mb) > 0.09, (
            "Tracker is erroneously computing decimal megabytes (10^6) instead of binary (1024^2)!"
        )

    def test_bandwidth_throttling_delay_mathematical_precision(self):
        """NetworkImpairmentEngine._apply_bandwidth_throttle converts bytes to megabits
        via (bytes * 8) / (1024 * 1024) and computes delay_seconds = megabits / bandwidth_mbps.
        """
        # Test with 2.0 Mbps bandwidth
        profile = NetworkProfile(
            name="throttling_profile",
            latency_min_ms=0.0,
            latency_max_ms=0.0,
            bandwidth_mbps=2.0,
            packet_loss_rate=0.0,
            dropout_probability=0.0,
            dropout_duration_rounds=0,
        )
        engine = NetworkImpairmentEngine(profile)

        # Payload of exactly 512 KiB = 524,288 bytes
        # 524,288 bytes * 8 = 4,194,304 bits = 4.0 Megabits (binary)
        # At 2.0 Mbps, delay = 4.0 / 2.0 = 2.0 seconds = 2000.0 milliseconds
        num_bytes = 512 * 1024
        update = ModelUpdate(
            client_id=0,
            weights={"w": torch.zeros(1)},
            num_samples=10,
            loss=0.1,
            round_num=1,
            training_time_seconds=0.1,
            model_size_bytes=num_bytes,
        )

        delay_ms = engine._apply_bandwidth_throttle(update)

        assert delay_ms == pytest.approx(2000.0, rel=1e-5), (
            f"Expected 2000.0 ms delay for 512 KiB over 2.0 Mbps, got {delay_ms} ms"
        )

    def test_zero_bandwidth_profile_yields_zero_throttling_delay(self):
        """Profiles with bandwidth <= 0 (e.g. unconstrained ideal network) should incur zero delay."""
        profile = NetworkProfile(
            name="zero_bw_profile",
            latency_min_ms=0.0,
            latency_max_ms=0.0,
            bandwidth_mbps=0.0,
            packet_loss_rate=0.0,
            dropout_probability=0.0,
            dropout_duration_rounds=0,
        )
        engine = NetworkImpairmentEngine(profile)

        update = ModelUpdate(
            client_id=0,
            weights={"w": torch.zeros(1)},
            num_samples=10,
            loss=0.1,
            round_num=1,
            training_time_seconds=0.1,
            model_size_bytes=1048576,
        )

        assert engine._apply_bandwidth_throttle(update) == 0.0
