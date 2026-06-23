"""
Unit Tests — Data Partitioning
================================
Tests for ``create_client_partitions()`` in ``src/data/loader.py``.

Validates both IID (uniform random) and non-IID (Dirichlet α=0.5)
partitioning strategies across properties: correct count, equal sizing,
no overlap, full coverage, and seed reproducibility.

Uses torchvision MNIST (downloaded once via a session-scoped fixture
to a temp directory) as the reference dataset.

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""

import numpy as np
import pytest

from src.data.loader import create_client_partitions, download_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mnist_dataset():
    """Load MNIST from the project's existing data/ directory.

    Uses ``session`` scope so the dataset is loaded only once across
    all test functions. Relies on MNIST already being downloaded
    (via prior experiment runs) at ``./data/``.
    """
    return download_dataset("mnist", data_dir="./data")


NUM_CLIENTS = 10


# ---------------------------------------------------------------------------
# IID Partitioning Tests
# ---------------------------------------------------------------------------

class TestIIDPartitioning:
    """Tests for IID (uniform random shuffle-and-split) partitioning."""

    def test_iid_partition_count(self, mnist_dataset):
        """Verify that the number of partitions equals num_clients.

        This is a basic structural check — if the count is wrong the
        entire FL simulation will assign data to the wrong clients.
        """
        partitions = create_client_partitions(
            mnist_dataset, num_clients=NUM_CLIENTS, iid=True, seed=42
        )
        assert len(partitions) == NUM_CLIENTS

    def test_iid_partition_sizes_equal(self, mnist_dataset):
        """Verify that all IID partitions have (approximately) equal size.

        With 60 000 MNIST samples and 10 clients, each should get
        exactly 6 000. The implementation uses np.array_split, which
        distributes remainders one-at-a-time, so sizes differ by at
        most 1 when the split is not perfectly even.
        """
        partitions = create_client_partitions(
            mnist_dataset, num_clients=NUM_CLIENTS, iid=True, seed=42
        )
        sizes = [len(p) for p in partitions]
        assert max(sizes) - min(sizes) <= 1, (
            f"IID partition sizes vary too much: {sizes}"
        )

    def test_iid_no_overlap(self, mnist_dataset):
        """Verify that no sample index appears in more than one partition.

        Overlapping indices would mean some samples are counted multiple
        times during federated training, biasing the aggregated model.
        """
        partitions = create_client_partitions(
            mnist_dataset, num_clients=NUM_CLIENTS, iid=True, seed=42
        )
        all_indices = []
        for p in partitions:
            all_indices.extend(p.indices)

        assert len(all_indices) == len(set(all_indices)), (
            "Some indices appear in multiple partitions"
        )

    def test_iid_covers_all_samples(self, mnist_dataset):
        """Verify that the union of all partition indices covers every
        sample in the original dataset.

        Missing indices would mean some training data is never used,
        wasting information and potentially biasing results.
        """
        partitions = create_client_partitions(
            mnist_dataset, num_clients=NUM_CLIENTS, iid=True, seed=42
        )
        all_indices = set()
        for p in partitions:
            all_indices.update(p.indices)

        expected = set(range(len(mnist_dataset)))
        assert all_indices == expected, (
            f"Missing {len(expected - all_indices)} indices from partitions"
        )


# ---------------------------------------------------------------------------
# Non-IID (Dirichlet) Partitioning Tests
# ---------------------------------------------------------------------------

class TestNonIIDPartitioning:
    """Tests for non-IID (Dirichlet distribution) partitioning."""

    def test_noniid_partition_count(self, mnist_dataset):
        """Verify that the number of non-IID partitions equals num_clients.

        Same structural check as IID — the Dirichlet strategy must
        produce the right number of client subsets.
        """
        partitions = create_client_partitions(
            mnist_dataset, num_clients=NUM_CLIENTS, iid=False, alpha=0.5, seed=42
        )
        assert len(partitions) == NUM_CLIENTS

    def test_noniid_covers_all_samples(self, mnist_dataset):
        """Verify that non-IID partitions collectively cover every sample.

        The Dirichlet allocation must not silently drop samples due to
        rounding errors in the proportion → count conversion.
        """
        partitions = create_client_partitions(
            mnist_dataset, num_clients=NUM_CLIENTS, iid=False, alpha=0.5, seed=42
        )
        all_indices = set()
        for p in partitions:
            all_indices.update(p.indices)

        expected = set(range(len(mnist_dataset)))
        assert all_indices == expected, (
            f"Missing {len(expected - all_indices)} indices from partitions"
        )

    def test_noniid_no_overlap(self, mnist_dataset):
        """Verify that no sample index appears in more than one non-IID
        partition.

        The Dirichlet partitioning splits each class's indices
        sequentially, so overlaps would indicate an implementation bug
        in the index slicing.
        """
        partitions = create_client_partitions(
            mnist_dataset, num_clients=NUM_CLIENTS, iid=False, alpha=0.5, seed=42
        )
        all_indices = []
        for p in partitions:
            all_indices.extend(p.indices)

        assert len(all_indices) == len(set(all_indices)), (
            "Some indices appear in multiple non-IID partitions"
        )


# ---------------------------------------------------------------------------
# Reproducibility Tests
# ---------------------------------------------------------------------------

class TestPartitionReproducibility:
    """Tests for deterministic seeding of partitions."""

    def test_reproducibility_with_seed(self, mnist_dataset):
        """Verify that the same seed produces identical partitions, and
        different seeds produce different partitions.

        Reproducibility is critical for scientific experiments — without
        it, results cannot be replicated or compared across runs.
        """
        # Same seed → identical partitions
        partitions_a = create_client_partitions(
            mnist_dataset, num_clients=NUM_CLIENTS, iid=True, seed=42
        )
        partitions_b = create_client_partitions(
            mnist_dataset, num_clients=NUM_CLIENTS, iid=True, seed=42
        )

        for pa, pb in zip(partitions_a, partitions_b):
            assert pa.indices == pb.indices, (
                "Same seed should produce identical partitions"
            )

        # Different seed → different partitions
        partitions_c = create_client_partitions(
            mnist_dataset, num_clients=NUM_CLIENTS, iid=True, seed=99
        )

        any_different = any(
            pa.indices != pc.indices
            for pa, pc in zip(partitions_a, partitions_c)
        )
        assert any_different, (
            "Different seeds should produce different partitions"
        )
