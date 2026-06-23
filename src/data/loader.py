"""
Data Loading and Client Partitioning
=======================================
Downloads MNIST / CIFAR-10 via ``torchvision`` and partitions the training
set across federated clients using either IID (uniform random) or non-IID
(Dirichlet) strategies.

Non-IID partitioning uses a Dirichlet distribution with concentration
parameter ``alpha`` following the approach from the FedProx paper
(Li et al., 2020). Lower ``alpha`` values produce more heterogeneous
(non-IID) partitions across clients.

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""

import logging
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def download_dataset(dataset: str, data_dir: str = "./data") -> Dataset:
    """Download and return the training split of the specified dataset.

    Parameters
    ----------
    dataset : str
        Dataset name — ``"mnist"`` or ``"cifar10"``.
    data_dir : str, optional
        Directory where dataset files will be stored (default: ``"./data"``).

    Returns
    -------
    Dataset
        The training dataset with appropriate transforms applied.

    Raises
    ------
    ValueError
        If the dataset name is not recognised.
    """
    dataset = dataset.lower()

    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
        ])
        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        logger.info(
            "MNIST training set downloaded — %d samples", len(train_dataset)
        )
        return train_dataset

    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),   # CIFAR-10 mean
                (0.2470, 0.2435, 0.2616),   # CIFAR-10 std
            ),
        ])
        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        logger.info(
            "CIFAR-10 training set downloaded — %d samples", len(train_dataset)
        )
        return train_dataset

    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Supported: 'mnist', 'cifar10'"
        )


# ---------------------------------------------------------------------------
# Client partitioning
# ---------------------------------------------------------------------------

def create_client_partitions(
    dataset: Dataset,
    num_clients: int,
    iid: bool = True,
    alpha: float = 0.5,
    seed: Optional[int] = None,
) -> List[Subset]:
    """Partition a training dataset across federated clients.

    Parameters
    ----------
    dataset : Dataset
        The full training dataset to partition.
    num_clients : int
        Number of FL clients to split the data among.
    iid : bool, optional
        If ``True``, data is shuffled and split equally (IID).
        If ``False``, a Dirichlet distribution is used to create
        heterogeneous (non-IID) partitions (default: ``True``).
    alpha : float, optional
        Dirichlet concentration parameter (only used when ``iid=False``).
        Lower values produce more heterogeneous distributions.
        Default is ``0.5`` following the FedProx paper.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    List[Subset]
        A list of ``num_clients`` ``torch.utils.data.Subset`` objects,
        one per client.
    """
    rng = np.random.default_rng(seed)
    num_samples = len(dataset)

    if iid:
        partitions = _partition_iid(num_samples, num_clients, rng)
    else:
        partitions = _partition_dirichlet(dataset, num_clients, alpha, rng)

    subsets = [Subset(dataset, indices.tolist()) for indices in partitions]

    # Log partition statistics
    sizes = [len(s) for s in subsets]
    logger.info(
        "Created %d client partitions (iid=%s, alpha=%.2f) — "
        "min=%d, max=%d, mean=%.1f samples per client",
        num_clients, iid, alpha, min(sizes), max(sizes), np.mean(sizes),
    )

    return subsets


def _partition_iid(
    num_samples: int,
    num_clients: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Shuffle indices and split equally across clients.

    Parameters
    ----------
    num_samples : int
        Total number of training samples.
    num_clients : int
        Number of clients.
    rng : np.random.Generator
        Seeded random number generator.

    Returns
    -------
    List[np.ndarray]
        Per-client index arrays.
    """
    indices = rng.permutation(num_samples)
    return np.array_split(indices, num_clients)


def _partition_dirichlet(
    dataset: Dataset,
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Partition data using a Dirichlet distribution (non-IID).

    For each class, samples a proportion vector from Dir(alpha) and
    assigns that fraction of the class's indices to each client.
    This follows the non-IID partitioning strategy from the FedProx
    paper (Li et al., 2020).

    Parameters
    ----------
    dataset : Dataset
        The training dataset (must have a ``targets`` attribute).
    num_clients : int
        Number of clients.
    alpha : float
        Dirichlet concentration parameter. Lower = more heterogeneous.
    rng : np.random.Generator
        Seeded random number generator.

    Returns
    -------
    List[np.ndarray]
        Per-client index arrays.
    """
    # Extract labels — handle both Tensor and list targets
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()
        else:
            targets = np.array(targets)
    else:
        raise AttributeError(
            "Dataset must have a 'targets' attribute for non-IID partitioning."
        )

    num_classes = len(np.unique(targets))
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for class_idx in range(num_classes):
        # Indices belonging to this class
        class_mask = np.where(targets == class_idx)[0]
        rng.shuffle(class_mask)

        # Sample proportions from Dirichlet distribution
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        # Convert proportions to actual counts (ensuring all samples assigned)
        counts = (proportions * len(class_mask)).astype(int)
        # Assign any remainder due to rounding to a random client
        remainder = len(class_mask) - counts.sum()
        if remainder > 0:
            lucky_clients = rng.choice(num_clients, size=remainder, replace=False)
            for lc in lucky_clients:
                counts[lc] += 1

        # Distribute indices
        start = 0
        for client_id in range(num_clients):
            end = start + counts[client_id]
            client_indices[client_id].extend(class_mask[start:end].tolist())
            start = end

    return [np.array(indices) for indices in client_indices]


# ---------------------------------------------------------------------------
# Test set loader
# ---------------------------------------------------------------------------

def get_test_loader(
    dataset: str,
    data_dir: str = "./data",
    batch_size: int = 256,
) -> DataLoader:
    """Create a DataLoader for the test split of the specified dataset.

    Parameters
    ----------
    dataset : str
        Dataset name — ``"mnist"`` or ``"cifar10"``.
    data_dir : str, optional
        Directory where dataset files are stored (default: ``"./data"``).
    batch_size : int, optional
        Batch size for the test loader (default: ``256``).

    Returns
    -------
    DataLoader
        A DataLoader for the full test set.

    Raises
    ------
    ValueError
        If the dataset name is not recognised.
    """
    dataset = dataset.lower()

    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ])
        test_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Supported: 'mnist', 'cifar10'"
        )

    logger.info(
        "%s test set loaded — %d samples, batch_size=%d",
        dataset.upper(), len(test_dataset), batch_size,
    )

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,      # Safest for cross-platform reproducibility
        pin_memory=False,
    )
