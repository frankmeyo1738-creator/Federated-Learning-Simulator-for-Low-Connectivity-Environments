"""
CNN Model Architectures
=========================
Provides convolutional neural network architectures for MNIST and CIFAR-10
datasets, used as the global model in federated learning experiments.

Architectures
-------------
- **MNISTNet**: Lightweight 2-layer CNN for 28×28 grayscale images.
- **CIFAR10Net**: Deeper 3-layer CNN with batch normalisation for 32×32 RGB images.

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""

import torch.nn as nn


# ---------------------------------------------------------------------------
# MNIST CNN
# ---------------------------------------------------------------------------

class MNISTNet(nn.Module):
    """Convolutional neural network for MNIST digit classification.

    Architecture
    ------------
    Conv2d(1, 32, 3) → ReLU → Conv2d(32, 64, 3) → ReLU → MaxPool2d(2)
    → Dropout(0.25) → Flatten → Linear(9216, 128) → ReLU → Dropout(0.5)
    → Linear(128, 10)

    Input shape : (batch, 1, 28, 28)
    Output shape: (batch, 10) — logits for digits 0–9
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, 1, 28, 28)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, 10)``.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# CIFAR-10 CNN
# ---------------------------------------------------------------------------

class CIFAR10Net(nn.Module):
    """Convolutional neural network for CIFAR-10 image classification.

    Architecture
    ------------
    Conv2d(3, 32, 3, padding=1) → BatchNorm2d(32) → ReLU
    → Conv2d(32, 64, 3, padding=1) → BatchNorm2d(64) → ReLU → MaxPool2d(2)
    → Conv2d(64, 128, 3, padding=1) → BatchNorm2d(128) → ReLU → MaxPool2d(2)
    → Flatten → Linear(8192, 256) → ReLU → Dropout(0.5) → Linear(256, 10)

    Input shape : (batch, 3, 32, 32)
    Output shape: (batch, 10) — logits for 10 CIFAR-10 classes
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, 3, 32, 32)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, 10)``.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_model(dataset: str) -> nn.Module:
    """Return the appropriate CNN architecture for the given dataset.

    Parameters
    ----------
    dataset : str
        Dataset name — ``"mnist"`` or ``"cifar10"``.

    Returns
    -------
    nn.Module
        An uninitialised (random weights) CNN model.

    Raises
    ------
    ValueError
        If the dataset name is not recognised.
    """
    dataset = dataset.lower()
    if dataset == "mnist":
        return MNISTNet()
    elif dataset == "cifar10":
        return CIFAR10Net()
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Supported: 'mnist', 'cifar10'"
        )
