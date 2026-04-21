"""
Layer 1: Configuration Layer
=============================
Reads and validates YAML experiment files. Exposes a clean config object
to all other layers. Single source of truth for all experiment parameters.

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NetworkProfile:
    """Describes the network conditions for a simulation run.

    All values are sourced from GSMA Mobile Economy SSA 2023 report
    and represent realistic Sub-Saharan African connectivity conditions.
    """
    name: str
    latency_min_ms: float
    latency_max_ms: float
    bandwidth_mbps: float
    packet_loss_rate: float       # 0.0 – 1.0
    dropout_probability: float    # per-round probability of a client going offline
    dropout_duration_rounds: int  # how many rounds the client stays offline


@dataclass
class ExperimentConfig:
    """Top-level configuration object consumed by every other layer."""
    experiment_name: str
    num_clients: int
    num_rounds: int
    fraction_fit: float           # fraction of clients selected per round
    algorithm: str                # "fedavg" or "fedprox"
    mu: float                    # FedProx proximal term (ignored for FedAvg)
    dataset: str                 # "mnist" or "cifar10"
    network_profile: NetworkProfile
    output_dir: str
    # Optional fields with sensible defaults
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

class ConfigLoader:
    """Loads and validates YAML experiment configuration files.

    Usage::

        loader = ConfigLoader()
        config = loader.load("config/experiments/my_experiment.yaml")
    """

    # Allowed values for constrained fields
    VALID_ALGORITHMS = {"fedavg", "fedprox"}
    VALID_DATASETS = {"mnist", "cifar10"}

    def load(self, path: str) -> ExperimentConfig:
        """Read a YAML file and return a validated ``ExperimentConfig``.

        Parameters
        ----------
        path : str
            Path to the YAML experiment configuration file.

        Returns
        -------
        ExperimentConfig
            Fully validated configuration object.

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        ValueError
            If any configuration value fails validation.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, "r") as fh:
            raw = yaml.safe_load(fh)

        config = self._parse(raw)
        self.validate(config)
        return config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse(self, raw: dict) -> ExperimentConfig:
        """Convert raw YAML dict into typed dataclass instances."""
        net_raw = raw.get("network_profile", {})
        network_profile = NetworkProfile(
            name=net_raw.get("name", "unnamed"),
            latency_min_ms=float(net_raw.get("latency_min_ms", 0)),
            latency_max_ms=float(net_raw.get("latency_max_ms", 0)),
            bandwidth_mbps=float(net_raw.get("bandwidth_mbps", 0)),
            packet_loss_rate=float(net_raw.get("packet_loss_rate", 0)),
            dropout_probability=float(net_raw.get("dropout_probability", 0)),
            dropout_duration_rounds=int(net_raw.get("dropout_duration_rounds", 0)),
        )

        return ExperimentConfig(
            experiment_name=raw.get("experiment_name", "unnamed_experiment"),
            num_clients=int(raw.get("num_clients", 10)),
            num_rounds=int(raw.get("num_rounds", 10)),
            fraction_fit=float(raw.get("fraction_fit", 1.0)),
            algorithm=str(raw.get("algorithm", "fedavg")).lower(),
            mu=float(raw.get("mu", 0.0)),
            dataset=str(raw.get("dataset", "mnist")).lower(),
            network_profile=network_profile,
            output_dir=raw.get("output_dir", "experiments/results"),
            local_epochs=int(raw.get("local_epochs", 1)),
            batch_size=int(raw.get("batch_size", 32)),
            learning_rate=float(raw.get("learning_rate", 0.01)),
            seed=raw.get("seed"),
        )

    def validate(self, config: ExperimentConfig) -> None:
        """Validate all config values. Raises ``ValueError`` on failure.

        Checks performed:
        - num_clients > 0
        - num_rounds > 0
        - 0 < fraction_fit <= 1
        - algorithm in {fedavg, fedprox}
        - dataset in {mnist, cifar10}
        - Network profile ranges are sane
        - mu >= 0 when using fedprox
        """
        errors: list[str] = []

        # --- Simulation parameters ---
        if config.num_clients <= 0:
            errors.append("num_clients must be > 0")
        if config.num_rounds <= 0:
            errors.append("num_rounds must be > 0")
        if not (0.0 < config.fraction_fit <= 1.0):
            errors.append("fraction_fit must be in (0, 1]")
        if config.algorithm not in self.VALID_ALGORITHMS:
            errors.append(
                f"algorithm must be one of {self.VALID_ALGORITHMS}, "
                f"got '{config.algorithm}'"
            )
        if config.dataset not in self.VALID_DATASETS:
            errors.append(
                f"dataset must be one of {self.VALID_DATASETS}, "
                f"got '{config.dataset}'"
            )
        if config.algorithm == "fedprox" and config.mu < 0:
            errors.append("mu must be >= 0 for FedProx")

        # --- Network profile validation ---
        np = config.network_profile
        if np.latency_min_ms < 0:
            errors.append("latency_min_ms must be >= 0")
        if np.latency_max_ms < np.latency_min_ms:
            errors.append("latency_max_ms must be >= latency_min_ms")
        if np.bandwidth_mbps <= 0:
            errors.append("bandwidth_mbps must be > 0")
        if not (0.0 <= np.packet_loss_rate <= 1.0):
            errors.append("packet_loss_rate must be in [0, 1]")
        if not (0.0 <= np.dropout_probability <= 1.0):
            errors.append("dropout_probability must be in [0, 1]")
        if np.dropout_duration_rounds < 0:
            errors.append("dropout_duration_rounds must be >= 0")

        if errors:
            raise ValueError(
                "Configuration validation failed:\n  • "
                + "\n  • ".join(errors)
            )
