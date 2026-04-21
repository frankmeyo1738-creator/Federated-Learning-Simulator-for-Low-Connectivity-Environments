"""
Main Entry Point
==================
CLI entry point for running FL experiments.

Usage::

    python -m src.main --config config/experiments/my_experiment.yaml

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""

import argparse
import logging
import sys

from src.config import ConfigLoader


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the simulator."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    """Parse CLI arguments, load config, and launch the experiment."""
    parser = argparse.ArgumentParser(
        description="Federated Learning Network Simulator for Low-Connectivity Environments",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML experiment configuration file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Layer 1: Load and validate config
    logger.info("Loading configuration from: %s", args.config)
    loader = ConfigLoader()
    config = loader.load(args.config)
    logger.info("Configuration loaded: %s", config.experiment_name)
    logger.info("Algorithm: %s | Dataset: %s | Clients: %d | Rounds: %d",
                config.algorithm, config.dataset, config.num_clients, config.num_rounds)
    logger.info("Network profile: %s", config.network_profile.name)

    # TODO: Layer 2 — Initialise model, create data partitions, build clients, run server
    # This will be connected once the data loading and model modules are implemented.
    logger.info("⚠️  Simulation engine not yet connected. Config validation passed.")
    logger.info("Next step: Implement data loading and model definitions.")


if __name__ == "__main__":
    main()
