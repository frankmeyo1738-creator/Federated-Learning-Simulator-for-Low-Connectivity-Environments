"""
Main Entry Point
==================
CLI entry point for running FL experiments. Wires together all four layers:

1. **Config Layer** — loads and validates the YAML experiment file.
2. **Model + Data** — instantiates the CNN architecture and partitions data.
3. **Core Layer** — creates FL clients and server, runs the training loop.
4. **Analytics Layer** — automatically invoked by the server on completion.

Usage::

    python -m src.main --config config/experiments/example_experiment.yaml
    python -m src.main --config config/experiments/example_experiment.yaml --iid

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""

import argparse
import logging
import sys

from src.config import ConfigLoader
from src.core.client import FLClient
from src.core.server import FLServer
from src.data.loader import create_client_partitions, download_dataset, get_test_loader
from src.models.cnn import get_model


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
    parser.add_argument(
        "--iid",
        action="store_true",
        default=False,
        help="Use IID data partitioning (default: non-IID Dirichlet α=0.5)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # ── Layer 1: Load and validate config ──────────────────────────
        logger.info("Loading configuration from: %s", args.config)
        loader = ConfigLoader()
        config = loader.load(args.config)
        logger.info("Configuration loaded: %s", config.experiment_name)
        logger.info(
            "Algorithm: %s | Dataset: %s | Clients: %d | Rounds: %d",
            config.algorithm, config.dataset, config.num_clients, config.num_rounds,
        )
        logger.info("Network profile: %s", config.network_profile.name)

        # ── Model architecture ─────────────────────────────────────────
        logger.info("Initialising model for dataset: %s", config.dataset)
        model = get_model(config.dataset)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info("Model created — %s parameters", f"{param_count:,}")

        # ── Data loading & partitioning ────────────────────────────────
        logger.info("Downloading dataset: %s", config.dataset)
        train_dataset = download_dataset(config.dataset, "./data")

        iid_mode = args.iid
        alpha = 0.5
        logger.info(
            "Partitioning data — iid=%s, alpha=%.2f, seed=%s",
            iid_mode, alpha, config.seed,
        )
        partitions = create_client_partitions(
            dataset=train_dataset,
            num_clients=config.num_clients,
            iid=iid_mode,
            alpha=alpha,
            seed=config.seed,
        )

        logger.info("Creating test loader for global evaluation")
        test_loader = get_test_loader(
            dataset=config.dataset,
            data_dir="./data",
            batch_size=config.batch_size,
        )

        # ── Create FL clients ──────────────────────────────────────────
        clients = []
        for i in range(config.num_clients):
            client = FLClient(
                client_id=i,
                dataset_partition=partitions[i],
                config=config,
            )
            clients.append(client)
        logger.info("Created %d FL clients", len(clients))

        # ── Create FL server and run ───────────────────────────────────
        server = FLServer(
            config=config,
            model=model,
            clients=clients,
            test_loader=test_loader,
        )

        logger.info("🚀 Starting federated learning experiment...")
        server.run()

        logger.info(
            "✅ Experiment complete. Results saved to %s", config.output_dir
        )
        print(f"\nExperiment complete. Results saved to {config.output_dir}")

    except KeyboardInterrupt:
        logger.info("⛔ Experiment interrupted by user. Exiting cleanly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
