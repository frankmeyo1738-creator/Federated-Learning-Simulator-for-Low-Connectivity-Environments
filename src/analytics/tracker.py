"""
Layer 4 — Analytics Layer
==========================
Records all per-round metrics, generates visualisations, exports CSV,
and prints console summaries. Knows nothing about FL — just records
what it's given.

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""

import csv
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics data class
# ---------------------------------------------------------------------------

@dataclass
class RoundMetrics:
    """Holds all measurable quantities for a single FL round."""
    round_num: int
    global_accuracy: float
    global_loss: float
    participating_clients: int
    dropped_updates: int
    avg_latency_ms: float
    total_bytes_transmitted: int
    round_duration_seconds: float


# ---------------------------------------------------------------------------
# Analytics tracker
# ---------------------------------------------------------------------------

class AnalyticsTracker:
    """Records round metrics, exports CSV, and generates matplotlib plots.

    Parameters
    ----------
    output_dir : str
        Directory where CSV and plots will be saved.
    experiment_name : str
        Name used as prefix for all output files.
    """

    def __init__(self, output_dir: str, experiment_name: str) -> None:
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.history: List[RoundMetrics] = []

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Seaborn styling for publication-quality plots
        sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

    def record_round(self, metrics: RoundMetrics) -> None:
        """Append a round's metrics to the history.

        Parameters
        ----------
        metrics : RoundMetrics
            Metrics for the completed round.
        """
        self.history.append(metrics)
        logger.info(
            "Round %d recorded — acc=%.4f, loss=%.4f, clients=%d, dropped=%d",
            metrics.round_num,
            metrics.global_accuracy,
            metrics.global_loss,
            metrics.participating_clients,
            metrics.dropped_updates,
        )

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def export_csv(self) -> str:
        """Export all recorded metrics to a CSV file.

        Returns
        -------
        str
            Path to the generated CSV file.
        """
        filepath = self.output_dir / f"{self.experiment_name}_metrics.csv"
        fieldnames = [
            "round_num",
            "global_accuracy",
            "global_loss",
            "participating_clients",
            "dropped_updates",
            "avg_latency_ms",
            "total_bytes_transmitted",
            "round_duration_seconds",
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in self.history:
                writer.writerow(asdict(m))

        logger.info("CSV exported to %s", filepath)
        return str(filepath)

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------

    def plot_accuracy_curve(self) -> str:
        """Plot global accuracy and loss over rounds.

        Returns
        -------
        str
            Path to the saved figure.
        """
        rounds = [m.round_num for m in self.history]
        accuracies = [m.global_accuracy for m in self.history]
        losses = [m.global_loss for m in self.history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy curve
        ax1.plot(rounds, accuracies, marker="o", linewidth=2, color="#2196F3")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Global Accuracy")
        ax1.set_title(f"{self.experiment_name} — Accuracy per Round")
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)

        # Loss curve
        ax2.plot(rounds, losses, marker="s", linewidth=2, color="#F44336")
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Global Loss")
        ax2.set_title(f"{self.experiment_name} — Loss per Round")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        filepath = self.output_dir / f"{self.experiment_name}_accuracy_loss.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Accuracy/Loss plot saved to %s", filepath)
        return str(filepath)

    def plot_communication_cost(self) -> str:
        """Plot communication cost (bytes transmitted and latency) over rounds.

        Returns
        -------
        str
            Path to the saved figure.
        """
        rounds = [m.round_num for m in self.history]
        bytes_tx = [m.total_bytes_transmitted / (1024 * 1024) for m in self.history]  # MB
        latencies = [m.avg_latency_ms for m in self.history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Bytes transmitted
        ax1.bar(rounds, bytes_tx, color="#4CAF50", alpha=0.8)
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Data Transmitted (MB)")
        ax1.set_title(f"{self.experiment_name} — Communication Cost per Round")
        ax1.grid(True, alpha=0.3, axis="y")

        # Average latency
        ax2.plot(rounds, latencies, marker="^", linewidth=2, color="#FF9800")
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Avg Latency (ms)")
        ax2.set_title(f"{self.experiment_name} — Avg Latency per Round")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        filepath = self.output_dir / f"{self.experiment_name}_communication.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Communication cost plot saved to %s", filepath)
        return str(filepath)

    def plot_client_participation(self) -> str:
        """Plot client participation and dropout rates over rounds.

        Returns
        -------
        str
            Path to the saved figure.
        """
        rounds = [m.round_num for m in self.history]
        participating = [m.participating_clients for m in self.history]
        dropped = [m.dropped_updates for m in self.history]

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(rounds, participating, label="Participated", color="#2196F3", alpha=0.8)
        ax.bar(
            rounds, dropped,
            bottom=participating,
            label="Dropped",
            color="#F44336",
            alpha=0.8,
        )
        ax.set_xlabel("Round")
        ax.set_ylabel("Number of Clients")
        ax.set_title(f"{self.experiment_name} — Client Participation per Round")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        filepath = self.output_dir / f"{self.experiment_name}_participation.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Client participation plot saved to %s", filepath)
        return str(filepath)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def generate_summary(self) -> Dict:
        """Generate a summary dictionary of the experiment.

        Returns
        -------
        Dict
            Key aggregate metrics across all rounds.
        """
        if not self.history:
            return {"status": "no data recorded"}

        total_rounds = len(self.history)
        final = self.history[-1]

        total_dropped = sum(m.dropped_updates for m in self.history)
        total_participated = sum(m.participating_clients for m in self.history)
        total_bytes = sum(m.total_bytes_transmitted for m in self.history)
        avg_latency = (
            sum(m.avg_latency_ms for m in self.history) / total_rounds
        )
        total_time = sum(m.round_duration_seconds for m in self.history)

        best_accuracy = max(m.global_accuracy for m in self.history)
        best_round = next(
            m.round_num for m in self.history
            if m.global_accuracy == best_accuracy
        )

        summary = {
            "experiment_name": self.experiment_name,
            "total_rounds": total_rounds,
            "final_accuracy": final.global_accuracy,
            "final_loss": final.global_loss,
            "best_accuracy": best_accuracy,
            "best_accuracy_round": best_round,
            "total_participated": total_participated,
            "total_dropped": total_dropped,
            "drop_rate": total_dropped / max(total_participated + total_dropped, 1),
            "total_bytes_transmitted": total_bytes,
            "total_bytes_transmitted_mb": total_bytes / (1024 * 1024),
            "avg_latency_ms": avg_latency,
            "total_experiment_time_seconds": total_time,
        }

        # Print to console
        print("\n" + "=" * 60)
        print(f"  EXPERIMENT SUMMARY: {self.experiment_name}")
        print("=" * 60)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key:>35s}: {value:.4f}")
            else:
                print(f"  {key:>35s}: {value}")
        print("=" * 60 + "\n")

        return summary
