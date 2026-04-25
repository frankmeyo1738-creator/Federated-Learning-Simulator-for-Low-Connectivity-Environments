# Component Specification
## Federated Learning Network Simulator for Low-Connectivity Environments

**Project:** CSC 4004 Final Year Project — UNZA CS 2026  
**Supervisor:** Mr. Mofya Phiri  
**Author:** Frank Meyoyo Chisanga (2022067479)  
**Version:** 1.0 — Week 1 Setup  
**Last Updated:** April 2026

---

## Overview

The simulator is built around a clean **4-layer architecture**. Each layer has a single, well-defined responsibility and communicates with adjacent layers through explicit interfaces. No layer reaches across to a non-adjacent layer.

```
┌─────────────────────────────────────────────────────┐
│           Layer 1: Configuration Layer              │
│         (YAML loading, validation, config)          │
└─────────────────────┬───────────────────────────────┘
                      │ ExperimentConfig
┌─────────────────────▼───────────────────────────────┐
│           Layer 2: Simulation Engine                │
│         (FL training loop, FedAvg / FedProx)        │
└──────┬──────────────────────────────────┬───────────┘
       │ ModelUpdate                      │ RoundMetrics
┌──────▼──────────────┐      ┌────────────▼────────────┐
│  Layer 3: Network   │      │  Layer 4: Analytics     │
│  Impairment Engine  │      │       Layer             │
│ (latency, dropout,  │      │  (CSV, plots, summary)  │
│  packet loss)       │      │                         │
└─────────────────────┘      └─────────────────────────┘
```

---

## Layer 1 — Configuration Layer

### Responsibility
Read and validate YAML experiment files. Expose a clean config object to all other layers. This is the **single source of truth** for all experiment parameters — no other layer touches YAML directly.

### Inputs
- YAML file path (provided as CLI argument)

### Outputs
- Validated `ExperimentConfig` dataclass object

### Key Classes

```python
from dataclasses import dataclass

@dataclass
class NetworkProfile:
    name: str
    latency_min_ms: float
    latency_max_ms: float
    bandwidth_mbps: float
    packet_loss_rate: float      # 0.0 – 1.0
    dropout_probability: float   # per round, 0.0 – 1.0
    dropout_duration_rounds: int

@dataclass
class ExperimentConfig:
    experiment_name: str
    num_clients: int
    num_rounds: int
    fraction_fit: float          # fraction of clients selected per round
    algorithm: str               # "fedavg" or "fedprox"
    mu: float                    # FedProx proximal term (ignored for FedAvg)
    dataset: str                 # "mnist" or "cifar10"
    network_profile: NetworkProfile
    output_dir: str

class ConfigLoader:
    def load(self, path: str) -> ExperimentConfig:
        """Parse YAML file and return validated ExperimentConfig."""
        ...

    def validate(self, config: ExperimentConfig) -> None:
        """Raise ValueError if any config field is invalid."""
        ...
```

### Interface with Adjacent Layers
- **Simulation Engine** imports `ExperimentConfig` at startup
- No other layer imports from or calls into this layer after startup

### Validation Rules
| Field | Rule |
|---|---|
| `num_clients` | Must be >= 2 |
| `num_rounds` | Must be >= 1 |
| `fraction_fit` | Must be in range (0.0, 1.0] |
| `algorithm` | Must be `"fedavg"` or `"fedprox"` |
| `mu` | Must be >= 0.0 |
| `dataset` | Must be `"mnist"` or `"cifar10"` |
| `packet_loss_rate` | Must be in range [0.0, 1.0] |
| `dropout_probability` | Must be in range [0.0, 1.0] |

---

## Layer 2 — Simulation Engine

### Responsibility
Orchestrate the FL training loop. Manage global model state. Select clients per round. Coordinate aggregation. **This layer knows nothing about how the network works** — that is exclusively the Network Impairment Layer's job.

### Inputs
- `ExperimentConfig` from Configuration Layer
- `ModelUpdate` objects from FL clients (after passing through Network Impairment Layer)

### Outputs
- `RoundMetrics` objects sent to Analytics Layer after each round
- Final trained global model

### Key Classes

```python
import torch.nn as nn
from typing import List, Optional, Dict

class FLServer:
    def __init__(self, config: ExperimentConfig, model: nn.Module):
        """Initialise server with experiment config and global model."""
        ...

    def run(self) -> List[RoundMetrics]:
        """Execute full FL training loop. Returns list of per-round metrics."""
        ...

    def select_clients(self, round_num: int) -> List[int]:
        """Randomly select fraction_fit * num_clients client IDs for this round."""
        ...

    def aggregate(self, updates: List[ModelUpdate]) -> nn.Module:
        """Run FedAvg or FedProx aggregation on received updates."""
        ...


class FLClient:
    def __init__(self, client_id: int, dataset_partition, config: ExperimentConfig):
        """Initialise client with its local data partition."""
        ...

    async def train(self, global_model: nn.Module) -> ModelUpdate:
        """Train on local data, return model update."""
        ...

    def get_data_size(self) -> int:
        """Return number of local training samples."""
        ...


@dataclass
class ModelUpdate:
    client_id: int
    weights: Dict[str, torch.Tensor]   # state_dict of trained local model
    num_samples: int                    # used for weighted averaging
    loss: float
    round_num: int
```

### FedAvg Aggregation Formula
The global model at round `t` is computed as:

```
w_global = Σ (n_i / N_total) * w_i
```

Where `n_i` is the number of samples for client `i` and `N_total` is the total samples across all participating clients.

### FedProx Modification
FedProx adds a proximal term to each client's local loss function:

```
L_fedprox(w) = L(w) + (μ/2) * ||w - w_global||²
```

The `mu` parameter controls how far local updates can deviate from the global model. When `mu = 0`, FedProx reduces to FedAvg.

### Interface with Adjacent Layers
- Calls **Network Impairment Layer** when transmitting model updates: `impairment_engine.transmit(update)`
- Sends `RoundMetrics` to **Analytics Layer** after each round: `tracker.record_round(metrics)`
- Receives `ExperimentConfig` from **Configuration Layer** at startup

---

## Layer 3 — Network Impairment Engine

### Responsibility
Intercept all model update transmissions and apply configured impairment conditions. **This is the core research contribution of the project.** It simulates the effect of Sub-Saharan African network conditions on FL convergence.

### Inputs
- `ModelUpdate` object (from Simulation Engine)
- `NetworkProfile` config (from Configuration Layer via constructor)

### Outputs
- `ModelUpdate` (possibly delayed) — if the transmission succeeds
- `None` — if the update is dropped (packet loss or client dropout)

### Key Classes

```python
import asyncio
import random
from enum import Enum
from typing import Optional

class NetworkCondition(Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"


class NetworkImpairmentEngine:
    def __init__(self, profile: NetworkProfile):
        self.profile = profile
        self._dropout_tracker: Dict[int, int] = {}  # client_id -> rounds_remaining

    async def transmit(self, update: ModelUpdate) -> Optional[ModelUpdate]:
        """
        Simulate network transmission of a model update.
        Returns the update after delay, or None if dropped.
        """
        # 1. Check client dropout state
        if not self._check_dropout(update.client_id, update.round_num):
            return None

        # 2. Apply packet loss
        if not self._apply_packet_loss():
            return None

        # 3. Apply bandwidth throttling (adds delay)
        bandwidth_delay = self._apply_bandwidth_throttle(update)

        # 4. Apply latency (Gaussian distribution)
        await self._apply_latency()

        # 5. Apply bandwidth delay
        await asyncio.sleep(bandwidth_delay)

        return update

    async def _apply_latency(self) -> None:
        """Simulate network latency using Gaussian distribution."""
        mu = (self.profile.latency_min_ms + self.profile.latency_max_ms) / 2
        sigma = (self.profile.latency_max_ms - self.profile.latency_min_ms) * 0.15
        latency_ms = max(self.profile.latency_min_ms,
                         random.gauss(mu, sigma))
        await asyncio.sleep(latency_ms / 1000)

    def _apply_packet_loss(self) -> bool:
        """Returns False if update should be dropped due to packet loss."""
        return random.random() > self.profile.packet_loss_rate

    def _apply_bandwidth_throttle(self, update: ModelUpdate) -> float:
        """
        Calculate transmission delay based on model size and bandwidth.
        Returns delay in seconds.
        """
        # Estimate model size in bytes (sum of all parameter tensors)
        total_bytes = sum(
            t.nelement() * t.element_size()
            for t in update.weights.values()
        )
        bandwidth_bytes_per_sec = self.profile.bandwidth_mbps * 1e6 / 8
        return total_bytes / bandwidth_bytes_per_sec

    def _check_dropout(self, client_id: int, round_num: int) -> bool:
        """
        Returns False if client is in a dropout period.
        Manages dropout state across rounds.
        """
        # If client is already in dropout, decrement counter
        if client_id in self._dropout_tracker:
            self._dropout_tracker[client_id] -= 1
            if self._dropout_tracker[client_id] <= 0:
                del self._dropout_tracker[client_id]
            return False

        # Randomly trigger new dropout
        if random.random() < self.profile.dropout_probability:
            self._dropout_tracker[client_id] = self.profile.dropout_duration_rounds
            return False

        return True
```

### Latency Distribution Rationale
Latency uses a **Gaussian distribution** rather than uniform random. The midpoint of `[latency_min_ms, latency_max_ms]` is the mean (μ), and 15% of the range is the standard deviation (σ). This produces realistic variance — most transmissions cluster near the typical latency, with occasional outliers.

### Interface with Adjacent Layers
- Called **exclusively** by `FLServer` during each round's aggregation phase
- Returns `None` for dropped updates — server handles missing updates gracefully by proceeding with available updates only
- Never calls back into any other layer

---

## Layer 4 — Analytics Layer

### Responsibility
Record all metrics per round. Generate visualisations. Export CSV. **This layer knows nothing about FL** — it just records what it is given and produces outputs.

### Inputs
- `RoundMetrics` objects from the Simulation Engine

### Outputs
- CSV file of per-round metrics
- matplotlib graphs (accuracy curve, communication cost, client participation)
- Console summary at end of experiment

### Key Classes

```python
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class RoundMetrics:
    round_num: int
    global_accuracy: float
    global_loss: float
    participating_clients: int       # clients that successfully submitted updates
    dropped_updates: int             # updates lost to packet loss or dropout
    avg_latency_ms: float
    total_bytes_transmitted: int
    round_duration_seconds: float


class AnalyticsTracker:
    def __init__(self, output_dir: str, experiment_name: str):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.history: List[RoundMetrics] = []

    def record_round(self, metrics: RoundMetrics) -> None:
        """Append round metrics to history. Called after every round."""
        self.history.append(metrics)

    def export_csv(self) -> None:
        """Write all round metrics to a CSV file in output_dir."""
        ...

    def plot_accuracy_curve(self) -> None:
        """
        Line chart: global_accuracy vs round_num.
        Useful for comparing FedAvg vs FedProx, and urban vs rural profiles.
        """
        ...

    def plot_communication_cost(self) -> None:
        """
        Line chart: cumulative total_bytes_transmitted vs round_num.
        Shows communication overhead under different network profiles.
        """
        ...

    def plot_client_participation(self) -> None:
        """
        Bar chart: participating_clients and dropped_updates per round.
        Shows the effect of dropout_probability on training stability.
        """
        ...

    def generate_summary(self) -> Dict:
        """
        Return dict with key experiment statistics:
        - final_accuracy
        - rounds_to_convergence (first round >= 90% of final accuracy)
        - total_bytes_transmitted
        - total_dropped_updates
        - avg_participation_rate
        """
        ...
```

### Output Files Structure
For each experiment run, outputs are saved to:
```
experiments/results/{experiment_name}/
├── metrics.csv
├── accuracy_curve.png
├── communication_cost.png
├── client_participation.png
└── summary.json
```

### Interface with Adjacent Layers
- Receives `RoundMetrics` from **Simulation Engine** only
- Never calls into any other layer
- Has no knowledge of FL algorithms, network profiles, or model architecture

---

## Data Flow Summary

```
CLI args
   │
   ▼
ConfigLoader.load() ──────────────────────────────┐
   │                                               │
   │ ExperimentConfig                              │ NetworkProfile
   ▼                                               ▼
FLServer.run()                         NetworkImpairmentEngine
   │                                               ▲
   │ for each round:                               │
   │   1. select_clients()                         │
   │   2. clients train locally                    │
   │   3. transmit updates ─────────────────────── ┘
   │      (returns ModelUpdate or None)
   │   4. aggregate surviving updates
   │   5. record_round(RoundMetrics) ──────────────┐
   │                                               │
   ▼                                               ▼
Global model saved                      AnalyticsTracker
                                               │
                                    export_csv(), plots, summary
```

---

## Experiment Configuration Schema

Full YAML schema for an experiment file:

```yaml
experiment:
  name: "fedavg_urban_mnist"
  num_clients: 10
  num_rounds: 50
  fraction_fit: 0.5
  algorithm: "fedavg"      # "fedavg" or "fedprox"
  mu: 0.01                 # only used if algorithm = "fedprox"
  dataset: "mnist"         # "mnist" or "cifar10"
  output_dir: "experiments/results/"

network_profile: "config/profiles/urban_zambia.yaml"
```

---

## Planned Experiments

| Experiment ID | Algorithm | Network Profile | Dataset | Purpose |
|---|---|---|---|---|
| EXP-01 | FedAvg | baseline | MNIST | Control — ideal conditions |
| EXP-02 | FedAvg | urban_zambia | MNIST | Urban network impact |
| EXP-03 | FedAvg | rural_zambia | MNIST | Rural network impact |
| EXP-04 | FedAvg | severe_disruption | MNIST | Extreme conditions impact |
| EXP-05 | FedProx | urban_zambia | MNIST | FedProx vs FedAvg (urban) |
| EXP-06 | FedProx | rural_zambia | MNIST | FedProx vs FedAvg (rural) |
| EXP-07 | FedProx | severe_disruption | MNIST | FedProx vs FedAvg (extreme) |
| EXP-08 | FedAvg | rural_zambia | CIFAR-10 | Complexity increase |
| EXP-09 | FedProx | rural_zambia | CIFAR-10 | FedProx on harder task |

**Key research questions these experiments answer:**
1. How much does model accuracy degrade as network conditions worsen (EXP-01 → 04)?
2. Does FedProx outperform FedAvg under low-connectivity conditions (EXP-02 vs 05, EXP-03 vs 06)?
3. Does the advantage of FedProx increase as conditions worsen (EXP-05 → 07)?

---

## Technology Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.11+ |
| ML Framework | PyTorch | >= 2.0.0 |
| Async I/O | asyncio | stdlib |
| Config Parsing | PyYAML | >= 6.0 |
| Data Analysis | pandas | >= 2.0.0 |
| Visualisation | matplotlib + seaborn | >= 3.7.0 |
| Testing | pytest + pytest-asyncio | >= 7.3.0 |
| GPU Backend (M2 Air) | MPS (Metal Performance Shaders) | via PyTorch |
| Free GPU (training) | Google Colab | T4 / A100 |

---

## Development Milestones

| Milestone | Target Date | Deliverable |
|---|---|---|
| Week 1–4 | April 2026 | Papers read, repo set up, spec written |
| Week 5–8 | May 2026 | Core simulator — Layers 1, 2, 4 working |
| Week 9–10 | June 2026 | Network Impairment Engine (Layer 3) |
| **Prototype** | **9 June 2026** | **End-to-end simulation running** |
| Week 11 | June 2026 | All 9 experiments run and logged |
| Week 12–13 | July 2026 | Analysis, graphs, write-up |
| **Final Submission** | **29 September 2026** | Complete project |
