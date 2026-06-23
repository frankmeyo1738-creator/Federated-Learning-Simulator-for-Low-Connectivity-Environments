"""
Unit Tests — Configuration Layer
==================================
Tests for ``ConfigLoader.load()`` and ``ConfigLoader.validate()``
in ``src/config.py``.

Validates that YAML configs are correctly parsed into typed dataclasses,
and that invalid values are caught with descriptive errors.

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""

import pytest
import yaml

from src.config import ConfigLoader, ExperimentConfig, NetworkProfile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_config_dict():
    """Return a minimal valid experiment configuration dictionary."""
    return {
        "experiment_name": "test_experiment",
        "num_clients": 10,
        "num_rounds": 20,
        "fraction_fit": 0.5,
        "algorithm": "fedavg",
        "mu": 0.0,
        "dataset": "mnist",
        "local_epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.01,
        "seed": 42,
        "output_dir": "experiments/results",
        "network_profile": {
            "name": "baseline",
            "latency_min_ms": 10.0,
            "latency_max_ms": 50.0,
            "bandwidth_mbps": 5.0,
            "packet_loss_rate": 0.0,
            "dropout_probability": 0.0,
            "dropout_duration_rounds": 0,
        },
    }


@pytest.fixture
def loader():
    """Return a fresh ConfigLoader instance."""
    return ConfigLoader()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConfigLoader:
    """Tests for ConfigLoader.load() and ConfigLoader.validate()."""

    def test_valid_config_loads_successfully(
        self, tmp_path, valid_config_dict, loader
    ):
        """Verify that a well-formed YAML config file is loaded and parsed
        into a valid ExperimentConfig object without raising any errors.

        This is the happy-path smoke test — if this fails, nothing else
        will work.
        """
        config_file = tmp_path / "valid_config.yaml"
        config_file.write_text(yaml.dump(valid_config_dict))

        config = loader.load(str(config_file))

        assert isinstance(config, ExperimentConfig)
        assert config.experiment_name == "test_experiment"
        assert config.num_clients == 10
        assert config.num_rounds == 20
        assert config.algorithm == "fedavg"
        assert config.dataset == "mnist"

    def test_invalid_algorithm_raises_error(
        self, tmp_path, valid_config_dict, loader
    ):
        """Verify that an unrecognised algorithm value (anything other than
        'fedavg' or 'fedprox') causes validate() to raise ValueError.

        Prevents silent misconfiguration if a user mistyped the algorithm.
        """
        valid_config_dict["algorithm"] = "invalid"
        config_file = tmp_path / "bad_algo.yaml"
        config_file.write_text(yaml.dump(valid_config_dict))

        with pytest.raises(ValueError, match="algorithm must be one of"):
            loader.load(str(config_file))

    def test_invalid_num_clients_raises_error(
        self, tmp_path, valid_config_dict, loader
    ):
        """Verify that num_clients <= 0 is rejected during validation.

        A federated learning simulation with zero (or negative) clients
        is meaningless, so the loader must catch this early.
        """
        valid_config_dict["num_clients"] = 0
        config_file = tmp_path / "bad_clients.yaml"
        config_file.write_text(yaml.dump(valid_config_dict))

        with pytest.raises(ValueError, match="num_clients must be > 0"):
            loader.load(str(config_file))

    def test_invalid_fraction_fit_raises_error(
        self, tmp_path, valid_config_dict, loader
    ):
        """Verify that fraction_fit > 1.0 is rejected during validation.

        Selecting more than 100% of clients per round is not valid.
        """
        valid_config_dict["fraction_fit"] = 1.5
        config_file = tmp_path / "bad_fraction.yaml"
        config_file.write_text(yaml.dump(valid_config_dict))

        with pytest.raises(ValueError, match="fraction_fit must be in"):
            loader.load(str(config_file))

    def test_network_profile_loaded_correctly(
        self, tmp_path, valid_config_dict, loader
    ):
        """Verify that all NetworkProfile fields are correctly mapped from
        the YAML file to the dataclass.

        This catches any key-name mismatches between the YAML schema and
        the NetworkProfile dataclass constructor.
        """
        config_file = tmp_path / "net_profile.yaml"
        config_file.write_text(yaml.dump(valid_config_dict))

        config = loader.load(str(config_file))
        np = config.network_profile

        assert isinstance(np, NetworkProfile)
        assert np.name == "baseline"
        assert np.latency_min_ms == 10.0
        assert np.latency_max_ms == 50.0
        assert np.bandwidth_mbps == 5.0
        assert np.packet_loss_rate == 0.0
        assert np.dropout_probability == 0.0
        assert np.dropout_duration_rounds == 0
