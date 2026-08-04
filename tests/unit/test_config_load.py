"""Config loader must work without local private datasets present."""
from __future__ import annotations

from pathlib import Path

from cmcc.config import load_config

REPO_CONFIG = Path(__file__).resolve().parents[2] / "code" / "config.yaml"


def test_defaults_load_without_data_on_disk() -> None:
    config = load_config()
    assert config["random_seed"] == 42
    assert config["preprocessing"]["high_gamma_passband"] == [70, 150]


def test_repo_config_yaml_loads() -> None:
    assert REPO_CONFIG.is_file()
    config = load_config(REPO_CONFIG)
    assert config["data"]["subjects"]
    assert config["avalanche"]["threshold_sd"] == 3.0
