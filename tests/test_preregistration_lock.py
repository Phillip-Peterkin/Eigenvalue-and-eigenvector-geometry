"""Ensure anti-leakage artifacts stay committed and internally consistent."""
from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_agents_md_exists_and_states_leakage_rules() -> None:
    text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "Leakage prevention" in text
    assert "No silent averaging" in text
    assert "subject" in text.lower()


def test_preregistration_spec_exists(preregistration: dict) -> None:
    assert preregistration["citation_key"] == "Peterkin2026"
    assert preregistration["canonical_config"] == "code/config.yaml"
    assert len(preregistration["primary_predictions"]) >= 5


def test_preregistration_frozen_seed_matches_config(preregistration: dict) -> None:
    config_path = REPO_ROOT / "code" / "config.yaml"
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert config["random_seed"] == preregistration["frozen_parameters"]["random_seed"]


def test_primary_prediction_artifacts_exist(preregistration: dict) -> None:
    for prediction in preregistration["primary_predictions"]:
        artifact = REPO_ROOT / prediction["adjudication_artifact"]
        assert artifact.is_file(), f"Missing locked artifact for {prediction['id']}: {artifact}"


def test_zurich_dataset_marked_exploratory(preregistration: dict) -> None:
    zurich = preregistration["datasets"]["zurich_seeg"]
    assert zurich["label"] == "exploratory"
