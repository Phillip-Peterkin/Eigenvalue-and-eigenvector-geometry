"""Ensure anti-leakage artifacts stay committed and internally consistent."""
from __future__ import annotations

import json
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
    assert preregistration["canonical_cohort_manifest"] == "cohorts/cogitate_primary.json"
    assert preregistration["canonical_execution_mode"] == "strict_reproduction"
    assert len(preregistration["primary_predictions"]) >= 5


def test_preregistration_frozen_config_contract_matches_files(preregistration: dict) -> None:
    config_path = REPO_ROOT / preregistration["canonical_config"]
    cohort_path = REPO_ROOT / preregistration["canonical_cohort_manifest"]
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with cohort_path.open(encoding="utf-8") as handle:
        cohort = json.load(handle)

    frozen = preregistration["frozen_parameters"]
    assert config["random_seed"] == frozen["random_seed"]
    assert config["preprocessing"]["high_gamma_passband"] == frozen["high_gamma_passband_hz"]
    assert config["preprocessing"]["broadband_passband"] == frozen["broadband_passband_hz"]
    assert config["statistics"]["n_perm"] == frozen["n_permutations_default"]
    assert config["statistics"]["alpha"] == frozen["alpha"]
    assert config["statistics"]["correction"] == frozen["multiple_comparison_correction"]
    assert config["data"]["subjects"] == cohort["subjects"]
    assert config["data"]["runs"] == cohort["expected_runs"]
    assert config["data"]["strict_reproduction"] is True


def test_primary_prediction_artifacts_exist(preregistration: dict) -> None:
    for prediction in preregistration["primary_predictions"]:
        artifact = REPO_ROOT / prediction["adjudication_artifact"]
        assert artifact.is_file(), f"Missing locked artifact for {prediction['id']}: {artifact}"


def test_zurich_dataset_marked_exploratory(preregistration: dict) -> None:
    zurich = preregistration["datasets"]["zurich_seeg"]
    assert zurich["label"] == "exploratory"
