"""Engineering contracts that prevent reproducibility and provenance regressions."""
from __future__ import annotations

import json
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

import cmcc
from cmcc.config import CANONICAL_COGITATE_SUBJECTS, DEFAULTS, load_config
from cmcc.features.operator_geometry import minimum_eigenvalue_gap
from cmcc.provenance import _get_package_versions

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_yaml_and_package_defaults_are_synchronized() -> None:
    with (REPO_ROOT / "code" / "config.yaml").open(encoding="utf-8") as handle:
        yaml_config = yaml.safe_load(handle)
    assert yaml_config == DEFAULTS
    assert load_config(REPO_ROOT / "code" / "config.yaml") == DEFAULTS


def test_primary_cohort_manifest_matches_canonical_config() -> None:
    path = REPO_ROOT / "cohorts" / "cogitate_primary.json"
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["selection_policy"] == "fixed_manifest"
    assert manifest["filesystem_discovery_allowed_for_canonical_reproduction"] is False
    assert manifest["expected_subject_count"] == len(CANONICAL_COGITATE_SUBJECTS) == 18
    assert manifest["subjects"] == CANONICAL_COGITATE_SUBJECTS
    assert DEFAULTS["data"]["subjects"] == manifest["subjects"]
    assert DEFAULTS["data"]["runs"] == manifest["expected_runs"]
    assert DEFAULTS["data"]["strict_reproduction"] is True


def test_canonical_broadband_runner_is_strict_by_default() -> None:
    source = (
        REPO_ROOT / "code" / "analysis_pipeline" / "scripts" / "run_all_subjects_broadband_canonical.py"
    ).read_text(encoding="utf-8")
    assert "_configured_subjects(config) if strict else _discover_subjects(data_root)" in source
    assert "Strict canonical reproduction incomplete" in source
    assert "require_all_runs=strict" in source
    assert "--best-effort" in source


def test_distribution_and_package_version_agree() -> None:
    assert cmcc.__version__ == metadata.version("operator-geometry")


def test_provenance_uses_distribution_names_correctly() -> None:
    versions = _get_package_versions()
    assert versions["scikit-learn"] != "not installed"
    assert versions["pyyaml"] != "not installed"
    assert versions["statsmodels"] != "not installed"
    assert versions["operator-geometry"] == metadata.version("operator-geometry")


def test_release_contract_manifest_matches_git_blob_hashes() -> None:
    with (REPO_ROOT / "release_contract_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["hash_type"] == "git_blob_sha1"
    for relative_path, expected_hash in manifest["files"].items():
        path = REPO_ROOT / relative_path
        assert path.is_file(), relative_path
        actual = subprocess.run(
            ["git", "hash-object", str(path)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert actual == expected_hash, f"Scientific contract drift: {relative_path}"


def test_minimum_gap_is_invariant_to_input_order_with_mode_limit() -> None:
    dominant = np.linspace(5.0, 1.0, 18).astype(complex)
    close_pair = np.array([10.0 + 0.0j, 10.0 + 1e-6j])
    small = np.array([0.1 + 0.0j, 0.2 + 0.0j, 0.3 + 0.0j, 0.4 + 0.0j])
    evals = np.concatenate([dominant, small, close_pair])
    gap_a, i_a, j_a = minimum_eigenvalue_gap(evals, max_modes=20)

    permutation = np.array(list(range(18)) + [22, 23, 18, 19, 20, 21])
    shuffled = evals[permutation]
    gap_b, i_b, j_b = minimum_eigenvalue_gap(shuffled, max_modes=20)

    assert gap_a == gap_b
    assert {evals[i_a], evals[j_a]} == {shuffled[i_b], shuffled[j_b]}
