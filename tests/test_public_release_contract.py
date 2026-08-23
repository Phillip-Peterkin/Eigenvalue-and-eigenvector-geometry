"""Public-release regression checks for scientific documentation contracts."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _canonical_broadband_source() -> str:
    return (
        ROOT
        / "code"
        / "analysis_pipeline"
        / "scripts"
        / "run_all_subjects_broadband_canonical.py"
    ).read_text(encoding="utf-8")


def test_canonical_config_declares_distinct_band_passbands() -> None:
    config = yaml.safe_load((ROOT / "code" / "config.yaml").read_text(encoding="utf-8"))
    preprocessing = config["preprocessing"]
    assert preprocessing["high_gamma_passband"] == [70, 150]
    assert preprocessing["broadband_passband"] == [1, 200]
    assert preprocessing["high_gamma_passband"] != preprocessing["broadband_passband"]


def test_canonical_broadband_runner_requires_broadband_key() -> None:
    source = _canonical_broadband_source()
    assert 'preprocessing"]["broadband_passband"' in source
    assert "Canonical broadband reproduction requires" in source


def test_canonical_broadband_runner_uses_documented_data_root_contract() -> None:
    source = _canonical_broadband_source()
    assert 'require_data_root("IEEG_DATA_ROOT")' in source
    assert 'config["data"]["root"] = str(data_root)' in source


def test_canonical_broadband_runner_persists_band_provenance() -> None:
    source = _canonical_broadband_source()
    assert 'summary["analysis_type"]' in source
    assert 'summary["configured_broadband_passband_hz"]' in source
    assert 'summary["effective_broadband_passband_by_run_hz"]' in source


def test_canonical_broadband_runner_fails_empty_success_set() -> None:
    source = _canonical_broadband_source()
    assert "if not successful:" in source
    assert "produced no successful subject summaries" in source


def test_readme_exports_canonical_data_roots() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "export IEEG_DATA_ROOT=" in readme
    assert "export DS004752_DATA_ROOT=" in readme
    assert "export PROPOFOL_DATA_ROOT=" in readme
    assert "export SLEEP_DATA_ROOT=" in readme


def test_public_contract_does_not_equate_legacy_ep_score_with_final_nd() -> None:
    spec = json.loads((ROOT / "preregistration_spec.json").read_text(encoding="utf-8"))
    frozen = spec["frozen_parameters"]
    assert "legacy_ep_score_definition" in frozen
    assert "current_manuscript_nd_definition" in frozen
    assert "not the same statistic" in frozen["metric_equivalence_warning"]


def test_historical_geometry_correlation_is_marked_as_legacy_metric() -> None:
    spec = json.loads((ROOT / "preregistration_spec.json").read_text(encoding="utf-8"))
    prediction = next(
        item for item in spec["primary_predictions"] if item["id"] == "P2_legacy_geometry_sigma"
    )
    assert prediction["status"] == "legacy_metric_requires_final_nd_recomputation"


def test_public_audit_exposes_open_alignment_items() -> None:
    audit = (ROOT / "PUBLIC_AUDIT.md").read_text(encoding="utf-8")
    assert "Legacy `ep_score` versus current Near-Degeneracy score" in audit
    assert "Broadband versus high-gamma configuration" in audit
    assert "Public-release gate" in audit
