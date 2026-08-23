"""Public entry point for the historical geometry brain-state test battery.

The original runner is retained unchanged in ``_geometry_brain_states_legacy.py``
for computational provenance. That historical runner populated its feature named
``nd_score`` from ``mean_ep_score``. The numerical feature is therefore the
legacy overlap/gap proximity statistic, not the current PC1-based
Near-Degeneracy (ND) score.

This entry point preserves the historical point computations while patching in
current semantic extraction adapters and corrected subject-level uncertainty
routines. A fresh run writes a new corrected-inference artifact and restores the
locked historical JSON byte-for-byte, even if the delegated historical runner
fails after touching that path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Support direct execution from the scripts directory without requiring an
# editable install first, matching the historical runner's behavior.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import _geometry_brain_states_legacy as _legacy  # noqa: E402

from cmcc.analysis.geometry_embedding import (  # noqa: E402
    classify_states_loso,
    compare_geometry_vs_power,
    extract_propofol_features_semantic,
    extract_sleep_features_semantic,
)

HISTORICAL_ARTIFACT = "geometry_brain_states.json"
CORRECTED_ARTIFACT = "geometry_brain_states_legacy_proximity_corrected_inference.json"


def _run_preserving_historical_artifact() -> Path:
    """Run the delegated battery without allowing it to overwrite locked history."""
    historical_path = _legacy.RESULTS_JSON / HISTORICAL_ARTIFACT
    corrected_path = _legacy.RESULTS_JSON / CORRECTED_ARTIFACT
    if corrected_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite {corrected_path}. Archive or remove it explicitly first."
        )

    original_bytes = historical_path.read_bytes() if historical_path.exists() else None
    try:
        _legacy.main()
        if not historical_path.exists():
            raise RuntimeError(
                "Delegated geometry-state runner completed without producing its expected JSON"
            )
        payload = json.loads(historical_path.read_text(encoding="utf-8"))
        payload["public_rerun_contract"] = {
            "feature_semantics": "legacy_proximity_score",
            "historical_artifact_preserved": HISTORICAL_ARTIFACT,
            "bootstrap": "subject_block_with_replacement_preserving_multiplicity",
            "finite_permutation_p": "(exceedances + 1) / (B + 1)",
            "current_nd_validation": False,
        }
        corrected_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    finally:
        if original_bytes is None:
            if historical_path.exists():
                historical_path.unlink()
        else:
            historical_path.write_bytes(original_bytes)

    return corrected_path


def main() -> None:
    """Run the battery with corrected semantics, inference, and artifact preservation."""
    _legacy.extract_propofol_features = extract_propofol_features_semantic
    _legacy.extract_sleep_features = extract_sleep_features_semantic
    _legacy.classify_states_loso = classify_states_loso
    _legacy.compare_geometry_vs_power = compare_geometry_vs_power
    output_path = _run_preserving_historical_artifact()
    print(f"Corrected public rerun written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
