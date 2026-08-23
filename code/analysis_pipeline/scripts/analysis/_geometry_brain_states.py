"""Public entry point for the historical geometry brain-state test battery.

The original runner is retained unchanged in ``_geometry_brain_states_legacy.py``
for computational provenance. That historical runner populated its feature named
``nd_score`` from ``mean_ep_score``. The numerical feature is therefore the
legacy overlap/gap proximity statistic, not the current PC1-based
Near-Degeneracy (ND) score.

This entry point preserves the historical point computations while patching in
current semantic extraction adapters and corrected subject-level uncertainty
routines. A fresh run therefore writes ``legacy_proximity_score`` in the output
schema, preserves subject multiplicity in bootstrap confidence intervals, and
uses finite-permutation +1 correction for classifier null p-values.
"""
from __future__ import annotations

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


def main() -> None:
    """Run the battery with corrected public semantics and uncertainty routines."""
    _legacy.extract_propofol_features = extract_propofol_features_semantic
    _legacy.extract_sleep_features = extract_sleep_features_semantic
    _legacy.classify_states_loso = classify_states_loso
    _legacy.compare_geometry_vs_power = compare_geometry_vs_power
    _legacy.main()


if __name__ == "__main__":
    main()
