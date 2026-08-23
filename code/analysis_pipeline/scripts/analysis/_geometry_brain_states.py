"""Public entry point for the historical geometry brain-state test battery.

The original runner is retained unchanged in ``_geometry_brain_states_legacy.py``
for computational provenance. That historical runner populated its feature named
``nd_score`` from ``mean_ep_score``. The numerical feature is therefore the
legacy overlap/gap proximity statistic, not the current PC1-based
Near-Degeneracy (ND) score.

This entry point preserves the historical computations while replacing only the
feature-name metadata through the semantic extraction adapters. A fresh run
therefore writes ``legacy_proximity_score`` in the output schema instead of
recreating the misleading historical ``nd_score`` label.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Support direct execution from the scripts directory without requiring an
# editable install first, matching the historical runner's behavior.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cmcc.analysis.geometry_embedding import (  # noqa: E402
    extract_propofol_features_semantic,
    extract_sleep_features_semantic,
)

import _geometry_brain_states_legacy as _legacy  # noqa: E402


def main() -> None:
    """Run the historical battery with corrected feature-name semantics."""
    _legacy.extract_propofol_features = extract_propofol_features_semantic
    _legacy.extract_sleep_features = extract_sleep_features_semantic
    _legacy.main()


if __name__ == "__main__":
    main()
