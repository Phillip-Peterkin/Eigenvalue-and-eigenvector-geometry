"""Compatibility surface for the historical geometry state-space analysis.

The original implementation is retained unchanged in
``geometry_embedding_legacy.py`` for computational provenance. That historical
implementation reads ``mean_ep_score`` from result JSON files. Despite an older
``nd_score`` column label, those values are the legacy proximity statistic

    eigenvector_overlap / (minimum_eigenvalue_gap + 1e-10)

and are NOT the current manuscript PC1-based Near-Degeneracy (ND) score.

Existing public functions are re-exported here so historical result reproduction
and tests remain stable. New code that needs semantically explicit feature names
can call ``extract_propofol_features_semantic`` or
``extract_sleep_features_semantic``; these adapters change only the feature
label, not any numerical values.
"""
from __future__ import annotations

from . import geometry_embedding_legacy as _legacy
from .geometry_embedding_legacy import *  # noqa: F401,F403

# Private helpers are imported explicitly because star imports intentionally omit
# underscore-prefixed names and the historical unit tests exercise these helpers.
_angle_between_2d = _legacy._angle_between_2d
_cohens_d = _legacy._cohens_d

HISTORICAL_SCHEMA_KEY = "nd_score"
HISTORICAL_SCHEMA_SEMANTICS = "legacy_proximity_score"


def _semantic_feature_names(table: GeometryFeatureTable) -> GeometryFeatureTable:
    """Return the historical table with its legacy proximity column named honestly.

    Numerical values, row order, subject identifiers, and conditions are left
    unchanged. Only the historical ``nd_score`` feature-name string is replaced.
    """
    names = list(table.feature_names)
    if HISTORICAL_SCHEMA_KEY not in names:
        raise RuntimeError(
            "Historical geometry table does not contain the expected nd_score schema key"
        )
    table.feature_names = [
        HISTORICAL_SCHEMA_SEMANTICS if name == HISTORICAL_SCHEMA_KEY else name
        for name in names
    ]
    return table


def extract_propofol_features_semantic(
    ep_data: dict,
    amplification_data: dict,
) -> GeometryFeatureTable:
    """Extract propofol historical features with corrected semantic labels."""
    return _semantic_feature_names(_legacy.extract_propofol_features(ep_data, amplification_data))


def extract_sleep_features_semantic(
    sleep_ep_data: dict,
    sleep_amplification_data: dict,
) -> GeometryFeatureTable:
    """Extract sleep historical features with corrected semantic labels."""
    return _semantic_feature_names(
        _legacy.extract_sleep_features(sleep_ep_data, sleep_amplification_data)
    )
