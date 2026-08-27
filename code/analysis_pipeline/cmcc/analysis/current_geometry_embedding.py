"""Current semantic API for geometry state-space inference.

This module deliberately exposes only the corrected public operations. The
historical implementation and the broad compatibility module remain available
for provenance, but new/canonical code should import from here.
"""
from __future__ import annotations

from cmcc.analysis.geometry_embedding import (
    HISTORICAL_SCHEMA_KEY,
    HISTORICAL_SCHEMA_SEMANTICS,
    _rows_for_sampled_subjects,
    _semantic_feature_names,
    analyze_geometric_structure,
    classify_states_loso,
    compare_geometry_vs_power,
    extract_propofol_features_semantic,
    extract_sleep_features_semantic,
)
from cmcc.analysis.geometry_embedding_legacy import (
    CollatedResult,
    GeometryFeatureTable,
    GeometryTestBattery,
    IncrementalValueResult,
    OrthogonalityResult,
    StructureResult,
    SufficiencyResult,
    assemble_test_battery,
    check_orthogonality,
    collate_existing_results,
    compute_overall_verdict,
)

__all__ = [
    "HISTORICAL_SCHEMA_KEY",
    "HISTORICAL_SCHEMA_SEMANTICS",
    "GeometryFeatureTable",
    "SufficiencyResult",
    "IncrementalValueResult",
    "OrthogonalityResult",
    "StructureResult",
    "CollatedResult",
    "GeometryTestBattery",
    "extract_propofol_features_semantic",
    "extract_sleep_features_semantic",
    "classify_states_loso",
    "compare_geometry_vs_power",
    "analyze_geometric_structure",
    "check_orthogonality",
    "collate_existing_results",
    "compute_overall_verdict",
    "assemble_test_battery",
    "_semantic_feature_names",
    "_rows_for_sampled_subjects",
]
