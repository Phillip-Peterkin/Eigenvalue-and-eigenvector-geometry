"""Regression tests for corrected subject-level geometry inference."""
from __future__ import annotations

import numpy as np
import pytest

from cmcc.analysis.geometry_embedding import (
    GeometryFeatureTable,
    _rows_for_sampled_subjects,
    _semantic_feature_names,
    analyze_geometric_structure,
    classify_states_loso,
    compare_geometry_vs_power,
)


def test_semantic_adapter_is_pure_and_idempotent() -> None:
    original = GeometryFeatureTable(
        subjects=["s1"],
        conditions=["awake"],
        features=np.array([[1.0, 2.0]]),
        feature_names=["nd_score", "spectral_radius"],
        dataset="propofol",
    )

    corrected = _semantic_feature_names(original)
    corrected_again = _semantic_feature_names(corrected)

    assert original.feature_names == ["nd_score", "spectral_radius"]
    assert corrected is not original
    assert corrected.feature_names == ["legacy_proximity_score", "spectral_radius"]
    assert corrected_again is not corrected
    assert corrected_again.feature_names == corrected.feature_names


def test_subject_bootstrap_preserves_duplicate_blocks() -> None:
    subject_ids = np.array(["s1", "s1", "s2", "s2", "s3", "s3"])
    sampled_subjects = np.array(["s1", "s1", "s3"])
    rows = _rows_for_sampled_subjects(subject_ids, sampled_subjects)
    assert rows.tolist() == [0, 1, 0, 1, 4, 5]


def test_finite_permutation_p_value_cannot_be_zero() -> None:
    rng = np.random.default_rng(7)
    n_subjects = 12
    class_zero = rng.normal(0.0, 0.1, (n_subjects, 2))
    class_one = rng.normal(4.0, 0.1, (n_subjects, 2))
    features = np.vstack([class_zero, class_one])
    labels = np.array([0] * n_subjects + [1] * n_subjects)
    subjects = np.array(
        [f"s{index}" for index in range(n_subjects)]
        + [f"s{index}" for index in range(n_subjects)]
    )

    n_permutations = 5
    result = classify_states_loso(
        features,
        labels,
        subjects,
        seed=11,
        n_bootstrap=20,
        n_null_permutations=n_permutations,
    )

    assert result.auc_loso > 0.95
    assert result.null_auc_p >= 1.0 / (n_permutations + 1)
    assert result.null_auc_p <= 1.0


def test_subject_block_bootstrap_produces_finite_interval() -> None:
    rng = np.random.default_rng(19)
    n_subjects = 10
    class_zero = rng.normal(0.0, 0.5, (n_subjects, 3))
    class_one = rng.normal(1.2, 0.5, (n_subjects, 3))
    features = np.vstack([class_zero, class_one])
    labels = np.array([0] * n_subjects + [1] * n_subjects)
    subjects = np.array(
        [f"s{index}" for index in range(n_subjects)]
        + [f"s{index}" for index in range(n_subjects)]
    )

    result = classify_states_loso(
        features,
        labels,
        subjects,
        seed=23,
        n_bootstrap=100,
        n_null_permutations=3,
    )

    assert np.isfinite(result.auc_ci_lower)
    assert np.isfinite(result.auc_ci_upper)
    assert 0.0 <= result.auc_ci_lower <= result.auc_ci_upper <= 1.0


def test_geometry_vs_power_null_baseline_averages_deterministic_permutations() -> None:
    rng = np.random.default_rng(31)
    n_subjects = 10
    labels = np.array([0] * n_subjects + [1] * n_subjects)
    subjects = np.array(
        [f"s{index}" for index in range(n_subjects)]
        + [f"s{index}" for index in range(n_subjects)]
    )
    geometry = np.vstack(
        [rng.normal(0.0, 0.5, (n_subjects, 2)), rng.normal(1.0, 0.5, (n_subjects, 2))]
    )
    power = rng.normal(0.0, 1.0, (2 * n_subjects, 1))

    first = compare_geometry_vs_power(
        geometry,
        power,
        labels,
        subjects,
        seed=37,
        n_bootstrap=20,
        n_null_permutations=7,
    )
    second = compare_geometry_vs_power(
        geometry,
        power,
        labels,
        subjects,
        seed=37,
        n_bootstrap=20,
        n_null_permutations=7,
    )

    assert 0.0 <= first.null_auc_geometry <= 1.0
    assert 0.0 <= first.null_auc_power <= 1.0
    assert first.null_auc_geometry == pytest.approx(second.null_auc_geometry)
    assert first.null_auc_power == pytest.approx(second.null_auc_power)


def test_unscorable_loso_design_fails_instead_of_scoring_default_predictions() -> None:
    features = np.array(
        [
            [0.0, 0.0],
            [2.0, 2.0],
            [0.1, 0.0],
            [0.0, 0.1],
        ]
    )
    labels = np.array([0, 1, 0, 0])
    subjects = np.array(["s1", "s1", "s2", "s3"])

    with pytest.raises(ValueError, match="insufficient fitted folds"):
        classify_states_loso(
            features,
            labels,
            subjects,
            n_bootstrap=10,
            n_null_permutations=2,
        )


def test_subject_consistency_wraps_across_plus_minus_pi_boundary() -> None:
    small_angle = np.deg2rad(1.0)
    features = np.array(
        [
            [0.0, 0.0],
            [-1.0, small_angle],
            [0.0, 0.0],
            [-1.0, -small_angle],
            [0.0, 0.0],
            [-1.0, 0.0],
        ]
    )
    labels = np.array(["awake", "state", "awake", "state", "awake", "state"])
    subjects = np.array(["s1", "s1", "s2", "s2", "s3", "s3"])

    result = analyze_geometric_structure(
        features,
        labels,
        subjects,
        feature_names=["x", "y"],
        seed=3,
        n_bootstrap=10,
    )

    assert result.subject_consistency["state"] == pytest.approx(1.0)
