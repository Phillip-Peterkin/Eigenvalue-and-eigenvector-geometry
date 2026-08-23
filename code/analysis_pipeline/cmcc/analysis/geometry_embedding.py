"""Compatibility and corrected execution surface for geometry state-space analysis.

The original implementation is retained unchanged in
``geometry_embedding_legacy.py`` for computational provenance. That historical
implementation reads ``mean_ep_score`` from result JSON files. Despite an older
``nd_score`` column label, those values are the legacy proximity statistic

    eigenvector_overlap / (minimum_eigenvalue_gap + 1e-10)

and are NOT the current manuscript PC1-based Near-Degeneracy (ND) score.

Most historical public functions are re-exported unchanged. New execution uses
semantic extraction adapters and corrected subject-level uncertainty routines:
subject bootstrap samples preserve multiplicity and finite permutation p-values
use the conventional +1 correction. Historical checked-in artifacts are not
silently recomputed or overwritten by these corrections.
"""
from __future__ import annotations

import numpy as np

from . import geometry_embedding_legacy as _legacy
from .geometry_embedding_legacy import *  # noqa: F401,F403

# Private helpers are imported explicitly because star imports intentionally omit
# underscore-prefixed names and the historical unit tests exercise these helpers.
_angle_between_2d = _legacy._angle_between_2d
_cohens_d = _legacy._cohens_d

HISTORICAL_SCHEMA_KEY = "nd_score"
HISTORICAL_SCHEMA_SEMANTICS = "legacy_proximity_score"


def _semantic_feature_names(
    table: _legacy.GeometryFeatureTable,
) -> _legacy.GeometryFeatureTable:
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
) -> _legacy.GeometryFeatureTable:
    """Extract propofol historical features with corrected semantic labels."""
    return _semantic_feature_names(_legacy.extract_propofol_features(ep_data, amplification_data))


def extract_sleep_features_semantic(
    sleep_ep_data: dict,
    sleep_amplification_data: dict,
) -> _legacy.GeometryFeatureTable:
    """Extract sleep historical features with corrected semantic labels."""
    return _semantic_feature_names(
        _legacy.extract_sleep_features(sleep_ep_data, sleep_amplification_data)
    )


def _rows_for_sampled_subjects(
    subject_ids: np.ndarray,
    sampled_subjects: np.ndarray,
) -> np.ndarray:
    """Return row indices for a subject bootstrap while preserving multiplicity.

    If a subject is sampled twice, that subject's complete observation block is
    appended twice. This differs intentionally from ``np.isin``, which collapses
    duplicate subject draws and therefore is not a with-replacement bootstrap.
    """
    subject_ids = np.asarray(subject_ids)
    parts = [np.flatnonzero(subject_ids == subject) for subject in sampled_subjects]
    parts = [part for part in parts if part.size]
    if not parts:
        return np.array([], dtype=int)
    return np.concatenate(parts).astype(int, copy=False)


def _loso_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fold-internal-standardized LOSO probabilities and predictions."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    subject_ids = np.asarray(subject_ids)
    all_probs = np.zeros(len(labels), dtype=float)
    all_preds = np.zeros(len(labels), dtype=int)

    for held_out in np.unique(subject_ids):
        test_mask = subject_ids == held_out
        train_mask = ~test_mask
        x_train = features[train_mask]
        y_train = labels[train_mask]
        x_test = features[test_mask]
        if len(np.unique(y_train)) < 2:
            all_probs[test_mask] = 0.5
            continue

        train_mean = x_train.mean(axis=0)
        train_std = x_train.std(axis=0)
        train_std[train_std == 0] = 1.0
        x_train_z = (x_train - train_mean) / train_std
        x_test_z = (x_test - train_mean) / train_std

        classifier = LinearDiscriminantAnalysis(solver="svd")
        classifier.fit(x_train_z, y_train)
        all_probs[test_mask] = classifier.predict_proba(x_test_z)[:, 1]
        all_preds[test_mask] = classifier.predict(x_test_z)

    return all_probs, all_preds


def classify_states_loso(
    features: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    contrast_name: str = "",
    dataset: str = "",
    feature_names: list[str] | None = None,
    seed: int = 42,
    n_bootstrap: int = 1000,
    n_null_permutations: int = 100,
) -> _legacy.SufficiencyResult:
    """Run subject-preserving LOSO classification with corrected uncertainty.

    The point prediction procedure matches the historical analysis: each subject
    is held out as a block and z-scoring is fitted only on the training fold.
    The confidence interval is corrected to a genuine subject-block bootstrap:
    duplicate sampled subjects contribute duplicate observation blocks. The
    finite permutation p-value uses ``(exceedances + 1) / (B + 1)`` so it cannot
    report an impossible zero probability from a finite null sample.
    """
    from sklearn.metrics import roc_auc_score

    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    subject_ids = np.asarray(subject_ids)
    unique_subjects = np.unique(subject_ids)
    if len(unique_subjects) < 3:
        raise ValueError(f"Need >= 3 subjects for LOSO, got {len(unique_subjects)}.")
    if len(np.unique(labels)) < 2:
        raise ValueError("Labels must contain at least 2 classes.")

    all_probs, all_preds = _loso_predictions(features, labels, subject_ids)
    auc = float(roc_auc_score(labels, all_probs))
    accuracy = float(np.mean(all_preds == labels))

    per_subject: list[dict] = []
    for subject in unique_subjects:
        rows = np.flatnonzero(subject_ids == subject)
        row_correct = all_preds[rows] == labels[rows]
        per_subject.append(
            {
                "subject": str(subject),
                "true_labels": labels[rows].astype(int).tolist(),
                "predicted_probs": all_probs[rows].astype(float).tolist(),
                "predicted_labels": all_preds[rows].astype(int).tolist(),
                "accuracy": float(np.mean(row_correct)),
                "correct": bool(np.all(row_correct)),
            }
        )
    subject_consistency = float(np.mean([item["correct"] for item in per_subject]))

    rng = np.random.default_rng(seed)
    bootstrap_aucs: list[float] = []
    for _ in range(n_bootstrap):
        sampled_subjects = rng.choice(
            unique_subjects,
            size=len(unique_subjects),
            replace=True,
        )
        rows = _rows_for_sampled_subjects(subject_ids, sampled_subjects)
        if rows.size == 0 or len(np.unique(labels[rows])) < 2:
            continue
        bootstrap_aucs.append(float(roc_auc_score(labels[rows], all_probs[rows])))

    if bootstrap_aucs:
        ci_lower = float(np.percentile(bootstrap_aucs, 2.5))
        ci_upper = float(np.percentile(bootstrap_aucs, 97.5))
    else:
        ci_lower = float("nan")
        ci_upper = float("nan")

    null_aucs: list[float] = []
    for permutation_index in range(n_null_permutations):
        perm_rng = np.random.default_rng(seed + permutation_index + 1)
        permuted_labels = labels.copy()
        for subject in unique_subjects:
            rows = np.flatnonzero(subject_ids == subject)
            shuffled = permuted_labels[rows].copy()
            perm_rng.shuffle(shuffled)
            permuted_labels[rows] = shuffled
        try:
            permuted_probs, _ = _loso_predictions(features, permuted_labels, subject_ids)
            null_aucs.append(float(roc_auc_score(permuted_labels, permuted_probs)))
        except (ValueError, np.linalg.LinAlgError):
            continue

    if null_aucs:
        null_mean = float(np.mean(null_aucs))
        null_std = float(np.std(null_aucs))
        exceedances = sum(value >= auc for value in null_aucs)
        null_p = float((exceedances + 1) / (len(null_aucs) + 1))
    else:
        null_mean = 0.5
        null_std = 0.0
        null_p = 1.0

    return _legacy.SufficiencyResult(
        contrast_name=contrast_name,
        dataset=dataset,
        auc_loso=auc,
        auc_ci_lower=ci_lower,
        auc_ci_upper=ci_upper,
        accuracy_loso=accuracy,
        per_subject_predictions=per_subject,
        n_subjects=len(unique_subjects),
        n_features=features.shape[1],
        feature_names=feature_names or [],
        subject_consistency=subject_consistency,
        null_auc_mean=null_mean,
        null_auc_std=null_std,
        null_auc_p=null_p,
        passes_threshold=auc >= 0.80,
    )


def compare_geometry_vs_power(
    geometry_features: np.ndarray,
    power_features: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    seed: int = 42,
    n_bootstrap: int = 1000,
) -> _legacy.IncrementalValueResult:
    """Compare LOSO feature families with a multiplicity-preserving subject CI."""
    from sklearn.metrics import roc_auc_score

    geometry_features = np.asarray(geometry_features, dtype=float)
    power_features = np.asarray(power_features, dtype=float)
    labels = np.asarray(labels)
    subject_ids = np.asarray(subject_ids)
    combined = np.hstack([geometry_features, power_features])

    geometry_probs, _ = _loso_predictions(geometry_features, labels, subject_ids)
    power_probs, _ = _loso_predictions(power_features, labels, subject_ids)
    combined_probs, _ = _loso_predictions(combined, labels, subject_ids)
    auc_geometry = float(roc_auc_score(labels, geometry_probs))
    auc_power = float(roc_auc_score(labels, power_probs))
    auc_combined = float(roc_auc_score(labels, combined_probs))

    unique_subjects = np.unique(subject_ids)
    rng = np.random.default_rng(seed)
    bootstrap_deltas: list[float] = []
    for _ in range(n_bootstrap):
        sampled_subjects = rng.choice(
            unique_subjects,
            size=len(unique_subjects),
            replace=True,
        )
        rows = _rows_for_sampled_subjects(subject_ids, sampled_subjects)
        if rows.size == 0 or len(np.unique(labels[rows])) < 2:
            continue
        bootstrap_deltas.append(
            float(
                roc_auc_score(labels[rows], geometry_probs[rows])
                - roc_auc_score(labels[rows], power_probs[rows])
            )
        )

    if bootstrap_deltas:
        ci = (
            float(np.percentile(bootstrap_deltas, 2.5)),
            float(np.percentile(bootstrap_deltas, 97.5)),
        )
    else:
        ci = (float("nan"), float("nan"))

    perm_rng = np.random.default_rng(seed + 999)
    null_labels = labels.copy()
    for subject in unique_subjects:
        rows = np.flatnonzero(subject_ids == subject)
        shuffled = null_labels[rows].copy()
        perm_rng.shuffle(shuffled)
        null_labels[rows] = shuffled
    null_geometry_probs, _ = _loso_predictions(
        geometry_features,
        null_labels,
        subject_ids,
    )
    null_power_probs, _ = _loso_predictions(power_features, null_labels, subject_ids)
    null_geometry = float(roc_auc_score(null_labels, null_geometry_probs))
    null_power = float(roc_auc_score(null_labels, null_power_probs))

    delta = auc_geometry - auc_power
    delta_combined = auc_combined - auc_power
    return _legacy.IncrementalValueResult(
        auc_geometry_only=auc_geometry,
        auc_power_only=auc_power,
        auc_combined=auc_combined,
        delta_auc_vs_power=delta,
        delta_auc_combined_vs_power=delta_combined,
        bootstrap_ci_delta=ci,
        null_auc_geometry=null_geometry,
        null_auc_power=null_power,
        passes_threshold=(delta >= 0.05) or (delta_combined >= 0.03),
    )
