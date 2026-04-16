"""Operator-geometry brain state test battery.

Purpose
-------
Eight quantitative tests evaluating whether operator geometry (eigenvalue gap,
condition number, ND score, spectral radius) constitutes a sufficient,
independent, and structured coordinate system for brain state.

Scientific context
------------------
VAR(1) Jacobians fitted in sliding windows yield per-window eigenvalue spectra.
Summary statistics (mean gap, mean condition number, mean EP/ND score, mean
spectral radius) per subject per condition serve as the unit of observation.
All cross-subject analyses use per-subject condition means — never raw windows
— to avoid pseudoreplication from overlapping windows (80% overlap at 500 ms
window / 100 ms step).

EP score = ND score (renamed in manuscript; JSON keys use ``mean_ep_score``).

Expected inputs
---------------
Parsed JSON dicts from existing result files:
- ep_propofol_eeg.json (propofol EP metrics)
- amplification_propofol.json (propofol condition numbers)
- ep_sleep_dynamics.json (sleep EP metrics)
- amplification_sleep_convergence.json (sleep condition numbers)
- exceptional_points.json, jackknife_sensitivity.json (criticality coupling)
- ep_shared_subspace_propofol.json, ep_shared_subspace_sleep.json (stability)

Assumptions
-----------
- All datasets use 15 PCA components, 0.5 s window, 0.1 s step.
- Condition number comes from amplification result files, not EP files.
- Alpha power available for propofol awake only; no power metrics for sleep.
- Ridge regularization sensitivity established upstream.

Known limitations
-----------------
- Test 2 (incremental value) and Test 3 (orthogonality) limited to propofol
  dataset due to missing power metrics in sleep results.
- Test 7 (temporal precedence) deferred — requires new data processing.
- Small sample sizes (n=10 sleep, n=20 propofol) limit statistical power.

Validation strategy
-------------------
- Unit tests on synthetic data with known separability, known distances,
  known angles, and known correlations.
- Negative controls: shuffled labels produce chance-level classification.
- Outlier sensitivity: top 1% condition number exclusion stability.
- Subject consistency: fraction of subjects showing same-direction effects.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats as sp_stats


@dataclass
class GeometryFeatureTable:
    """Per-subject, per-condition geometry features extracted from JSON results.

    Attributes
    ----------
    subjects : list[str]
        Subject identifiers, one per observation row.
    conditions : list[str]
        Condition labels, one per observation row.
    features : np.ndarray, shape (n_obs, n_features)
        Feature matrix. Rows = subject-condition pairs.
    feature_names : list[str]
        Column names for features.
    dataset : str
        "propofol" or "sleep".
    alpha_power : np.ndarray or None, shape (n_obs,)
        Alpha power per observation (propofol awake only; None for sleep).
    excluded_subjects : list[str]
        Subjects excluded due to missing data or NaN.
    """

    subjects: list[str]
    conditions: list[str]
    features: np.ndarray
    feature_names: list[str]
    dataset: str
    alpha_power: np.ndarray | None = None
    excluded_subjects: list[str] = field(default_factory=list)


@dataclass
class SufficiencyResult:
    """Test 1: Leave-one-subject-out classification.

    Units: AUC is dimensionless [0, 1]. Accuracy is proportion [0, 1].
    """

    contrast_name: str
    dataset: str
    auc_loso: float
    auc_ci_lower: float
    auc_ci_upper: float
    accuracy_loso: float
    per_subject_predictions: list[dict]
    n_subjects: int
    n_features: int
    feature_names: list[str]
    subject_consistency: float
    null_auc_mean: float
    null_auc_std: float
    null_auc_p: float
    passes_threshold: bool


@dataclass
class IncrementalValueResult:
    """Test 2: Geometry vs power comparison (propofol only).

    Units: AUC dimensionless [0, 1]. ΔAUC dimensionless.
    """

    auc_geometry_only: float
    auc_power_only: float
    auc_combined: float
    delta_auc_vs_power: float
    delta_auc_combined_vs_power: float
    bootstrap_ci_delta: tuple[float, float]
    null_auc_geometry: float
    null_auc_power: float
    passes_threshold: bool


@dataclass
class OrthogonalityResult:
    """Test 3: Independence from spectral features (propofol only).

    Two-tier reporting:
    - Primary (strict): median |r| < 0.20 AND 2/3 primary features |d| >= 0.5
    - Sensitivity: median |r| < 0.30 AND 2/3 primary features |d| >= 0.5

    Units: correlations dimensionless [-1, 1]. Cohen's d dimensionless.
    """

    median_abs_correlation: float
    per_feature_correlations: dict[str, float]
    residualized_effect_sizes: dict[str, float]
    passes_threshold: bool
    passes_sensitivity_threshold: bool = False
    n_primary_features_passing_d: int = 0
    interpretation: str = ""


@dataclass
class StructureResult:
    """Test 4: Multi-dimensional geometric state space.

    Units: Mahalanobis distance dimensionless. Angles in degrees.
    """

    state_centroids: dict[str, list[float]]
    pairwise_mahalanobis: dict[str, float]
    state_change_vectors: dict[str, list[float]]
    angular_separations: dict[str, float]
    angular_separation_cis: dict[str, tuple[float, float]]
    subject_consistency: dict[str, float]
    passes_threshold: bool


@dataclass
class CollatedResult:
    """Tests 5, 6, 8: Collated from existing results."""

    test_name: str
    metric_name: str
    value: float
    threshold: float
    passes: bool
    details: dict = field(default_factory=dict)


@dataclass
class TemporalPrecedenceSummary:
    """Test 7: Temporal precedence summary for battery integration.

    Extracted from temporal_precedence.json output.

    Two-tier pass system:
    - any_metric_passes: lenient (overlapping windows OK)
    - any_metric_passes_strict: requires non-overlapping window survival
    """

    any_metric_passes: bool
    any_metric_passes_strict: bool
    best_metric: str
    best_slope_p: float
    best_consistency: float
    best_early_late_d: float
    details: dict = field(default_factory=dict)


@dataclass
class GeometryTestBattery:
    """Master result combining all tests."""

    sufficiency: list[SufficiencyResult]
    incremental: IncrementalValueResult | None
    orthogonality: OrthogonalityResult | None
    structure: StructureResult
    stability: list[CollatedResult]
    criticality_coupling: list[CollatedResult]
    amplification_link: list[CollatedResult]
    temporal_precedence: TemporalPrecedenceSummary | str
    overall_verdict: str
    n_tests_passed: int
    n_tests_total: int


def extract_propofol_features(
    ep_data: dict,
    amplification_data: dict,
) -> GeometryFeatureTable:
    """Extract geometry features from propofol JSON results.

    Parameters
    ----------
    ep_data : dict
        Parsed ep_propofol_eeg.json.
    amplification_data : dict
        Parsed amplification_propofol.json.

    Returns
    -------
    GeometryFeatureTable
        Two rows per subject (awake, sedation_run1).

    Raises
    ------
    KeyError
        If required keys are missing from either JSON.
    ValueError
        If subject ID join fails.
    """
    ep_subjects = {s["subject"]: s for s in ep_data["subjects"]}
    amp_subjects = {s["subject"]: s for s in amplification_data["subjects"]}

    common_ids = sorted(set(ep_subjects.keys()) & set(amp_subjects.keys()))
    if not common_ids:
        raise ValueError("No common subject IDs between EP and amplification data.")

    orphan_ep = set(ep_subjects.keys()) - set(amp_subjects.keys())
    orphan_amp = set(amp_subjects.keys()) - set(ep_subjects.keys())
    if orphan_ep or orphan_amp:
        warnings.warn(
            f"Subject ID join: {len(orphan_ep)} EP-only, {len(orphan_amp)} amp-only. "
            f"Using {len(common_ids)} common subjects.",
            stacklevel=2,
        )

    feature_names = ["eigenvalue_gap", "condition_number", "nd_score", "spectral_radius"]
    subjects = []
    conditions = []
    rows = []
    alpha_powers = []
    excluded = []

    for sid in common_ids:
        ep_s = ep_subjects[sid]
        amp_s = amp_subjects[sid]

        try:
            awake_row = [
                ep_s["awake"]["mean_eigenvalue_gap"],
                amp_s["awake"]["condition_number_mean"],
                ep_s["awake"]["mean_ep_score"],
                ep_s["awake"]["mean_spectral_radius"],
            ]
            sed_row = [
                ep_s["sedation_run1"]["mean_eigenvalue_gap"],
                amp_s["sedation_run1"]["condition_number_mean"],
                ep_s["sedation_run1"]["mean_ep_score"],
                ep_s["sedation_run1"]["mean_spectral_radius"],
            ]
        except KeyError as e:
            warnings.warn(f"Subject {sid}: missing key {e}. Excluding.", stacklevel=2)
            excluded.append(sid)
            continue

        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in awake_row + sed_row):
            warnings.warn(f"Subject {sid}: NaN/None in features. Excluding.", stacklevel=2)
            excluded.append(sid)
            continue

        subjects.extend([sid, sid])
        conditions.extend(["awake", "propofol"])
        rows.append(awake_row)
        rows.append(sed_row)

        awake_alpha = ep_s["awake"].get("mean_alpha_power")
        alpha_powers.extend([awake_alpha, awake_alpha])

    if not rows:
        raise ValueError("No valid subjects after extraction and validation.")

    features = np.array(rows, dtype=float)
    alpha_arr = np.array(alpha_powers, dtype=float)

    if not np.all(np.isfinite(features)):
        bad_mask = ~np.all(np.isfinite(features), axis=1)
        n_bad = int(bad_mask.sum())
        warnings.warn(f"{n_bad} rows have non-finite values after extraction.", stacklevel=2)

    return GeometryFeatureTable(
        subjects=subjects,
        conditions=conditions,
        features=features,
        feature_names=feature_names,
        dataset="propofol",
        alpha_power=alpha_arr,
        excluded_subjects=excluded,
    )


def extract_sleep_features(
    sleep_ep_data: dict,
    sleep_amplification_data: dict,
) -> GeometryFeatureTable:
    """Extract geometry features from sleep JSON results.

    Parameters
    ----------
    sleep_ep_data : dict
        Parsed ep_sleep_dynamics.json.
    sleep_amplification_data : dict
        Parsed amplification_sleep_convergence.json.

    Returns
    -------
    GeometryFeatureTable
        Three rows per subject (W, N3, R).

    Raises
    ------
    KeyError
        If required keys are missing.
    ValueError
        If subject ID join fails.
    """
    ep_by_subject: dict[str, dict[str, dict]] = {}
    for r in sleep_ep_data["per_state_results"]:
        sid = r["subject"]
        state = r["state"]
        if sid not in ep_by_subject:
            ep_by_subject[sid] = {}
        ep_by_subject[sid][state] = r

    amp_by_subject: dict[str, dict[str, dict]] = {}
    for s in sleep_amplification_data["subjects"]:
        sid = s["subject"]
        amp_by_subject[sid] = s["states"]

    common_ids = sorted(set(ep_by_subject.keys()) & set(amp_by_subject.keys()))
    if not common_ids:
        raise ValueError("No common subject IDs between sleep EP and amplification data.")

    orphan_ep = set(ep_by_subject.keys()) - set(amp_by_subject.keys())
    orphan_amp = set(amp_by_subject.keys()) - set(ep_by_subject.keys())
    if orphan_ep or orphan_amp:
        warnings.warn(
            f"Sleep subject join: {len(orphan_ep)} EP-only, {len(orphan_amp)} amp-only. "
            f"Using {len(common_ids)} common subjects.",
            stacklevel=2,
        )

    target_states = ["W", "N3", "R"]
    state_to_label = {"W": "awake", "N3": "N3", "R": "REM"}
    feature_names = ["eigenvalue_gap", "condition_number", "nd_score", "spectral_radius"]
    subjects = []
    conditions = []
    rows = []
    excluded = []

    for sid in common_ids:
        ep_states = ep_by_subject[sid]
        amp_states = amp_by_subject[sid]

        missing_states = [st for st in target_states if st not in ep_states or st not in amp_states]
        if missing_states:
            warnings.warn(
                f"Subject {sid}: missing states {missing_states}. Excluding.",
                stacklevel=2,
            )
            excluded.append(sid)
            continue

        subject_rows = []
        valid = True
        for st in target_states:
            try:
                row = [
                    ep_states[st]["mean_eigenvalue_gap"],
                    amp_states[st]["condition_number_median"],
                    ep_states[st]["mean_ep_score"],
                    ep_states[st]["mean_spectral_radius"],
                ]
            except KeyError as e:
                warnings.warn(f"Subject {sid}, state {st}: missing key {e}.", stacklevel=2)
                excluded.append(sid)
                valid = False
                break

            if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in row):
                warnings.warn(f"Subject {sid}, state {st}: NaN/None.", stacklevel=2)
                excluded.append(sid)
                valid = False
                break

            subject_rows.append((st, row))

        if not valid:
            continue

        for st, row in subject_rows:
            subjects.append(sid)
            conditions.append(state_to_label[st])
            rows.append(row)

    if not rows:
        raise ValueError("No valid subjects after sleep extraction and validation.")

    features = np.array(rows, dtype=float)

    return GeometryFeatureTable(
        subjects=subjects,
        conditions=conditions,
        features=features,
        feature_names=feature_names,
        dataset="sleep",
        alpha_power=None,
        excluded_subjects=excluded,
    )


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d (pooled SD denominator).

    Parameters
    ----------
    a, b : np.ndarray
        Two groups.

    Returns
    -------
    float
        Positive means a > b.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    pooled_var = ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)) / (n_a + n_b - 2)
    pooled_sd = np.sqrt(pooled_var)
    if pooled_sd == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


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
) -> SufficiencyResult:
    """Leave-one-subject-out classification with LDA.

    Z-scoring is performed INSIDE each fold (fit on train, transform test)
    to prevent data leakage.

    Parameters
    ----------
    features : np.ndarray, shape (n_obs, n_features)
    labels : np.ndarray, shape (n_obs,), binary 0/1
    subject_ids : np.ndarray, shape (n_obs,), subject identifiers
    contrast_name : str
    dataset : str
    feature_names : list[str] or None
    seed : int
    n_bootstrap : int
        Bootstrap iterations for AUC confidence interval.
    n_null_permutations : int
        Shuffled-label permutations for null AUC distribution.

    Returns
    -------
    SufficiencyResult
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score

    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)

    if n_subjects < 3:
        raise ValueError(f"Need >= 3 subjects for LOSO, got {n_subjects}.")

    all_probs = np.zeros(len(labels))
    all_preds = np.zeros(len(labels), dtype=int)
    per_subject = []

    for held_out in unique_subjects:
        test_mask = subject_ids == held_out
        train_mask = ~test_mask

        X_train, y_train = features[train_mask], labels[train_mask]
        X_test, y_test = features[test_mask], labels[test_mask]

        if len(np.unique(y_train)) < 2:
            continue

        train_mean = X_train.mean(axis=0)
        train_std = X_train.std(axis=0)
        train_std[train_std == 0] = 1.0
        X_train_z = (X_train - train_mean) / train_std
        X_test_z = (X_test - train_mean) / train_std

        clf = LinearDiscriminantAnalysis(solver="svd")
        clf.fit(X_train_z, y_train)

        prob = clf.predict_proba(X_test_z)[:, 1]
        pred = clf.predict(X_test_z)

        all_probs[test_mask] = prob
        all_preds[test_mask] = pred

        per_subject.append({
            "subject": str(held_out),
            "true_label": int(y_test[0]),
            "predicted_prob": float(prob[0]),
            "predicted_label": int(pred[0]),
            "correct": bool(pred[0] == y_test[0]),
        })

    if len(np.unique(labels)) < 2:
        raise ValueError("Labels must contain at least 2 classes.")

    auc = float(roc_auc_score(labels, all_probs))
    accuracy = float(np.mean(all_preds == labels))
    consistency = float(np.mean([p["correct"] for p in per_subject]))

    rng = np.random.default_rng(seed)
    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(unique_subjects), size=len(unique_subjects), replace=True)
        boot_subjects = unique_subjects[idx]
        boot_mask = np.isin(subject_ids, boot_subjects)
        if len(np.unique(labels[boot_mask])) < 2:
            continue
        try:
            boot_aucs.append(roc_auc_score(labels[boot_mask], all_probs[boot_mask]))
        except ValueError:
            continue

    if boot_aucs:
        ci_lower = float(np.percentile(boot_aucs, 2.5))
        ci_upper = float(np.percentile(boot_aucs, 97.5))
    else:
        ci_lower, ci_upper = float("nan"), float("nan")

    null_aucs = []
    for i in range(n_null_permutations):
        perm_rng = np.random.default_rng(seed + i + 1)
        perm_labels = labels.copy()
        for subj in unique_subjects:
            subj_mask = subject_ids == subj
            subj_labels = perm_labels[subj_mask]
            perm_rng.shuffle(subj_labels)
            perm_labels[subj_mask] = subj_labels

        perm_probs = np.zeros(len(perm_labels))
        for held_out in unique_subjects:
            test_mask = subject_ids == held_out
            train_mask = ~test_mask
            X_train, y_train = features[train_mask], perm_labels[train_mask]
            X_test = features[test_mask]
            if len(np.unique(y_train)) < 2:
                perm_probs[test_mask] = 0.5
                continue
            tm = X_train.mean(axis=0)
            ts = X_train.std(axis=0)
            ts[ts == 0] = 1.0
            clf = LinearDiscriminantAnalysis(solver="svd")
            clf.fit((X_train - tm) / ts, y_train)
            perm_probs[test_mask] = clf.predict_proba((X_test - tm) / ts)[:, 1]

        try:
            null_aucs.append(roc_auc_score(perm_labels, perm_probs))
        except ValueError:
            continue

    null_mean = float(np.mean(null_aucs)) if null_aucs else 0.5
    null_std = float(np.std(null_aucs)) if null_aucs else 0.0
    null_p = float(np.mean([na >= auc for na in null_aucs])) if null_aucs else 1.0

    return SufficiencyResult(
        contrast_name=contrast_name,
        dataset=dataset,
        auc_loso=auc,
        auc_ci_lower=ci_lower,
        auc_ci_upper=ci_upper,
        accuracy_loso=accuracy,
        per_subject_predictions=per_subject,
        n_subjects=n_subjects,
        n_features=features.shape[1],
        feature_names=feature_names or [],
        subject_consistency=consistency,
        null_auc_mean=null_mean,
        null_auc_std=null_std,
        null_auc_p=null_p,
        passes_threshold=auc >= 0.80,
    )


def analyze_geometric_structure(
    features: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    feature_names: list[str],
    seed: int = 42,
    n_bootstrap: int = 5000,
) -> StructureResult:
    """Multi-dimensional geometric state space analysis.

    Computes state centroids, pairwise Mahalanobis distances, per-subject
    state-change vectors, and angular separations with bootstrap CIs.

    Parameters
    ----------
    features : np.ndarray, shape (n_obs, n_features)
    labels : np.ndarray, shape (n_obs,), state labels (str)
    subject_ids : np.ndarray, shape (n_obs,)
    feature_names : list[str]
    seed : int
    n_bootstrap : int

    Returns
    -------
    StructureResult
    """
    unique_states = sorted(np.unique(labels))
    n_features = features.shape[1]

    centroids: dict[str, list[float]] = {}
    state_data: dict[str, np.ndarray] = {}
    for state in unique_states:
        mask = labels == state
        state_data[state] = features[mask]
        centroids[state] = features[mask].mean(axis=0).tolist()

    pooled_cov = np.zeros((n_features, n_features))
    total_n = 0
    for state in unique_states:
        n_s = len(state_data[state])
        if n_s > 1:
            pooled_cov += (n_s - 1) * np.cov(state_data[state], rowvar=False)
            total_n += n_s - 1
    if total_n > 0:
        pooled_cov /= total_n

    reg = 1e-6 * np.eye(n_features)
    pooled_cov_inv = np.linalg.inv(pooled_cov + reg)

    pairwise_mah: dict[str, float] = {}
    for i, s1 in enumerate(unique_states):
        for s2 in unique_states[i + 1:]:
            diff = np.array(centroids[s1]) - np.array(centroids[s2])
            d = float(np.sqrt(diff @ pooled_cov_inv @ diff))
            pairwise_mah[f"{s1}_vs_{s2}"] = d

    awake_label = None
    for candidate in ["awake", "W"]:
        if candidate in unique_states:
            awake_label = candidate
            break

    change_vectors: dict[str, list[float]] = {}
    angular_seps: dict[str, float] = {}
    angular_cis: dict[str, tuple[float, float]] = {}
    subj_consistency: dict[str, float] = {}

    if awake_label is not None:
        unique_subjs = np.unique(subject_ids)
        non_awake_states = [s for s in unique_states if s != awake_label]

        per_subject_deltas: dict[str, list[np.ndarray]] = {s: [] for s in non_awake_states}

        for subj in unique_subjs:
            subj_mask = subject_ids == subj
            awake_mask = subj_mask & (labels == awake_label)
            if not np.any(awake_mask):
                continue
            awake_feat = features[awake_mask].mean(axis=0)

            for state in non_awake_states:
                state_mask = subj_mask & (labels == state)
                if not np.any(state_mask):
                    continue
                state_feat = features[state_mask].mean(axis=0)
                delta = state_feat - awake_feat
                per_subject_deltas[state].append(delta)

        for state in non_awake_states:
            deltas = per_subject_deltas[state]
            if not deltas:
                continue
            deltas_arr = np.array(deltas)
            mean_delta = deltas_arr.mean(axis=0)
            change_vectors[state] = mean_delta.tolist()

            if n_features >= 2:
                mean_angle = np.arctan2(mean_delta[1], mean_delta[0])
                same_quadrant = 0
                for d in deltas:
                    angle = np.arctan2(d[1], d[0])
                    if abs(angle - mean_angle) < np.pi / 2:
                        same_quadrant += 1
                subj_consistency[state] = float(same_quadrant / len(deltas))

        rng = np.random.default_rng(seed)
        state_pairs = []
        for i, s1 in enumerate(non_awake_states):
            for s2 in non_awake_states[i + 1:]:
                if s1 in change_vectors and s2 in change_vectors:
                    state_pairs.append((s1, s2))

        for s1, s2 in state_pairs:
            v1 = np.array(change_vectors[s1][:2])
            v2 = np.array(change_vectors[s2][:2])
            angle = _angle_between_2d(v1, v2)
            angular_seps[f"{s1}_vs_{s2}"] = angle

            d1_arr = np.array(per_subject_deltas[s1])
            d2_arr = np.array(per_subject_deltas[s2])
            n1, n2 = len(d1_arr), len(d2_arr)

            boot_angles = []
            for _ in range(n_bootstrap):
                idx1 = rng.choice(n1, size=n1, replace=True)
                idx2 = rng.choice(n2, size=n2, replace=True)
                bv1 = d1_arr[idx1].mean(axis=0)[:2]
                bv2 = d2_arr[idx2].mean(axis=0)[:2]
                boot_angles.append(_angle_between_2d(bv1, bv2))

            if boot_angles:
                angular_cis[f"{s1}_vs_{s2}"] = (
                    float(np.percentile(boot_angles, 2.5)),
                    float(np.percentile(boot_angles, 97.5)),
                )

    passes = False
    if pairwise_mah:
        min_mah = min(pairwise_mah.values())
        passes = min_mah > 1.5

    return StructureResult(
        state_centroids=centroids,
        pairwise_mahalanobis=pairwise_mah,
        state_change_vectors=change_vectors,
        angular_separations=angular_seps,
        angular_separation_cis=angular_cis,
        subject_consistency=subj_consistency,
        passes_threshold=passes,
    )


def _angle_between_2d(v1: np.ndarray, v2: np.ndarray) -> float:
    """Unsigned angle between two 2D vectors in degrees.

    Returns value in [0, 90] because abs(cos) is used, collapsing
    anti-parallel vectors to 0 degrees.
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-15 or norm2 < 1e-15:
        return float("nan")
    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(abs(cos_angle))))


def compare_geometry_vs_power(
    geometry_features: np.ndarray,
    power_features: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    seed: int = 42,
    n_bootstrap: int = 1000,
) -> IncrementalValueResult:
    """Compare LOSO AUC of geometry-only, power-only, and combined models.

    Parameters
    ----------
    geometry_features : np.ndarray, shape (n_obs, n_geom)
    power_features : np.ndarray, shape (n_obs, n_power)
    labels : np.ndarray, shape (n_obs,), binary 0/1
    subject_ids : np.ndarray, shape (n_obs,)
    seed : int
    n_bootstrap : int

    Returns
    -------
    IncrementalValueResult
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score

    combined = np.hstack([geometry_features, power_features])

    def _loso_auc(X, y, subj_ids):
        unique_subj = np.unique(subj_ids)
        all_probs = np.zeros(len(y))
        for held_out in unique_subj:
            test_mask = subj_ids == held_out
            train_mask = ~test_mask
            Xtr, ytr = X[train_mask], y[train_mask]
            Xte = X[test_mask]
            if len(np.unique(ytr)) < 2:
                all_probs[test_mask] = 0.5
                continue
            tm, ts = Xtr.mean(0), Xtr.std(0)
            ts[ts == 0] = 1.0
            clf = LinearDiscriminantAnalysis(solver="svd")
            clf.fit((Xtr - tm) / ts, ytr)
            all_probs[test_mask] = clf.predict_proba((Xte - tm) / ts)[:, 1]
        return roc_auc_score(y, all_probs), all_probs

    auc_geom, _ = _loso_auc(geometry_features, labels, subject_ids)
    auc_pow, _ = _loso_auc(power_features, labels, subject_ids)
    auc_comb, _ = _loso_auc(combined, labels, subject_ids)

    rng = np.random.default_rng(seed)
    unique_subj = np.unique(subject_ids)
    boot_deltas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(unique_subj), size=len(unique_subj), replace=True)
        boot_subj = unique_subj[idx]
        boot_mask = np.isin(subject_ids, boot_subj)
        if len(np.unique(labels[boot_mask])) < 2:
            continue
        try:
            bg, _ = _loso_auc(geometry_features[boot_mask], labels[boot_mask],
                              subject_ids[boot_mask])
            bp, _ = _loso_auc(power_features[boot_mask], labels[boot_mask],
                              subject_ids[boot_mask])
            boot_deltas.append(bg - bp)
        except (ValueError, np.linalg.LinAlgError):
            continue

    ci = (float(np.percentile(boot_deltas, 2.5)), float(np.percentile(boot_deltas, 97.5))) \
        if boot_deltas else (float("nan"), float("nan"))

    rng2 = np.random.default_rng(seed + 999)
    null_labels = labels.copy()
    for subj in np.unique(subject_ids):
        subj_mask = subject_ids == subj
        subj_labels = null_labels[subj_mask]
        rng2.shuffle(subj_labels)
        null_labels[subj_mask] = subj_labels
    null_geom, _ = _loso_auc(geometry_features, null_labels, subject_ids)
    null_pow, _ = _loso_auc(power_features, null_labels, subject_ids)

    delta = auc_geom - auc_pow
    delta_comb = auc_comb - auc_pow
    passes = (delta >= 0.05) or (delta_comb >= 0.03)

    return IncrementalValueResult(
        auc_geometry_only=float(auc_geom),
        auc_power_only=float(auc_pow),
        auc_combined=float(auc_comb),
        delta_auc_vs_power=float(delta),
        delta_auc_combined_vs_power=float(delta_comb),
        bootstrap_ci_delta=ci,
        null_auc_geometry=float(null_geom),
        null_auc_power=float(null_pow),
        passes_threshold=passes,
    )


_PRIMARY_GEOMETRY_FEATURES = {"eigenvalue_gap", "condition_number", "spectral_radius"}


def check_orthogonality(
    geometry_features: np.ndarray,
    power_features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    seed: int = 42,
) -> OrthogonalityResult:
    """Test independence of geometry from spectral features.

    Correlations are computed on baseline-condition rows only (labels == 0)
    to avoid pseudo-replication when power features are constant within
    subject across conditions. Residualization uses all rows so that
    Cohen's d can be computed between conditions.

    Pass criterion (two-level rule):
    1. median |r| < 0.20 across geometry-power pairs (baseline only)
    2. At least 2 of 3 primary primitive geometry features (eigenvalue_gap,
       condition_number, spectral_radius) retain |d| >= 0.5 after
       regressing out power. ND score is a secondary composite and is
       excluded from the orthogonality gate.

    Parameters
    ----------
    geometry_features : np.ndarray, shape (n_obs, n_geom)
    power_features : np.ndarray, shape (n_obs, n_power)
    labels : np.ndarray, shape (n_obs,), binary 0/1
    feature_names : list[str]
    seed : int

    Returns
    -------
    OrthogonalityResult
    """
    n_geom = geometry_features.shape[1]
    n_power = power_features.shape[1]

    baseline_mask = labels == 0

    correlations: dict[str, float] = {}
    for g in range(n_geom):
        for p in range(n_power):
            key = feature_names[g] if g < len(feature_names) else f"geom_{g}"
            pwr_key = f"power_{p}" if n_power > 1 else "power"
            r, _ = sp_stats.pearsonr(
                geometry_features[baseline_mask, g],
                power_features[baseline_mask, p],
            )
            corr_key = f"{key}_vs_{pwr_key}" if n_power > 1 else key
            correlations[corr_key] = float(r)

    median_abs_r = float(np.median([abs(v) for v in correlations.values()]))

    residualized_d: dict[str, float] = {}
    unique_labels = np.unique(labels)
    if len(unique_labels) == 2:
        mask_a = labels == unique_labels[0]
        mask_b = labels == unique_labels[1]

        for g in range(n_geom):
            y = geometry_features[:, g]
            X = np.column_stack([power_features, np.ones(len(y))])
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                residuals = y - X @ beta
            except np.linalg.LinAlgError:
                residuals = y

            key = feature_names[g] if g < len(feature_names) else f"geom_{g}"
            residualized_d[key] = _cohens_d(residuals[mask_a], residuals[mask_b])

    n_primary_passing = 0
    if residualized_d:
        primary_ds = [
            abs(v) for k, v in residualized_d.items()
            if k in _PRIMARY_GEOMETRY_FEATURES and np.isfinite(v)
        ]
        n_primary_passing = sum(1 for d in primary_ds if d >= 0.5)

    d_criterion = n_primary_passing >= 2
    passes_strict = median_abs_r < 0.20 and d_criterion
    passes_sensitivity = median_abs_r < 0.30 and d_criterion

    if passes_strict:
        interpretation = "Geometry is independent of spectral power."
    elif passes_sensitivity:
        interpretation = (
            "Geometry shows partial spectral entanglement "
            "(driven by condition number), but state information "
            "remains robust after residualization."
        )
    else:
        interpretation = "Geometry is partially confounded with spectral power."

    return OrthogonalityResult(
        median_abs_correlation=median_abs_r,
        per_feature_correlations=correlations,
        residualized_effect_sizes=residualized_d,
        passes_threshold=passes_strict,
        passes_sensitivity_threshold=passes_sensitivity,
        n_primary_features_passing_d=n_primary_passing,
        interpretation=interpretation,
    )


def collate_existing_results(
    results: dict[str, dict],
) -> tuple[list[CollatedResult], list[CollatedResult], list[CollatedResult]]:
    """Extract pass/fail metrics from existing JSON results.

    Parameters
    ----------
    results : dict[str, dict]
        Keys are filenames (without path), values are parsed JSON dicts.
        Expected keys: "exceptional_points", "jackknife_sensitivity",
        "ep_shared_subspace_propofol", "ep_shared_subspace_sleep",
        "transient_amplification" (from summary_statistics.csv data),
        "amplification_propofol".

    Returns
    -------
    stability : list[CollatedResult]
        Test 5 results.
    criticality : list[CollatedResult]
        Test 6 results.
    amplification : list[CollatedResult]
        Test 8 results.
    """
    stability = []
    criticality = []
    amplification = []

    ss_prop = results.get("ep_shared_subspace_propofol")
    if ss_prop and "group_statistics" in ss_prop:
        gs = ss_prop["group_statistics"]
        comp = gs.get("comparison_to_original", {})
        for metric in ["spectral_radius", "min_eigenvalue_gap"]:
            if metric in comp:
                orig_d = comp[metric].get("original_d", 0)
                shared_d = comp[metric].get("shared_d", 0)
                if orig_d != 0:
                    shrinkage = 1.0 - abs(shared_d) / abs(orig_d)
                    same_sign = np.sign(orig_d) == np.sign(shared_d)
                    passes = same_sign and shrinkage < 0.25
                    stability.append(CollatedResult(
                        test_name="stability_propofol",
                        metric_name=f"shared_subspace_{metric}",
                        value=float(shared_d),
                        threshold=float(orig_d * 0.75),
                        passes=bool(passes),
                        details={
                            "original_d": float(orig_d),
                            "shared_d": float(shared_d),
                            "shrinkage_fraction": float(shrinkage),
                            "same_sign": bool(same_sign),
                        },
                    ))

    ss_sleep = results.get("ep_shared_subspace_sleep")
    if ss_sleep and "group_statistics" in ss_sleep:
        gs = ss_sleep["group_statistics"]
        comp = gs.get("comparison_to_original", {})
        for metric in ["min_gap_n3_vs_rem", "min_gap_awake_vs_rem"]:
            if metric in comp:
                orig_d = comp[metric].get("original_d", 0)
                shared_d = comp[metric].get("shared_d", 0)
                if orig_d != 0:
                    shrinkage = 1.0 - abs(shared_d) / abs(orig_d)
                    same_sign = np.sign(orig_d) == np.sign(shared_d)
                    passes = same_sign and shrinkage < 0.25
                    stability.append(CollatedResult(
                        test_name="stability_sleep",
                        metric_name=f"shared_subspace_{metric}",
                        value=float(shared_d),
                        threshold=float(orig_d * 0.75),
                        passes=bool(passes),
                        details={
                            "original_d": float(orig_d),
                            "shared_d": float(shared_d),
                            "shrinkage_fraction": float(shrinkage),
                            "same_sign": bool(same_sign),
                        },
                    ))

    ep = results.get("exceptional_points")
    jk = results.get("jackknife_sensitivity")
    if ep and "correlations" in ep:
        sigma_nd = ep["correlations"].get("sigma_vs_ep_score", {})
        r_val = sigma_nd.get("r", 0)
        criticality.append(CollatedResult(
            test_name="criticality_coupling",
            metric_name="sigma_vs_nd_score_r",
            value=float(r_val),
            threshold=0.80,
            passes=abs(r_val) >= 0.80,
            details={"p": sigma_nd.get("p", None), "n": sigma_nd.get("n", None)},
        ))

    if jk and "correlations" in jk:
        sigma_jk = jk["correlations"].get("sigma_vs_ep_score", {})
        jk_data = sigma_jk.get("jackknife", {})
        all_sig = jk_data.get("all_significant_at_0.05", False)
        r_min = jk_data.get("r_min", 0)
        criticality.append(CollatedResult(
            test_name="criticality_coupling",
            metric_name="sigma_vs_nd_jackknife_stable",
            value=float(r_min),
            threshold=0.70,
            passes=bool(all_sig) and abs(r_min) >= 0.70,
            details={
                "all_significant": bool(all_sig),
                "r_min": float(r_min),
                "r_max": float(jk_data.get("r_max", 0)),
                "most_influential": jk_data.get("most_influential_subject", ""),
            },
        ))

    amp_kn_r = results.get("condition_number_vs_kreiss_r")
    if amp_kn_r is not None:
        amplification.append(CollatedResult(
            test_name="amplification_link",
            metric_name="condition_number_vs_kreiss_r",
            value=float(amp_kn_r),
            threshold=0.80,
            passes=abs(amp_kn_r) >= 0.80,
            details={},
        ))

    return stability, criticality, amplification


def compute_overall_verdict(
    sufficiency: list[SufficiencyResult],
    incremental: IncrementalValueResult | None,
    orthogonality: OrthogonalityResult | None,
    structure: StructureResult,
    stability: list[CollatedResult],
    criticality: list[CollatedResult],
    amplification: list[CollatedResult],
    temporal: TemporalPrecedenceSummary | str | None = None,
) -> tuple[str, int, int]:
    """Compute overall verdict from individual test results.

    Returns
    -------
    verdict : str
        "strong", "complementary", or "insufficient".
    n_passed : int
    n_total : int
    """
    passes = []

    any_suff = any(s.passes_threshold for s in sufficiency)
    passes.append(any_suff)

    if incremental is not None:
        passes.append(incremental.passes_threshold)

    if orthogonality is not None:
        passes.append(orthogonality.passes_threshold)

    passes.append(structure.passes_threshold)

    if stability:
        passes.append(all(s.passes for s in stability))
    else:
        passes.append(False)

    if criticality:
        passes.append(all(c.passes for c in criticality))
    else:
        passes.append(False)

    if isinstance(temporal, TemporalPrecedenceSummary):
        passes.append(temporal.any_metric_passes_strict)

    if amplification:
        passes.append(all(a.passes for a in amplification))
    else:
        passes.append(False)

    n_passed = sum(passes)
    n_total = len(passes)

    if n_passed >= 6:
        verdict = "strong"
    elif n_passed >= 4:
        verdict = "complementary"
    else:
        verdict = "insufficient"

    return verdict, n_passed, n_total


def load_temporal_precedence_summary(
    temporal_data: dict | None,
) -> TemporalPrecedenceSummary | str:
    """Build TemporalPrecedenceSummary from temporal_precedence.json data.

    Parameters
    ----------
    temporal_data : dict or None
        Parsed temporal_precedence.json. If None or missing transitions,
        returns "DEFERRED".

    Returns
    -------
    TemporalPrecedenceSummary or "DEFERRED"
    """
    if temporal_data is None:
        return "DEFERRED"

    transitions = temporal_data.get("transitions", {})
    if not transitions:
        return "DEFERRED"

    best_p = 1.0
    best_metric = ""
    best_consistency = 0.0
    best_d = 0.0
    any_passes = temporal_data.get("overall_passes", False)
    any_passes_strict = temporal_data.get("overall_passes_strict", False)
    details = {}

    for ttype, metrics in transitions.items():
        for metric_name, r in metrics.items():
            p = r.get("slope_p_value", 1.0)
            details[f"{ttype}_{metric_name}"] = {
                "slope_p": p,
                "consistency": r.get("subject_consistency", 0),
                "early_late_d": r.get("early_vs_late_d", 0),
                "passes": r.get("passes_threshold", False),
                "passes_strict": r.get("passes_strict", False),
                "nonoverlap_survives": r.get("nonoverlap_survives", False),
            }
            if p < best_p:
                best_p = p
                best_metric = f"{ttype}_{metric_name}"
                best_consistency = r.get("subject_consistency", 0)
                best_d = r.get("early_vs_late_d", 0)

    return TemporalPrecedenceSummary(
        any_metric_passes=any_passes,
        any_metric_passes_strict=any_passes_strict,
        best_metric=best_metric,
        best_slope_p=best_p,
        best_consistency=best_consistency,
        best_early_late_d=best_d,
        details=details,
    )


def assemble_test_battery(
    sufficiency: list[SufficiencyResult],
    incremental: IncrementalValueResult | None,
    orthogonality: OrthogonalityResult | None,
    structure: StructureResult,
    stability: list[CollatedResult],
    criticality: list[CollatedResult],
    amplification: list[CollatedResult],
    temporal: TemporalPrecedenceSummary | str | None = None,
) -> GeometryTestBattery:
    """Assemble all test results into a single battery."""
    if temporal is None:
        temporal = "DEFERRED"

    verdict, n_passed, n_total = compute_overall_verdict(
        sufficiency, incremental, orthogonality, structure,
        stability, criticality, amplification,
        temporal=temporal,
    )

    return GeometryTestBattery(
        sufficiency=sufficiency,
        incremental=incremental,
        orthogonality=orthogonality,
        structure=structure,
        stability=stability,
        criticality_coupling=criticality,
        amplification_link=amplification,
        temporal_precedence=temporal,
        overall_verdict=verdict,
        n_tests_passed=n_passed,
        n_tests_total=n_total,
    )
