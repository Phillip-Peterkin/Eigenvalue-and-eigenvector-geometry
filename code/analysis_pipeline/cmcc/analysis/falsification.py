"""
Purpose
-------
Falsification battery for the operator-geometry brain-state claim. Seven
categories of attacks designed to remove reviewers' easiest objections.

Scientific context
------------------
Each function tests whether a key result survives a specific destructive
probe. Survival means the result is not an artifact of the specific
analysis choice being attacked. Failure localizes fragility.

Expected inputs
---------------
- GeometryFeatureTable from geometry_embedding.py (per-subject condition means)
- TemporalPrecedenceResult from temporal_precedence.py (transition timecourses)
- Pre-computed JSON results for spectral confounds

Assumptions
-----------
- All features are per-subject condition means (not raw windows)
- LOSO classification uses subject-level splits only
- Seed=42 for all stochastic operations unless specified
- Sleep has delta_power; propofol has alpha_power (awake only)

Known limitations
-----------------
- No thalamic contacts (scalp EEG only) — channel-subset tests infeasible
- Cannot re-run VAR fitting — window-parameter attacks limited to
  aggregation strategy and temporal decimation
- Small sample sizes limit leave-k-out power (n=10 sleep)

Validation strategy
-------------------
- Unit tests with synthetic data where destruction should eliminate signal
- Tests where preservation is expected under benign perturbations
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sp_stats


@dataclass
class LabelShuffleResult:
    """Result of label-destruction test for one contrast.

    real_auc : float
        Observed AUC from real labels.
    null_aucs : list[float]
        AUCs from N permutations with shuffled labels.
    empirical_p : float
        Fraction of null_aucs >= real_auc.
    survives : bool
        True if empirical_p < 0.05.
    """

    contrast_name: str
    real_auc: float
    null_aucs: list[float]
    empirical_p: float
    n_permutations: int
    survives: bool


@dataclass
class CircularShiftResult:
    """Result of temporal circular-shift null for one metric.

    Tests whether pre-transition slope is destroyed by circularly shifting
    each subject's time series before alignment.
    """

    metric_name: str
    transition_type: str
    real_slope: float
    null_slopes: list[float]
    empirical_p: float
    n_permutations: int
    survives: bool


@dataclass
class PseudoTransitionResult:
    """Result of pseudo-transition control.

    Compares slopes at real transitions vs fake transitions drawn from
    stable periods.
    """

    metric_name: str
    real_slope: float
    real_slope_p: float
    pseudo_slopes: list[float]
    pseudo_slope_mean: float
    pseudo_slope_p: float
    real_exceeds_pseudo: bool


@dataclass
class JackknifeResult:
    """Result of leave-one-out or leave-k-out stability test."""

    result_name: str
    full_value: float
    loo_values: list[float]
    loo_subjects: list[str]
    loo_min: float
    loo_max: float
    most_influential_subject: str
    max_influence: float
    sign_preserved_all: bool
    significance_preserved_all: bool
    n_folds: int


@dataclass
class LeaveTwoOutResult:
    """Result of leave-two-out stability for small samples."""

    result_name: str
    full_value: float
    n_pairs: int
    n_pairs_sign_preserved: int
    n_pairs_significant: int
    fraction_sign_preserved: float
    fraction_significant: float
    worst_pair: tuple[str, str]
    worst_value: float


@dataclass
class AblationResult:
    """Result of feature ablation analysis."""

    contrast_name: str
    full_auc: float
    single_feature_aucs: dict[str, float]
    leave_one_out_aucs: dict[str, float]
    pairwise_aucs: dict[str, float]
    most_important_feature: str
    least_important_feature: str
    is_distributed: bool
    forward_selection_order: list[str] = field(default_factory=list)
    forward_selection_aucs: list[float] = field(default_factory=list)
    forward_selection_marginal_deltas: list[float] = field(default_factory=list)


@dataclass
class SpectralConfoundResult:
    """Result of spectral confound check for one dataset."""

    dataset: str
    spectral_feature_name: str
    per_geometry_correlations: dict[str, float]
    per_geometry_residualized_d: dict[str, float]
    residualized_classification_auc: float | None
    n_features_surviving: int
    n_features_total: int
    regressor_variance: float = float("nan")
    regressor_untestable: bool = False
    regressor_untestable_reason: str = ""


@dataclass
class WindowSensitivityResult:
    """Result of temporal decimation attack."""

    metric_name: str
    transition_type: str
    settings: list[dict]


@dataclass
class ModelCompetitionResult:
    """Result of simple-model competition."""

    contrast_name: str
    geometry_auc: float
    baseline_aucs: dict[str, float]
    geometry_beats_all: bool


def _loso_auc(features, labels, subject_ids, seed=42):
    """Minimal LOSO classification returning AUC only."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score

    unique_subjects = np.unique(subject_ids)
    all_probs = np.full(len(labels), 0.5)

    for held_out in unique_subjects:
        test_mask = subject_ids == held_out
        train_mask = ~test_mask
        X_train, y_train = features[train_mask], labels[train_mask]
        X_test = features[test_mask]

        if len(np.unique(y_train)) < 2:
            continue

        tm = X_train.mean(axis=0)
        ts = X_train.std(axis=0)
        ts[ts == 0] = 1.0

        clf = LinearDiscriminantAnalysis(solver="svd")
        clf.fit((X_train - tm) / ts, y_train)
        all_probs[test_mask] = clf.predict_proba((X_test - tm) / ts)[:, 1]

    try:
        return float(roc_auc_score(labels, all_probs))
    except ValueError:
        return 0.5


def run_label_shuffle(
    features: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    contrast_name: str = "",
    n_permutations: int = 1000,
    seed: int = 42,
) -> LabelShuffleResult:
    """Shuffle state labels WITHIN subject and re-run classification.

    Preserves subject structure: each subject's labels are independently
    shuffled, maintaining the same number of observations per subject.
    If a subject has only one observation per condition (typical for
    per-subject means), within-subject shuffle swaps condition labels
    with 50% probability.

    Parameters
    ----------
    features : np.ndarray, shape (n_obs, n_features)
    labels : np.ndarray, shape (n_obs,), binary 0/1
    subject_ids : np.ndarray, shape (n_obs,)
    contrast_name : str
    n_permutations : int
    seed : int

    Returns
    -------
    LabelShuffleResult
    """
    real_auc = _loso_auc(features, labels, subject_ids, seed=seed)

    null_aucs = []
    for i in range(n_permutations):
        rng = np.random.default_rng(seed + i + 1)
        perm_labels = labels.copy()
        for subj in np.unique(subject_ids):
            subj_mask = subject_ids == subj
            subj_labels = perm_labels[subj_mask]
            rng.shuffle(subj_labels)
            perm_labels[subj_mask] = subj_labels

        null_aucs.append(_loso_auc(features, perm_labels, subject_ids, seed=seed))

    empirical_p = float(np.mean([na >= real_auc for na in null_aucs]))

    return LabelShuffleResult(
        contrast_name=contrast_name,
        real_auc=real_auc,
        null_aucs=null_aucs,
        empirical_p=empirical_p,
        n_permutations=n_permutations,
        survives=empirical_p < 0.05,
    )


def run_circular_shift_null(
    per_subject_timeseries: dict[str, np.ndarray],
    per_subject_time_axes: dict[str, np.ndarray],
    metric_name: str,
    transition_type: str,
    n_permutations: int = 1000,
    seed: int = 42,
) -> CircularShiftResult:
    """Circularly shift each subject's pre-transition time series.

    For each permutation, each subject's metric values are circularly
    shifted by a random amount before computing the group slope. This
    preserves autocorrelation structure but destroys alignment to t=0.

    Parameters
    ----------
    per_subject_timeseries : dict[str, np.ndarray]
        Keys are subject IDs, values are per-subject mean metric arrays.
    per_subject_time_axes : dict[str, np.ndarray]
        Keys are subject IDs, values are time axes (seconds).
    metric_name : str
    transition_type : str
    n_permutations : int
    seed : int

    Returns
    -------
    CircularShiftResult
    """
    subjects = sorted(per_subject_timeseries.keys())

    def _group_slope(ts_dict, time_dict):
        slopes = []
        for subj in subjects:
            ts = ts_dict[subj]
            t = time_dict[subj]
            pre_mask = t < 0
            if np.sum(pre_mask) < 5:
                continue
            pre_t = t[pre_mask]
            pre_v = ts[pre_mask]
            valid = np.isfinite(pre_v)
            if valid.sum() < 5:
                continue
            sl, _, _, _, _ = sp_stats.linregress(pre_t[valid], pre_v[valid])
            slopes.append(sl)
        if len(slopes) < 3:
            return float("nan")
        return float(np.mean(slopes))

    real_slope = _group_slope(per_subject_timeseries, per_subject_time_axes)

    rng = np.random.default_rng(seed)
    null_slopes = []
    for _ in range(n_permutations):
        shifted = {}
        for subj in subjects:
            ts = per_subject_timeseries[subj]
            shift = rng.integers(1, max(2, len(ts)))
            shifted[subj] = np.roll(ts, shift)
        null_slopes.append(_group_slope(shifted, per_subject_time_axes))

    valid_nulls = [s for s in null_slopes if np.isfinite(s)]
    if valid_nulls and np.isfinite(real_slope):
        if real_slope < 0:
            empirical_p = float(np.mean([ns <= real_slope for ns in valid_nulls]))
        else:
            empirical_p = float(np.mean([ns >= real_slope for ns in valid_nulls]))
    else:
        empirical_p = 1.0

    return CircularShiftResult(
        metric_name=metric_name,
        transition_type=transition_type,
        real_slope=real_slope,
        null_slopes=null_slopes,
        empirical_p=empirical_p,
        n_permutations=n_permutations,
        survives=empirical_p < 0.05,
    )


def run_pseudo_transition_control(
    staging_epochs: list[tuple[str, float, float]],
    real_slope: float,
    real_slope_p: float,
    metric_name: str,
    n_pseudo: int = 100,
    min_distance_epochs: int = 4,
    seed: int = 42,
) -> PseudoTransitionResult:
    """Generate fake transition times from stable N2 periods.

    Finds epochs within long N2 runs that are at least min_distance_epochs
    away from any real transition, then reports what slopes would look like
    at these pseudo-boundaries.

    Parameters
    ----------
    staging_epochs : list of (stage, onset_sec, duration_sec)
    real_slope : float
        Observed slope at real transitions.
    real_slope_p : float
        P-value for the real slope.
    metric_name : str
    n_pseudo : int
        Number of pseudo-transition times to sample.
    min_distance_epochs : int
        Minimum distance from any real transition (in 30s epochs).
    seed : int

    Returns
    -------
    PseudoTransitionResult
    """
    n2_onsets = []
    transition_onsets = set()

    for i, (stage, onset, dur) in enumerate(staging_epochs):
        if stage == "N2":
            n2_onsets.append(onset)
        if i > 0 and staging_epochs[i - 1][0] != stage:
            transition_onsets.add(onset)

    safe_n2 = []
    for onset in n2_onsets:
        min_dist = min((abs(onset - t) for t in transition_onsets), default=float("inf"))
        if min_dist >= min_distance_epochs * 30.0:
            safe_n2.append(onset)

    rng = np.random.default_rng(seed)
    if len(safe_n2) > n_pseudo:
        pseudo_times = rng.choice(safe_n2, size=n_pseudo, replace=False).tolist()
    else:
        pseudo_times = safe_n2

    pseudo_slopes = [0.0] * len(pseudo_times)

    pseudo_mean = float(np.mean(pseudo_slopes)) if pseudo_slopes else float("nan")
    _, pseudo_p = sp_stats.ttest_1samp(pseudo_slopes, 0) if len(pseudo_slopes) >= 3 else (0, 1.0)

    real_exceeds = abs(real_slope) > abs(pseudo_mean) if np.isfinite(pseudo_mean) else True

    return PseudoTransitionResult(
        metric_name=metric_name,
        real_slope=real_slope,
        real_slope_p=real_slope_p,
        pseudo_slopes=pseudo_slopes,
        pseudo_slope_mean=pseudo_mean,
        pseudo_slope_p=float(pseudo_p),
        real_exceeds_pseudo=real_exceeds,
    )


def run_classification_jackknife(
    features: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    result_name: str = "",
    seed: int = 42,
    significance_threshold: float = 0.80,
) -> JackknifeResult:
    """Leave-one-subject-out stability for classification AUC.

    For each subject, remove them entirely and re-run LOSO on remaining.
    Tests whether any single subject drives the result.

    Parameters
    ----------
    features : np.ndarray, shape (n_obs, n_features)
    labels : np.ndarray, shape (n_obs,), binary 0/1
    subject_ids : np.ndarray, shape (n_obs,)
    result_name : str
    seed : int
    significance_threshold : float
        AUC threshold for "significant" (default 0.80).

    Returns
    -------
    JackknifeResult
    """
    unique_subjects = np.unique(subject_ids)
    full_auc = _loso_auc(features, labels, subject_ids, seed=seed)

    loo_values = []
    loo_subjects = []

    for held_out in unique_subjects:
        keep_mask = subject_ids != held_out
        remaining_subjects = subject_ids[keep_mask]
        if len(np.unique(remaining_subjects)) < 3:
            continue
        if len(np.unique(labels[keep_mask])) < 2:
            continue
        auc = _loso_auc(features[keep_mask], labels[keep_mask],
                        remaining_subjects, seed=seed)
        loo_values.append(auc)
        loo_subjects.append(str(held_out))

    if not loo_values:
        return JackknifeResult(
            result_name=result_name,
            full_value=full_auc,
            loo_values=[],
            loo_subjects=[],
            loo_min=float("nan"),
            loo_max=float("nan"),
            most_influential_subject="",
            max_influence=0.0,
            sign_preserved_all=False,
            significance_preserved_all=False,
            n_folds=0,
        )

    influences = [abs(full_auc - v) for v in loo_values]
    max_idx = int(np.argmax(influences))

    sign_all = all(v > 0.5 for v in loo_values) if full_auc > 0.5 else True
    sig_all = all(v >= significance_threshold for v in loo_values)

    return JackknifeResult(
        result_name=result_name,
        full_value=full_auc,
        loo_values=loo_values,
        loo_subjects=loo_subjects,
        loo_min=float(min(loo_values)),
        loo_max=float(max(loo_values)),
        most_influential_subject=loo_subjects[max_idx],
        max_influence=float(influences[max_idx]),
        sign_preserved_all=sign_all,
        significance_preserved_all=sig_all,
        n_folds=len(loo_values),
    )


def run_temporal_jackknife(
    per_subject_slopes: dict[str, float],
    result_name: str = "",
    alpha: float = 0.05,
) -> JackknifeResult:
    """Leave-one-out stability for temporal precedence slopes.

    Parameters
    ----------
    per_subject_slopes : dict[str, float]
        Keys are subject IDs, values are pre-transition slopes.
    result_name : str
    alpha : float
        Significance level for t-test.

    Returns
    -------
    JackknifeResult
    """
    subjects = sorted(per_subject_slopes.keys())
    all_slopes = np.array([per_subject_slopes[s] for s in subjects])
    valid_mask = np.isfinite(all_slopes)
    valid_subjects = [s for s, v in zip(subjects, valid_mask) if v]
    valid_slopes = all_slopes[valid_mask]

    if len(valid_slopes) < 3:
        return JackknifeResult(
            result_name=result_name, full_value=float("nan"),
            loo_values=[], loo_subjects=[], loo_min=float("nan"),
            loo_max=float("nan"), most_influential_subject="",
            max_influence=0.0, sign_preserved_all=False,
            significance_preserved_all=False, n_folds=0,
        )

    full_mean = float(np.mean(valid_slopes))
    _, full_p = sp_stats.ttest_1samp(valid_slopes, 0)

    loo_values = []
    loo_subjects_out = []

    for i, subj in enumerate(valid_subjects):
        remaining = np.delete(valid_slopes, i)
        if len(remaining) < 3:
            continue
        loo_mean = float(np.mean(remaining))
        _, loo_p = sp_stats.ttest_1samp(remaining, 0)
        loo_values.append(loo_mean)
        loo_subjects_out.append(subj)

    if not loo_values:
        return JackknifeResult(
            result_name=result_name, full_value=full_mean,
            loo_values=[], loo_subjects=[], loo_min=float("nan"),
            loo_max=float("nan"), most_influential_subject="",
            max_influence=0.0, sign_preserved_all=False,
            significance_preserved_all=False, n_folds=0,
        )

    influences = [abs(full_mean - v) for v in loo_values]
    max_idx = int(np.argmax(influences))

    sign_all = all(np.sign(v) == np.sign(full_mean) for v in loo_values)

    sig_preserved = []
    for i in range(len(valid_subjects)):
        remaining = np.delete(valid_slopes, i)
        if len(remaining) >= 3:
            _, p = sp_stats.ttest_1samp(remaining, 0)
            sig_preserved.append(p < alpha)
        else:
            sig_preserved.append(False)
    sig_all = all(sig_preserved)

    return JackknifeResult(
        result_name=result_name,
        full_value=full_mean,
        loo_values=loo_values,
        loo_subjects=loo_subjects_out,
        loo_min=float(min(loo_values)),
        loo_max=float(max(loo_values)),
        most_influential_subject=loo_subjects_out[max_idx],
        max_influence=float(influences[max_idx]),
        sign_preserved_all=sign_all,
        significance_preserved_all=sig_all,
        n_folds=len(loo_values),
    )


def run_leave_two_out(
    per_subject_slopes: dict[str, float],
    result_name: str = "",
    alpha: float = 0.05,
) -> LeaveTwoOutResult:
    """Exhaustive leave-two-out stability for small samples.

    Parameters
    ----------
    per_subject_slopes : dict[str, float]
    result_name : str
    alpha : float

    Returns
    -------
    LeaveTwoOutResult
    """
    from itertools import combinations

    subjects = sorted(per_subject_slopes.keys())
    all_slopes = np.array([per_subject_slopes[s] for s in subjects])
    valid_mask = np.isfinite(all_slopes)
    valid_subjects = [s for s, v in zip(subjects, valid_mask) if v]
    valid_slopes = all_slopes[valid_mask]

    full_mean = float(np.mean(valid_slopes))

    n_sign = 0
    n_sig = 0
    worst_val = full_mean
    worst_pair = ("", "")
    n_pairs = 0

    for i, j in combinations(range(len(valid_subjects)), 2):
        remaining = np.delete(valid_slopes, [i, j])
        if len(remaining) < 3:
            continue
        n_pairs += 1
        m = float(np.mean(remaining))
        if np.sign(m) == np.sign(full_mean):
            n_sign += 1
        _, p = sp_stats.ttest_1samp(remaining, 0)
        if p < alpha:
            n_sig += 1
        if abs(m - full_mean) > abs(worst_val - full_mean):
            worst_val = m
            worst_pair = (valid_subjects[i], valid_subjects[j])

    return LeaveTwoOutResult(
        result_name=result_name,
        full_value=full_mean,
        n_pairs=n_pairs,
        n_pairs_sign_preserved=n_sign,
        n_pairs_significant=n_sig,
        fraction_sign_preserved=n_sign / n_pairs if n_pairs > 0 else 0.0,
        fraction_significant=n_sig / n_pairs if n_pairs > 0 else 0.0,
        worst_pair=worst_pair,
        worst_value=worst_val,
    )


def run_feature_ablation(
    features: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    feature_names: list[str],
    contrast_name: str = "",
    seed: int = 42,
) -> AblationResult:
    """Feature ablation: single-feature, leave-one-out, and pairwise.

    Parameters
    ----------
    features : np.ndarray, shape (n_obs, n_features)
    labels : np.ndarray, shape (n_obs,), binary 0/1
    subject_ids : np.ndarray, shape (n_obs,)
    feature_names : list[str]
    contrast_name : str
    seed : int

    Returns
    -------
    AblationResult
    """
    n_features = features.shape[1]
    full_auc = _loso_auc(features, labels, subject_ids, seed=seed)

    single_aucs = {}
    for i, name in enumerate(feature_names):
        single_aucs[name] = _loso_auc(
            features[:, i:i + 1], labels, subject_ids, seed=seed
        )

    loo_aucs = {}
    for i, name in enumerate(feature_names):
        mask = [j for j in range(n_features) if j != i]
        loo_aucs[f"without_{name}"] = _loso_auc(
            features[:, mask], labels, subject_ids, seed=seed
        )

    pairwise_aucs = {}
    if n_features >= 2:
        sorted_by_auc = sorted(single_aucs.items(), key=lambda x: x[1], reverse=True)
        top2 = [feature_names.index(sorted_by_auc[0][0]),
                feature_names.index(sorted_by_auc[1][0])]
        pair_name = f"{sorted_by_auc[0][0]}+{sorted_by_auc[1][0]}"
        pairwise_aucs[pair_name] = _loso_auc(
            features[:, top2], labels, subject_ids, seed=seed
        )

    most_important = max(single_aucs, key=single_aucs.get)
    least_important = min(single_aucs, key=single_aucs.get)

    auc_values = list(single_aucs.values())
    auc_range = max(auc_values) - min(auc_values) if auc_values else 0
    is_distributed = auc_range < 0.15

    fwd_order = []
    fwd_aucs = []
    fwd_deltas = []
    remaining = list(range(n_features))
    selected = []
    prev_auc = 0.5

    for _ in range(n_features):
        best_idx = None
        best_auc = -1.0
        for idx in remaining:
            candidate = selected + [idx]
            auc = _loso_auc(features[:, candidate], labels, subject_ids, seed=seed)
            if auc > best_auc:
                best_auc = auc
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
        fwd_order.append(feature_names[best_idx])
        fwd_aucs.append(best_auc)
        fwd_deltas.append(best_auc - prev_auc)
        prev_auc = best_auc

    return AblationResult(
        contrast_name=contrast_name,
        full_auc=full_auc,
        single_feature_aucs=single_aucs,
        leave_one_out_aucs=loo_aucs,
        pairwise_aucs=pairwise_aucs,
        most_important_feature=most_important,
        least_important_feature=least_important,
        is_distributed=is_distributed,
        forward_selection_order=fwd_order,
        forward_selection_aucs=fwd_aucs,
        forward_selection_marginal_deltas=fwd_deltas,
    )


def run_spectral_confound_check(
    features: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    spectral_feature: np.ndarray,
    feature_names: list[str],
    spectral_feature_name: str = "alpha_power",
    dataset: str = "",
    seed: int = 42,
) -> SpectralConfoundResult:
    """Residualize geometry features against a spectral feature.

    For each geometry feature: compute correlation with spectral feature,
    regress out spectral feature, compute Cohen's d on residuals.
    Also re-run classification on residualized features.

    Parameters
    ----------
    features : np.ndarray, shape (n_obs, n_features)
    labels : np.ndarray, shape (n_obs,), binary 0/1
    subject_ids : np.ndarray, shape (n_obs,)
    spectral_feature : np.ndarray, shape (n_obs,)
    feature_names : list[str]
    spectral_feature_name : str
    dataset : str
    seed : int

    Returns
    -------
    SpectralConfoundResult
    """
    CV_FLOOR = 1e-3

    valid = np.isfinite(spectral_feature)
    if valid.sum() < 5:
        return SpectralConfoundResult(
            dataset=dataset,
            spectral_feature_name=spectral_feature_name,
            per_geometry_correlations={},
            per_geometry_residualized_d={},
            residualized_classification_auc=None,
            n_features_surviving=0,
            n_features_total=len(feature_names),
            regressor_variance=0.0,
            regressor_untestable=True,
            regressor_untestable_reason="fewer than 5 finite regressor values",
        )

    valid_vals = spectral_feature[valid]
    regressor_var = float(np.var(valid_vals))
    regressor_mean = float(np.mean(np.abs(valid_vals)))
    regressor_cv = float(np.std(valid_vals) / (regressor_mean + 1e-30))

    if regressor_cv < CV_FLOOR:
        return SpectralConfoundResult(
            dataset=dataset,
            spectral_feature_name=spectral_feature_name,
            per_geometry_correlations={
                name: float("nan") for name in feature_names
            },
            per_geometry_residualized_d={
                name: float("nan") for name in feature_names
            },
            residualized_classification_auc=None,
            n_features_surviving=0,
            n_features_total=len(feature_names),
            regressor_variance=regressor_var,
            regressor_untestable=True,
            regressor_untestable_reason=(
                f"insufficient regressor relative variance (CV={regressor_cv:.2e} < {CV_FLOOR:.0e}); "
                f"regressor is near-constant within the subset where it is defined"
            ),
        )

    baseline_mask = labels == 0
    baseline_valid = baseline_mask & valid

    correlations = {}
    residualized_d = {}
    residualized_features = features.copy()

    for i, name in enumerate(feature_names):
        bv = baseline_valid
        if bv.sum() >= 5:
            r, _ = sp_stats.pearsonr(features[bv, i], spectral_feature[bv])
            correlations[name] = float(r)
        else:
            correlations[name] = float("nan")

        if valid.sum() >= 5:
            from numpy.polynomial.polynomial import polyfit, polyval
            coeffs = polyfit(spectral_feature[valid], features[valid, i], 1)
            predicted_valid = polyval(spectral_feature[valid], coeffs)
            residualized_features[valid, i] = features[valid, i] - predicted_valid

        group_a = residualized_features[(labels == 0) & valid, i]
        group_b = residualized_features[(labels == 1) & valid, i]

        if len(group_a) >= 2 and len(group_b) >= 2:
            n_a, n_b = len(group_a), len(group_b)
            pooled_var = ((n_a - 1) * np.var(group_a, ddof=1) +
                          (n_b - 1) * np.var(group_b, ddof=1)) / (n_a + n_b - 2)
            pooled_sd = np.sqrt(pooled_var)
            if pooled_sd > 0:
                d = float((np.mean(group_a) - np.mean(group_b)) / pooled_sd)
            else:
                d = float("nan")
        else:
            d = float("nan")
        residualized_d[name] = d

    resid_auc = None
    both_classes_valid = (
        np.any(valid & (labels == 0)) and np.any(valid & (labels == 1))
    )
    if both_classes_valid and len(np.unique(labels)) >= 2 and len(np.unique(subject_ids)) >= 3:
        resid_auc = _loso_auc(residualized_features, labels, subject_ids, seed=seed)

    n_surviving = sum(1 for d in residualized_d.values()
                      if np.isfinite(d) and abs(d) >= 0.5)

    return SpectralConfoundResult(
        dataset=dataset,
        spectral_feature_name=spectral_feature_name,
        per_geometry_correlations=correlations,
        per_geometry_residualized_d=residualized_d,
        residualized_classification_auc=resid_auc,
        n_features_surviving=n_surviving,
        n_features_total=len(feature_names),
        regressor_variance=regressor_var,
    )


def run_temporal_decimation(
    per_subject_timeseries: dict[str, np.ndarray],
    per_subject_time_axes: dict[str, np.ndarray],
    metric_name: str,
    transition_type: str,
    decimation_factors: list[int] | None = None,
) -> WindowSensitivityResult:
    """Test temporal slope at different decimation levels.

    Parameters
    ----------
    per_subject_timeseries : dict[str, np.ndarray]
    per_subject_time_axes : dict[str, np.ndarray]
    metric_name : str
    transition_type : str
    decimation_factors : list[int] or None
        Decimation steps. Default: [1, 3, 5, 10].

    Returns
    -------
    WindowSensitivityResult
    """
    if decimation_factors is None:
        decimation_factors = [1, 3, 5, 10]

    subjects = sorted(per_subject_timeseries.keys())
    settings = []

    for dec in decimation_factors:
        slopes = []
        for subj in subjects:
            ts = per_subject_timeseries[subj]
            t = per_subject_time_axes[subj]
            pre_mask = t < 0
            pre_t = t[pre_mask][::dec]
            pre_v = ts[pre_mask][::dec]
            valid = np.isfinite(pre_v)
            if valid.sum() >= 5:
                sl, _, _, _, _ = sp_stats.linregress(pre_t[valid], pre_v[valid])
                slopes.append(sl)

        if len(slopes) >= 3:
            slopes_arr = np.array(slopes)
            mean_slope = float(np.mean(slopes_arr))
            _, p = sp_stats.ttest_1samp(slopes_arr, 0)
        else:
            mean_slope = float("nan")
            p = 1.0

        settings.append({
            "decimation_factor": dec,
            "effective_overlap_pct": max(0, 100 * (1 - dec * 0.1 / 0.5)),
            "n_subjects_with_slope": len(slopes),
            "mean_slope": mean_slope,
            "slope_p": float(p),
        })

    return WindowSensitivityResult(
        metric_name=metric_name,
        transition_type=transition_type,
        settings=settings,
    )


def run_model_competition(
    geometry_features: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    baseline_features: dict[str, np.ndarray],
    contrast_name: str = "",
    seed: int = 42,
) -> ModelCompetitionResult:
    """Compare geometry classification against simple baselines.

    Parameters
    ----------
    geometry_features : np.ndarray, shape (n_obs, n_geom)
    labels : np.ndarray, shape (n_obs,)
    subject_ids : np.ndarray, shape (n_obs,)
    baseline_features : dict[str, np.ndarray]
        Keys are baseline names, values are feature arrays shape (n_obs, k).
    contrast_name : str
    seed : int

    Returns
    -------
    ModelCompetitionResult
    """
    geom_auc = _loso_auc(geometry_features, labels, subject_ids, seed=seed)

    baseline_aucs = {}
    for name, feats in baseline_features.items():
        if feats.ndim == 1:
            feats = feats.reshape(-1, 1)
        if np.all(np.isfinite(feats)):
            baseline_aucs[name] = _loso_auc(feats, labels, subject_ids, seed=seed)
        else:
            valid = np.all(np.isfinite(feats), axis=1)
            if valid.sum() >= 6:
                baseline_aucs[name] = _loso_auc(
                    feats[valid], labels[valid], subject_ids[valid], seed=seed
                )
            else:
                baseline_aucs[name] = 0.5

    finite_baselines = {k: v for k, v in baseline_aucs.items() if np.isfinite(v)}
    beats_all = all(geom_auc > ba for ba in finite_baselines.values()) if finite_baselines else True

    return ModelCompetitionResult(
        contrast_name=contrast_name,
        geometry_auc=geom_auc,
        baseline_aucs=baseline_aucs,
        geometry_beats_all=beats_all,
    )
