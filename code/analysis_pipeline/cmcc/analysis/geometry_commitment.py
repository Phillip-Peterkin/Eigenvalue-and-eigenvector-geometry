"""
Geometric Transition Commitment Analysis (GCA).

Purpose
-------
Tests whether operator geometry shows an identifiable commitment moment
before N2->N3 sleep transitions -- a statistical point within N2 at which
geometry irreversibly begins drifting toward N3 values.

Scientific context
------------------
Extends the temporal precedence analysis (temporal_precedence.py) by:
  Q1: Testing whether the 120-s pre-transition window captures the full drift
      or only its tail (extended window to 600 s).
  Q2: Detecting the changepoint within N2 at which slope becomes non-zero.
  Q3: Testing whether N2 geometry is unimodal or bistable.
  Q4: Testing whether geometry drift direction predicts N3 vs REM outcome.
  Q5: Synthesizing into a "geometric commitment window" characterization.

Expected inputs / units
-----------------------
- Timecourse arrays: float, geometry metric values (unitless)
- Time axes: float, seconds relative to transition (t=0 = staging boundary)
- Subject IDs: str
- Stage labels: str ("N2", "N3", "R", etc.)

Assumptions
-----------
- Unit of observation is the subject (per-subject mean before group stats)
- Multiple transitions per subject are averaged within subject
- 30-s epoch staging resolution introduces +/-15 s timing uncertainty
- PCA fitted per segment (enforced by calling code, not this library)

Potential confounds
-------------------
- Global slow drift in EEG across the night may produce apparent slopes
  unrelated to transition-specific geometry changes.
- Subjects with few qualifying transitions have noisier per-subject means.
- Class imbalance in N2->N3 vs N2->REM (62 vs 37 transitions) affects Q4.

Validation strategy
-------------------
- Unit tests with synthetic TransitionTimecourse objects (no raw data needed)
- Property tests on output invariants (CI containment, range constraints)
- Integration test using actual temporal_precedence.json

Known limitations
-----------------
- n=10 subjects -> low power for all group tests
- Changepoint detection is grid-search piecewise OLS (no PELT)
- Bistability test is exploratory (Hartigan dip + GMM; not pre-registered)
- Q1/Q2 require raw EEG (SLEEP_DATA_ROOT); Q3/Q4/Q5 use existing JSONs
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


@dataclass
class ExtendedWindowResult:
    """Result of Q1: extended drift window analysis.

    Attributes
    ----------
    metric_name : str
        e.g., "spectral_radius"
    n_qualifying_transitions : int
        Transitions with sufficient pre-transition coverage.
    n_subjects : int
        Subjects with at least one qualifying transition.
    mean_slope : float
        Group mean OLS slope over pre-transition segment (units/sec).
    slope_p_value : float
        One-sample t-test p-value (H0: mean slope = 0).
    slope_ci : tuple[float, float]
        Bootstrap 95% CI on mean slope.
    cohens_d : float
        Paired Cohen's d: early (-pre_sec to -pre_sec/2) vs late (-120 to 0 s).
    subject_consistency : float
        Fraction of subjects with slope in same direction as group mean.
    linear_aic : float
        AIC of linear fit to group-mean pre-transition trajectory.
    quadratic_aic : float
        AIC of quadratic fit (tests for acceleration).
    aic_delta : float
        quadratic_aic - linear_aic. Negative = quadratic fits better (acceleration).
    group_time_axis : np.ndarray
        Common time axis (seconds, relative to transition).
    group_mean_trajectory : np.ndarray
        Cross-subject mean trajectory on group_time_axis.
    group_ci_lower : np.ndarray
        Bootstrap 2.5th percentile of group mean.
    group_ci_upper : np.ndarray
        Bootstrap 97.5th percentile of group mean.
    passes_threshold : bool
        slope_p_value < 0.05 AND subject_consistency >= 0.70 AND |cohens_d| >= 0.5
    """

    metric_name: str
    n_qualifying_transitions: int
    n_subjects: int
    mean_slope: float
    slope_p_value: float
    slope_ci: tuple[float, float]
    cohens_d: float
    subject_consistency: float
    linear_aic: float
    quadratic_aic: float
    aic_delta: float
    group_time_axis: np.ndarray
    group_mean_trajectory: np.ndarray
    group_ci_lower: np.ndarray
    group_ci_upper: np.ndarray
    passes_threshold: bool
    data_constraint: dict = field(default_factory=dict)


@dataclass
class ChangepointResult:
    """Result of Q2: changepoint detection within N2 episodes.

    Attributes
    ----------
    metric_name : str
    n_qualifying_transitions : int
        Transitions with sufficient N2 coverage for changepoint detection.
    n_subjects : int
    per_transition_latencies_sec : list[float]
        Detected changepoint time for each transition (seconds, negative = before boundary).
        NaN if detection failed for that transition.
    per_subject_mean_latencies_sec : list[float]
        Per-subject mean of per_transition_latencies_sec.
    group_mean_latency_sec : float
        Group mean of per_subject_mean_latencies_sec.
    group_latency_p_value : float
        One-sample t-test p-value (H0: mean latency = 0).
    group_latency_ci : tuple[float, float]
        Bootstrap 95% CI on group mean latency.
    latency_sd_sec : float
        SD of per-subject mean latencies.
    latency_hist_60s_bins : list[int]
        Count of transitions in each 60-s bin from -600 to 0.
    latency_hist_bin_edges : list[float]
        Bin edges (length = len(latency_hist_60s_bins) + 1).
    fraction_within_300s : float
        Fraction of detected changepoints in the -300 to 0 s window.
    penalty_sensitivity : dict[str, dict]
        Sensitivity results keyed by "min_seg_{N}s" for each penalty value.
    """

    metric_name: str
    n_qualifying_transitions: int
    n_subjects: int
    per_transition_latencies_sec: list[float]
    per_subject_mean_latencies_sec: list[float]
    group_mean_latency_sec: float
    group_latency_p_value: float
    group_latency_ci: tuple[float, float]
    latency_sd_sec: float
    latency_hist_60s_bins: list[int]
    latency_hist_bin_edges: list[float]
    fraction_within_300s: float
    penalty_sensitivity: dict = field(default_factory=dict)
    edge_proximity_ratio: float = float("nan")
    data_constraint: dict = field(default_factory=dict)
    window_truncation_sensitivity: dict = field(default_factory=dict)


@dataclass
class BistabilityResult:
    """Result of Q3: N2 geometry bistability test.

    Attributes
    ----------
    n_early_windows : int
        Total windows classified as "early N2" (far from transitions).
    n_late_windows : int
        Total windows classified as "late N2" (approaching N3).
    dip_test_statistic : float
        Hartigan dip statistic on pooled 1D projection.
    dip_test_p : float
        Bootstrap p-value under uniform null.
    gmm_bic_1 : float
        BIC for 1-component GMM.
    gmm_bic_2 : float
        BIC for 2-component GMM.
    gmm_bic_delta : float
        gmm_bic_2 - gmm_bic_1. Negative = 2 components fits better.
    early_vs_late_d_spectral_radius : float
        Paired Cohen's d (late - early) for spectral_radius.
    early_vs_late_d_eigenvalue_gap : float
        Paired Cohen's d (late - early) for eigenvalue_gap.
    hotelling_p : float or None
        Hotelling's T^2 p-value; None if pingouin unavailable.
    pca_coords_early : np.ndarray
        2D PCA coordinates of early windows, shape (n_early, 2).
    pca_coords_late : np.ndarray
        2D PCA coordinates of late windows, shape (n_late, 2).
    gmm_means : np.ndarray
        GMM component means, shape (2, n_metrics).
    gmm_covariances : np.ndarray
        GMM component covariances, shape (2, n_metrics, n_metrics).
    """

    n_early_windows: int
    n_late_windows: int
    dip_test_statistic: float
    dip_test_p: float
    bc_gaussian_null_p: float = float("nan")
    silverman_p: float = float("nan")
    gmm_bic_1: float = float("nan")
    gmm_bic_2: float = float("nan")
    gmm_bic_delta: float = float("nan")
    gmm_cv_ll_delta: float = float("nan")
    early_vs_late_d_spectral_radius: float = float("nan")
    early_vs_late_d_eigenvalue_gap: float = float("nan")
    hotelling_p: float | None = None
    pca_coords_early: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    pca_coords_late: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    gmm_means: np.ndarray = field(default_factory=lambda: np.full((2, 1), np.nan))
    gmm_covariances: np.ndarray = field(default_factory=lambda: np.full((2, 1, 1), np.nan))
    bc_note: str = ""


@dataclass
class TrajectoryPredictionResult:
    """Result of Q4: trajectory direction predicts upcoming sleep state.

    Attributes
    ----------
    metrics : list[str]
        Metric names used as features.
    time_bins_sec : list[float]
        Pre-transition latencies evaluated (seconds, negative).
    per_bin_cohens_d : dict[str, list[float]]
        metric_name -> Cohen's d (N3-bound minus REM-bound slope) at each time bin.
    loso_auc : float
        LOSO logistic regression AUC (chance = 0.5).
    loso_balanced_accuracy : float
        LOSO balanced accuracy.
    permutation_null_auc : list[float]
        1000-sample permutation null AUC distribution.
    permutation_p : float
        Fraction of null >= observed AUC.
    n_n3_bound : int
        Number of subjects with N2->N3 transitions.
    n_rem_bound : int
        Number of subjects with N2->REM transitions.
    """

    metrics: list[str]
    time_bins_sec: list[float]
    per_bin_cohens_d: dict[str, list[float]]
    loso_auc: float
    loso_balanced_accuracy: float
    permutation_null_auc: list[float]
    permutation_p: float
    n_n3_bound: int
    n_rem_bound: int
    per_fold_predictions: list[tuple] = field(default_factory=list)
    feature_scale_ratio: float = float("nan")
    n_convergence_failures: int = 0
    majority_baseline_auc: float = float("nan")
    per_bin_loso_auc: list[float] = field(default_factory=list)
    per_subject_slopes: dict = field(default_factory=dict)


@dataclass
class CommitmentWindowResult:
    """Result of Q5: commitment window characterization.

    Attributes
    ----------
    discrimination_onset_sec : float or None
        Earliest pre-transition latency where Cohen's d >= primary threshold
        for at least one metric. None if threshold never reached.
    commitment_window_sec : float or None
        abs(discrimination_onset_sec). Duration of detectable commitment window.
    geometry_at_onset : dict[str, float]
        Mean geometry metric values at the discrimination onset time bin.
    inter_subject_sd_latency_sec : float or None
        SD of per-subject changepoint latencies (from Q2). None if Q2 skipped.
    changepoint_overlap : bool or None
        Whether changepoint CI overlaps with discrimination_onset +/- 60 s.
    thresholds_tested : list[float]
        Cohen's d thresholds evaluated.
    onset_by_threshold : dict[float, float | None]
        threshold -> detected onset latency (or None if not reached).
    """

    discrimination_onset_sec: float | None
    commitment_window_sec: float | None
    geometry_at_onset: dict[str, float]
    inter_subject_sd_latency_sec: float | None
    changepoint_overlap: bool | None
    thresholds_tested: list[float]
    onset_by_threshold: dict
    onset_ci_sec: tuple[float, float] | None = None
    onset_permutation_p: float = float("nan")
    monotonicity_fraction: float = float("nan")
    n_bootstrap: int = 0
    n_permutations: int = 0


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples: mean(a - b) / std(a - b, ddof=1)."""
    diff = np.asarray(a) - np.asarray(b)
    n = len(diff)
    if n < 2:
        return float("nan")
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return float("nan")
    return float(np.mean(diff) / sd)


def _cohens_d_independent(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for independent samples: (mean_a - mean_b) / pooled_sd.

    Uses the pooled standard deviation:
        pooled_sd = sqrt(((n_a-1)*var_a + (n_b-1)*var_b) / (n_a + n_b - 2))
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled_var <= 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled_var))


def _interpolate_to_grid(
    time_sec: np.ndarray,
    values: np.ndarray,
    common_time: np.ndarray,
    tol: float = 0.15,
) -> np.ndarray:
    """Nearest-neighbour interpolation onto a common time grid.

    Parameters
    ----------
    time_sec : np.ndarray
        Source time axis (seconds).
    values : np.ndarray
        Source values, same length as time_sec.
    common_time : np.ndarray
        Target time axis.
    tol : float
        Maximum distance (s) for a valid match; beyond this, NaN is returned.

    Returns
    -------
    np.ndarray
        Values on common_time grid.
    """
    interp = np.full(len(common_time), np.nan)
    for i, t in enumerate(common_time):
        diffs = np.abs(time_sec - t)
        idx = np.argmin(diffs)
        if diffs[idx] <= tol:
            interp[i] = values[idx]
    return interp


def _compute_aic(y: np.ndarray, x: np.ndarray, degree: int) -> float:
    """AIC of polynomial regression of given degree.

    AIC = n * log(RSS / n) + 2 * (degree + 1)
    where RSS is residual sum of squares.

    Parameters
    ----------
    y : np.ndarray
        Observed values.
    x : np.ndarray
        Predictor values.
    degree : int
        Polynomial degree (1 = linear, 2 = quadratic).

    Returns
    -------
    float
        AIC value.
    """
    valid = np.isfinite(y) & np.isfinite(x)
    if valid.sum() < degree + 2:
        return float("nan")
    yv = y[valid]
    xv = x[valid]
    coeffs = np.polyfit(xv, yv, degree)
    yhat = np.polyval(coeffs, xv)
    rss = float(np.sum((yv - yhat) ** 2))
    n = len(yv)
    k = degree + 1
    if rss <= 0:
        return float("-inf")
    return float(n * np.log(rss / n) + 2 * k)


def analyze_extended_window(
    timecourses_by_subject: dict,
    metric_name: str,
    pre_sec: float = 600.0,
    post_sec: float = 60.0,
    min_qualifying_pre_sec: float = 600.0,
    seed: int = 42,
    n_bootstrap: int = 5000,
) -> ExtendedWindowResult:
    """Q1: Analyze geometry timecourse over extended pre-transition window.

    Only includes transitions where the timecourse actually covers
    min_qualifying_pre_sec of pre-transition data. Per-subject slopes
    are computed via OLS on the pre-transition segment (time_sec < 0)
    and tested with a one-sample t-test across subjects.

    Parameters
    ----------
    timecourses_by_subject : dict[str, list[TransitionTimecourse]]
        Keys: subject IDs. Each TransitionTimecourse must have .time_sec
        and an attribute named metric_name.
    metric_name : str
        One of: "spectral_radius", "eigenvalue_gap", "condition_number", "nd_score".
    pre_sec : float
        Expected pre-transition window duration (seconds).
    post_sec : float
        Expected post-transition window duration (seconds; used only for time axis).
    min_qualifying_pre_sec : float
        Minimum pre-transition coverage required (seconds). Timecourses where
        min(time_sec) > -(min_qualifying_pre_sec - 15.0) are excluded.
        The 15-s tolerance accounts for 30-s epoch staging resolution.
    seed : int
        Random seed for bootstrap.
    n_bootstrap : int
        Number of bootstrap resamples for CI and trajectory CI.

    Returns
    -------
    ExtendedWindowResult

    Notes
    -----
    Slope sign convention: positive slope means metric increases toward transition.
    Early/late Cohen's d uses: early = [-pre_sec, -pre_sec/2], late = [-120, 0] s.
    """
    tol_sec = 15.0
    coverage_threshold = -(min_qualifying_pre_sec - tol_sec)

    time_step = 0.5
    common_time = np.arange(-pre_sec, post_sec + time_step, time_step)

    subject_mean_timecourses: dict[str, np.ndarray] = {}
    n_qualifying_transitions = 0

    for subj, tcs in timecourses_by_subject.items():
        qualifying = []
        for tc in tcs:
            if len(tc.time_sec) == 0:
                continue
            if np.min(tc.time_sec) > coverage_threshold:
                continue
            values = getattr(tc, metric_name, None)
            if values is None or len(values) == 0:
                continue
            qualifying.append(_interpolate_to_grid(tc.time_sec, values, common_time))

        if not qualifying:
            continue

        n_qualifying_transitions += len(qualifying)
        stacked = np.array(qualifying)
        subject_mean_timecourses[subj] = np.nanmean(stacked, axis=0)

    n_subjects = len(subject_mean_timecourses)

    _nan_result = ExtendedWindowResult(
        metric_name=metric_name,
        n_qualifying_transitions=n_qualifying_transitions,
        n_subjects=n_subjects,
        mean_slope=float("nan"),
        slope_p_value=1.0,
        slope_ci=(float("nan"), float("nan")),
        cohens_d=float("nan"),
        subject_consistency=float("nan"),
        linear_aic=float("nan"),
        quadratic_aic=float("nan"),
        aic_delta=float("nan"),
        group_time_axis=common_time,
        group_mean_trajectory=np.full(len(common_time), np.nan),
        group_ci_lower=np.full(len(common_time), np.nan),
        group_ci_upper=np.full(len(common_time), np.nan),
        passes_threshold=False,
    )

    if n_subjects < 3:
        return _nan_result

    subject_ids = sorted(subject_mean_timecourses.keys())
    subject_matrix = np.array([subject_mean_timecourses[s] for s in subject_ids])

    pre_mask = common_time < 0
    pre_time = common_time[pre_mask]

    per_subject_slopes = []
    for row in subject_matrix:
        pre_vals = row[pre_mask]
        valid = np.isfinite(pre_vals)
        if valid.sum() < 5:
            per_subject_slopes.append(float("nan"))
            continue
        slope, _, _, _, _ = sp_stats.linregress(pre_time[valid], pre_vals[valid])
        per_subject_slopes.append(float(slope))

    valid_slopes = np.array([s for s in per_subject_slopes if np.isfinite(s)])
    if len(valid_slopes) < 3:
        return _nan_result

    mean_slope = float(np.mean(valid_slopes))
    _, slope_p = sp_stats.ttest_1samp(valid_slopes, 0.0)
    slope_p = float(slope_p)
    if not np.isfinite(slope_p):
        slope_p = 1.0 if mean_slope == 0.0 else float("nan")

    if mean_slope != 0:
        subject_consistency = float(np.mean(np.sign(valid_slopes) == np.sign(mean_slope)))
    else:
        subject_consistency = 0.0

    early_boundary = -pre_sec / 2.0
    early_mask = (common_time >= -pre_sec) & (common_time < early_boundary)
    late_mask = (common_time >= -120.0) & (common_time < 0.0)

    early_means, late_means = [], []
    for row in subject_matrix:
        e = row[early_mask]
        la = row[late_mask]
        if np.sum(np.isfinite(e)) >= 3 and np.sum(np.isfinite(la)) >= 3:
            early_means.append(float(np.nanmean(e)))
            late_means.append(float(np.nanmean(la)))

    if len(early_means) >= 3:
        cohens_d = _cohens_d_paired(np.array(late_means), np.array(early_means))
    else:
        cohens_d = float("nan")

    group_mean = np.nanmean(subject_matrix, axis=0)

    lin_aic = _compute_aic(group_mean[pre_mask], pre_time, degree=1)
    quad_aic = _compute_aic(group_mean[pre_mask], pre_time, degree=2)
    aic_delta = (quad_aic - lin_aic) if np.isfinite(lin_aic) and np.isfinite(quad_aic) else float("nan")

    rng = np.random.default_rng(seed)
    boot_slopes = []
    boot_trajectories = []
    n_subj = len(subject_ids)
    for _ in range(n_bootstrap):
        idx = rng.choice(n_subj, size=n_subj, replace=True)
        boot_mat = subject_matrix[idx]
        boot_mean = np.nanmean(boot_mat, axis=0)
        boot_trajectories.append(boot_mean)
        boot_subj_slopes = [per_subject_slopes[i] for i in idx
                            if np.isfinite(per_subject_slopes[i])]
        if len(boot_subj_slopes) >= 2:
            boot_slopes.append(float(np.mean(boot_subj_slopes)))

    if len(boot_slopes) >= 10:
        slope_ci = (float(np.percentile(boot_slopes, 2.5)),
                    float(np.percentile(boot_slopes, 97.5)))
    else:
        slope_ci = (float("nan"), float("nan"))

    boot_traj_arr = np.array(boot_trajectories)
    ci_lower = np.nanpercentile(boot_traj_arr, 2.5, axis=0)
    ci_upper = np.nanpercentile(boot_traj_arr, 97.5, axis=0)

    passes = (
        slope_p < 0.05
        and subject_consistency >= 0.70
        and np.isfinite(cohens_d)
        and abs(cohens_d) >= 0.5
    )

    actual_pre_coverage = []
    for subj, tcs in timecourses_by_subject.items():
        for tc in tcs:
            if len(tc.time_sec) > 0:
                actual_pre_coverage.append(float(-np.min(tc.time_sec)))
    max_actual_pre = float(np.max(actual_pre_coverage)) if actual_pre_coverage else 0.0
    is_truncated = max_actual_pre < (pre_sec - tol_sec)
    analysis_type = "constrained_replication" if is_truncated else "full_extended_window"

    data_constraint = {
        "window_sec_requested": float(pre_sec),
        "window_sec_available": max_actual_pre,
        "is_truncated": is_truncated,
        "analysis_type": analysis_type,
        "interpretation_caveat": (
            f"Data covers {max_actual_pre:.0f} s pre-transition vs {pre_sec:.0f} s requested. "
            "Results replicate existing temporal precedence findings within the available window "
            "but cannot test whether the drift extends further."
        ) if is_truncated else "",
    }

    return ExtendedWindowResult(
        metric_name=metric_name,
        n_qualifying_transitions=n_qualifying_transitions,
        n_subjects=n_subjects,
        mean_slope=mean_slope,
        slope_p_value=slope_p,
        slope_ci=slope_ci,
        cohens_d=cohens_d,
        subject_consistency=subject_consistency,
        linear_aic=lin_aic,
        quadratic_aic=quad_aic,
        aic_delta=aic_delta,
        group_time_axis=common_time,
        group_mean_trajectory=group_mean,
        group_ci_lower=ci_lower,
        group_ci_upper=ci_upper,
        passes_threshold=bool(passes),
        data_constraint=data_constraint,
    )


def _piecewise_ols_rss(
    t: np.ndarray,
    y: np.ndarray,
    breakpoint_idx: int,
) -> float:
    """Residual sum of squares for two-segment piecewise OLS at breakpoint_idx.

    Segment 1: t[:breakpoint_idx+1], y[:breakpoint_idx+1]
    Segment 2: t[breakpoint_idx+1:], y[breakpoint_idx+1:]

    Returns total RSS = RSS_1 + RSS_2. NaN if either segment has < 2 valid points.
    """
    t1, y1 = t[: breakpoint_idx + 1], y[: breakpoint_idx + 1]
    t2, y2 = t[breakpoint_idx + 1 :], y[breakpoint_idx + 1 :]
    v1 = np.isfinite(y1)
    v2 = np.isfinite(y2)
    if v1.sum() < 2 or v2.sum() < 2:
        return float("nan")
    res1 = sp_stats.linregress(t1[v1], y1[v1])
    pred1 = res1.intercept + res1.slope * t1[v1]
    rss1 = float(np.sum((y1[v1] - pred1) ** 2))
    res2 = sp_stats.linregress(t2[v2], y2[v2])
    pred2 = res2.intercept + res2.slope * t2[v2]
    rss2 = float(np.sum((y2[v2] - pred2) ** 2))
    return rss1 + rss2


def _detect_single_changepoint(
    time_sec: np.ndarray,
    values: np.ndarray,
    min_segment_sec: float = 30.0,
) -> float:
    """Find the breakpoint time (seconds) that minimises piecewise-OLS RSS.

    Parameters
    ----------
    time_sec : np.ndarray
        Time axis (seconds, pre-transition only, should be negative).
    values : np.ndarray
        Metric values aligned to time_sec.
    min_segment_sec : float
        Minimum duration on each side of the breakpoint.

    Returns
    -------
    float
        Breakpoint time in seconds (negative = before boundary). NaN if
        fewer than 3 candidate breakpoints exist or data is insufficient.
    """
    valid = np.isfinite(values) & np.isfinite(time_sec)
    if valid.sum() < 6:
        return float("nan")

    t = time_sec[valid]
    y = values[valid]

    t_min, t_max = float(t.min()), float(t.max())
    cand_lo = t_min + min_segment_sec
    cand_hi = t_max - min_segment_sec
    if cand_lo >= cand_hi:
        return float("nan")

    candidate_times = np.arange(cand_lo, cand_hi + min_segment_sec / 2, min_segment_sec)
    if len(candidate_times) < 3:
        return float("nan")

    best_rss = float("inf")
    best_t = float("nan")
    for ct in candidate_times:
        bp_idx = int(np.searchsorted(t, ct, side="right")) - 1
        if bp_idx < 1 or bp_idx >= len(t) - 2:
            continue
        rss = _piecewise_ols_rss(t, y, bp_idx)
        if np.isfinite(rss) and rss < best_rss:
            best_rss = rss
            best_t = float(t[bp_idx])

    return best_t


def detect_commitment_changepoints(
    timecourses_by_subject: dict,
    metric_name: str,
    pre_sec: float = 600.0,
    min_segment_sec: float = 30.0,
    min_qualifying_pre_sec: float = 240.0,
    penalty_values: list[float] | None = None,
    seed: int = 42,
) -> ChangepointResult:
    """Q2: Detect slope-onset changepoint in pre-transition N2 geometry.

    Uses grid-search piecewise linear regression: for each candidate breakpoint
    position (spaced at min_segment_sec resolution), fit two OLS line segments
    and select the breakpoint minimising total residual SS. Reports sensitivity
    across penalty_values (which control minimum segment length).

    Parameters
    ----------
    timecourses_by_subject : dict[str, list[TransitionTimecourse]]
    metric_name : str
    pre_sec : float
        Total pre-transition window to search.
    min_segment_sec : float
        Minimum segment length on each side of changepoint (seconds).
    min_qualifying_pre_sec : float
        Minimum actual pre-transition coverage to include a transition.
    penalty_values : list[float] or None
        Candidate minimum segment lengths for sensitivity analysis.
        Default: [20.0, 30.0, 60.0].
    seed : int

    Returns
    -------
    ChangepointResult

    Notes
    -----
    Unit of observation: per transition -> per subject mean -> group t-test.
    Changepoint latency is in seconds relative to transition (negative = before).
    """
    if penalty_values is None:
        penalty_values = [20.0, 30.0, 60.0]

    tol_sec = 15.0
    coverage_threshold = -(min_qualifying_pre_sec - tol_sec)

    per_transition_latencies: list[float] = []
    per_subject_mean_latencies: list[float] = []
    n_qualifying = 0

    subject_ids = sorted(timecourses_by_subject.keys())
    for subj in subject_ids:
        tcs = timecourses_by_subject[subj]
        subj_latencies = []
        for tc in tcs:
            if len(tc.time_sec) == 0:
                continue
            if np.min(tc.time_sec) > coverage_threshold:
                continue
            values = getattr(tc, metric_name, None)
            if values is None or len(values) == 0:
                continue

            pre_mask = tc.time_sec < 0
            t_pre = tc.time_sec[pre_mask]
            v_pre = values[pre_mask]

            if len(t_pre) < 6:
                per_transition_latencies.append(float("nan"))
                continue

            n_qualifying += 1
            lat = _detect_single_changepoint(t_pre, v_pre, min_segment_sec)
            per_transition_latencies.append(lat)
            if np.isfinite(lat):
                subj_latencies.append(lat)

        if subj_latencies:
            per_subject_mean_latencies.append(float(np.mean(subj_latencies)))

    n_subjects = len(per_subject_mean_latencies)

    if n_subjects >= 3:
        valid_lats = np.array(per_subject_mean_latencies)
        group_mean = float(np.mean(valid_lats))
        _, group_p = sp_stats.ttest_1samp(valid_lats, 0.0)
        group_p = float(group_p) if np.isfinite(group_p) else 1.0
        lat_sd = float(np.std(valid_lats, ddof=1))

        rng = np.random.default_rng(seed)
        boot_means = []
        for _ in range(5000):
            idx = rng.choice(n_subjects, size=n_subjects, replace=True)
            boot_means.append(float(np.mean(valid_lats[idx])))
        group_ci = (float(np.percentile(boot_means, 2.5)),
                    float(np.percentile(boot_means, 97.5)))
    else:
        group_mean = float("nan")
        group_p = 1.0
        group_ci = (float("nan"), float("nan"))
        lat_sd = float("nan")

    bin_edges = list(np.arange(-pre_sec, 0 + 60, 60.0))
    valid_lats_all = [x for x in per_transition_latencies if np.isfinite(x)]
    hist_counts, _ = np.histogram(valid_lats_all, bins=bin_edges) if valid_lats_all else (
        np.zeros(len(bin_edges) - 1, dtype=int), bin_edges
    )

    within_300 = sum(1 for x in valid_lats_all if -300.0 <= x <= 0.0)
    frac_300 = within_300 / len(valid_lats_all) if valid_lats_all else float("nan")

    penalty_sensitivity: dict[str, dict] = {}
    for pv in penalty_values:
        pv_lats: list[list[float]] = []
        for subj in subject_ids:
            tcs = timecourses_by_subject[subj]
            s_lats = []
            for tc in tcs:
                if len(tc.time_sec) == 0:
                    continue
                if np.min(tc.time_sec) > coverage_threshold:
                    continue
                vals = getattr(tc, metric_name, None)
                if vals is None or len(vals) == 0:
                    continue
                pre_mask = tc.time_sec < 0
                t_pre = tc.time_sec[pre_mask]
                v_pre = vals[pre_mask]
                if len(t_pre) < 6:
                    continue
                lat = _detect_single_changepoint(t_pre, v_pre, pv)
                if np.isfinite(lat):
                    s_lats.append(lat)
            if s_lats:
                pv_lats.append(s_lats)

        pv_subj_means = [float(np.mean(sl)) for sl in pv_lats]
        if len(pv_subj_means) >= 3:
            pv_arr = np.array(pv_subj_means)
            pv_mean = float(np.mean(pv_arr))
            _, pv_p = sp_stats.ttest_1samp(pv_arr, 0.0)
            pv_p = float(pv_p) if np.isfinite(pv_p) else 1.0
        else:
            pv_mean = float("nan")
            pv_p = 1.0
        penalty_sensitivity[f"min_seg_{int(pv)}s"] = {
            "group_mean_latency_sec": pv_mean,
            "group_p": pv_p,
        }

    actual_pre_coverage = []
    for subj in subject_ids:
        for tc in timecourses_by_subject[subj]:
            if len(tc.time_sec) > 0:
                actual_pre_coverage.append(float(-np.min(tc.time_sec)))
    max_actual_pre = float(np.max(actual_pre_coverage)) if actual_pre_coverage else 0.0
    is_truncated = max_actual_pre < (pre_sec - tol_sec)

    edge_ratio = float("nan")
    if np.isfinite(group_mean) and max_actual_pre > 0:
        edge_ratio = float(abs(group_mean) / max_actual_pre)

    data_constraint = {
        "window_sec_requested": float(pre_sec),
        "window_sec_available": max_actual_pre,
        "is_truncated": is_truncated,
        "interpretation_caveat": (
            f"Changepoint search bounded by {max_actual_pre:.0f} s available data "
            f"(requested {pre_sec:.0f} s). Changepoint at {group_mean:.0f} s may be "
            "constrained by window edge rather than biology."
        ) if is_truncated else "",
    }

    trunc_sensitivity: dict[str, dict] = {}
    if np.isfinite(group_mean):
        for trunc_window in [60.0, 90.0, 120.0]:
            if trunc_window >= max_actual_pre:
                continue
            trunc_subj_means = []
            for subj in subject_ids:
                tcs = timecourses_by_subject[subj]
                s_lats = []
                for tc in tcs:
                    if len(tc.time_sec) == 0:
                        continue
                    vals = getattr(tc, metric_name, None)
                    if vals is None or len(vals) == 0:
                        continue
                    pre_mask = (tc.time_sec < 0) & (tc.time_sec >= -trunc_window)
                    t_pre = tc.time_sec[pre_mask]
                    v_pre = vals[pre_mask]
                    if len(t_pre) < 6:
                        continue
                    lat = _detect_single_changepoint(t_pre, v_pre, min_segment_sec)
                    if np.isfinite(lat):
                        s_lats.append(lat)
                if s_lats:
                    trunc_subj_means.append(float(np.mean(s_lats)))
            if len(trunc_subj_means) >= 3:
                t_arr = np.array(trunc_subj_means)
                t_mean = float(np.mean(t_arr))
                _, t_p = sp_stats.ttest_1samp(t_arr, 0.0)
                trunc_sensitivity[f"{int(trunc_window)}s"] = {
                    "group_mean_latency_sec": t_mean,
                    "group_p": float(t_p) if np.isfinite(t_p) else 1.0,
                    "n_subjects": len(trunc_subj_means),
                }

    return ChangepointResult(
        metric_name=metric_name,
        n_qualifying_transitions=n_qualifying,
        n_subjects=n_subjects,
        per_transition_latencies_sec=[float(x) for x in per_transition_latencies],
        per_subject_mean_latencies_sec=per_subject_mean_latencies,
        group_mean_latency_sec=group_mean,
        group_latency_p_value=group_p,
        group_latency_ci=group_ci,
        latency_sd_sec=lat_sd,
        latency_hist_60s_bins=hist_counts.tolist(),
        latency_hist_bin_edges=[float(e) for e in bin_edges],
        fraction_within_300s=frac_300,
        penalty_sensitivity=penalty_sensitivity,
        edge_proximity_ratio=edge_ratio,
        data_constraint=data_constraint,
        window_truncation_sensitivity=trunc_sensitivity,
    )


def _bimodality_coefficient(x: np.ndarray) -> float:
    """Compute the bimodality coefficient (BC) for 1D data.

    BC = (skewness^2 + 1) / (excess_kurtosis + 3).
    Under unimodality, BC <= 5/9 ≈ 0.555 (achieved by the uniform distribution).
    Values > 0.555 suggest bimodality (Freeman & Dale, 2013).

    Parameters
    ----------
    x : np.ndarray
        1D array of observations.

    Returns
    -------
    float
        Bimodality coefficient in (0, inf). Higher values indicate bimodality.
    """
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return float("nan")
    skew = float(sp_stats.skew(x))
    kurt = float(sp_stats.kurtosis(x, fisher=True))
    denom = kurt + 3.0
    if denom <= 0:
        return float("nan")
    return float((skew ** 2 + 1.0) / denom)


def _silverman_test(
    x: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = 500,
) -> float:
    """Silverman bandwidth test for unimodality.

    Finds the critical bandwidth h_crit at which a Gaussian KDE of x has
    exactly one mode. Under H0 (unimodal), bootstrapping from a Gaussian
    with matched mean/std should produce critical bandwidths >= h_crit at
    least 5% of the time.

    Returns p-value (fraction of null h_crits >= observed h_crit).
    """
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float("nan")

    def _count_modes_vec(data: np.ndarray, bandwidth: float) -> int:
        grid = np.linspace(data.min() - 3 * bandwidth, data.max() + 3 * bandwidth, 256)
        diff = grid[:, None] - data[None, :]
        kde = np.sum(np.exp(-0.5 * (diff / bandwidth) ** 2), axis=1)
        sign_changes = np.diff(np.sign(np.diff(kde)))
        return int(np.sum(sign_changes < 0))

    def _find_h_crit(data: np.ndarray, sd_est: float) -> float:
        lo_b, hi_b = 0.01 * sd_est, 2.0 * sd_est
        for _ in range(30):
            mid_b = (lo_b + hi_b) / 2
            if _count_modes_vec(data, mid_b) > 1:
                lo_b = mid_b
            else:
                hi_b = mid_b
        return hi_b

    sd_x = float(np.std(x, ddof=1))
    h_crit = _find_h_crit(x, sd_x)

    mu, sd = float(np.mean(x)), sd_x
    null_h_crits = []
    for _ in range(n_bootstrap):
        null_sample = rng.normal(mu, sd, size=n)
        null_h_crits.append(_find_h_crit(null_sample, sd))

    return float(np.mean(np.array(null_h_crits) >= h_crit))


def _gmm_cv_log_likelihood_delta(
    X: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> float:
    """Cross-validated log-likelihood difference: GMM k=2 minus k=1.

    Positive values indicate k=2 generalises better than k=1 (bimodal).
    """
    if X.shape[0] < n_folds * 2:
        return float("nan")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    ll_k1, ll_k2 = [], []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        try:
            g1 = GaussianMixture(n_components=1, covariance_type="full",
                                 random_state=seed, max_iter=200)
            g1.fit(X_tr)
            ll_k1.append(float(g1.score(X_te)))
        except Exception:
            ll_k1.append(float("nan"))
        try:
            g2 = GaussianMixture(n_components=2, covariance_type="full",
                                 random_state=seed, max_iter=200)
            g2.fit(X_tr)
            ll_k2.append(float(g2.score(X_te)))
        except Exception:
            ll_k2.append(float("nan"))

    arr1 = np.array(ll_k1)
    arr2 = np.array(ll_k2)
    valid = np.isfinite(arr1) & np.isfinite(arr2)
    if valid.sum() < 2:
        return float("nan")
    return float(np.mean(arr2[valid] - arr1[valid]))


def analyze_n2_bistability(
    early_windows: dict,
    late_windows: dict,
    metric_names: list[str],
    seed: int = 42,
    n_bootstrap_dip: int = 2000,
) -> BistabilityResult:
    """Q3: Test whether N2 geometry is unimodal or bistable.

    Compares geometry distributions between "early N2" windows (far from any
    transition, representing the stable attractor) and "late N2" windows
    (approaching an N3 transition, representing the pre-commitment state).

    Uses three complementary tests:
    1. Bimodality coefficient (BC) of the pooled 1D projection — BC > 5/9
       suggests bimodality. Bootstrap p-value under uniform null (the most
       conservative unimodal null distribution). Note: BC-based test used
       in place of the Hartigan dip test because diptest is not installed.
    2. GMM BIC comparison (1 vs 2 components on 4D feature space).
    3. Paired Cohen's d and Hotelling's T^2 on per-subject early vs late means.

    Parameters
    ----------
    early_windows : dict[str, np.ndarray]
        Subject -> array shape (n_windows, n_metrics) for "stable N2"
        (e.g., windows from -120 to -60 s before transition).
        The column order must match metric_names.
    late_windows : dict[str, np.ndarray]
        Subject -> array shape (n_windows, n_metrics) for "approaching N3"
        (e.g., windows from -60 to 0 s before transition).
    metric_names : list[str]
        Names corresponding to columns in the arrays.
    seed : int
        Random seed for GMM and bootstrap.
    n_bootstrap_dip : int
        Number of bootstrap samples for bimodality coefficient p-value.

    Returns
    -------
    BistabilityResult

    Notes
    -----
    Units: early/late windows are in native metric units (unitless geometry values).
    Assumptions: per-subject arrays have been cleaned of NaN before passing.
    Limitations: with n=10 subjects, Hotelling's T^2 has limited power. Report
    per-metric Cohen's d as the primary subject-level statistic.
    What this test cannot establish: even a significant result only shows that
    early and late N2 geometry differs — it does not prove bistability or
    attractor separation in a dynamical-systems sense.
    """
    all_early = []
    all_late = []
    subjects = sorted(set(list(early_windows.keys()) + list(late_windows.keys())))

    for subj in subjects:
        e = early_windows.get(subj)
        la = late_windows.get(subj)
        if e is not None and len(e) > 0:
            all_early.append(e)
        if la is not None and len(la) > 0:
            all_late.append(la)

    if not all_early or not all_late:
        _empty = np.zeros((0, len(metric_names)))
        return BistabilityResult(
            n_early_windows=0, n_late_windows=0,
            dip_test_statistic=float("nan"), dip_test_p=float("nan"),
            gmm_bic_1=float("nan"), gmm_bic_2=float("nan"), gmm_bic_delta=float("nan"),
            early_vs_late_d_spectral_radius=float("nan"),
            early_vs_late_d_eigenvalue_gap=float("nan"),
            hotelling_p=None,
            pca_coords_early=_empty[:, :2], pca_coords_late=_empty[:, :2],
            gmm_means=np.full((2, len(metric_names)), np.nan),
            gmm_covariances=np.full((2, len(metric_names), len(metric_names)), np.nan),
        )

    early_arr = np.vstack(all_early)
    late_arr = np.vstack(all_late)
    n_early = len(early_arr)
    n_late = len(late_arr)
    pooled = np.vstack([early_arr, late_arr])

    finite_rows = np.all(np.isfinite(pooled), axis=1)
    pooled_clean = pooled[finite_rows]
    labels_clean = np.concatenate([
        np.zeros(n_early), np.ones(n_late)
    ])[finite_rows]

    n_metrics = pooled_clean.shape[1] if pooled_clean.ndim == 2 else 1

    rng = np.random.default_rng(seed)

    bc_obs = float("nan")
    bc_p = float("nan")
    bc_gaussian_p = float("nan")
    silverman_p_val = float("nan")
    proj_1d = None

    if pooled_clean.shape[0] >= 10 and n_metrics >= 1:
        n_pca = min(1, n_metrics)
        pca_1d = PCA(n_components=n_pca, random_state=seed)
        proj_1d = pca_1d.fit_transform(pooled_clean)[:, 0]
        bc_obs = _bimodality_coefficient(proj_1d)

        if np.isfinite(bc_obs):
            x_min, x_max = proj_1d.min(), proj_1d.max()
            null_bcs_uniform = []
            for _ in range(n_bootstrap_dip):
                null_sample = rng.uniform(x_min, x_max, size=len(proj_1d))
                null_bcs_uniform.append(_bimodality_coefficient(null_sample))
            bc_p = float(np.mean(np.array(null_bcs_uniform) >= bc_obs))

            mu_1d = float(np.mean(proj_1d))
            sd_1d = float(np.std(proj_1d, ddof=1))
            null_bcs_gauss = []
            for _ in range(n_bootstrap_dip):
                null_sample = rng.normal(mu_1d, sd_1d, size=len(proj_1d))
                null_bcs_gauss.append(_bimodality_coefficient(null_sample))
            bc_gaussian_p = float(np.mean(np.array(null_bcs_gauss) >= bc_obs))

        silverman_n = min(n_bootstrap_dip, 500)
        silverman_input = proj_1d
        if len(proj_1d) > 500:
            silverman_input = rng.choice(proj_1d, size=500, replace=False)
        silverman_p_val = _silverman_test(silverman_input, rng, n_bootstrap=silverman_n)

    gmm_bic_1, gmm_bic_2 = float("nan"), float("nan")
    gmm_means = np.full((2, n_metrics), np.nan)
    gmm_covs = np.full((2, n_metrics, n_metrics), np.nan)
    gmm_cv_ll = float("nan")

    if pooled_clean.shape[0] >= 10 and n_metrics >= 1:
        try:
            g1 = GaussianMixture(n_components=1, covariance_type="full",
                                 random_state=seed, max_iter=200)
            g1.fit(pooled_clean)
            gmm_bic_1 = float(g1.bic(pooled_clean))

            g2 = GaussianMixture(n_components=2, covariance_type="full",
                                 random_state=seed, max_iter=200)
            g2.fit(pooled_clean)
            gmm_bic_2 = float(g2.bic(pooled_clean))
            gmm_means = g2.means_
            gmm_covs = g2.covariances_
        except Exception:
            pass

        if pooled_clean.shape[0] >= 20:
            gmm_cv_ll = _gmm_cv_log_likelihood_delta(pooled_clean, seed=seed)

    gmm_bic_delta = (gmm_bic_2 - gmm_bic_1) if (np.isfinite(gmm_bic_1) and np.isfinite(gmm_bic_2)) else float("nan")

    n_pca_2d = min(2, n_metrics)
    pca_2d = PCA(n_components=n_pca_2d, random_state=seed)
    if pooled_clean.shape[0] >= 4 and n_metrics >= 2:
        all_2d = pca_2d.fit_transform(pooled_clean)
        early_mask_clean = labels_clean == 0
        coords_early = all_2d[early_mask_clean]
        coords_late = all_2d[~early_mask_clean]
    else:
        coords_early = np.zeros((n_early, 2))
        coords_late = np.zeros((n_late, 2))

    per_subject_early_means: dict[str, np.ndarray] = {}
    per_subject_late_means: dict[str, np.ndarray] = {}
    for subj in subjects:
        e = early_windows.get(subj)
        la = late_windows.get(subj)
        if e is not None and len(e) > 0:
            valid = np.all(np.isfinite(e), axis=1)
            if valid.sum() > 0:
                per_subject_early_means[subj] = np.nanmean(e[valid], axis=0)
        if la is not None and len(la) > 0:
            valid = np.all(np.isfinite(la), axis=1)
            if valid.sum() > 0:
                per_subject_late_means[subj] = np.nanmean(la[valid], axis=0)

    common_subjs = sorted(set(per_subject_early_means.keys()) & set(per_subject_late_means.keys()))

    d_sr = float("nan")
    d_eg = float("nan")
    hotelling_p = None

    if len(common_subjs) >= 3:
        early_mat = np.array([per_subject_early_means[s] for s in common_subjs])
        late_mat = np.array([per_subject_late_means[s] for s in common_subjs])

        if "spectral_radius" in metric_names:
            idx_sr = metric_names.index("spectral_radius")
            d_sr = _cohens_d_paired(late_mat[:, idx_sr], early_mat[:, idx_sr])

        if "eigenvalue_gap" in metric_names:
            idx_eg = metric_names.index("eigenvalue_gap")
            d_eg = _cohens_d_paired(late_mat[:, idx_eg], early_mat[:, idx_eg])

        if n_metrics >= 2 and len(common_subjs) > n_metrics:
            try:
                import pingouin as pg
                diff_mat = late_mat - early_mat
                result = pg.multivariate_ttest(diff_mat)
                hotelling_p = float(result["pval"].iloc[0])
            except Exception:
                hotelling_p = None

    return BistabilityResult(
        n_early_windows=n_early,
        n_late_windows=n_late,
        dip_test_statistic=bc_obs,
        dip_test_p=bc_p,
        bc_gaussian_null_p=bc_gaussian_p,
        silverman_p=silverman_p_val,
        gmm_bic_1=gmm_bic_1,
        gmm_bic_2=gmm_bic_2,
        gmm_bic_delta=gmm_bic_delta,
        gmm_cv_ll_delta=gmm_cv_ll,
        early_vs_late_d_spectral_radius=d_sr,
        early_vs_late_d_eigenvalue_gap=d_eg,
        hotelling_p=hotelling_p,
        pca_coords_early=coords_early,
        pca_coords_late=coords_late,
        gmm_means=gmm_means,
        gmm_covariances=gmm_covs,
        bc_note="dip_test_p uses uniform null (uninformative); bc_gaussian_null_p uses Gaussian null (correct test)",
    )


def _local_slope(
    time_sec: np.ndarray,
    values: np.ndarray,
    t_bin: float,
    halfwindow: float = 30.0,
) -> float:
    """OLS slope of values vs time in [t_bin - halfwindow, t_bin + halfwindow].

    Parameters
    ----------
    time_sec : np.ndarray
        Time axis (seconds).
    values : np.ndarray
        Metric values, same length as time_sec.
    t_bin : float
        Center of the local window (seconds).
    halfwindow : float
        Half-width of the local window (seconds).

    Returns
    -------
    float
        OLS slope, or NaN if fewer than 3 valid points in window.
    """
    mask = (time_sec >= t_bin - halfwindow) & (time_sec <= t_bin + halfwindow)
    t_win = time_sec[mask]
    v_win = values[mask]
    valid = np.isfinite(v_win)
    if valid.sum() < 3:
        return float("nan")
    slope, _, _, _, _ = sp_stats.linregress(t_win[valid], v_win[valid])
    return float(slope)


def predict_transition_type_from_trajectory(
    n3_timecourses_by_subject: dict,
    rem_timecourses_by_subject: dict,
    metric_names: list[str],
    time_bins_sec: list[float] | None = None,
    slope_halfwindow_sec: float = 30.0,
    n_permutations: int = 1000,
    seed: int = 42,
) -> TrajectoryPredictionResult:
    """Q4: Test whether geometry drift direction at each pre-transition latency
    distinguishes N2->N3 from N2->REM transitions.

    For each time bin and metric, computes the local OLS slope per subject
    (averaged across transitions). Subjects are the unit of analysis.
    LOSO logistic regression classifies N3-bound vs REM-bound subjects from
    their slope-feature vectors. Permutation null permutes subject-level labels.

    Parameters
    ----------
    n3_timecourses_by_subject : dict[str, list[TransitionTimecourse]]
        Subject -> list of N2->N3 transition timecourses.
    rem_timecourses_by_subject : dict[str, list[TransitionTimecourse]]
        Subject -> list of N2->REM transition timecourses.
    metric_names : list[str]
        Metric attributes to extract from TransitionTimecourse.
    time_bins_sec : list[float] or None
        Pre-transition latencies to evaluate (seconds, must be negative).
        Default: [-120, -100, -80, -60, -40, -20].
    slope_halfwindow_sec : float
        Half-window for local slope estimation (seconds).
    n_permutations : int
        Number of permutation samples for null AUC distribution.
    seed : int

    Returns
    -------
    TrajectoryPredictionResult

    Notes
    -----
    Classification: LOSO logistic regression on per-subject mean slope vectors.
    Class imbalance: AUC (not accuracy) is the primary metric.
    Permutation: labels shuffled at the subject level, not the transition level.
    Missing data: subjects missing coverage at a given time bin are excluded
    from that bin's Cohen's d computation only; LOSO uses the primary time bin
    (default -120 s) and excludes subjects with NaN features.

    What this test can establish:
    - Whether the direction of geometry drift at a given pre-transition latency
      is systematically different for N3-bound vs REM-bound N2 episodes.
    What it cannot establish:
    - Causality or mechanism.
    - Individual-level prediction with n=10 subjects.
    """
    if time_bins_sec is None:
        time_bins_sec = [-120.0, -100.0, -80.0, -60.0, -40.0, -20.0]

    rng = np.random.default_rng(seed)

    def _subject_mean_slopes(tcs_by_subj: dict, t_bins: list[float]) -> dict[str, dict[str, list[float]]]:
        """Returns {subject -> {metric -> [slope_at_bin0, slope_at_bin1, ...]}}."""
        out = {}
        for subj, tcs in tcs_by_subj.items():
            metric_slopes = {m: [] for m in metric_names}
            for t_bin in t_bins:
                for metric in metric_names:
                    bin_slopes = []
                    for tc in tcs:
                        values = getattr(tc, metric, None)
                        if values is None:
                            continue
                        slope = _local_slope(tc.time_sec, values, t_bin, slope_halfwindow_sec)
                        if np.isfinite(slope):
                            bin_slopes.append(slope)
                    metric_slopes[metric].append(float(np.mean(bin_slopes)) if bin_slopes else float("nan"))
            out[subj] = metric_slopes
        return out

    n3_slopes = _subject_mean_slopes(n3_timecourses_by_subject, time_bins_sec)
    rem_slopes = _subject_mean_slopes(rem_timecourses_by_subject, time_bins_sec)

    n3_subjects = sorted(n3_slopes.keys())
    rem_subjects = sorted(rem_slopes.keys())
    n_n3 = len(n3_subjects)
    n_rem = len(rem_subjects)

    per_bin_d: dict[str, list[float]] = {m: [] for m in metric_names}
    for bin_idx in range(len(time_bins_sec)):
        for metric in metric_names:
            n3_vals = [n3_slopes[s][metric][bin_idx] for s in n3_subjects
                       if np.isfinite(n3_slopes[s][metric][bin_idx])]
            rem_vals = [rem_slopes[s][metric][bin_idx] for s in rem_subjects
                        if np.isfinite(rem_slopes[s][metric][bin_idx])]
            if len(n3_vals) >= 3 and len(rem_vals) >= 3:
                d = _cohens_d_independent(
                    np.array(n3_vals),
                    np.array(rem_vals),
                )
                per_bin_d[metric].append(d)
            else:
                per_bin_d[metric].append(float("nan"))

    def _build_feature_matrix(slopes_dict: dict[str, dict], subjects: list[str], bin_idx: int) -> np.ndarray:
        rows = []
        for s in subjects:
            row = [slopes_dict[s][m][bin_idx] for m in metric_names]
            rows.append(row)
        return np.array(rows)

    def _loso_classify(X: np.ndarray, y: np.ndarray, subject_ids: list[str],
                       rng_local: np.random.Generator, do_permutation: bool = False):
        """Run LOSO logistic regression with fold-internal z-scoring.

        Returns (auc, ba, per_fold_preds, n_failures).
        """
        n = len(y)
        y_pred_proba = np.full(n, float("nan"))
        y_pred_class = np.full(n, -1)
        n_fail = 0
        per_fold = []

        for i in range(n):
            train_mask = np.ones(n, dtype=bool)
            train_mask[i] = False
            X_tr, y_tr = X[train_mask], y[train_mask]
            if len(np.unique(y_tr)) < 2:
                per_fold.append((subject_ids[i] if i < len(subject_ids) else str(i),
                                 int(y[i]), float("nan")))
                continue
            try:
                scaler = StandardScaler()
                X_tr_z = scaler.fit_transform(X_tr)
                X_te_z = scaler.transform(X[[i]])
                clf = LogisticRegression(max_iter=1000, random_state=int(seed),
                                         class_weight="balanced")
                clf.fit(X_tr_z, y_tr)
                prob = clf.predict_proba(X_te_z)[0, 1]
                pred = clf.predict(X_te_z)[0]
                y_pred_proba[i] = prob
                y_pred_class[i] = pred
                per_fold.append((subject_ids[i] if i < len(subject_ids) else str(i),
                                 int(y[i]), float(prob)))
            except Exception:
                n_fail += 1
                per_fold.append((subject_ids[i] if i < len(subject_ids) else str(i),
                                 int(y[i]), float("nan")))

        valid_preds = np.isfinite(y_pred_proba)
        auc = float("nan")
        ba = float("nan")
        if valid_preds.sum() >= 4 and len(np.unique(y[valid_preds])) == 2:
            auc = float(roc_auc_score(y[valid_preds], y_pred_proba[valid_preds]))
            ba = float(balanced_accuracy_score(y[valid_preds], y_pred_class[valid_preds]))

        return auc, ba, per_fold, n_fail

    all_subject_ids = n3_subjects + rem_subjects

    primary_bin_idx = 0
    X_n3 = _build_feature_matrix(n3_slopes, n3_subjects, primary_bin_idx)
    X_rem = _build_feature_matrix(rem_slopes, rem_subjects, primary_bin_idx)

    X_all = np.vstack([X_n3, X_rem])
    y_all = np.concatenate([np.ones(n_n3), np.zeros(n_rem)])

    finite_rows = np.all(np.isfinite(X_all), axis=1)
    X_clean = X_all[finite_rows]
    y_clean = y_all[finite_rows]
    subj_clean = [all_subject_ids[i] for i in range(len(finite_rows)) if finite_rows[i]]

    feat_stds = np.std(X_clean, axis=0) if X_clean.shape[0] > 1 else np.ones(X_clean.shape[1])
    feat_stds = np.where(feat_stds == 0, 1.0, feat_stds)
    feature_scale_ratio = float(np.max(feat_stds) / np.min(feat_stds))

    loso_auc = float("nan")
    loso_ba = float("nan")
    per_fold_preds = []
    n_conv_fail = 0

    if len(X_clean) >= 4 and len(np.unique(y_clean)) == 2:
        loso_auc, loso_ba, per_fold_preds, n_conv_fail = _loso_classify(
            X_clean, y_clean, subj_clean, rng)

    majority_auc = float("nan")
    if len(y_clean) >= 4 and len(np.unique(y_clean)) == 2:
        majority_proba = np.full(len(y_clean), float(np.mean(y_clean)))
        majority_auc = float(roc_auc_score(y_clean, majority_proba))

    per_bin_auc: list[float] = []
    for bin_idx in range(len(time_bins_sec)):
        X_n3_b = _build_feature_matrix(n3_slopes, n3_subjects, bin_idx)
        X_rem_b = _build_feature_matrix(rem_slopes, rem_subjects, bin_idx)
        X_b = np.vstack([X_n3_b, X_rem_b])
        y_b = np.concatenate([np.ones(n_n3), np.zeros(n_rem)])
        fin_b = np.all(np.isfinite(X_b), axis=1)
        X_b_c = X_b[fin_b]
        y_b_c = y_b[fin_b]
        s_b_c = [all_subject_ids[i] for i in range(len(fin_b)) if fin_b[i]]
        if len(X_b_c) >= 4 and len(np.unique(y_b_c)) == 2:
            auc_b, _, _, _ = _loso_classify(X_b_c, y_b_c, s_b_c, rng)
            per_bin_auc.append(auc_b)
        else:
            per_bin_auc.append(float("nan"))

    null_aucs: list[float] = []
    if len(X_clean) >= 4 and len(np.unique(y_clean)) == 2:
        for _ in range(n_permutations):
            y_perm = rng.permutation(y_clean)
            perm_auc, _, _, _ = _loso_classify(X_clean, y_perm, subj_clean, rng)
            null_aucs.append(perm_auc if np.isfinite(perm_auc) else 0.5)

    perm_p = float("nan")
    if null_aucs and np.isfinite(loso_auc):
        perm_p = float(np.mean(np.array(null_aucs) >= loso_auc))

    per_subj_slopes_out = {
        "n3": {s: n3_slopes[s] for s in n3_subjects},
        "rem": {s: rem_slopes[s] for s in rem_subjects},
    }

    return TrajectoryPredictionResult(
        metrics=list(metric_names),
        time_bins_sec=list(time_bins_sec),
        per_bin_cohens_d=per_bin_d,
        loso_auc=loso_auc,
        loso_balanced_accuracy=loso_ba,
        permutation_null_auc=null_aucs,
        permutation_p=perm_p,
        n_n3_bound=n_n3,
        n_rem_bound=n_rem,
        per_fold_predictions=per_fold_preds,
        feature_scale_ratio=feature_scale_ratio,
        n_convergence_failures=n_conv_fail,
        majority_baseline_auc=majority_auc,
        per_bin_loso_auc=per_bin_auc,
        per_subject_slopes=per_subj_slopes_out,
    )


def _find_onset_from_d_matrix(
    per_bin_d: dict[str, list[float]],
    time_bins: list[float],
    sorted_indices: np.ndarray,
    threshold: float,
) -> float | None:
    """Find earliest time bin where any metric's |d| >= threshold."""
    for sorted_pos, orig_idx in enumerate(sorted_indices):
        t_bin = time_bins[sorted_pos]
        for metric, d_vals in per_bin_d.items():
            if orig_idx < len(d_vals) and np.isfinite(d_vals[orig_idx]) and abs(d_vals[orig_idx]) >= threshold:
                return t_bin
    return None


def _compute_per_bin_d_from_slopes(
    n3_slopes: dict[str, dict[str, list[float]]],
    rem_slopes: dict[str, dict[str, list[float]]],
    metric_names: list[str],
    n_bins: int,
) -> dict[str, list[float]]:
    """Recompute per-bin Cohen's d from per-subject slope dicts."""
    n3_subjects = sorted(n3_slopes.keys())
    rem_subjects = sorted(rem_slopes.keys())
    per_bin_d: dict[str, list[float]] = {m: [] for m in metric_names}
    for bin_idx in range(n_bins):
        for metric in metric_names:
            n3_vals = [n3_slopes[s][metric][bin_idx] for s in n3_subjects
                       if bin_idx < len(n3_slopes[s][metric]) and np.isfinite(n3_slopes[s][metric][bin_idx])]
            rem_vals = [rem_slopes[s][metric][bin_idx] for s in rem_subjects
                        if bin_idx < len(rem_slopes[s][metric]) and np.isfinite(rem_slopes[s][metric][bin_idx])]
            if len(n3_vals) >= 3 and len(rem_vals) >= 3:
                d = _cohens_d_independent(np.array(n3_vals), np.array(rem_vals))
                per_bin_d[metric].append(d)
            else:
                per_bin_d[metric].append(float("nan"))
    return per_bin_d


def characterize_commitment_window(
    trajectory_result: TrajectoryPredictionResult,
    changepoint_result: ChangepointResult | None = None,
    extended_window_result: ExtendedWindowResult | None = None,
    d_thresholds: list[float] | None = None,
    n_bootstrap: int = 5000,
    n_permutations: int = 1000,
    seed: int = 42,
) -> CommitmentWindowResult:
    """Q5: Synthesize Q2 and Q4 results into a commitment window characterization.

    Scans per-bin Cohen's d (from trajectory_result) to identify the earliest
    pre-transition latency at which geometry drift direction becomes discriminative
    (d >= threshold) for at least one metric. Reports the commitment window as the
    absolute pre-transition interval from that onset to the staging boundary.

    Parameters
    ----------
    trajectory_result : TrajectoryPredictionResult
        Output from predict_transition_type_from_trajectory() (Q4). Must contain
        per_bin_cohens_d and time_bins_sec.
    changepoint_result : ChangepointResult or None
        Output from detect_commitment_changepoints() (Q2). Used to assess whether
        the changepoint CI overlaps with the identified discrimination onset.
    extended_window_result : ExtendedWindowResult or None
        Output from analyze_extended_window() (Q1). Used to extract geometry
        values at the identified onset time.
    d_thresholds : list[float] or None
        Cohen's d thresholds defining "discriminative". Default: [0.3, 0.5, 0.8].
        The primary result uses threshold=0.5.

    Returns
    -------
    CommitmentWindowResult

    Notes
    -----
    The discrimination onset is the EARLIEST (most negative) time bin at which
    d >= threshold for at least one metric. "Earliest" = furthest before boundary.
    The commitment window is abs(onset_sec): larger = earlier discrimination.
    Changepoint overlap: True if onset_sec is within changepoint group_latency_ci
    +/- 60 s (one epoch's worth of staging uncertainty).

    What this cannot establish:
    - The commitment window is an operational construct defined by the chosen
      threshold. It does not identify a biological "point of no return" in the
      mechanistic sense. Multiple thresholds are reported to show sensitivity.
    """
    if d_thresholds is None:
        d_thresholds = [0.3, 0.5, 0.8]

    rng = np.random.default_rng(seed)

    sorted_indices = np.argsort(trajectory_result.time_bins_sec)
    time_bins = [trajectory_result.time_bins_sec[i] for i in sorted_indices]
    per_bin_d = trajectory_result.per_bin_cohens_d
    metric_names = trajectory_result.metrics
    n_bins = len(time_bins)

    onset_by_threshold: dict = {}
    for thresh in d_thresholds:
        onset_by_threshold[thresh] = _find_onset_from_d_matrix(
            per_bin_d, time_bins, sorted_indices, thresh)

    primary_threshold = 0.5
    if primary_threshold not in d_thresholds:
        primary_threshold = d_thresholds[0]
    primary_onset = onset_by_threshold.get(primary_threshold)

    commitment_window_sec = abs(primary_onset) if primary_onset is not None else None

    mono_frac = float("nan")
    if primary_onset is not None:
        onset_sorted_pos = time_bins.index(primary_onset) if primary_onset in time_bins else None
        if onset_sorted_pos is not None:
            post_onset_bins = list(range(onset_sorted_pos, len(time_bins)))
            if len(post_onset_bins) > 0:
                exceeds = 0
                for sp in post_onset_bins:
                    oi = sorted_indices[sp]
                    for metric, d_vals in per_bin_d.items():
                        if oi < len(d_vals) and np.isfinite(d_vals[oi]) and abs(d_vals[oi]) >= primary_threshold:
                            exceeds += 1
                            break
                mono_frac = float(exceeds / len(post_onset_bins))

    onset_ci = None
    onset_perm_p = float("nan")
    per_subj_slopes = trajectory_result.per_subject_slopes

    has_slopes = (per_subj_slopes
                  and "n3" in per_subj_slopes and "rem" in per_subj_slopes
                  and len(per_subj_slopes["n3"]) >= 3 and len(per_subj_slopes["rem"]) >= 3)

    if has_slopes and primary_onset is not None:
        n3_slopes = per_subj_slopes["n3"]
        rem_slopes = per_subj_slopes["rem"]
        n3_subjs = sorted(n3_slopes.keys())
        rem_subjs = sorted(rem_slopes.keys())

        boot_onsets = []
        for _ in range(n_bootstrap):
            n3_idx = rng.choice(len(n3_subjs), size=len(n3_subjs), replace=True)
            rem_idx = rng.choice(len(rem_subjs), size=len(rem_subjs), replace=True)
            boot_n3 = {f"b{i}": n3_slopes[n3_subjs[j]] for i, j in enumerate(n3_idx)}
            boot_rem = {f"b{i}": rem_slopes[rem_subjs[j]] for i, j in enumerate(rem_idx)}
            boot_d = _compute_per_bin_d_from_slopes(boot_n3, boot_rem, metric_names, n_bins)
            boot_onset = _find_onset_from_d_matrix(boot_d, time_bins, sorted_indices, primary_threshold)
            boot_onsets.append(boot_onset if boot_onset is not None else float("nan"))

        finite_onsets = [x for x in boot_onsets if np.isfinite(x)]
        if len(finite_onsets) >= 100:
            onset_ci = (float(np.percentile(finite_onsets, 2.5)),
                        float(np.percentile(finite_onsets, 97.5)))

        all_keys = []
        all_slopes_list = []
        all_labels = []
        for s in n3_subjs:
            all_keys.append(f"n3_{s}")
            all_slopes_list.append(n3_slopes[s])
            all_labels.append("n3")
        for s in rem_subjs:
            all_keys.append(f"rem_{s}")
            all_slopes_list.append(rem_slopes[s])
            all_labels.append("rem")

        null_onsets = []
        for _ in range(n_permutations):
            perm_labels = rng.permutation(all_labels)
            perm_n3 = {}
            perm_rem = {}
            for idx_s in range(len(all_keys)):
                key = all_keys[idx_s]
                if perm_labels[idx_s] == "n3":
                    perm_n3[key] = all_slopes_list[idx_s]
                else:
                    perm_rem[key] = all_slopes_list[idx_s]
            if len(perm_n3) >= 3 and len(perm_rem) >= 3:
                perm_d = _compute_per_bin_d_from_slopes(perm_n3, perm_rem, metric_names, n_bins)
                perm_onset = _find_onset_from_d_matrix(perm_d, time_bins, sorted_indices, primary_threshold)
                null_onsets.append(perm_onset if perm_onset is not None else float("inf"))
            else:
                null_onsets.append(float("inf"))

        if primary_onset is not None:
            onset_perm_p = float(np.mean(np.array(null_onsets) <= primary_onset))

    geometry_at_onset: dict[str, float] = {}
    if primary_onset is not None and extended_window_result is not None:
        time_axis = extended_window_result.group_time_axis
        trajectory = extended_window_result.group_mean_trajectory
        if len(time_axis) > 0 and len(trajectory) > 0:
            idx = int(np.argmin(np.abs(time_axis - primary_onset)))
            if np.isfinite(trajectory[idx]):
                geometry_at_onset[extended_window_result.metric_name] = float(trajectory[idx])

    inter_subject_sd = None
    changepoint_overlap = None
    if changepoint_result is not None:
        if len(changepoint_result.per_subject_mean_latencies_sec) >= 2:
            valid_lats = [x for x in changepoint_result.per_subject_mean_latencies_sec
                          if np.isfinite(x)]
            if len(valid_lats) >= 2:
                inter_subject_sd = float(np.std(valid_lats, ddof=1))

        if primary_onset is not None:
            lo, hi = changepoint_result.group_latency_ci
            tol = 60.0
            if np.isfinite(lo) and np.isfinite(hi):
                changepoint_overlap = bool(
                    (primary_onset >= lo - tol) and (primary_onset <= hi + tol)
                )

    return CommitmentWindowResult(
        discrimination_onset_sec=primary_onset,
        commitment_window_sec=commitment_window_sec,
        geometry_at_onset=geometry_at_onset,
        inter_subject_sd_latency_sec=inter_subject_sd,
        changepoint_overlap=changepoint_overlap,
        thresholds_tested=list(d_thresholds),
        onset_by_threshold={float(k): v for k, v in onset_by_threshold.items()},
        onset_ci_sec=onset_ci,
        onset_permutation_p=onset_perm_p,
        monotonicity_fraction=mono_frac,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
    )
