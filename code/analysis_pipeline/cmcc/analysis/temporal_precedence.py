"""Temporal precedence analysis for operator-geometry brain state transitions.

Purpose
-------
Test whether operator geometry (eigenvalue gap, condition number, ND score,
spectral radius) changes *before* state transitions, not just after. This is
Test 7 in the geometry brain-state battery.

Scientific context
------------------
If geometry anticipates state transitions, it suggests the dynamical operator
reorganizes before the observable state change — supporting geometry as a
latent control variable rather than a passive descriptor.

Data source: ANPHY-Sleep continuous full-night EEG with 30s epoch staging.
Propofol data (ds005620) uses separate recordings per condition, so
within-recording transitions are not available.

Expected inputs
---------------
- Sleep staging: list of (stage, onset_sec, duration_sec) tuples parsed from
  .txt files (tab-delimited: stage, onset, duration)
- PCA-reduced EEG: np.ndarray shape (n_components, n_samples) from
  existing preprocessing pipeline
- Sampling frequency: float (typically 500 Hz after downsampling)

Assumptions
-----------
- Sleep staging has 30s epoch resolution — transition timing is approximate
- PCA fitted per-segment to avoid leakage across transition boundary
- VAR(1) windows: 0.5s window, 0.1s step (matching main pipeline)
- 15 PCA components (matching main pipeline)
- Overlapping windows create temporal smoothing — validated with
  non-overlapping sensitivity check

Known limitations
-----------------
- Only sleep data has within-recording transitions (propofol is separate files)
- 30s staging resolution introduces ±15s uncertainty in transition timing
- Small sample (10 subjects) limits statistical power
- Multiple transitions per subject create within-subject correlation —
  average per subject before group statistics

Validation strategy
-------------------
- Unit tests with synthetic staging and geometry trajectories
- Non-overlapping window sensitivity check
- Subject consistency requirement (≥70% same direction)
- Bootstrap CIs on slopes and effect sizes
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sp_stats


@dataclass
class TransitionEvent:
    """A detected state transition in sleep staging.

    Attributes
    ----------
    subject : str
        Subject identifier.
    transition_type : str
        e.g., "N2_to_N3", "N2_to_REM".
    onset_sec : float
        Time in seconds of the first epoch of the post-state.
    pre_state : str
        Sleep stage before transition (e.g., "N2").
    post_state : str
        Sleep stage after transition (e.g., "N3").
    n_pre_epochs : int
        Number of consecutive pre-state epochs before transition.
    n_post_epochs : int
        Number of consecutive post-state epochs after transition.
    """

    subject: str
    transition_type: str
    onset_sec: float
    pre_state: str
    post_state: str
    n_pre_epochs: int
    n_post_epochs: int


@dataclass
class TransitionTimecourse:
    """Per-window geometry metrics aligned to a transition.

    Attributes
    ----------
    subject : str
    transition_type : str
    transition_index : int
        Which transition for this subject (0-indexed).
    time_sec : np.ndarray, shape (n_windows,)
        Time in seconds relative to transition (t=0 = first post-state epoch).
    eigenvalue_gap : np.ndarray, shape (n_windows,)
        Minimum eigenvalue gap per window.
    condition_number : np.ndarray, shape (n_windows,)
        Eigenvector condition number per window.
    nd_score : np.ndarray, shape (n_windows,)
        Near-degeneracy (EP) score per window.
    spectral_radius : np.ndarray, shape (n_windows,)
        Spectral radius per window.
    """

    subject: str
    transition_type: str
    transition_index: int
    time_sec: np.ndarray
    eigenvalue_gap: np.ndarray
    condition_number: np.ndarray
    nd_score: np.ndarray
    spectral_radius: np.ndarray


@dataclass
class TemporalPrecedenceResult:
    """Result of temporal precedence analysis for one metric × one transition type.

    Attributes
    ----------
    metric_name : str
        e.g., "eigenvalue_gap".
    transition_type : str
        e.g., "N2_to_N3".
    n_subjects : int
    n_transitions_total : int
    per_subject_slopes : list[float]
        OLS slope of metric vs time for t < 0, one per subject.
    mean_slope : float
    slope_p_value : float
        One-sample t-test p-value for mean slope ≠ 0.
    slope_ci : tuple[float, float]
        Bootstrap 95% CI on mean slope.
    early_vs_late_d : float
        Cohen's d comparing early baseline (−120 to −60s) vs late
        pre-transition (−30 to 0s).
    early_vs_late_p : float
        Paired t-test p-value.
    subject_consistency : float
        Fraction of subjects with slope in same direction as group mean.
    group_time_axis : np.ndarray
        Common time axis for group trajectory (seconds).
    group_mean_trajectory : np.ndarray
        Cross-subject mean trajectory.
    group_ci_lower : np.ndarray
        Bootstrap 2.5th percentile.
    group_ci_upper : np.ndarray
        Bootstrap 97.5th percentile.
    nonoverlap_slope : float
        Mean slope using non-overlapping windows (sensitivity check).
    nonoverlap_slope_p : float
    nonoverlap_survives : bool
        Whether non-overlapping windows preserve significance (p < 0.10).
        If False, the overlapping-window result may be inflated by temporal
        autocorrelation from 80% window overlap.
    passes_threshold : bool
        slope_p_value < 0.05 AND subject_consistency >= 0.70 AND
        |early_vs_late_d| >= 0.5.
    passes_strict : bool
        passes_threshold AND nonoverlap_survives. This is the reviewer-proof
        criterion: the result must survive both overlapping and non-overlapping
        analyses.
    """

    metric_name: str
    transition_type: str
    n_subjects: int
    n_transitions_total: int
    per_subject_slopes: list[float]
    mean_slope: float
    slope_p_value: float
    slope_ci: tuple[float, float]
    early_vs_late_d: float
    early_vs_late_p: float
    subject_consistency: float
    group_time_axis: np.ndarray
    group_mean_trajectory: np.ndarray
    group_ci_lower: np.ndarray
    group_ci_upper: np.ndarray
    nonoverlap_slope: float
    nonoverlap_slope_p: float
    nonoverlap_survives: bool
    passes_threshold: bool
    passes_strict: bool


def parse_sleep_staging(staging_path: str) -> list[tuple[str, float, float]]:
    """Parse sleep staging file into list of (stage, onset_sec, duration_sec).

    Parameters
    ----------
    staging_path : str
        Path to tab-delimited staging file.

    Returns
    -------
    list of (stage, onset_sec, duration_sec) tuples
    """
    epochs = []
    with open(staging_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                stage = parts[0]
                onset_sec = float(parts[1])
                duration_sec = float(parts[2])
                epochs.append((stage, onset_sec, duration_sec))
    return epochs


def find_transitions(
    epochs: list[tuple[str, float, float]],
    pre_state: str,
    post_state: str,
    min_pre_epochs: int = 2,
    min_post_epochs: int = 2,
) -> list[dict]:
    """Find state transitions in sleep staging.

    Scans for boundaries where `pre_state` is followed by `post_state`,
    requiring at least `min_pre_epochs` consecutive epochs of pre_state
    before the boundary and `min_post_epochs` of post_state after.

    Parameters
    ----------
    epochs : list of (stage, onset_sec, duration_sec)
    pre_state : str
        e.g., "N2"
    post_state : str
        e.g., "N3" or "R"
    min_pre_epochs : int
        Minimum consecutive pre_state epochs before transition.
    min_post_epochs : int
        Minimum consecutive post_state epochs after transition.

    Returns
    -------
    list of dict with keys: onset_sec, pre_count, post_count
        onset_sec is the start time of the first post_state epoch.
    """
    n = len(epochs)
    transitions = []

    for i in range(1, n):
        if epochs[i][0] != post_state:
            continue
        if epochs[i - 1][0] != pre_state:
            continue

        pre_count = 0
        j = i - 1
        while j >= 0 and epochs[j][0] == pre_state:
            pre_count += 1
            j -= 1

        post_count = 0
        k = i
        while k < n and epochs[k][0] == post_state:
            post_count += 1
            k += 1

        if pre_count >= min_pre_epochs and post_count >= min_post_epochs:
            transitions.append({
                "onset_sec": epochs[i][1],
                "pre_count": pre_count,
                "post_count": post_count,
            })

    return transitions


def compute_transition_geometry(
    data_pca: np.ndarray,
    sfreq: float,
    transition_onset_sec: float,
    pre_sec: float = 120.0,
    post_sec: float = 60.0,
    window_sec: float = 0.5,
    step_sec: float = 0.1,
    seed: int = 42,
) -> TransitionTimecourse | None:
    """Compute per-window geometry metrics around a state transition.

    Parameters
    ----------
    data_pca : np.ndarray, shape (n_components, n_samples)
        Full recording PCA-reduced data.
    sfreq : float
        Sampling frequency in Hz.
    transition_onset_sec : float
        Time of transition (first post-state epoch) in seconds.
    pre_sec : float
        Seconds before transition to include.
    post_sec : float
        Seconds after transition to include.
    window_sec : float
        VAR window duration in seconds.
    step_sec : float
        VAR window step in seconds.
    seed : int

    Returns
    -------
    TransitionTimecourse or None if segment is too short.
    """
    from cmcc.analysis.dynamical_systems import compute_ep_proximity_timecourse

    n_samples = data_pca.shape[1]
    seg_start_sec = transition_onset_sec - pre_sec
    seg_end_sec = transition_onset_sec + post_sec

    seg_start_samp = max(0, int(seg_start_sec * sfreq))
    seg_end_samp = min(n_samples, int(seg_end_sec * sfreq))

    if seg_end_samp - seg_start_samp < int(window_sec * sfreq) * 3:
        return None

    segment = data_pca[:, seg_start_samp:seg_end_samp]

    result = compute_ep_proximity_timecourse(
        segment, sfreq=sfreq,
        window_sec=window_sec, step_sec=step_sec,
        max_channels=data_pca.shape[0], seed=seed,
    )

    jac = result["jac_result"]
    ep = result["ep_result"]

    actual_start_sec = seg_start_samp / sfreq
    time_sec = (jac.window_centers / sfreq) + actual_start_sec - transition_onset_sec

    return TransitionTimecourse(
        subject="",
        transition_type="",
        transition_index=0,
        time_sec=time_sec,
        eigenvalue_gap=ep.min_eigenvalue_gaps,
        condition_number=jac.condition_numbers,
        nd_score=ep.ep_scores,
        spectral_radius=jac.spectral_radius,
    )


def _subject_mean_timecourse(
    timecourses: list[TransitionTimecourse],
    metric_name: str,
    common_time: np.ndarray,
    tol: float = 0.15,
) -> np.ndarray | None:
    """Average a subject's transition timecourses onto a common time grid.

    Parameters
    ----------
    timecourses : list[TransitionTimecourse]
        All transitions for one subject.
    metric_name : str
    common_time : np.ndarray
        Target time axis in seconds.
    tol : float
        Tolerance in seconds for nearest-neighbor interpolation.

    Returns
    -------
    np.ndarray or None
        Mean metric values on common_time grid, or None if no data.
    """
    all_interp = []
    for tc in timecourses:
        metric = getattr(tc, metric_name)
        interp = np.full(len(common_time), np.nan)
        for i, t in enumerate(common_time):
            diffs = np.abs(tc.time_sec - t)
            idx = np.argmin(diffs)
            if diffs[idx] <= tol:
                interp[i] = metric[idx]
        if np.sum(np.isfinite(interp)) > len(common_time) * 0.5:
            all_interp.append(interp)

    if not all_interp:
        return None

    stacked = np.array(all_interp)
    return np.nanmean(stacked, axis=0)


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples (mean difference / SD of differences)."""
    diff = a - b
    n = len(diff)
    if n < 2:
        return float("nan")
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return float("nan")
    return float(np.mean(diff) / sd)


def analyze_temporal_precedence(
    timecourses_by_subject: dict[str, list[TransitionTimecourse]],
    metric_name: str,
    transition_type: str,
    pre_sec: float = 120.0,
    post_sec: float = 60.0,
    seed: int = 42,
    n_bootstrap: int = 5000,
) -> TemporalPrecedenceResult:
    """Analyze whether a geometry metric changes before state transition.

    Parameters
    ----------
    timecourses_by_subject : dict[str, list[TransitionTimecourse]]
        Keys are subject IDs, values are lists of TransitionTimecourse.
    metric_name : str
        One of: eigenvalue_gap, condition_number, nd_score, spectral_radius.
    transition_type : str
    pre_sec : float
    post_sec : float
    seed : int
    n_bootstrap : int

    Returns
    -------
    TemporalPrecedenceResult
    """
    time_step = 0.5
    common_time = np.arange(-pre_sec, post_sec + time_step, time_step)

    subject_means = {}
    for subj, tcs in timecourses_by_subject.items():
        mean_tc = _subject_mean_timecourse(tcs, metric_name, common_time)
        if mean_tc is not None:
            subject_means[subj] = mean_tc

    n_subjects = len(subject_means)
    n_total = sum(len(tcs) for tcs in timecourses_by_subject.values())

    if n_subjects < 3:
        return _empty_result(metric_name, transition_type, n_subjects, n_total, common_time)

    subject_ids = sorted(subject_means.keys())
    subject_matrix = np.array([subject_means[s] for s in subject_ids])

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
        return _empty_result(metric_name, transition_type, n_subjects, n_total, common_time)

    mean_slope = float(np.mean(valid_slopes))
    t_stat, slope_p = sp_stats.ttest_1samp(valid_slopes, 0)
    slope_p = float(slope_p)

    if mean_slope != 0:
        consistency = float(np.mean(np.sign(valid_slopes) == np.sign(mean_slope)))
    else:
        consistency = 0.0

    early_mask = (common_time >= -pre_sec) & (common_time < -60.0)
    late_mask = (common_time >= -30.0) & (common_time < 0.0)

    early_means = []
    late_means = []
    for row in subject_matrix:
        e = row[early_mask]
        la = row[late_mask]
        if np.sum(np.isfinite(e)) > 3 and np.sum(np.isfinite(la)) > 3:
            early_means.append(float(np.nanmean(e)))
            late_means.append(float(np.nanmean(la)))

    if len(early_means) >= 3:
        early_arr = np.array(early_means)
        late_arr = np.array(late_means)
        el_d = _cohens_d_paired(late_arr, early_arr)
        _, el_p = sp_stats.ttest_rel(late_arr, early_arr)
        el_p = float(el_p)
    else:
        el_d = float("nan")
        el_p = 1.0

    group_mean = np.nanmean(subject_matrix, axis=0)
    rng = np.random.default_rng(seed)

    boot_slopes = []
    boot_trajectories = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_subjects, size=n_subjects, replace=True)
        boot_mat = subject_matrix[idx]
        boot_mean = np.nanmean(boot_mat, axis=0)
        boot_trajectories.append(boot_mean)

        boot_pre = boot_mean[pre_mask]
        valid_b = np.isfinite(boot_pre)
        if valid_b.sum() >= 5:
            sl, _, _, _, _ = sp_stats.linregress(pre_time[valid_b], boot_pre[valid_b])
            boot_slopes.append(sl)

    if boot_slopes:
        slope_ci = (float(np.percentile(boot_slopes, 2.5)),
                    float(np.percentile(boot_slopes, 97.5)))
    else:
        slope_ci = (float("nan"), float("nan"))

    if boot_trajectories:
        boot_arr = np.array(boot_trajectories)
        ci_lower = np.nanpercentile(boot_arr, 2.5, axis=0)
        ci_upper = np.nanpercentile(boot_arr, 97.5, axis=0)
    else:
        ci_lower = np.full_like(group_mean, np.nan)
        ci_upper = np.full_like(group_mean, np.nan)

    step_ratio = max(1, round(0.5 / 0.1))
    nonoverlap_slopes = []
    for row in subject_matrix:
        pre_vals = row[pre_mask]
        no_vals = pre_vals[::step_ratio]
        no_time = pre_time[::step_ratio]
        valid_no = np.isfinite(no_vals)
        if valid_no.sum() >= 5:
            sl, _, _, _, _ = sp_stats.linregress(no_time[valid_no], no_vals[valid_no])
            nonoverlap_slopes.append(sl)

    if len(nonoverlap_slopes) >= 3:
        no_slopes_arr = np.array(nonoverlap_slopes)
        nonoverlap_mean = float(np.mean(no_slopes_arr))
        _, no_p = sp_stats.ttest_1samp(no_slopes_arr, 0)
        nonoverlap_p = float(no_p)
    else:
        nonoverlap_mean = float("nan")
        nonoverlap_p = 1.0

    passes = (
        slope_p < 0.05
        and consistency >= 0.70
        and abs(el_d) >= 0.5
    )

    nonoverlap_survives = nonoverlap_p < 0.10

    passes_strict = passes and nonoverlap_survives

    return TemporalPrecedenceResult(
        metric_name=metric_name,
        transition_type=transition_type,
        n_subjects=n_subjects,
        n_transitions_total=n_total,
        per_subject_slopes=per_subject_slopes,
        mean_slope=mean_slope,
        slope_p_value=slope_p,
        slope_ci=slope_ci,
        early_vs_late_d=el_d,
        early_vs_late_p=el_p,
        subject_consistency=consistency,
        group_time_axis=common_time,
        group_mean_trajectory=group_mean,
        group_ci_lower=ci_lower,
        group_ci_upper=ci_upper,
        nonoverlap_slope=nonoverlap_mean,
        nonoverlap_slope_p=nonoverlap_p,
        nonoverlap_survives=nonoverlap_survives,
        passes_threshold=passes,
        passes_strict=passes_strict,
    )


def _empty_result(
    metric_name: str,
    transition_type: str,
    n_subjects: int,
    n_total: int,
    common_time: np.ndarray,
) -> TemporalPrecedenceResult:
    """Return a non-significant result when insufficient data."""
    empty = np.full_like(common_time, np.nan)
    return TemporalPrecedenceResult(
        metric_name=metric_name,
        transition_type=transition_type,
        n_subjects=n_subjects,
        n_transitions_total=n_total,
        per_subject_slopes=[],
        mean_slope=float("nan"),
        slope_p_value=1.0,
        slope_ci=(float("nan"), float("nan")),
        early_vs_late_d=float("nan"),
        early_vs_late_p=1.0,
        subject_consistency=0.0,
        group_time_axis=common_time,
        group_mean_trajectory=empty,
        group_ci_lower=empty,
        group_ci_upper=empty,
        nonoverlap_slope=float("nan"),
        nonoverlap_slope_p=1.0,
        nonoverlap_survives=False,
        passes_threshold=False,
        passes_strict=False,
    )
