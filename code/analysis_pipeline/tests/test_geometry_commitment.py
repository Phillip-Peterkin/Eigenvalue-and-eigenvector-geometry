"""
Tests for cmcc.analysis.geometry_commitment

Validates all 5 sub-analyses of the Geometric Transition Commitment Analysis:
  Q1 (ExtendedWindow)     — Steps 29
  Q2 (Changepoint)        — Steps 30
  Q3 (Bistability)        — Steps 31
  Q4 (TrajectoryPredict)  — Steps 32
  Q5 (Commitment)         — Steps 33
"""
from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmcc.analysis.temporal_precedence import TransitionTimecourse
from cmcc.analysis.geometry_commitment import (
    ExtendedWindowResult,
    ChangepointResult,
    BistabilityResult,
    TrajectoryPredictionResult,
    CommitmentWindowResult,
    analyze_extended_window,
    detect_commitment_changepoints,
    analyze_n2_bistability,
    predict_transition_type_from_trajectory,
    characterize_commitment_window,
    _cohens_d_independent,
    _cohens_d_paired,
    _silverman_test,
    _gmm_cv_log_likelihood_delta,
    _compute_per_bin_d_from_slopes,
)


def _make_tc(
    subject: str,
    transition_index: int,
    time_start: float,
    time_end: float,
    step: float,
    spectral_radius_fn=None,
    eigenvalue_gap_fn=None,
    condition_number_fn=None,
    nd_score_fn=None,
) -> TransitionTimecourse:
    """Build a synthetic TransitionTimecourse for testing."""
    t = np.arange(time_start, time_end + step / 2, step)
    if spectral_radius_fn is None:
        spectral_radius_fn = lambda tt: np.ones_like(tt) * 0.95
    if eigenvalue_gap_fn is None:
        eigenvalue_gap_fn = lambda tt: np.ones_like(tt) * 0.001
    if condition_number_fn is None:
        condition_number_fn = lambda tt: np.ones_like(tt) * 5.0
    if nd_score_fn is None:
        nd_score_fn = lambda tt: np.ones_like(tt) * 0.1
    return TransitionTimecourse(
        subject=subject,
        transition_type="N2_to_N3",
        transition_index=transition_index,
        time_sec=t,
        eigenvalue_gap=eigenvalue_gap_fn(t),
        condition_number=condition_number_fn(t),
        nd_score=nd_score_fn(t),
        spectral_radius=spectral_radius_fn(t),
    )


class TestExtendedWindow:

    def _make_subjects(self, n_subjects=4, time_start=-620.0, time_end=60.0,
                       step=0.5, sr_fn=None):
        """Create synthetic subject dict with linear increasing spectral_radius."""
        subjects = {}
        if sr_fn is None:
            sr_fn = lambda t: 0.95 + 1e-5 * (t - time_start)
        for i in range(n_subjects):
            sid = f"S{i:02d}"
            tc = _make_tc(sid, 0, time_start, time_end, step, spectral_radius_fn=sr_fn)
            subjects[sid] = [tc]
        return subjects

    def test_extended_window_filters_short_timecourses(self):
        """Timecourses that don't reach -600 s should be excluded."""
        sr_fn = lambda t: 0.95 + 1e-5 * t
        subjects = {}
        for i in range(5):
            sid = f"S{i:02d}"
            if i < 3:
                tc = _make_tc(sid, 0, -620.0, 60.0, 0.5, spectral_radius_fn=sr_fn)
            else:
                tc = _make_tc(sid, 0, -100.0, 60.0, 0.5, spectral_radius_fn=sr_fn)
            subjects[sid] = [tc]

        result = analyze_extended_window(subjects, "spectral_radius",
                                         pre_sec=600.0, min_qualifying_pre_sec=600.0,
                                         seed=0, n_bootstrap=100)
        assert result.n_subjects == 3, (
            f"Expected 3 qualifying subjects, got {result.n_subjects}"
        )

    def test_extended_window_slope_direction(self):
        """Monotonically increasing trajectory -> positive mean slope."""
        subjects = self._make_subjects(
            n_subjects=5,
            sr_fn=lambda t: 0.9 + 5e-5 * (t + 600),
        )
        result = analyze_extended_window(subjects, "spectral_radius",
                                         pre_sec=600.0, min_qualifying_pre_sec=600.0,
                                         seed=0, n_bootstrap=200)
        assert result.mean_slope > 0, f"Expected positive slope, got {result.mean_slope}"
        assert result.subject_consistency == 1.0

    def test_extended_window_flat_trajectory(self):
        """Flat trajectory -> slope near zero and p-value not significant."""
        subjects = self._make_subjects(
            n_subjects=5,
            sr_fn=lambda t: np.ones_like(t) * 0.95,
        )
        result = analyze_extended_window(subjects, "spectral_radius",
                                         pre_sec=600.0, min_qualifying_pre_sec=600.0,
                                         seed=0, n_bootstrap=200)
        assert abs(result.mean_slope) < 1e-10, f"Expected ~0 slope, got {result.mean_slope}"
        assert result.slope_p_value > 0.05, f"Expected non-significant, got p={result.slope_p_value}"
        assert not result.passes_threshold

    def test_extended_window_aic_quadratic_wins(self):
        """Quadratic (accelerating) signal -> aic_delta < 0 (quadratic AIC lower)."""
        def sr_fn(t):
            t_norm = (t + 600) / 600.0
            return 0.9 + 0.1 * t_norm ** 2

        subjects = self._make_subjects(n_subjects=5, sr_fn=sr_fn)
        result = analyze_extended_window(subjects, "spectral_radius",
                                         pre_sec=600.0, min_qualifying_pre_sec=600.0,
                                         seed=0, n_bootstrap=100)
        assert result.aic_delta < 0, (
            f"Expected quadratic to win (aic_delta < 0), got {result.aic_delta}"
        )

    def test_extended_window_linear_wins_for_linear(self):
        """Linear signal -> quadratic should not appreciably beat linear."""
        subjects = self._make_subjects(
            n_subjects=5,
            sr_fn=lambda t: 0.9 + 5e-5 * (t + 600),
        )
        result = analyze_extended_window(subjects, "spectral_radius",
                                         pre_sec=600.0, min_qualifying_pre_sec=600.0,
                                         seed=0, n_bootstrap=100)
        assert np.isfinite(result.linear_aic)
        assert np.isfinite(result.quadratic_aic)

    def test_extended_window_insufficient_subjects_returns_nan_result(self):
        """Fewer than 3 subjects -> nan result with passes_threshold=False."""
        subjects = self._make_subjects(n_subjects=2)
        result = analyze_extended_window(subjects, "spectral_radius",
                                         pre_sec=600.0, min_qualifying_pre_sec=600.0,
                                         seed=0, n_bootstrap=100)
        assert not result.passes_threshold
        assert np.isnan(result.mean_slope)

    def test_extended_window_metric_eigenvalue_gap(self):
        """Function works for eigenvalue_gap metric."""
        def eg_fn(t): return 0.001 + 5e-8 * (t + 600)
        subjects = {}
        for i in range(4):
            sid = f"S{i:02d}"
            tc = _make_tc(sid, 0, -620.0, 60.0, 0.5, eigenvalue_gap_fn=eg_fn)
            subjects[sid] = [tc]

        result = analyze_extended_window(subjects, "eigenvalue_gap",
                                         pre_sec=600.0, min_qualifying_pre_sec=600.0,
                                         seed=0, n_bootstrap=100)
        assert result.mean_slope > 0
        assert result.n_subjects == 4


class TestExtendedWindowProperties:

    def _make_noisy_subjects(self, n_subjects=6, seed=7):
        rng = np.random.default_rng(seed)
        t = np.arange(-620.0, 60.5, 0.5)
        subjects = {}
        for i in range(n_subjects):
            sid = f"S{i:02d}"
            signal = 0.9 + 3e-5 * (t + 600) + rng.normal(0, 0.002, size=len(t))
            tc = TransitionTimecourse(
                subject=sid, transition_type="N2_to_N3", transition_index=0,
                time_sec=t.copy(), eigenvalue_gap=np.ones_like(t) * 0.001,
                condition_number=np.ones_like(t) * 5.0, nd_score=np.ones_like(t) * 0.1,
                spectral_radius=signal,
            )
            subjects[sid] = [tc]
        return subjects

    def test_slope_ci_contains_mean_slope(self):
        """Bootstrap CI must contain the mean slope."""
        subjects = self._make_noisy_subjects()
        result = analyze_extended_window(subjects, "spectral_radius",
                                         pre_sec=600.0, min_qualifying_pre_sec=600.0,
                                         seed=42, n_bootstrap=1000)
        assert np.isfinite(result.mean_slope)
        assert np.isfinite(result.slope_ci[0]) and np.isfinite(result.slope_ci[1])
        assert result.slope_ci[0] <= result.mean_slope <= result.slope_ci[1], (
            f"CI [{result.slope_ci[0]:.2e}, {result.slope_ci[1]:.2e}] "
            f"does not contain mean slope {result.mean_slope:.2e}"
        )

    def test_subject_consistency_in_unit_interval(self):
        """subject_consistency must be in [0, 1]."""
        subjects = self._make_noisy_subjects()
        result = analyze_extended_window(subjects, "spectral_radius",
                                         pre_sec=600.0, min_qualifying_pre_sec=600.0,
                                         seed=42, n_bootstrap=200)
        assert 0.0 <= result.subject_consistency <= 1.0

    def test_group_trajectory_shape_matches_time_axis(self):
        """group_mean_trajectory, ci_lower, ci_upper must match group_time_axis."""
        subjects = self._make_noisy_subjects()
        result = analyze_extended_window(subjects, "spectral_radius",
                                         pre_sec=600.0, min_qualifying_pre_sec=600.0,
                                         seed=42, n_bootstrap=100)
        n = len(result.group_time_axis)
        assert result.group_mean_trajectory.shape == (n,)
        assert result.group_ci_lower.shape == (n,)
        assert result.group_ci_upper.shape == (n,)

    def test_ci_lower_leq_ci_upper(self):
        """Pointwise: CI lower <= CI upper everywhere finite."""
        subjects = self._make_noisy_subjects()
        result = analyze_extended_window(subjects, "spectral_radius",
                                         pre_sec=600.0, min_qualifying_pre_sec=600.0,
                                         seed=42, n_bootstrap=100)
        finite = np.isfinite(result.group_ci_lower) & np.isfinite(result.group_ci_upper)
        assert np.all(result.group_ci_lower[finite] <= result.group_ci_upper[finite])


class TestChangepointDetection:

    def _make_piecewise_subjects(self, n_subjects=5, breakpoint_sec=-200.0,
                                  flat_val=0.95, slope_after=5e-4, seed=0):
        rng = np.random.default_rng(seed)
        subjects = {}
        for i in range(n_subjects):
            sid = f"S{i:02d}"
            t = np.arange(-500.0, 0.5, 0.5)
            y = np.where(t < breakpoint_sec,
                         flat_val + rng.normal(0, 1e-4, size=len(t)),
                         flat_val + slope_after * (t - breakpoint_sec) + rng.normal(0, 1e-4, size=len(t)))
            tc = TransitionTimecourse(
                subject=sid, transition_type="N2_to_N3", transition_index=0,
                time_sec=t, eigenvalue_gap=np.ones_like(t) * 0.001,
                condition_number=np.ones_like(t) * 5.0, nd_score=np.ones_like(t) * 0.1,
                spectral_radius=y,
            )
            subjects[sid] = [tc]
        return subjects

    def test_changepoint_detects_midpoint(self):
        """Piecewise signal with breakpoint at -200 s should be detected near -200 s."""
        subjects = self._make_piecewise_subjects(breakpoint_sec=-200.0)
        result = detect_commitment_changepoints(
            subjects, "spectral_radius", pre_sec=500.0,
            min_segment_sec=30.0, min_qualifying_pre_sec=240.0, seed=0,
        )
        assert isinstance(result, ChangepointResult)
        assert result.n_subjects >= 3
        assert np.isfinite(result.group_mean_latency_sec)
        assert -300.0 <= result.group_mean_latency_sec <= -100.0, (
            f"Expected changepoint near -200 s, got {result.group_mean_latency_sec:.1f}"
        )

    def test_changepoint_flat_before_breakpoint(self):
        """Breakpoint near end (-50 s): flat before, steep after."""
        subjects = self._make_piecewise_subjects(breakpoint_sec=-50.0, slope_after=1e-3)
        result = detect_commitment_changepoints(
            subjects, "spectral_radius", pre_sec=500.0,
            min_segment_sec=30.0, min_qualifying_pre_sec=240.0, seed=0,
        )
        assert result.n_qualifying_transitions >= 3
        if np.isfinite(result.group_mean_latency_sec):
            assert result.group_mean_latency_sec > -200.0

    def test_changepoint_skips_insufficient_coverage(self):
        """Short timecourses (< min_qualifying_pre_sec) should be excluded."""
        subjects = {}
        for i in range(4):
            sid = f"S{i:02d}"
            t = np.arange(-100.0, 0.5, 0.5)
            y = np.ones_like(t) * 0.95
            tc = TransitionTimecourse(
                subject=sid, transition_type="N2_to_N3", transition_index=0,
                time_sec=t, eigenvalue_gap=np.ones_like(t) * 0.001,
                condition_number=np.ones_like(t) * 5.0, nd_score=np.ones_like(t) * 0.1,
                spectral_radius=y,
            )
            subjects[sid] = [tc]
        result = detect_commitment_changepoints(
            subjects, "spectral_radius", pre_sec=500.0,
            min_segment_sec=30.0, min_qualifying_pre_sec=240.0, seed=0,
        )
        assert result.n_qualifying_transitions == 0
        assert result.n_subjects == 0

    def test_changepoint_sensitivity_consistent(self):
        """Sensitivity results should have entries for each penalty value."""
        subjects = self._make_piecewise_subjects()
        result = detect_commitment_changepoints(
            subjects, "spectral_radius", pre_sec=500.0,
            min_segment_sec=30.0, min_qualifying_pre_sec=240.0,
            penalty_values=[20.0, 30.0, 60.0], seed=0,
        )
        assert "min_seg_20s" in result.penalty_sensitivity
        assert "min_seg_30s" in result.penalty_sensitivity
        assert "min_seg_60s" in result.penalty_sensitivity
        for key, val in result.penalty_sensitivity.items():
            assert "group_mean_latency_sec" in val
            assert "group_p" in val

    def test_changepoint_latency_in_valid_range(self):
        """All finite per-transition latencies must be in [-pre_sec, 0]."""
        subjects = self._make_piecewise_subjects()
        result = detect_commitment_changepoints(
            subjects, "spectral_radius", pre_sec=500.0,
            min_segment_sec=30.0, min_qualifying_pre_sec=240.0, seed=0,
        )
        for lat in result.per_transition_latencies_sec:
            if np.isfinite(lat):
                assert -500.0 <= lat <= 0.0, f"Latency {lat} out of range [-500, 0]"


METRIC_NAMES = ["spectral_radius", "eigenvalue_gap", "condition_number", "nd_score"]


def _make_window_dict(subjects, n_windows_per_subj, means, stds, seed=0):
    """Build dict[str, np.ndarray] of shape (n_windows, 4) per subject."""
    rng = np.random.default_rng(seed)
    result = {}
    for i, s in enumerate(subjects):
        arr = rng.normal(means, stds, size=(n_windows_per_subj, len(means)))
        result[s] = arr
    return result


class TestBistability:

    def test_bistability_unimodal(self):
        """Gaussian early and late from same distribution -> GMM delta near 0, BC not significant."""
        subjs = [f"S{i:02d}" for i in range(8)]
        means = np.array([0.95, 0.002, 5.0, 0.1])
        stds = np.array([0.01, 0.0002, 0.5, 0.01])
        early = _make_window_dict(subjs, 30, means, stds, seed=1)
        late = _make_window_dict(subjs, 30, means, stds, seed=2)
        result = analyze_n2_bistability(early, late, METRIC_NAMES, seed=42, n_bootstrap_dip=200)
        assert result.n_early_windows == 8 * 30
        assert result.n_late_windows == 8 * 30
        assert np.isfinite(result.gmm_bic_1) and np.isfinite(result.gmm_bic_2)
        assert np.isfinite(result.gmm_bic_delta)

    def test_bistability_bimodal(self):
        """Well-separated early and late clusters -> GMM ΔBIC < -10."""
        subjs = [f"S{i:02d}" for i in range(8)]
        means_early = np.array([0.90, 0.001, 5.0, 0.1])
        means_late = np.array([0.98, 0.003, 8.0, 0.2])
        stds = np.array([0.005, 0.0002, 0.3, 0.005])
        early = _make_window_dict(subjs, 50, means_early, stds, seed=1)
        late = _make_window_dict(subjs, 50, means_late, stds, seed=2)
        result = analyze_n2_bistability(early, late, METRIC_NAMES, seed=42, n_bootstrap_dip=200)
        assert result.gmm_bic_delta < 0, (
            f"Expected GMM ΔBIC < 0 for bimodal data, got {result.gmm_bic_delta:.1f}"
        )

    def test_bistability_early_vs_late_d(self):
        """Late > early for spectral_radius -> d_spectral_radius > 0."""
        subjs = [f"S{i:02d}" for i in range(6)]
        means_early = np.array([0.90, 0.001, 5.0, 0.1])
        means_late = np.array([0.98, 0.003, 5.0, 0.1])
        stds = np.array([0.003, 0.0001, 0.3, 0.005])
        early = _make_window_dict(subjs, 30, means_early, stds, seed=1)
        late = _make_window_dict(subjs, 30, means_late, stds, seed=2)
        result = analyze_n2_bistability(early, late, METRIC_NAMES, seed=42, n_bootstrap_dip=100)
        assert result.early_vs_late_d_spectral_radius > 0, (
            f"Expected d > 0 (late > early), got {result.early_vs_late_d_spectral_radius:.3f}"
        )
        assert result.early_vs_late_d_eigenvalue_gap > 0

    def test_bistability_gmm_bic_positive(self):
        """GMM BIC values must be positive (finite, real-valued)."""
        subjs = [f"S{i:02d}" for i in range(5)]
        means = np.array([0.95, 0.002, 5.0, 0.1])
        stds = np.array([0.01, 0.0002, 0.5, 0.01])
        early = _make_window_dict(subjs, 20, means, stds, seed=0)
        late = _make_window_dict(subjs, 20, means, stds, seed=1)
        result = analyze_n2_bistability(early, late, METRIC_NAMES, seed=0, n_bootstrap_dip=100)
        assert np.isfinite(result.gmm_bic_1), f"gmm_bic_1 must be finite, got {result.gmm_bic_1}"
        assert np.isfinite(result.gmm_bic_2), f"gmm_bic_2 must be finite, got {result.gmm_bic_2}"

    def test_bistability_pca_coords_shape(self):
        """pca_coords_early and pca_coords_late have shape (n_windows, 2)."""
        subjs = [f"S{i:02d}" for i in range(5)]
        means = np.array([0.95, 0.002, 5.0, 0.1])
        stds = np.array([0.01, 0.0002, 0.5, 0.01])
        n = 20
        early = _make_window_dict(subjs, n, means, stds, seed=0)
        late = _make_window_dict(subjs, n, means, stds, seed=1)
        result = analyze_n2_bistability(early, late, METRIC_NAMES, seed=0, n_bootstrap_dip=50)
        assert result.pca_coords_early.shape[1] == 2
        assert result.pca_coords_late.shape[1] == 2
        assert result.pca_coords_early.shape[0] == 5 * n
        assert result.pca_coords_late.shape[0] == 5 * n

    def test_bistability_empty_input_returns_nan(self):
        """Empty input -> NaN result, no crash."""
        result = analyze_n2_bistability({}, {}, METRIC_NAMES, seed=0, n_bootstrap_dip=10)
        assert result.n_early_windows == 0
        assert result.n_late_windows == 0
        assert np.isnan(result.dip_test_statistic)


def _make_tc_with_slope(
    subject: str, transition_index: int,
    t_start: float, t_end: float, step: float,
    sr_slope: float = 0.0, eg_slope: float = 0.0,
    seed: int = 0,
) -> TransitionTimecourse:
    """Build a TC with a linear spectral_radius slope from t_start to t_end."""
    rng = np.random.default_rng(seed)
    t = np.arange(t_start, t_end + step / 2, step)
    sr = 0.95 + sr_slope * t + rng.normal(0, 1e-5, size=len(t))
    eg = 0.002 + eg_slope * t + rng.normal(0, 1e-7, size=len(t))
    return TransitionTimecourse(
        subject=subject, transition_type="N2_to_N3",
        transition_index=transition_index, time_sec=t,
        eigenvalue_gap=eg, condition_number=np.ones_like(t) * 5.0,
        nd_score=np.ones_like(t) * 0.1, spectral_radius=sr,
    )


class TestTrajectoryPrediction:

    def _make_n3_rem_subjects(self, n_subj=8, n3_sr_slope=1e-4, rem_sr_slope=-1e-4):
        """Create N3-bound and REM-bound subjects with distinct slope directions."""
        n3_dict = {}
        rem_dict = {}
        for i in range(n_subj):
            sid = f"S{i:02d}"
            n3_tc = _make_tc_with_slope(sid, 0, -130.0, 0.0, 0.5,
                                         sr_slope=n3_sr_slope, seed=i)
            n3_dict[sid] = [n3_tc]
            rem_tc = _make_tc_with_slope(sid + "_R", 0, -130.0, 0.0, 0.5,
                                          sr_slope=rem_sr_slope, seed=i + 100)
            rem_dict[sid + "_R"] = [rem_tc]
        return n3_dict, rem_dict

    def test_trajectory_prediction_perfect_separation(self):
        """N3 slopes strongly positive, REM strongly negative -> AUC > 0.7."""
        n3, rem = self._make_n3_rem_subjects(n_subj=6, n3_sr_slope=5e-4, rem_sr_slope=-5e-4)
        result = predict_transition_type_from_trajectory(
            n3, rem, METRIC_NAMES,
            time_bins_sec=[-120.0, -60.0], slope_halfwindow_sec=30.0,
            n_permutations=20, seed=42,
        )
        assert result.n_n3_bound == 6
        assert result.n_rem_bound == 6
        assert np.isfinite(result.loso_auc)
        assert result.loso_auc > 0.6, f"Expected AUC > 0.6 for separated data, got {result.loso_auc:.3f}"

    def test_trajectory_prediction_no_separation(self):
        """Same slope for N3 and REM -> AUC near 0.5."""
        n3, rem = self._make_n3_rem_subjects(n_subj=6, n3_sr_slope=1e-5, rem_sr_slope=1e-5)
        result = predict_transition_type_from_trajectory(
            n3, rem, METRIC_NAMES,
            time_bins_sec=[-120.0, -60.0], slope_halfwindow_sec=30.0,
            n_permutations=20, seed=42,
        )
        assert np.isfinite(result.loso_auc)
        assert result.loso_auc < 0.8, (
            f"Expected AUC < 0.8 for unseparated data, got {result.loso_auc:.3f}"
        )

    def test_trajectory_prediction_permutation_null_centered(self):
        """Permutation null AUC should be near 0.5 on average."""
        n3, rem = self._make_n3_rem_subjects(n_subj=8, n3_sr_slope=1e-5, rem_sr_slope=1e-5)
        result = predict_transition_type_from_trajectory(
            n3, rem, METRIC_NAMES,
            time_bins_sec=[-120.0], slope_halfwindow_sec=30.0,
            n_permutations=50, seed=0,
        )
        if result.permutation_null_auc:
            null_mean = float(np.mean(result.permutation_null_auc))
            assert 0.25 <= null_mean <= 0.75, (
                f"Permutation null mean AUC expected in [0.25, 0.75], got {null_mean:.3f}"
            )

    def test_trajectory_prediction_loso_respects_subject_boundary(self):
        """Each subject is held out exactly once and only contributes one test prediction."""
        n3, rem = self._make_n3_rem_subjects(n_subj=5, n3_sr_slope=3e-4, rem_sr_slope=-3e-4)
        result = predict_transition_type_from_trajectory(
            n3, rem, METRIC_NAMES,
            time_bins_sec=[-120.0], slope_halfwindow_sec=30.0,
            n_permutations=5, seed=42,
        )
        assert result.n_n3_bound == 5
        assert result.n_rem_bound == 5
        assert len(result.per_bin_cohens_d[METRIC_NAMES[0]]) == 1

    def test_trajectory_prediction_cohens_d_direction(self):
        """N3-bound has positive sr slope, REM-bound negative -> d > 0 for spectral_radius."""
        n3, rem = self._make_n3_rem_subjects(n_subj=8, n3_sr_slope=3e-4, rem_sr_slope=-3e-4)
        result = predict_transition_type_from_trajectory(
            n3, rem, METRIC_NAMES,
            time_bins_sec=[-120.0], slope_halfwindow_sec=30.0,
            n_permutations=5, seed=0,
        )
        d_sr = result.per_bin_cohens_d["spectral_radius"][0]
        assert np.isfinite(d_sr)
        assert d_sr > 0, f"Expected d > 0 for N3 positive / REM negative slopes, got {d_sr:.3f}"

    def test_trajectory_prediction_result_fields(self):
        """TrajectoryPredictionResult has all expected fields with correct types."""
        n3, rem = self._make_n3_rem_subjects(n_subj=4)
        result = predict_transition_type_from_trajectory(
            n3, rem, METRIC_NAMES,
            time_bins_sec=[-120.0, -60.0], slope_halfwindow_sec=30.0,
            n_permutations=5, seed=0,
        )
        assert isinstance(result, TrajectoryPredictionResult)
        assert result.metrics == METRIC_NAMES
        assert len(result.time_bins_sec) == 2
        assert all(m in result.per_bin_cohens_d for m in METRIC_NAMES)
        assert len(result.permutation_null_auc) == 5


def _make_trajectory_result_with_d(time_bins, d_values_by_metric):
    """Build a minimal TrajectoryPredictionResult with specified per_bin_cohens_d."""
    return TrajectoryPredictionResult(
        metrics=list(d_values_by_metric.keys()),
        time_bins_sec=list(time_bins),
        per_bin_cohens_d={m: list(v) for m, v in d_values_by_metric.items()},
        loso_auc=float("nan"),
        loso_balanced_accuracy=float("nan"),
        permutation_null_auc=[],
        permutation_p=float("nan"),
        n_n3_bound=5,
        n_rem_bound=5,
    )


class TestCommitmentWindow:

    def test_commitment_window_onset_detection(self):
        """Earliest bin with d >= 0.5 is correctly identified as discrimination onset."""
        time_bins = [-120.0, -90.0, -60.0, -30.0]
        d_vals = {
            "spectral_radius": [0.2, 0.2, 0.6, 0.9],
            "eigenvalue_gap": [0.1, 0.1, 0.1, 0.3],
        }
        traj = _make_trajectory_result_with_d(time_bins, d_vals)
        result = characterize_commitment_window(traj, d_thresholds=[0.5])
        assert result.discrimination_onset_sec == -60.0, (
            f"Expected onset at -60.0 s (first bin where d>=0.5), got {result.discrimination_onset_sec}"
        )
        assert result.commitment_window_sec == 60.0

    def test_commitment_window_no_onset(self):
        """If no bin reaches threshold, onset is None and window is None."""
        time_bins = [-120.0, -90.0, -60.0]
        d_vals = {"spectral_radius": [0.1, 0.2, 0.3]}
        traj = _make_trajectory_result_with_d(time_bins, d_vals)
        result = characterize_commitment_window(traj, d_thresholds=[0.5])
        assert result.discrimination_onset_sec is None
        assert result.commitment_window_sec is None

    def test_commitment_window_multi_threshold(self):
        """Multiple thresholds produce independent onset estimates."""
        time_bins = [-120.0, -90.0, -60.0, -30.0]
        d_vals = {"spectral_radius": [0.1, 0.4, 0.6, 0.9]}
        traj = _make_trajectory_result_with_d(time_bins, d_vals)
        result = characterize_commitment_window(traj, d_thresholds=[0.3, 0.5, 0.8])
        assert result.onset_by_threshold[0.3] == -90.0, (
            f"Threshold 0.3 onset expected at -90.0, got {result.onset_by_threshold[0.3]}"
        )
        assert result.onset_by_threshold[0.5] == -60.0
        assert result.onset_by_threshold[0.8] == -30.0

    def test_commitment_window_returns_correct_type(self):
        """characterize_commitment_window always returns CommitmentWindowResult."""
        time_bins = [-120.0]
        d_vals = {"spectral_radius": [0.7]}
        traj = _make_trajectory_result_with_d(time_bins, d_vals)
        result = characterize_commitment_window(traj)
        assert isinstance(result, CommitmentWindowResult)
        assert 0.5 in result.thresholds_tested

    def test_commitment_window_changepoint_overlap(self):
        """Changepoint CI overlapping with onset (within 60 s) is flagged."""
        time_bins = [-120.0, -90.0, -60.0]
        d_vals = {"spectral_radius": [0.1, 0.1, 0.7]}
        traj = _make_trajectory_result_with_d(time_bins, d_vals)

        cp = ChangepointResult(
            metric_name="spectral_radius",
            n_qualifying_transitions=5,
            n_subjects=5,
            per_transition_latencies_sec=[-70.0, -80.0, -50.0, -65.0, -75.0],
            per_subject_mean_latencies_sec=[-70.0, -80.0, -50.0, -65.0, -75.0],
            group_mean_latency_sec=-68.0,
            group_latency_p_value=0.01,
            group_latency_ci=(-90.0, -40.0),
            latency_sd_sec=12.0,
            latency_hist_60s_bins=[0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
            latency_hist_bin_edges=[-600.0, -540.0, -480.0, -420.0, -360.0,
                                    -300.0, -240.0, -180.0, -120.0, -60.0, 0.0],
            fraction_within_300s=1.0,
        )
        result = characterize_commitment_window(traj, changepoint_result=cp, d_thresholds=[0.5])
        assert result.changepoint_overlap is True, (
            f"Expected overlap=True (onset=-60, CI=[-90,-40]+/-60 tol), got {result.changepoint_overlap}"
        )

    def test_commitment_window_no_changepoint_overlap(self):
        """Changepoint CI far from onset -> overlap=False."""
        time_bins = [-120.0, -30.0]
        d_vals = {"spectral_radius": [0.1, 0.9]}
        traj = _make_trajectory_result_with_d(time_bins, d_vals)
        cp = ChangepointResult(
            metric_name="spectral_radius",
            n_qualifying_transitions=5,
            n_subjects=5,
            per_transition_latencies_sec=[-400.0, -450.0, -420.0, -380.0, -430.0],
            per_subject_mean_latencies_sec=[-400.0, -450.0, -420.0, -380.0, -430.0],
            group_mean_latency_sec=-416.0,
            group_latency_p_value=0.001,
            group_latency_ci=(-500.0, -300.0),
            latency_sd_sec=30.0,
            latency_hist_60s_bins=[3, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            latency_hist_bin_edges=[-600.0, -540.0, -480.0, -420.0, -360.0,
                                    -300.0, -240.0, -180.0, -120.0, -60.0, 0.0],
            fraction_within_300s=0.0,
        )
        result = characterize_commitment_window(traj, changepoint_result=cp, d_thresholds=[0.5])
        assert result.changepoint_overlap is False, (
            f"Expected overlap=False (onset=-30, CI=[-500,-300]+/-60 tol), got {result.changepoint_overlap}"
        )


class TestGeometryCommitmentSmoke:

    def test_geometry_commitment_smoke(self):
        """End-to-end smoke test: Q3, Q4, Q5 with 2 synthetic subjects, 3 transitions each."""
        rng = np.random.default_rng(99)
        n_subj = 2
        n_tc_each = 3

        early_windows = {}
        late_windows = {}
        n3_dict = {}
        rem_dict = {}

        for i in range(n_subj):
            sid = f"SS{i:02d}"
            early_windows[sid] = rng.normal(
                [0.90, 0.001, 5.0, 0.1], [0.01, 0.0002, 0.5, 0.01],
                size=(n_tc_each * 5, 4),
            )
            late_windows[sid] = rng.normal(
                [0.95, 0.002, 5.5, 0.12], [0.01, 0.0002, 0.5, 0.01],
                size=(n_tc_each * 5, 4),
            )
            n3_tcs = []
            rem_tcs = []
            for j in range(n_tc_each):
                t = np.arange(-130.0, 10.0, 0.5)
                sr_n3 = 0.93 + 3e-4 * t + rng.normal(0, 1e-4, size=len(t))
                sr_rem = 0.93 - 3e-4 * t + rng.normal(0, 1e-4, size=len(t))
                eg = np.ones_like(t) * 0.002
                n3_tcs.append(TransitionTimecourse(
                    subject=sid, transition_type="N2_to_N3",
                    transition_index=j, time_sec=t,
                    eigenvalue_gap=eg, condition_number=np.ones_like(t) * 5.0,
                    nd_score=np.ones_like(t) * 0.1, spectral_radius=sr_n3,
                ))
                rem_tcs.append(TransitionTimecourse(
                    subject=sid + "_R", transition_type="N2_to_REM",
                    transition_index=j, time_sec=t,
                    eigenvalue_gap=eg, condition_number=np.ones_like(t) * 5.0,
                    nd_score=np.ones_like(t) * 0.1, spectral_radius=sr_rem,
                ))
            n3_dict[sid] = n3_tcs
            rem_dict[sid + "_R"] = rem_tcs

        bistability = analyze_n2_bistability(
            early_windows, late_windows, METRIC_NAMES, seed=0, n_bootstrap_dip=50
        )
        assert isinstance(bistability, BistabilityResult)
        assert bistability.n_early_windows == n_subj * n_tc_each * 5
        assert bistability.n_late_windows == n_subj * n_tc_each * 5

        trajectory = predict_transition_type_from_trajectory(
            n3_dict, rem_dict, METRIC_NAMES,
            time_bins_sec=[-120.0, -60.0],
            slope_halfwindow_sec=30.0,
            n_permutations=5, seed=0,
        )
        assert isinstance(trajectory, TrajectoryPredictionResult)
        assert trajectory.n_n3_bound == n_subj
        assert trajectory.n_rem_bound == n_subj

        commitment = characterize_commitment_window(trajectory, d_thresholds=[0.3, 0.5, 0.8])
        assert isinstance(commitment, CommitmentWindowResult)
        assert set(commitment.thresholds_tested) == {0.3, 0.5, 0.8}
        assert all(t in commitment.onset_by_threshold for t in [0.3, 0.5, 0.8])


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TP_JSON = REPO_ROOT / "results" / "json_results" / "temporal_precedence.json"
GC_JSON = REPO_ROOT / "results" / "json_results" / "geometry_commitment.json"


class TestGeometryCommitmentIntegration:

    @pytest.mark.skipif(not TP_JSON.exists(), reason="temporal_precedence.json not available")
    def test_q3_q4_with_real_data(self):
        """Integration: run Q3 and Q4 on actual temporal_precedence.json data."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "analysis"))
        from _geometry_commitment import load_json_timecourses, extract_bistability_windows

        n3_by_subj, rem_by_subj = load_json_timecourses(TP_JSON)
        assert len(n3_by_subj) >= 1
        assert len(rem_by_subj) >= 1

        early, late = extract_bistability_windows(n3_by_subj)
        assert len(early) >= 1 and len(late) >= 1

        metrics = ["spectral_radius", "eigenvalue_gap", "condition_number", "nd_score"]
        result_q3 = analyze_n2_bistability(early, late, metrics, seed=42, n_bootstrap_dip=50)
        assert isinstance(result_q3, BistabilityResult)
        assert result_q3.n_early_windows > 0
        assert result_q3.n_late_windows > 0
        assert np.isfinite(result_q3.gmm_bic_delta)

        result_q4 = predict_transition_type_from_trajectory(
            n3_by_subj, rem_by_subj, metrics,
            time_bins_sec=[-120.0, -60.0], slope_halfwindow_sec=30.0,
            n_permutations=10, seed=42,
        )
        assert isinstance(result_q4, TrajectoryPredictionResult)
        assert result_q4.n_n3_bound >= 1
        assert result_q4.n_rem_bound >= 1

    @pytest.mark.skipif(not GC_JSON.exists(), reason="geometry_commitment.json not available")
    def test_json_schema_validation(self):
        """Validate geometry_commitment.json has required top-level keys."""
        import json
        with open(GC_JSON) as f:
            data = json.load(f)

        required_keys = [
            "analysis", "dataset", "run_timestamp", "parameters",
            "q1_extended_window", "q2_changepoint",
            "q3_bistability", "q4_trajectory_prediction", "q5_commitment",
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

        assert data["analysis"] == "geometry_commitment"
        assert data["dataset"] == "ANPHY-Sleep"

        params = data["parameters"]
        assert params["seed"] == 42
        assert params["n_bootstrap"] == 5000

        q3 = data["q3_bistability"]
        if not q3.get("skipped"):
            assert "n_early_windows" in q3
            assert "gmm_bic_delta" in q3

        q4 = data["q4_trajectory_prediction"]
        if not q4.get("skipped"):
            assert "loso_auc" in q4
            assert "permutation_p" in q4

        q5 = data["q5_commitment"]
        assert "primary_onset_sec" in q5
        assert "d_thresholds_tested" in q5


class TestCohensD:

    def test_independent_d_known_value(self):
        """Independent Cohen's d with known separation."""
        a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        b = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        d = _cohens_d_independent(a, b)
        assert d > 0
        expected = (12.0 - 2.0) / np.sqrt(2.5)
        assert abs(d - expected) < 1e-10

    def test_independent_d_unequal_n(self):
        """Independent d works with unequal group sizes (no truncation needed)."""
        a = np.array([10.0, 11.0, 12.0])
        b = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        d = _cohens_d_independent(a, b)
        assert np.isfinite(d)
        assert d > 0

    def test_paired_vs_independent_differ(self):
        """Paired and independent d give different values for the same data."""
        a = np.array([10.0, 12.0, 11.0, 14.0])
        b = np.array([8.0, 9.5, 10.0, 11.0])
        d_paired = _cohens_d_paired(a, b)
        d_indep = _cohens_d_independent(a, b)
        assert np.isfinite(d_paired) and np.isfinite(d_indep)
        assert d_paired != d_indep


class TestCommitmentWindowRobust:

    def test_reversed_time_bins_same_onset(self):
        """Onset detection is independent of time_bins_sec iteration order."""
        d_vals = {"spectral_radius": [0.2, 0.2, 0.6, 0.9]}
        time_bins_fwd = [-120.0, -90.0, -60.0, -30.0]
        time_bins_rev = [-30.0, -60.0, -90.0, -120.0]
        d_vals_rev = {"spectral_radius": [0.9, 0.6, 0.2, 0.2]}
        traj_fwd = _make_trajectory_result_with_d(time_bins_fwd, d_vals)
        traj_rev = _make_trajectory_result_with_d(time_bins_rev, d_vals_rev)
        r_fwd = characterize_commitment_window(traj_fwd, d_thresholds=[0.5])
        r_rev = characterize_commitment_window(traj_rev, d_thresholds=[0.5])
        assert r_fwd.discrimination_onset_sec == r_rev.discrimination_onset_sec == -60.0

    def test_geometry_at_onset_populated(self):
        """When extended_window_result is provided, geometry_at_onset is populated."""
        time_bins = [-120.0, -60.0]
        d_vals = {"spectral_radius": [0.1, 0.7]}
        traj = _make_trajectory_result_with_d(time_bins, d_vals)
        ext = ExtendedWindowResult(
            metric_name="spectral_radius",
            n_qualifying_transitions=10, n_subjects=5,
            mean_slope=1e-5, slope_p_value=0.01,
            slope_ci=(5e-6, 2e-5), cohens_d=0.8,
            subject_consistency=0.9,
            linear_aic=-100.0, quadratic_aic=-110.0, aic_delta=-10.0,
            group_time_axis=np.arange(-120.0, 10.0, 0.5),
            group_mean_trajectory=np.linspace(0.90, 0.98, len(np.arange(-120.0, 10.0, 0.5))),
            group_ci_lower=np.linspace(0.89, 0.97, len(np.arange(-120.0, 10.0, 0.5))),
            group_ci_upper=np.linspace(0.91, 0.99, len(np.arange(-120.0, 10.0, 0.5))),
            passes_threshold=True,
        )
        result = characterize_commitment_window(traj, extended_window_result=ext, d_thresholds=[0.5])
        assert result.discrimination_onset_sec == -60.0
        assert "spectral_radius" in result.geometry_at_onset
        assert np.isfinite(result.geometry_at_onset["spectral_radius"])


class TestStep36BistabilityHardening:

    def _make_bimodal_windows(self, seed=0, n_per_subj=50):
        rng = np.random.default_rng(seed)
        early = {}
        late = {}
        for i in range(10):
            sid = f"S{i:02d}"
            early[sid] = rng.normal(loc=[0.0, 0.0, 0.0, 0.0], scale=0.5, size=(n_per_subj, 4))
            late[sid] = rng.normal(loc=[3.0, 3.0, 3.0, 3.0], scale=0.5, size=(n_per_subj, 4))
        return early, late

    def _make_unimodal_windows(self, seed=0, n_per_subj=50):
        rng = np.random.default_rng(seed)
        early = {}
        late = {}
        for i in range(10):
            sid = f"S{i:02d}"
            early[sid] = rng.normal(loc=[0.0, 0.0, 0.0, 0.0], scale=1.0, size=(n_per_subj, 4))
            late[sid] = rng.normal(loc=[0.0, 0.0, 0.0, 0.0], scale=1.0, size=(n_per_subj, 4))
        return early, late

    def test_bc_gaussian_null_detects_bimodal(self):
        early, late = self._make_bimodal_windows()
        result = analyze_n2_bistability(early, late, METRIC_NAMES, seed=42, n_bootstrap_dip=200)
        assert result.bc_gaussian_null_p < 0.05, (
            f"Gaussian-null BC should detect bimodality, got p={result.bc_gaussian_null_p}"
        )

    def test_bc_gaussian_null_accepts_unimodal(self):
        early, late = self._make_unimodal_windows()
        result = analyze_n2_bistability(early, late, METRIC_NAMES, seed=42, n_bootstrap_dip=200)
        assert result.bc_gaussian_null_p > 0.05, (
            f"Gaussian-null BC should accept unimodal data, got p={result.bc_gaussian_null_p}"
        )

    def test_gmm_cv_bimodal_prefers_k2(self):
        early, late = self._make_bimodal_windows()
        result = analyze_n2_bistability(early, late, METRIC_NAMES, seed=42, n_bootstrap_dip=50)
        assert result.gmm_cv_ll_delta > 0, (
            f"CV GMM should prefer k=2 for bimodal data, got delta={result.gmm_cv_ll_delta}"
        )

    def test_gmm_cv_unimodal_prefers_k1(self):
        early, late = self._make_unimodal_windows()
        result = analyze_n2_bistability(early, late, METRIC_NAMES, seed=42, n_bootstrap_dip=50)
        assert result.gmm_cv_ll_delta <= 0.5, (
            f"CV GMM should not strongly prefer k=2 for unimodal data, got delta={result.gmm_cv_ll_delta}"
        )

    def test_bc_note_populated(self):
        early, late = self._make_unimodal_windows()
        result = analyze_n2_bistability(early, late, METRIC_NAMES, seed=42, n_bootstrap_dip=50)
        assert "uniform" in result.bc_note.lower()
        assert "gaussian" in result.bc_note.lower()

    def test_silverman_bimodal(self):
        rng = np.random.default_rng(42)
        x = np.concatenate([rng.normal(-3, 0.5, 200), rng.normal(3, 0.5, 200)])
        p = _silverman_test(x, rng, n_bootstrap=200)
        assert p < 0.10, f"Silverman should detect clear bimodality, got p={p}"

    def test_silverman_unimodal(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 400)
        p = _silverman_test(x, rng, n_bootstrap=200)
        assert p > 0.05, f"Silverman should accept unimodal Gaussian, got p={p}"


class TestStep37ClassifierHardening:

    def _make_n3_rem_scaled(self, n_subj=5, seed=0):
        rng = np.random.default_rng(seed)
        n3 = {}
        rem = {}
        for i in range(n_subj):
            t = np.arange(-120.0, 0.5, 0.5)
            for group, target in [(n3, "N2_to_N3"), (rem, "N2_to_R")]:
                sid = f"S{i:02d}_{target[-2:]}"
                sr = rng.normal(1.006, 0.001, size=len(t))
                eg = rng.normal(0.012, 0.002, size=len(t))
                cn = rng.normal(55.0, 10.0, size=len(t))
                nd = rng.normal(100.0, 20.0, size=len(t))
                tc = TransitionTimecourse(
                    subject=sid, transition_type=target, transition_index=0,
                    time_sec=t, spectral_radius=sr, eigenvalue_gap=eg,
                    condition_number=cn, nd_score=nd,
                )
                group[sid] = [tc]
        return n3, rem

    def test_trajectory_z_scaling_prevents_inversion(self):
        n3, rem = self._make_n3_rem_scaled(n_subj=5)
        result = predict_transition_type_from_trajectory(
            n3, rem, METRIC_NAMES,
            time_bins_sec=[-120.0, -60.0], slope_halfwindow_sec=30.0,
            n_permutations=5, seed=42,
        )
        if np.isfinite(result.loso_auc):
            assert result.loso_auc >= 0.2, (
                f"With z-scaling, AUC should not be near 0 for random data, got {result.loso_auc}"
            )

    def test_per_fold_diagnostics_populated(self):
        n3, rem = self._make_n3_rem_scaled(n_subj=5)
        result = predict_transition_type_from_trajectory(
            n3, rem, METRIC_NAMES,
            time_bins_sec=[-120.0], slope_halfwindow_sec=30.0,
            n_permutations=5, seed=42,
        )
        assert len(result.per_fold_predictions) > 0
        for fold in result.per_fold_predictions:
            assert len(fold) == 3

    def test_majority_baseline_reported(self):
        n3, rem = self._make_n3_rem_scaled(n_subj=5)
        result = predict_transition_type_from_trajectory(
            n3, rem, METRIC_NAMES,
            time_bins_sec=[-120.0], slope_halfwindow_sec=30.0,
            n_permutations=5, seed=42,
        )
        assert np.isfinite(result.majority_baseline_auc)
        assert 0.0 <= result.majority_baseline_auc <= 1.0

    def test_feature_scale_ratio_reported(self):
        n3, rem = self._make_n3_rem_scaled(n_subj=5)
        result = predict_transition_type_from_trajectory(
            n3, rem, METRIC_NAMES,
            time_bins_sec=[-120.0], slope_halfwindow_sec=30.0,
            n_permutations=5, seed=42,
        )
        assert np.isfinite(result.feature_scale_ratio)
        assert result.feature_scale_ratio >= 1.0

    def test_per_bin_loso_auc_length(self):
        n3, rem = self._make_n3_rem_scaled(n_subj=5)
        result = predict_transition_type_from_trajectory(
            n3, rem, METRIC_NAMES,
            time_bins_sec=[-120.0, -60.0], slope_halfwindow_sec=30.0,
            n_permutations=5, seed=42,
        )
        assert len(result.per_bin_loso_auc) == 2

    def test_per_subject_slopes_populated(self):
        n3, rem = self._make_n3_rem_scaled(n_subj=5)
        result = predict_transition_type_from_trajectory(
            n3, rem, METRIC_NAMES,
            time_bins_sec=[-120.0], slope_halfwindow_sec=30.0,
            n_permutations=5, seed=42,
        )
        assert "n3" in result.per_subject_slopes
        assert "rem" in result.per_subject_slopes
        assert len(result.per_subject_slopes["n3"]) == 5
        assert len(result.per_subject_slopes["rem"]) == 5


class TestStep38CommitmentStatistical:

    def _make_trajectory_with_slopes(self, n_subj=5, seed=0):
        rng = np.random.default_rng(seed)
        time_bins = [-120.0, -90.0, -60.0, -30.0]
        n3_slopes = {}
        rem_slopes = {}
        for i in range(n_subj):
            n3_slopes[f"N3_{i}"] = {
                "spectral_radius": [
                    rng.normal(0.0, 0.01),
                    rng.normal(0.0, 0.01),
                    rng.normal(0.5, 0.1),
                    rng.normal(0.8, 0.1),
                ],
                "eigenvalue_gap": [
                    rng.normal(0.0, 0.01),
                    rng.normal(0.0, 0.01),
                    rng.normal(0.3, 0.1),
                    rng.normal(0.6, 0.1),
                ],
            }
            rem_slopes[f"REM_{i}"] = {
                "spectral_radius": [
                    rng.normal(0.0, 0.01),
                    rng.normal(0.0, 0.01),
                    rng.normal(-0.5, 0.1),
                    rng.normal(-0.8, 0.1),
                ],
                "eigenvalue_gap": [
                    rng.normal(0.0, 0.01),
                    rng.normal(0.0, 0.01),
                    rng.normal(-0.3, 0.1),
                    rng.normal(-0.6, 0.1),
                ],
            }

        d_vals = _compute_per_bin_d_from_slopes(n3_slopes, rem_slopes,
                                                 ["spectral_radius", "eigenvalue_gap"], 4)
        traj = TrajectoryPredictionResult(
            metrics=["spectral_radius", "eigenvalue_gap"],
            time_bins_sec=time_bins,
            per_bin_cohens_d=d_vals,
            loso_auc=0.5, loso_balanced_accuracy=0.5,
            permutation_null_auc=[], permutation_p=0.5,
            n_n3_bound=n_subj, n_rem_bound=n_subj,
            per_subject_slopes={"n3": n3_slopes, "rem": rem_slopes},
        )
        return traj

    def test_commitment_bootstrap_ci_contains_onset(self):
        traj = self._make_trajectory_with_slopes(n_subj=10, seed=42)
        result = characterize_commitment_window(
            traj, d_thresholds=[0.5], n_bootstrap=500, n_permutations=50, seed=42)
        assert result.discrimination_onset_sec is not None, "Expected an onset to be detected"
        if result.onset_ci_sec is not None:
            lo, hi = result.onset_ci_sec
            assert lo <= result.discrimination_onset_sec <= hi, (
                f"CI [{lo}, {hi}] should contain onset {result.discrimination_onset_sec}"
            )

    def test_commitment_permutation_null_centered(self):
        rng = np.random.default_rng(99)
        time_bins = [-120.0, -60.0]
        n3_slopes = {}
        rem_slopes = {}
        for i in range(8):
            n3_slopes[f"N3_{i}"] = {
                "m1": [rng.normal(0, 0.1) for _ in time_bins],
            }
            rem_slopes[f"REM_{i}"] = {
                "m1": [rng.normal(0, 0.1) for _ in time_bins],
            }
        d_vals = {"m1": [0.05, 0.05]}
        traj = TrajectoryPredictionResult(
            metrics=["m1"], time_bins_sec=time_bins,
            per_bin_cohens_d=d_vals,
            loso_auc=0.5, loso_balanced_accuracy=0.5,
            permutation_null_auc=[], permutation_p=0.5,
            n_n3_bound=8, n_rem_bound=8,
            per_subject_slopes={"n3": n3_slopes, "rem": rem_slopes},
        )
        result = characterize_commitment_window(
            traj, d_thresholds=[0.5], n_bootstrap=100, n_permutations=100, seed=42)
        assert result.discrimination_onset_sec is None
        assert np.isnan(result.onset_permutation_p) or result.onset_permutation_p > 0.05

    def test_commitment_monotonicity_perfect(self):
        time_bins = [-120.0, -90.0, -60.0, -30.0]
        d_vals = {"spectral_radius": [0.6, 0.7, 0.8, 0.9]}
        traj = _make_trajectory_result_with_d(time_bins, d_vals)
        result = characterize_commitment_window(traj, d_thresholds=[0.5])
        assert result.monotonicity_fraction == 1.0

    def test_commitment_monotonicity_partial(self):
        time_bins = [-120.0, -90.0, -60.0, -30.0]
        d_vals = {"spectral_radius": [0.6, 0.3, 0.8, 0.2]}
        traj = _make_trajectory_result_with_d(time_bins, d_vals)
        result = characterize_commitment_window(traj, d_thresholds=[0.5])
        assert 0.0 < result.monotonicity_fraction < 1.0

    def test_commitment_n_bootstrap_n_permutations_stored(self):
        traj = self._make_trajectory_with_slopes(n_subj=5, seed=0)
        result = characterize_commitment_window(
            traj, d_thresholds=[0.5], n_bootstrap=100, n_permutations=50, seed=42)
        assert result.n_bootstrap == 100
        assert result.n_permutations == 50


class TestStep39ConstraintDiagnostics:

    def _make_truncated_subjects(self, n_subjects=5, pre_sec=120.0):
        subjects = {}
        rng = np.random.default_rng(42)
        for i in range(n_subjects):
            sid = f"S{i:02d}"
            t = np.arange(-pre_sec, 0.5, 0.5)
            y = 0.95 + 1e-4 * (t + pre_sec) + rng.normal(0, 1e-4, size=len(t))
            tc = TransitionTimecourse(
                subject=sid, transition_type="N2_to_N3", transition_index=0,
                time_sec=t, eigenvalue_gap=np.ones_like(t) * 0.001,
                condition_number=np.ones_like(t) * 5.0, nd_score=np.ones_like(t) * 0.1,
                spectral_radius=y,
            )
            subjects[sid] = [tc]
        return subjects

    def _make_full_subjects(self, n_subjects=5, pre_sec=620.0):
        subjects = {}
        rng = np.random.default_rng(42)
        for i in range(n_subjects):
            sid = f"S{i:02d}"
            t = np.arange(-pre_sec, 0.5, 0.5)
            y = 0.95 + 1e-5 * (t + pre_sec) + rng.normal(0, 1e-4, size=len(t))
            tc = TransitionTimecourse(
                subject=sid, transition_type="N2_to_N3", transition_index=0,
                time_sec=t, eigenvalue_gap=np.ones_like(t) * 0.001,
                condition_number=np.ones_like(t) * 5.0, nd_score=np.ones_like(t) * 0.1,
                spectral_radius=y,
            )
            subjects[sid] = [tc]
        return subjects

    def test_q1_data_constraint_truncated(self):
        subjects = self._make_truncated_subjects(pre_sec=120.0)
        result = analyze_extended_window(
            subjects, "spectral_radius", pre_sec=600.0,
            min_qualifying_pre_sec=100.0, seed=0, n_bootstrap=50)
        dc = result.data_constraint
        assert dc["is_truncated"]
        assert dc["analysis_type"] == "constrained_replication"
        assert dc["window_sec_requested"] == 600.0
        assert dc["window_sec_available"] <= 121.0
        assert len(dc["interpretation_caveat"]) > 0

    def test_q1_data_constraint_full(self):
        subjects = self._make_full_subjects(pre_sec=620.0)
        result = analyze_extended_window(
            subjects, "spectral_radius", pre_sec=600.0,
            min_qualifying_pre_sec=600.0, seed=0, n_bootstrap=50)
        dc = result.data_constraint
        assert not dc["is_truncated"]
        assert dc["analysis_type"] == "full_extended_window"
        assert dc["interpretation_caveat"] == ""

    def test_q2_data_constraint_populated(self):
        subjects = self._make_truncated_subjects(pre_sec=120.0)
        result = detect_commitment_changepoints(
            subjects, "spectral_radius", pre_sec=600.0,
            min_segment_sec=20.0, min_qualifying_pre_sec=100.0, seed=0)
        dc = result.data_constraint
        assert "window_sec_requested" in dc
        assert "window_sec_available" in dc
        assert "is_truncated" in dc
        assert dc["is_truncated"]

    def test_q2_edge_proximity_ratio_valid(self):
        rng = np.random.default_rng(0)
        subjects = {}
        for i in range(5):
            sid = f"S{i:02d}"
            t = np.arange(-500.0, 0.5, 0.5)
            y = np.where(t < -200.0,
                         0.95 + rng.normal(0, 1e-4, size=len(t)),
                         0.95 + 5e-4 * (t + 200) + rng.normal(0, 1e-4, size=len(t)))
            tc = TransitionTimecourse(
                subject=sid, transition_type="N2_to_N3", transition_index=0,
                time_sec=t, eigenvalue_gap=np.ones_like(t) * 0.001,
                condition_number=np.ones_like(t) * 5.0, nd_score=np.ones_like(t) * 0.1,
                spectral_radius=y,
            )
            subjects[sid] = [tc]
        result = detect_commitment_changepoints(
            subjects, "spectral_radius", pre_sec=500.0,
            min_segment_sec=30.0, min_qualifying_pre_sec=240.0, seed=0)
        assert np.isfinite(result.edge_proximity_ratio)
        assert 0.0 <= result.edge_proximity_ratio <= 1.0

    def test_q2_window_truncation_sensitivity(self):
        rng = np.random.default_rng(0)
        subjects = {}
        for i in range(5):
            sid = f"S{i:02d}"
            t = np.arange(-500.0, 0.5, 0.5)
            y = np.where(t < -200.0,
                         0.95 + rng.normal(0, 1e-4, size=len(t)),
                         0.95 + 5e-4 * (t + 200) + rng.normal(0, 1e-4, size=len(t)))
            tc = TransitionTimecourse(
                subject=sid, transition_type="N2_to_N3", transition_index=0,
                time_sec=t, eigenvalue_gap=np.ones_like(t) * 0.001,
                condition_number=np.ones_like(t) * 5.0, nd_score=np.ones_like(t) * 0.1,
                spectral_radius=y,
            )
            subjects[sid] = [tc]
        result = detect_commitment_changepoints(
            subjects, "spectral_radius", pre_sec=500.0,
            min_segment_sec=30.0, min_qualifying_pre_sec=240.0, seed=0)
        wts = result.window_truncation_sensitivity
        assert len(wts) > 0
        for key, val in wts.items():
            assert "group_mean_latency_sec" in val
            assert "group_p" in val
            assert "n_subjects" in val
