"""Tests for temporal precedence analysis (Test 7).

Validates:
1. Sleep staging parsing
2. Transition detection from staging sequences
3. Transition-aligned geometry computation
4. Statistical analysis of pre-transition dynamics
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmcc.analysis.temporal_precedence import (
    TransitionEvent,
    TransitionTimecourse,
    TemporalPrecedenceResult,
    parse_sleep_staging,
    find_transitions,
    _subject_mean_timecourse,
    _cohens_d_paired,
    analyze_temporal_precedence,
)


class TestParseSleepStaging:
    def test_basic_parse(self, tmp_path):
        staging = tmp_path / "staging.txt"
        staging.write_text("W\t0\t30\nW\t30\t30\nN1\t60\t30\nN2\t90\t30\n")
        epochs = parse_sleep_staging(str(staging))
        assert len(epochs) == 4
        assert epochs[0] == ("W", 0.0, 30.0)
        assert epochs[2] == ("N1", 60.0, 30.0)

    def test_empty_file(self, tmp_path):
        staging = tmp_path / "empty.txt"
        staging.write_text("")
        epochs = parse_sleep_staging(str(staging))
        assert epochs == []


class TestFindTransitions:
    def _make_epochs(self, stages):
        return [(s, i * 30.0, 30.0) for i, s in enumerate(stages)]

    def test_simple_n2_to_n3(self):
        stages = ["W", "W", "N2", "N2", "N2", "N3", "N3", "N3"]
        epochs = self._make_epochs(stages)
        transitions = find_transitions(epochs, "N2", "N3", min_pre_epochs=2, min_post_epochs=2)
        assert len(transitions) == 1
        assert transitions[0]["onset_sec"] == 150.0
        assert transitions[0]["pre_count"] == 3
        assert transitions[0]["post_count"] == 3

    def test_insufficient_pre_epochs(self):
        stages = ["W", "N2", "N3", "N3", "N3"]
        epochs = self._make_epochs(stages)
        transitions = find_transitions(epochs, "N2", "N3", min_pre_epochs=2, min_post_epochs=2)
        assert len(transitions) == 0

    def test_insufficient_post_epochs(self):
        stages = ["N2", "N2", "N2", "N3"]
        epochs = self._make_epochs(stages)
        transitions = find_transitions(epochs, "N2", "N3", min_pre_epochs=2, min_post_epochs=2)
        assert len(transitions) == 0

    def test_multiple_transitions(self):
        stages = ["N2", "N2", "N3", "N3", "N2", "N2", "N3", "N3"]
        epochs = self._make_epochs(stages)
        transitions = find_transitions(epochs, "N2", "N3", min_pre_epochs=2, min_post_epochs=2)
        assert len(transitions) == 2
        assert transitions[0]["onset_sec"] == 60.0
        assert transitions[1]["onset_sec"] == 180.0

    def test_n2_to_rem(self):
        stages = ["N2", "N2", "N2", "R", "R", "R"]
        epochs = self._make_epochs(stages)
        transitions = find_transitions(epochs, "N2", "R", min_pre_epochs=2, min_post_epochs=2)
        assert len(transitions) == 1

    def test_no_matching_transition(self):
        stages = ["W", "W", "N1", "N1", "N2", "N2"]
        epochs = self._make_epochs(stages)
        transitions = find_transitions(epochs, "N2", "N3", min_pre_epochs=2, min_post_epochs=2)
        assert len(transitions) == 0

    def test_min_epochs_one(self):
        stages = ["N2", "N3"]
        epochs = self._make_epochs(stages)
        transitions = find_transitions(epochs, "N2", "N3", min_pre_epochs=1, min_post_epochs=1)
        assert len(transitions) == 1


class TestSubjectMeanTimecourse:
    def test_single_transition(self):
        time_sec = np.arange(-5.0, 5.0, 0.5)
        tc = TransitionTimecourse(
            subject="s01",
            transition_type="N2_to_N3",
            transition_index=0,
            time_sec=time_sec,
            eigenvalue_gap=np.ones(len(time_sec)),
            condition_number=np.ones(len(time_sec)) * 2,
            nd_score=np.zeros(len(time_sec)),
            spectral_radius=np.ones(len(time_sec)) * 0.95,
        )
        common = np.arange(-5.0, 5.0, 0.5)
        result = _subject_mean_timecourse([tc], "eigenvalue_gap", common, tol=0.3)
        assert result is not None
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_empty_list(self):
        common = np.arange(-5.0, 5.0, 0.5)
        result = _subject_mean_timecourse([], "eigenvalue_gap", common)
        assert result is None


class TestCohensD:
    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert _cohens_d_paired(a, b) != _cohens_d_paired(a, b) or _cohens_d_paired(a, b) == 0.0

    def test_known_effect(self):
        a = np.array([10.0, 12.0, 9.0, 11.0])
        b = np.array([0.0, 0.5, -0.5, 1.0])
        d = _cohens_d_paired(a, b)
        assert abs(d) > 3


class TestAnalyzeTemporalPrecedence:
    def _make_timecourses(self, n_subjects=8, slope=-0.001, seed=42):
        rng = np.random.default_rng(seed)
        by_subject = {}
        for i in range(n_subjects):
            subj = f"s{i:02d}"
            time_sec = np.arange(-120.0, 60.0, 0.1)
            trend = slope * time_sec
            noise = rng.normal(0, abs(slope) * 10, len(time_sec))
            gap = 0.01 + trend + noise
            tc = TransitionTimecourse(
                subject=subj,
                transition_type="N2_to_N3",
                transition_index=0,
                time_sec=time_sec,
                eigenvalue_gap=gap,
                condition_number=np.ones(len(time_sec)),
                nd_score=np.zeros(len(time_sec)),
                spectral_radius=np.ones(len(time_sec)) * 0.95,
            )
            by_subject[subj] = [tc]
        return by_subject

    def test_detects_negative_slope(self):
        tcs = self._make_timecourses(n_subjects=8, slope=-0.001)
        result = analyze_temporal_precedence(
            tcs, "eigenvalue_gap", "N2_to_N3",
            seed=42, n_bootstrap=100,
        )
        assert result.mean_slope < 0
        assert result.n_subjects == 8

    def test_no_slope_no_detection(self):
        tcs = self._make_timecourses(n_subjects=8, slope=0.0)
        result = analyze_temporal_precedence(
            tcs, "eigenvalue_gap", "N2_to_N3",
            seed=42, n_bootstrap=100,
        )
        assert abs(result.mean_slope) < 0.001

    def test_insufficient_subjects(self):
        tcs = self._make_timecourses(n_subjects=2, slope=-0.001)
        result = analyze_temporal_precedence(
            tcs, "eigenvalue_gap", "N2_to_N3",
            seed=42, n_bootstrap=100,
        )
        assert result.passes_threshold is False

    def test_result_structure(self):
        tcs = self._make_timecourses(n_subjects=5, slope=-0.001)
        result = analyze_temporal_precedence(
            tcs, "eigenvalue_gap", "N2_to_N3",
            seed=42, n_bootstrap=100,
        )
        assert result.metric_name == "eigenvalue_gap"
        assert result.transition_type == "N2_to_N3"
        assert len(result.group_time_axis) > 0
        assert len(result.group_mean_trajectory) == len(result.group_time_axis)
        assert len(result.group_ci_lower) == len(result.group_time_axis)
        assert len(result.group_ci_upper) == len(result.group_time_axis)
        assert isinstance(result.slope_ci, tuple)
        assert len(result.slope_ci) == 2
        assert 0 <= result.subject_consistency <= 1
        assert isinstance(result.nonoverlap_survives, bool)
        assert isinstance(result.passes_strict, bool)

    def test_strict_requires_nonoverlap(self):
        tcs = self._make_timecourses(n_subjects=8, slope=-0.001)
        result = analyze_temporal_precedence(
            tcs, "eigenvalue_gap", "N2_to_N3",
            seed=42, n_bootstrap=100,
        )
        if result.passes_threshold and not result.nonoverlap_survives:
            assert not result.passes_strict
        if result.passes_strict:
            assert result.nonoverlap_survives
