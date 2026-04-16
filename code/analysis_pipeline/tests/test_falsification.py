"""
Tests for falsification battery.
Validates: label destruction, jackknife, feature ablation, spectral confounds,
temporal decimation, model competition.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmcc.analysis.falsification import (
    LabelShuffleResult,
    CircularShiftResult,
    JackknifeResult,
    LeaveTwoOutResult,
    AblationResult,
    SpectralConfoundResult,
    WindowSensitivityResult,
    ModelCompetitionResult,
    run_label_shuffle,
    run_circular_shift_null,
    run_classification_jackknife,
    run_temporal_jackknife,
    run_leave_two_out,
    run_feature_ablation,
    run_spectral_confound_check,
    run_temporal_decimation,
    run_model_competition,
)


class TestLabelShuffle:
    def _make_separable(self, n_subjects=20, seed=42):
        rng = np.random.default_rng(seed)
        features = []
        labels = []
        subject_ids = []
        for i in range(n_subjects):
            features.append([rng.normal(0, 0.3), rng.normal(0, 0.3)])
            labels.append(0)
            subject_ids.append(f"s{i:02d}")
            features.append([rng.normal(3, 0.3), rng.normal(3, 0.3)])
            labels.append(1)
            subject_ids.append(f"s{i:02d}")
        return (np.array(features), np.array(labels), np.array(subject_ids))

    def test_separable_survives(self):
        features, labels, subjects = self._make_separable()
        result = run_label_shuffle(features, labels, subjects,
                                   contrast_name="test", n_permutations=50, seed=42)
        assert result.survives
        assert result.real_auc > 0.9
        assert result.empirical_p < 0.05
        assert len(result.null_aucs) == 50

    def test_random_does_not_survive(self):
        rng = np.random.default_rng(42)
        features = rng.normal(0, 1, (40, 2))
        labels = np.array([0, 1] * 20)
        subjects = np.array([f"s{i:02d}" for i in range(20) for _ in range(2)])
        result = run_label_shuffle(features, labels, subjects,
                                   n_permutations=50, seed=42)
        assert not result.survives or result.empirical_p > 0.01


class TestCircularShift:
    def test_planted_trend_survives(self):
        subjects = {}
        time_axes = {}
        for i in range(10):
            t = np.arange(-120.0, 60.0, 0.5)
            ts = -0.001 * t + np.random.default_rng(42 + i).normal(0, 0.01, len(t))
            subjects[f"s{i:02d}"] = ts
            time_axes[f"s{i:02d}"] = t
        result = run_circular_shift_null(subjects, time_axes, "test", "N2_to_N3",
                                         n_permutations=100, seed=42)
        assert result.real_slope < 0
        assert isinstance(result.survives, bool)

    def test_no_trend_does_not_survive(self):
        subjects = {}
        time_axes = {}
        for i in range(10):
            t = np.arange(-120.0, 60.0, 0.5)
            ts = np.random.default_rng(42 + i).normal(0, 0.01, len(t))
            subjects[f"s{i:02d}"] = ts
            time_axes[f"s{i:02d}"] = t
        result = run_circular_shift_null(subjects, time_axes, "test", "N2_to_N3",
                                         n_permutations=100, seed=42)
        assert not result.survives


class TestClassificationJackknife:
    def test_stable_result(self):
        rng = np.random.default_rng(42)
        features = []
        labels = []
        subjects = []
        for i in range(20):
            features.append([rng.normal(0, 0.3)])
            labels.append(0)
            subjects.append(f"s{i:02d}")
            features.append([rng.normal(3, 0.3)])
            labels.append(1)
            subjects.append(f"s{i:02d}")
        result = run_classification_jackknife(
            np.array(features), np.array(labels), np.array(subjects),
            result_name="test", seed=42,
        )
        assert result.sign_preserved_all
        assert result.n_folds == 20
        assert result.loo_min > 0.8

    def test_outlier_detected(self):
        rng = np.random.default_rng(42)
        features = []
        labels = []
        subjects = []
        for i in range(10):
            if i == 0:
                features.append([5.0])
                labels.append(0)
                features.append([-5.0])
                labels.append(1)
            else:
                features.append([rng.normal(0, 1.0)])
                labels.append(0)
                features.append([rng.normal(0.1, 1.0)])
                labels.append(1)
            subjects.extend([f"s{i:02d}"] * 2)
        result = run_classification_jackknife(
            np.array(features), np.array(labels), np.array(subjects),
            result_name="test", seed=42,
        )
        assert result.most_influential_subject != ""
        assert result.max_influence > 0


class TestTemporalJackknife:
    def test_consistent_slopes(self):
        slopes = {f"s{i:02d}": -0.001 + np.random.default_rng(42 + i).normal(0, 0.0002)
                  for i in range(10)}
        result = run_temporal_jackknife(slopes, result_name="test")
        assert result.sign_preserved_all
        assert result.n_folds == 10

    def test_one_outlier(self):
        slopes = {f"s{i:02d}": -0.001 for i in range(9)}
        slopes["s09"] = 0.01
        result = run_temporal_jackknife(slopes, result_name="test")
        assert result.most_influential_subject == "s09"


class TestLeaveTwoOut:
    def test_robust_result(self):
        slopes = {f"s{i:02d}": -0.001 + np.random.default_rng(42 + i).normal(0, 0.0001)
                  for i in range(10)}
        result = run_leave_two_out(slopes, result_name="test")
        assert result.n_pairs == 45
        assert result.fraction_sign_preserved > 0.9

    def test_fragile_result(self):
        slopes = {f"s{i:02d}": -0.001 for i in range(5)}
        slopes.update({f"s{i:02d}": 0.001 for i in range(5, 10)})
        result = run_leave_two_out(slopes, result_name="test")
        assert result.fraction_sign_preserved < 1.0


class TestFeatureAblation:
    def test_single_feature_dominant(self):
        rng = np.random.default_rng(42)
        n = 20
        informative = np.concatenate([rng.normal(0, 0.2, n), rng.normal(3, 0.2, n)])
        noise1 = rng.normal(0, 1, 2 * n)
        noise2 = rng.normal(0, 1, 2 * n)
        features = np.column_stack([informative, noise1, noise2])
        labels = np.array([0] * n + [1] * n)
        subjects = np.array([f"s{i:02d}" for i in range(n) for _ in range(2)])

        result = run_feature_ablation(features, labels, subjects,
                                      ["informative", "noise1", "noise2"],
                                      contrast_name="test", seed=42)
        assert result.most_important_feature == "informative"
        assert result.single_feature_aucs["informative"] > result.single_feature_aucs["noise1"]

    def test_result_structure(self):
        rng = np.random.default_rng(42)
        features = rng.normal(0, 1, (40, 4))
        labels = np.array([0, 1] * 20)
        subjects = np.array([f"s{i:02d}" for i in range(20) for _ in range(2)])
        result = run_feature_ablation(features, labels, subjects,
                                      ["a", "b", "c", "d"], seed=42)
        assert len(result.single_feature_aucs) == 4
        assert len(result.leave_one_out_aucs) == 4
        assert isinstance(result.is_distributed, bool)

    def test_forward_selection_order_and_length(self):
        rng = np.random.default_rng(42)
        n = 20
        informative = np.concatenate([rng.normal(0, 0.3, n), rng.normal(3, 0.3, n)])
        noise = rng.normal(0, 1, 2 * n)
        features = np.column_stack([informative, noise])
        labels = np.array([0] * n + [1] * n)
        subjects = np.array([f"s{i:02d}" for i in range(n) for _ in range(2)])
        result = run_feature_ablation(features, labels, subjects,
                                      ["informative", "noise"],
                                      contrast_name="test", seed=42)
        assert len(result.forward_selection_order) == 2
        assert len(result.forward_selection_aucs) == 2
        assert len(result.forward_selection_marginal_deltas) == 2
        assert result.forward_selection_order[0] == "informative"
        assert result.forward_selection_marginal_deltas[0] > result.forward_selection_marginal_deltas[1]


class TestSpectralConfound:
    def test_independent_features_survive(self):
        rng = np.random.default_rng(42)
        n = 20
        features_a = rng.normal(0, 0.3, (n, 2))
        features_b = rng.normal(3, 0.3, (n, 2))
        features = np.vstack([features_a, features_b])
        labels = np.array([0] * n + [1] * n)
        subjects = np.array([f"s{i:02d}" for i in range(n) for _ in range(2)])
        spectral = rng.normal(0, 1, 2 * n)

        result = run_spectral_confound_check(
            features, labels, subjects, spectral,
            ["f1", "f2"], "power", "test", seed=42,
        )
        assert result.n_features_surviving >= 1

    def test_confounded_features_reduced(self):
        rng = np.random.default_rng(42)
        n = 20
        spectral = np.concatenate([rng.normal(0, 0.1, n), rng.normal(3, 0.1, n)])
        features = spectral.reshape(-1, 1) + rng.normal(0, 0.01, (2 * n, 1))
        labels = np.array([0] * n + [1] * n)
        subjects = np.array([f"s{i:02d}" for i in range(n) for _ in range(2)])

        result = run_spectral_confound_check(
            features, labels, subjects, spectral,
            ["confounded"], "power", "test", seed=42,
        )
        assert abs(result.per_geometry_correlations["confounded"]) > 0.5

    def test_partial_nan_spectral_single_class_coverage(self):
        rng = np.random.default_rng(42)
        n = 10
        features_a = rng.normal(0, 0.3, (n, 2))
        features_b = rng.normal(3, 0.3, (n, 2))
        features = np.vstack([features_a, features_b])
        labels = np.array([0] * n + [1] * n)
        subjects = np.array([f"s{i:02d}" for i in range(n) for _ in range(2)])
        spectral = np.full(2 * n, np.nan)
        spectral[:n] = rng.normal(0, 1, n)

        result = run_spectral_confound_check(
            features, labels, subjects, spectral,
            ["f1", "f2"], "power", "test", seed=42,
        )
        for d in result.per_geometry_residualized_d.values():
            assert np.isnan(d), (
                "When spectral feature covers only one class, Cohen's d should be NaN "
                "(cannot compare residualized vs non-residualized groups)"
            )
        assert result.residualized_classification_auc is None, (
            "Should not run classification when spectral feature covers only one class"
        )


class TestVarianceFloor:
    def test_near_constant_regressor_flagged_untestable(self):
        rng = np.random.default_rng(42)
        n = 10
        features = np.vstack([rng.normal(0, 0.3, (n, 2)),
                              rng.normal(3, 0.3, (n, 2))])
        labels = np.array([0] * n + [1] * n)
        subjects = np.array([f"s{i:02d}" for i in range(n) for _ in range(2)])
        spectral = np.full(2 * n, 5.0)

        result = run_spectral_confound_check(
            features, labels, subjects, spectral,
            ["f1", "f2"], "delta_power", "test", seed=42,
        )
        assert result.regressor_untestable
        assert "insufficient regressor relative variance" in result.regressor_untestable_reason
        assert all(np.isnan(d) for d in result.per_geometry_residualized_d.values())
        assert result.residualized_classification_auc is None

    def test_normal_variance_not_flagged(self):
        rng = np.random.default_rng(42)
        n = 20
        features = np.vstack([rng.normal(0, 0.3, (n, 2)),
                              rng.normal(3, 0.3, (n, 2))])
        labels = np.array([0] * n + [1] * n)
        subjects = np.array([f"s{i:02d}" for i in range(n) for _ in range(2)])
        spectral = rng.normal(0, 1, 2 * n)

        result = run_spectral_confound_check(
            features, labels, subjects, spectral,
            ["f1", "f2"], "alpha_power", "test", seed=42,
        )
        assert not result.regressor_untestable
        assert result.regressor_variance > 1e-6

    def test_small_valued_real_variance_not_flagged(self):
        rng = np.random.default_rng(42)
        n = 20
        features = np.vstack([rng.normal(0, 0.3, (n, 2)),
                              rng.normal(3, 0.3, (n, 2))])
        labels = np.array([0] * n + [1] * n)
        subjects = np.array([f"s{i:02d}" for i in range(n) for _ in range(2)])
        spectral = rng.uniform(1e-6, 1e-4, 2 * n)

        result = run_spectral_confound_check(
            features, labels, subjects, spectral,
            ["f1", "f2"], "alpha_power", "test", seed=42,
        )
        assert not result.regressor_untestable, (
            "Small-valued but genuinely varying regressor should not be flagged untestable"
        )


class TestTemporalDecimation:
    def test_decimation_factors(self):
        subjects = {}
        time_axes = {}
        for i in range(8):
            t = np.arange(-120.0, 60.0, 0.5)
            ts = -0.001 * t + np.random.default_rng(42 + i).normal(0, 0.01, len(t))
            subjects[f"s{i:02d}"] = ts
            time_axes[f"s{i:02d}"] = t

        result = run_temporal_decimation(subjects, time_axes, "test", "N2_to_N3",
                                         decimation_factors=[1, 3, 5])
        assert len(result.settings) == 3
        assert result.settings[0]["decimation_factor"] == 1
        assert result.settings[2]["decimation_factor"] == 5


class TestModelCompetition:
    def test_geometry_beats_random_baseline(self):
        rng = np.random.default_rng(42)
        n = 20
        geom = np.vstack([rng.normal(0, 0.3, (n, 2)), rng.normal(3, 0.3, (n, 2))])
        labels = np.array([0] * n + [1] * n)
        subjects = np.array([f"s{i:02d}" for i in range(n) for _ in range(2)])
        baseline = {"random": rng.normal(0, 1, (2 * n, 1))}

        result = run_model_competition(geom, labels, subjects, baseline,
                                       contrast_name="test", seed=42)
        assert result.geometry_beats_all
        assert result.geometry_auc > result.baseline_aucs["random"]

    def test_nan_baseline_does_not_poison_verdict(self):
        rng = np.random.default_rng(42)
        n = 20
        geom = np.vstack([rng.normal(0, 0.3, (n, 2)), rng.normal(3, 0.3, (n, 2))])
        labels = np.array([0] * n + [1] * n)
        subjects = np.array([f"s{i:02d}" for i in range(n) for _ in range(2)])
        nan_baseline = np.full((2 * n, 1), np.nan)
        result = run_model_competition(geom, labels, subjects,
                                       {"nan_feature": nan_baseline},
                                       contrast_name="test", seed=42)
        assert result.geometry_beats_all, "NaN baseline should not force geometry_beats_all=False"


class TestSpectralConfoundPartialCoverage:
    def test_single_class_spectral_gives_nan_d(self):
        rng = np.random.default_rng(42)
        n = 10
        features_a = rng.normal(0, 0.3, (n, 2))
        features_b = rng.normal(3, 0.3, (n, 2))
        features = np.vstack([features_a, features_b])
        labels = np.array([0] * n + [1] * n)
        subjects = np.array([f"s{i:02d}" for i in range(n) for _ in range(2)])
        spectral = np.full(2 * n, np.nan)
        spectral[:n] = rng.normal(0, 1, n)

        result = run_spectral_confound_check(
            features, labels, subjects, spectral,
            ["f1", "f2"], "power", "test", seed=42,
        )
        for name, d in result.per_geometry_residualized_d.items():
            assert np.isnan(d) or abs(d) < 50, (
                f"Feature {name}: d={d} is implausibly large; "
                f"partial-coverage residualization should produce NaN, not inflated d"
            )
        assert result.residualized_classification_auc is None, (
            "Should not run classification when spectral feature covers only one class"
        )
