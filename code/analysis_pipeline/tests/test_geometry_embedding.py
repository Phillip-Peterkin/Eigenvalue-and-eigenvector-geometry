"""Tests for geometry embedding analysis module.

Validates:
1. Data extraction from JSON schemas (propofol and sleep)
2. LOSO classification on synthetic separable / random data
3. Geometric structure (Mahalanobis, angular separation)
4. Orthogonality and incremental value tests
5. Collation of existing results
6. Verdict logic
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmcc.analysis.geometry_embedding import (
    GeometryFeatureTable,
    SufficiencyResult,
    IncrementalValueResult,
    OrthogonalityResult,
    StructureResult,
    CollatedResult,
    GeometryTestBattery,
    extract_propofol_features,
    extract_sleep_features,
    classify_states_loso,
    analyze_geometric_structure,
    compare_geometry_vs_power,
    check_orthogonality,
    collate_existing_results,
    compute_overall_verdict,
    assemble_test_battery,
    _angle_between_2d,
    _cohens_d,
)


def _make_propofol_json(n_subjects=5, seed=42):
    rng = np.random.default_rng(seed)
    ep_subjects = []
    amp_subjects = []
    for i in range(n_subjects):
        sid = f"10{i:02d}"
        ep_subjects.append({
            "subject": sid,
            "awake": {
                "mean_eigenvalue_gap": float(rng.uniform(0.01, 0.02)),
                "mean_ep_score": float(rng.uniform(50, 100)),
                "mean_spectral_radius": float(rng.uniform(0.99, 1.01)),
                "mean_alpha_power": float(rng.uniform(1e-6, 5e-6)),
            },
            "sedation_run1": {
                "mean_eigenvalue_gap": float(rng.uniform(0.008, 0.015)),
                "mean_ep_score": float(rng.uniform(60, 120)),
                "mean_spectral_radius": float(rng.uniform(0.995, 1.005)),
            },
        })
        amp_subjects.append({
            "subject": sid,
            "awake": {"condition_number_mean": float(rng.uniform(50, 150))},
            "sedation_run1": {"condition_number_mean": float(rng.uniform(30, 100))},
        })
    return {"subjects": ep_subjects}, {"subjects": amp_subjects}


def _make_sleep_json(n_subjects=5, seed=42):
    rng = np.random.default_rng(seed)
    per_state = []
    amp_subjects = []
    for i in range(n_subjects):
        sid = f"EPCTL{i:02d}"
        states_data = {}
        for state in ["W", "N3", "R"]:
            per_state.append({
                "subject": sid,
                "state": state,
                "mean_eigenvalue_gap": float(rng.uniform(0.01, 0.02)),
                "mean_ep_score": float(rng.uniform(50, 100)),
                "mean_spectral_radius": float(rng.uniform(0.99, 1.01)),
            })
            states_data[state] = {
                "condition_number_median": float(rng.uniform(50, 150)),
            }
        amp_subjects.append({"subject": sid, "states": states_data})
    return {"per_state_results": per_state}, {"subjects": amp_subjects}


class TestExtractPropofolFeatures:
    def test_basic_extraction(self):
        ep, amp = _make_propofol_json(5)
        table = extract_propofol_features(ep, amp)
        assert table.dataset == "propofol"
        assert len(table.subjects) == 10
        assert len(table.conditions) == 10
        assert table.features.shape == (10, 4)
        assert table.feature_names == ["eigenvalue_gap", "condition_number", "nd_score", "spectral_radius"]
        assert table.alpha_power is not None
        assert len(table.alpha_power) == 10
        assert len(table.excluded_subjects) == 0

    def test_conditions_alternate(self):
        ep, amp = _make_propofol_json(3)
        table = extract_propofol_features(ep, amp)
        assert table.conditions[0] == "awake"
        assert table.conditions[1] == "propofol"
        assert table.conditions[2] == "awake"
        assert table.conditions[3] == "propofol"

    def test_missing_key_excludes_subject(self):
        ep, amp = _make_propofol_json(3)
        del ep["subjects"][0]["awake"]["mean_eigenvalue_gap"]
        table = extract_propofol_features(ep, amp)
        assert len(table.excluded_subjects) == 1
        assert table.features.shape[0] == 4

    def test_no_common_subjects_raises(self):
        ep, amp = _make_propofol_json(3)
        for s in amp["subjects"]:
            s["subject"] = "DIFFERENT_" + s["subject"]
        with pytest.raises(ValueError, match="No common subject IDs"):
            extract_propofol_features(ep, amp)

    def test_no_finite_features(self):
        ep, amp = _make_propofol_json(1)
        ep["subjects"][0]["awake"]["mean_eigenvalue_gap"] = None
        with pytest.raises(ValueError, match="No valid subjects"):
            extract_propofol_features(ep, amp)


class TestExtractSleepFeatures:
    def test_basic_extraction(self):
        ep, amp = _make_sleep_json(5)
        table = extract_sleep_features(ep, amp)
        assert table.dataset == "sleep"
        assert len(table.subjects) == 15
        assert table.features.shape == (15, 4)
        assert table.alpha_power is None

    def test_three_conditions_per_subject(self):
        ep, amp = _make_sleep_json(3)
        table = extract_sleep_features(ep, amp)
        assert table.conditions[:3] == ["awake", "N3", "REM"]
        assert table.conditions[3:6] == ["awake", "N3", "REM"]

    def test_missing_state_excludes_subject(self):
        ep, amp = _make_sleep_json(3)
        ep["per_state_results"] = [r for r in ep["per_state_results"]
                                    if not (r["subject"] == "EPCTL00" and r["state"] == "R")]
        table = extract_sleep_features(ep, amp)
        assert "EPCTL00" in table.excluded_subjects
        assert table.features.shape[0] == 6


class TestClassifyStatesLoso:
    def test_perfect_separation(self):
        rng = np.random.default_rng(42)
        n = 20
        features = np.vstack([
            rng.normal(0, 0.1, (n, 2)),
            rng.normal(5, 0.1, (n, 2)),
        ])
        labels = np.array([0] * n + [1] * n)
        subjects = np.array([f"s{i}" for i in range(n)] + [f"s{i}" for i in range(n)])

        result = classify_states_loso(
            features, labels, subjects,
            contrast_name="test", dataset="test",
            seed=42, n_bootstrap=50, n_null_permutations=10,
        )
        assert result.auc_loso > 0.95
        assert result.passes_threshold
        assert result.subject_consistency > 0.9

    def test_random_data_near_chance(self):
        rng = np.random.default_rng(42)
        n = 20
        features = rng.normal(0, 1, (2 * n, 2))
        labels = np.array([0] * n + [1] * n)
        subjects = np.array([f"s{i}" for i in range(n)] + [f"s{i}" for i in range(n)])

        result = classify_states_loso(
            features, labels, subjects,
            seed=42, n_bootstrap=50, n_null_permutations=10,
        )
        assert 0.2 < result.auc_loso < 0.8
        assert not result.passes_threshold

    def test_fold_internal_zscoring(self):
        rng = np.random.default_rng(42)
        features = rng.normal(100, 50, (20, 2))
        labels = np.array([0] * 10 + [1] * 10)
        subjects = np.array([f"s{i}" for i in range(10)] + [f"s{i}" for i in range(10)])

        result = classify_states_loso(
            features, labels, subjects,
            seed=42, n_bootstrap=10, n_null_permutations=5,
        )
        assert np.isfinite(result.auc_loso)

    def test_too_few_subjects_raises(self):
        features = np.array([[1, 2], [3, 4]])
        labels = np.array([0, 1])
        subjects = np.array(["s1", "s2"])
        with pytest.raises(ValueError, match="Need >= 3 subjects"):
            classify_states_loso(features, labels, subjects)


class TestAngleBetween2D:
    def test_orthogonal(self):
        angle = _angle_between_2d(np.array([1, 0]), np.array([0, 1]))
        np.testing.assert_allclose(angle, 90.0, atol=0.1)

    def test_parallel(self):
        angle = _angle_between_2d(np.array([1, 0]), np.array([2, 0]))
        np.testing.assert_allclose(angle, 0.0, atol=0.1)

    def test_antiparallel(self):
        angle = _angle_between_2d(np.array([1, 0]), np.array([-1, 0]))
        np.testing.assert_allclose(angle, 0.0, atol=0.1)

    def test_zero_vector(self):
        angle = _angle_between_2d(np.array([0, 0]), np.array([1, 0]))
        assert np.isnan(angle)


class TestAnalyzeGeometricStructure:
    def test_known_mahalanobis(self):
        rng = np.random.default_rng(42)
        n = 50
        g1 = rng.normal([0, 0], 1, (n, 2))
        g2 = rng.normal([3, 0], 1, (n, 2))
        features = np.vstack([g1, g2])
        labels = np.array(["awake"] * n + ["propofol"] * n)
        subjects = np.array([f"s{i}" for i in range(n)] + [f"s{i}" for i in range(n)])

        result = analyze_geometric_structure(
            features, labels, subjects,
            feature_names=["f1", "f2"], seed=42, n_bootstrap=100,
        )
        mah = result.pairwise_mahalanobis.get("awake_vs_propofol", 0)
        assert 2.0 < mah < 4.0

    def test_angular_separation(self):
        features = np.array([
            [0, 0], [1, 0], [0, 1],
            [0, 0], [1, 0], [0, 1],
        ], dtype=float)
        labels = np.array(["awake", "propofol", "REM", "awake", "propofol", "REM"])
        subjects = np.array(["s1", "s1", "s1", "s2", "s2", "s2"])

        result = analyze_geometric_structure(
            features, labels, subjects,
            feature_names=["gap", "cond"], seed=42, n_bootstrap=100,
        )
        assert "propofol" in result.state_change_vectors
        assert "REM" in result.state_change_vectors


class TestOrthogonality:
    def test_orthogonal_features(self):
        rng = np.random.default_rng(42)
        n = 40
        geom = rng.normal(0, 1, (n, 3))
        power = rng.normal(0, 1, (n, 1))
        labels = np.array([0] * 20 + [1] * 20)

        result = check_orthogonality(
            geom, power, labels,
            feature_names=["eigenvalue_gap", "condition_number", "spectral_radius"],
        )
        assert result.median_abs_correlation < 0.5

    def test_collinear_features(self):
        rng = np.random.default_rng(42)
        n = 40
        power = rng.normal(0, 1, (n, 1))
        geom = np.hstack([power * 2 + 0.01 * rng.normal(0, 1, (n, 1)),
                          power * -1 + 0.01 * rng.normal(0, 1, (n, 1)),
                          power * 0.5 + 0.01 * rng.normal(0, 1, (n, 1))])
        labels = np.array([0] * 20 + [1] * 20)

        result = check_orthogonality(
            geom, power, labels,
            feature_names=["eigenvalue_gap", "condition_number", "spectral_radius"],
        )
        assert result.median_abs_correlation > 0.5

    def test_two_level_pass_rule_primary_features(self):
        rng = np.random.default_rng(42)
        n = 40
        power = rng.normal(0, 1, (n, 1))
        gap = np.concatenate([rng.normal(0, 0.5, 20), rng.normal(3, 0.5, 20)])
        cond = np.concatenate([rng.normal(0, 0.5, 20), rng.normal(-3, 0.5, 20)])
        nd = np.concatenate([rng.normal(0, 0.5, 20), rng.normal(0.05, 0.5, 20)])
        sr = np.concatenate([rng.normal(0, 0.5, 20), rng.normal(-3, 0.5, 20)])
        geom = np.column_stack([gap, cond, nd, sr])
        labels = np.array([0] * 20 + [1] * 20)

        result = check_orthogonality(
            geom, power, labels,
            feature_names=["eigenvalue_gap", "condition_number", "nd_score", "spectral_radius"],
        )
        assert result.passes_threshold
        assert abs(result.residualized_effect_sizes["nd_score"]) < 0.5

    def test_correlations_use_baseline_only(self):
        rng = np.random.default_rng(42)
        n = 40
        geom = rng.normal(0, 1, (n, 3))
        power = np.concatenate([rng.normal(0, 1, (20, 1)), rng.normal(0, 1, (20, 1))])
        labels = np.array([0] * 20 + [1] * 20)

        result = check_orthogonality(
            geom, power, labels,
            feature_names=["eigenvalue_gap", "condition_number", "spectral_radius"],
        )
        assert len(result.per_feature_correlations) == 3


class TestCohensD:
    def test_known_effect(self):
        a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        b = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        d = _cohens_d(a, b)
        assert d > 3.0

    def test_no_effect(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        d = _cohens_d(a, b)
        np.testing.assert_allclose(d, 0.0, atol=1e-10)


class TestCollateExistingResults:
    def test_with_mock_data(self):
        results = {
            "exceptional_points": {
                "correlations": {
                    "sigma_vs_ep_score": {"r": 0.86, "p": 4.78e-06, "n": 18},
                },
            },
            "jackknife_sensitivity": {
                "correlations": {
                    "sigma_vs_ep_score": {
                        "full_sample": {"r": 0.86},
                        "jackknife": {
                            "r_min": 0.79,
                            "r_max": 0.89,
                            "all_significant_at_0.05": True,
                            "most_influential_subject": "CG103",
                        },
                    },
                },
            },
            "ep_shared_subspace_propofol": {
                "group_statistics": {
                    "comparison_to_original": {
                        "spectral_radius": {"original_d": -1.66, "shared_d": -1.72},
                        "min_eigenvalue_gap": {"original_d": 0.71, "shared_d": 0.78},
                    },
                },
            },
            "ep_shared_subspace_sleep": {
                "group_statistics": {
                    "comparison_to_original": {
                        "min_gap_n3_vs_rem": {"original_d": -2.51, "shared_d": -2.39},
                        "min_gap_awake_vs_rem": {"original_d": -2.13, "shared_d": -2.03},
                    },
                },
            },
            "condition_number_vs_kreiss_r": 0.912,
        }

        stability, criticality, amplification = collate_existing_results(results)

        assert len(stability) == 4
        assert all(s.passes for s in stability)

        assert len(criticality) == 2
        assert criticality[0].passes

        assert len(amplification) == 1
        assert amplification[0].passes

    def test_missing_data_graceful(self):
        stability, criticality, amplification = collate_existing_results({})
        assert len(stability) == 0
        assert len(criticality) == 0
        assert len(amplification) == 0


class TestVerdict:
    def test_strong_verdict(self):
        suff = [SufficiencyResult(
            contrast_name="test", dataset="test", auc_loso=0.90,
            auc_ci_lower=0.85, auc_ci_upper=0.95, accuracy_loso=0.85,
            per_subject_predictions=[], n_subjects=20, n_features=4,
            feature_names=[], subject_consistency=0.9,
            null_auc_mean=0.5, null_auc_std=0.05, null_auc_p=0.0,
            passes_threshold=True,
        )]
        incr = IncrementalValueResult(
            auc_geometry_only=0.9, auc_power_only=0.7, auc_combined=0.92,
            delta_auc_vs_power=0.2, delta_auc_combined_vs_power=0.22,
            bootstrap_ci_delta=(0.1, 0.3),
            null_auc_geometry=0.5, null_auc_power=0.5,
            passes_threshold=True,
        )
        orth = OrthogonalityResult(
            median_abs_correlation=0.1,
            per_feature_correlations={},
            residualized_effect_sizes={"gap": 0.8},
            passes_threshold=True,
        )
        struct = StructureResult(
            state_centroids={}, pairwise_mahalanobis={"a_vs_b": 2.5},
            state_change_vectors={}, angular_separations={},
            angular_separation_cis={}, subject_consistency={},
            passes_threshold=True,
        )
        stab = [CollatedResult("stab", "m", 1.0, 0.5, True)]
        crit = [CollatedResult("crit", "m", 0.86, 0.80, True)]
        amp = [CollatedResult("amp", "m", 0.91, 0.80, True)]

        verdict, n_pass, n_total = compute_overall_verdict(
            suff, incr, orth, struct, stab, crit, amp,
        )
        assert verdict == "strong"
        assert n_pass == 7

    def test_insufficient_verdict(self):
        suff = [SufficiencyResult(
            contrast_name="test", dataset="test", auc_loso=0.55,
            auc_ci_lower=0.4, auc_ci_upper=0.7, accuracy_loso=0.55,
            per_subject_predictions=[], n_subjects=20, n_features=4,
            feature_names=[], subject_consistency=0.5,
            null_auc_mean=0.5, null_auc_std=0.05, null_auc_p=0.3,
            passes_threshold=False,
        )]
        struct = StructureResult(
            state_centroids={}, pairwise_mahalanobis={"a_vs_b": 0.5},
            state_change_vectors={}, angular_separations={},
            angular_separation_cis={}, subject_consistency={},
            passes_threshold=False,
        )

        verdict, n_pass, n_total = compute_overall_verdict(
            suff, None, None, struct, [], [], [],
        )
        assert verdict == "insufficient"
        assert n_pass == 0

    def test_complementary_verdict(self):
        suff = [SufficiencyResult(
            contrast_name="test", dataset="test", auc_loso=0.85,
            auc_ci_lower=0.8, auc_ci_upper=0.9, accuracy_loso=0.8,
            per_subject_predictions=[], n_subjects=20, n_features=4,
            feature_names=[], subject_consistency=0.8,
            null_auc_mean=0.5, null_auc_std=0.05, null_auc_p=0.0,
            passes_threshold=True,
        )]
        struct = StructureResult(
            state_centroids={}, pairwise_mahalanobis={"a_vs_b": 2.0},
            state_change_vectors={}, angular_separations={},
            angular_separation_cis={}, subject_consistency={},
            passes_threshold=True,
        )
        stab = [CollatedResult("stab", "m", 1.0, 0.5, True)]
        crit = [CollatedResult("crit", "m", 0.86, 0.80, True)]

        verdict, n_pass, n_total = compute_overall_verdict(
            suff, None, None, struct, stab, crit, [],
        )
        assert verdict == "complementary"
        assert n_pass == 4
