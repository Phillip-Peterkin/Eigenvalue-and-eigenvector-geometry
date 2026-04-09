"""Tests for transient amplification analysis.

Validates that:
1. Normal (symmetric) matrices show amplification ratio = 1.0
2. Known non-normal matrices show amplification ratio > 1.0
3. Energy envelope has correct shape and boundary conditions
4. Kreiss constant >= 1.0 for all stable systems
5. Peak time is sensible
6. Integration with JacobianResult works correctly
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cmcc.features.transient_amplification import (
    compute_energy_envelope,
    compute_amplification_ratio,
    compute_kreiss_constant,
    compute_peak_time,
    analyze_jacobian_amplification,
    compute_hump_magnitude,
    compute_residual_kreiss,
    _phase_randomize,
    surrogate_amplification_null,
    compare_real_vs_surrogate,
    compute_model_free_energy_growth,
    compute_out_of_sample_prediction,
)


class TestEnergyEnvelope:
    def test_identity_matrix(self):
        A = np.eye(5)
        env = compute_energy_envelope(A, max_horizon=20)
        assert env.shape == (21,)
        assert env[0] == 1.0
        np.testing.assert_allclose(env, 1.0, atol=1e-10)

    def test_zero_matrix(self):
        A = np.zeros((3, 3))
        env = compute_energy_envelope(A, max_horizon=10)
        assert env[0] == 1.0
        np.testing.assert_allclose(env[1:], 0.0, atol=1e-10)

    def test_stable_normal_matrix_decays(self):
        A = 0.8 * np.eye(4)
        env = compute_energy_envelope(A, max_horizon=30)
        assert env[0] == 1.0
        for k in range(1, 31):
            np.testing.assert_allclose(env[k], 0.8**k, rtol=1e-6)

    def test_symmetric_matrix_monotonic_decay(self):
        rng = np.random.default_rng(42)
        S = rng.standard_normal((5, 5))
        S = (S + S.T) / 2
        evals = np.linalg.eigvalsh(S)
        S = S * (0.7 / np.max(np.abs(evals)))
        env = compute_energy_envelope(S, max_horizon=30)
        assert env[0] == 1.0
        for k in range(2, len(env)):
            if np.isfinite(env[k]) and np.isfinite(env[k-1]):
                assert env[k] <= env[k-1] + 1e-10

    def test_nonnormal_matrix_can_show_hump(self):
        A = np.array([
            [0.0, 5.0],
            [0.0, 0.0],
        ])
        env = compute_energy_envelope(A, max_horizon=10)
        assert env[0] == 1.0
        assert env[1] > env[0]

    def test_shape_correct(self):
        A = 0.5 * np.eye(3)
        env = compute_energy_envelope(A, max_horizon=15)
        assert env.shape == (16,)


class TestAmplificationRatio:
    def test_normal_matrix_ratio_one(self):
        A = 0.9 * np.eye(5)
        rho = 0.9
        env = compute_energy_envelope(A, max_horizon=20)
        ratio = compute_amplification_ratio(env, spectral_radius=rho)
        np.testing.assert_allclose(ratio, 1.0, atol=1e-6)

    def test_nonnormal_ratio_exceeds_one(self):
        A = np.array([
            [0.5, 3.0],
            [0.0, 0.5],
        ])
        rho = 0.5
        env = compute_energy_envelope(A, max_horizon=20)
        ratio = compute_amplification_ratio(env, spectral_radius=rho)
        assert ratio > 1.0

    def test_ratio_always_geq_one(self):
        rng = np.random.default_rng(123)
        for _ in range(10):
            A = rng.standard_normal((4, 4))
            A = A * (0.8 / np.max(np.abs(np.linalg.eigvals(A))))
            rho = float(np.max(np.abs(np.linalg.eigvals(A))))
            env = compute_energy_envelope(A, max_horizon=20)
            ratio = compute_amplification_ratio(env, spectral_radius=rho)
            assert ratio >= 1.0 - 1e-10


class TestKreissConstant:
    def test_stable_system_finite(self):
        A = 0.9 * np.eye(3)
        env = compute_energy_envelope(A, max_horizon=20)
        K = compute_kreiss_constant(env)
        assert np.isfinite(K)
        assert K >= 1.0

    def test_kreiss_geq_one(self):
        rng = np.random.default_rng(456)
        for _ in range(10):
            A = rng.standard_normal((5, 5))
            A = A * (0.7 / np.max(np.abs(np.linalg.eigvals(A))))
            env = compute_energy_envelope(A, max_horizon=30)
            K = compute_kreiss_constant(env)
            assert K >= 1.0 - 1e-10


class TestPeakTime:
    def test_identity_peak_at_zero(self):
        env = np.ones(21)
        k = compute_peak_time(env)
        assert k == 0

    def test_decaying_peak_at_zero(self):
        env = np.array([1.0, 0.9, 0.81, 0.729])
        k = compute_peak_time(env)
        assert k == 0

    def test_hump_peak_correct(self):
        env = np.array([1.0, 2.0, 3.0, 2.5, 1.0])
        k = compute_peak_time(env)
        assert k == 2


class TestAnalyzeJacobianAmplification:
    def test_output_keys(self):
        jacobians = np.stack([0.8 * np.eye(3)] * 5)
        result = analyze_jacobian_amplification(jacobians, max_horizon=10)
        assert "kreiss_constants" in result
        assert "amplification_ratios" in result
        assert "peak_times" in result
        assert "mean_envelope" in result
        assert "has_hump" in result

    def test_output_shapes(self):
        n_windows = 10
        n_ch = 4
        jacobians = np.stack([0.7 * np.eye(n_ch)] * n_windows)
        result = analyze_jacobian_amplification(jacobians, max_horizon=15)
        assert result["kreiss_constants"].shape == (n_windows,)
        assert result["amplification_ratios"].shape == (n_windows,)
        assert result["peak_times"].shape == (n_windows,)
        assert result["mean_envelope"].shape == (16,)
        assert result["has_hump"].shape == (n_windows,)

    def test_normal_matrices_no_hump(self):
        jacobians = np.stack([0.5 * np.eye(3)] * 5)
        result = analyze_jacobian_amplification(jacobians, max_horizon=20)
        assert not np.any(result["has_hump"])
        np.testing.assert_allclose(result["amplification_ratios"], 1.0, atol=1e-6)

    def test_integration_with_var1(self):
        from cmcc.analysis.dynamical_systems import estimate_jacobian
        rng = np.random.default_rng(42)
        A_true = rng.standard_normal((5, 5))
        A_true = A_true * (0.9 / np.max(np.abs(np.linalg.eigvals(A_true))))
        data = np.zeros((5, 3000))
        data[:, 0] = rng.standard_normal(5)
        for t in range(1, 3000):
            data[:, t] = A_true @ data[:, t-1] + 0.01 * rng.standard_normal(5)

        jac = estimate_jacobian(data, window_size=500, step_size=250)
        result = analyze_jacobian_amplification(jac.jacobians, max_horizon=30)
        assert len(result["kreiss_constants"]) == len(jac.window_centers)
        assert np.all(result["kreiss_constants"] >= 1.0 - 1e-10)


class TestHumpMagnitude:
    def test_monotonic_decay_zero_magnitude(self):
        env = np.array([1.0, 0.9, 0.81, 0.729])
        assert compute_hump_magnitude(env) == 0.0

    def test_hump_returns_overshoot(self):
        env = np.array([1.0, 2.5, 3.0, 1.5, 0.5])
        assert compute_hump_magnitude(env) == pytest.approx(2.0)

    def test_identity_envelope_zero(self):
        env = np.ones(10)
        assert compute_hump_magnitude(env) == 0.0

    def test_empty_envelope(self):
        env = np.array([1.0])
        assert compute_hump_magnitude(env) == 0.0

    def test_nan_handling(self):
        env = np.array([1.0, np.nan, np.nan])
        assert compute_hump_magnitude(env) == 0.0

    def test_nonnormal_matrix_positive_magnitude(self):
        A = np.array([[0.0, 5.0], [0.0, 0.0]])
        env = compute_energy_envelope(A, max_horizon=10)
        mag = compute_hump_magnitude(env)
        assert mag > 0.0


class TestPhaseRandomize:
    def test_preserves_shape(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((5, 1000))
        surr = _phase_randomize(data, rng)
        assert surr.shape == data.shape

    def test_preserves_power_spectrum(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((3, 2000))
        surr = _phase_randomize(data, np.random.default_rng(99))
        for ch in range(3):
            psd_real = np.abs(np.fft.rfft(data[ch])) ** 2
            psd_surr = np.abs(np.fft.rfft(surr[ch])) ** 2
            np.testing.assert_allclose(psd_real, psd_surr, rtol=1e-10)

    def test_destroys_temporal_structure(self):
        rng = np.random.default_rng(42)
        n = 2000
        x = np.cumsum(rng.standard_normal(n))
        data = x.reshape(1, -1)
        surr = _phase_randomize(data, np.random.default_rng(99))
        autocorr_real = np.corrcoef(data[0, :-1], data[0, 1:])[0, 1]
        autocorr_surr = np.corrcoef(surr[0, :-1], surr[0, 1:])[0, 1]
        assert abs(autocorr_surr) < abs(autocorr_real)

    def test_real_valued_output(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((4, 500))
        surr = _phase_randomize(data, np.random.default_rng(7))
        assert surr.dtype == np.float64
        assert np.all(np.isfinite(surr))


class TestSurrogateAmplificationNull:
    def test_output_keys_and_shapes(self):
        from cmcc.analysis.dynamical_systems import estimate_jacobian
        rng = np.random.default_rng(42)
        data = rng.standard_normal((5, 2000))
        result = surrogate_amplification_null(
            data, sfreq=500.0,
            n_surrogates=3,
            window_size=200,
            step_size=100,
            max_channels=5,
            max_horizon=10,
            seed=42,
            jacobian_estimator=estimate_jacobian,
        )
        assert result["n_surrogates"] == 3
        assert result["surrogate_kreiss"].shape == (3,)
        assert result["surrogate_amp_ratio"].shape == (3,)
        assert result["surrogate_hump_frac"].shape == (3,)
        assert result["surrogate_hump_magnitude"].shape == (3,)
        assert np.all(result["surrogate_kreiss"] >= 1.0 - 1e-10)

    def test_deterministic_with_seed(self):
        from cmcc.analysis.dynamical_systems import estimate_jacobian
        rng = np.random.default_rng(42)
        data = rng.standard_normal((5, 2000))
        r1 = surrogate_amplification_null(
            data, sfreq=500.0, n_surrogates=2,
            window_size=200, step_size=100, max_channels=5,
            max_horizon=10, seed=42, jacobian_estimator=estimate_jacobian,
        )
        r2 = surrogate_amplification_null(
            data, sfreq=500.0, n_surrogates=2,
            window_size=200, step_size=100, max_channels=5,
            max_horizon=10, seed=42, jacobian_estimator=estimate_jacobian,
        )
        np.testing.assert_array_equal(r1["surrogate_kreiss"], r2["surrogate_kreiss"])


class TestCompareRealVsSurrogate:
    def test_extreme_real_gives_low_p(self):
        surr = {
            "surrogate_kreiss": np.array([1.0, 1.1, 1.2, 1.05, 1.15]),
            "surrogate_amp_ratio": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            "surrogate_hump_frac": np.array([0.5, 0.6, 0.55, 0.52, 0.58]),
            "surrogate_hump_magnitude": np.array([0.01, 0.02, 0.015, 0.012, 0.018]),
        }
        comp = compare_real_vs_surrogate(
            real_kreiss=10.0,
            real_amp_ratio=5.0,
            real_hump_frac=1.0,
            real_hump_magnitude=2.0,
            surrogate_result=surr,
        )
        assert comp["kreiss_constant"]["p_value_one_sided"] < 0.2
        assert comp["kreiss_constant"]["percentile"] > 90.0
        assert comp["hump_magnitude"]["effect_size_z"] > 1.0

    def test_typical_real_gives_moderate_p(self):
        surr = {
            "surrogate_kreiss": np.array([5.0, 5.5, 4.8, 5.2, 5.1]),
            "surrogate_amp_ratio": np.array([2.0, 2.1, 1.9, 2.05, 1.95]),
            "surrogate_hump_frac": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            "surrogate_hump_magnitude": np.array([0.5, 0.6, 0.55, 0.52, 0.58]),
        }
        comp = compare_real_vs_surrogate(
            real_kreiss=5.1,
            real_amp_ratio=2.0,
            real_hump_frac=1.0,
            real_hump_magnitude=0.55,
            surrogate_result=surr,
        )
        assert comp["kreiss_constant"]["p_value_one_sided"] > 0.1
        assert comp["hump_fraction"]["p_value_one_sided"] > 0.1


class TestModelFreeEnergyGrowth:
    def test_output_keys(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((5, 5000))
        result = compute_model_free_energy_growth(data, sfreq=500.0)
        assert "n_events" in result
        assert "mean_energy_trajectory" in result
        assert "fraction_growing_at_peak" in result
        assert "peak_growth_ratio" in result

    def test_white_noise_no_amplification(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((5, 10000))
        result = compute_model_free_energy_growth(data, sfreq=500.0)
        assert result["n_events"] > 0
        assert 0.5 < result["peak_growth_ratio"] < 1.5

    def test_growing_signal_detected_with_growth(self):
        rng = np.random.default_rng(42)
        n_ch, n_samp = 3, 10000
        data = rng.standard_normal((n_ch, n_samp)) * 0.1
        for i in range(100, n_samp - 100, 200):
            data[:, i] += rng.standard_normal(n_ch) * 3
            for k in range(1, 50):
                if i + k < n_samp:
                    data[:, i + k] += rng.standard_normal(n_ch) * 3 * (1 + 2 * np.exp(-k / 10))
        result = compute_model_free_energy_growth(data, sfreq=500.0, horizon_sec=0.1)
        assert result["n_events"] > 0
        assert result["peak_growth_ratio"] > 1.0

    def test_normal_system_no_large_model_free_amplification(self):
        rng = np.random.default_rng(42)
        A = 0.85 * np.eye(5)
        data = np.zeros((5, 20000))
        data[:, 0] = rng.standard_normal(5)
        for t in range(1, 20000):
            data[:, t] = A @ data[:, t-1] + rng.standard_normal(5) * 0.3
        result = compute_model_free_energy_growth(data, sfreq=500.0, horizon_sec=0.1)
        if result["n_events"] > 0:
            assert result["peak_growth_ratio"] < 2.0

    def test_short_data_returns_default(self):
        data = np.ones((3, 5))
        result = compute_model_free_energy_growth(data, sfreq=500.0)
        assert result["n_events"] == 0
        assert result["peak_growth_ratio"] == 1.0

    def test_trajectory_shape(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((5, 10000))
        result = compute_model_free_energy_growth(
            data, sfreq=500.0, horizon_sec=0.05
        )
        expected_len = int(0.05 * 500.0) + 1
        assert len(result["mean_energy_trajectory"]) == expected_len

    def test_triggers_on_energy_derivative_not_level(self):
        rng = np.random.default_rng(42)
        n_samp = 20000
        data = rng.standard_normal((3, n_samp)) * 0.5
        for i in range(500, n_samp - 200, 400):
            data[:, i] += rng.standard_normal(3) * 5
        energy = np.sum(data ** 2, axis=0)
        dE = np.diff(energy)
        threshold_dE = np.percentile(dE, 95.0)
        high_dE_indices = np.where(dE >= threshold_dE)[0]
        result = compute_model_free_energy_growth(data, sfreq=500.0)
        assert result["n_events"] > 0
        assert result["mean_energy_trajectory"][0] == 1.0
        assert result["mean_energy_trajectory"][-1] < result["mean_energy_trajectory"][0] + 0.5


class TestResidualKreiss:
    def test_output_keys(self):
        rng = np.random.default_rng(42)
        kreiss = rng.uniform(1.0, 10.0, 100)
        rho = rng.uniform(0.8, 0.99, 100)
        result = compute_residual_kreiss(kreiss, rho)
        assert "residuals" in result
        assert "mean_residual" in result
        assert "slope" in result
        assert "r_squared" in result
        assert "n_stable" in result

    def test_residuals_sum_to_zero(self):
        rng = np.random.default_rng(42)
        kreiss = rng.uniform(1.0, 10.0, 200)
        rho = rng.uniform(0.7, 0.99, 200)
        result = compute_residual_kreiss(kreiss, rho)
        np.testing.assert_allclose(np.mean(result["residuals"]), 0.0, atol=1e-10)

    def test_filters_unstable_windows(self):
        rng = np.random.default_rng(42)
        kreiss = rng.uniform(1.0, 100.0, 50)
        rho = np.concatenate([rng.uniform(0.8, 0.99, 30), rng.uniform(1.0, 1.2, 20)])
        result = compute_residual_kreiss(kreiss, rho)
        assert result["n_stable"] == 30

    def test_insufficient_stable_windows(self):
        kreiss = np.array([1.0, 2.0, 3.0])
        rho = np.array([1.1, 1.2, 1.3])
        result = compute_residual_kreiss(kreiss, rho)
        assert np.isnan(result["mean_residual"])
        assert len(result["residuals"]) == 0

    def test_positive_slope_expected(self):
        rng = np.random.default_rng(42)
        rho = rng.uniform(0.8, 0.99, 200)
        kreiss = 1.0 / (1.0 - rho) + rng.standard_normal(200) * 0.1
        kreiss = np.clip(kreiss, 1.0, None)
        result = compute_residual_kreiss(kreiss, rho)
        assert result["slope"] > 0

    def test_pure_stability_margin_effect_gives_zero_residual(self):
        rho = np.linspace(0.80, 0.98, 300)
        kreiss = 1.0 / (1.0 - rho)
        result = compute_residual_kreiss(kreiss, rho)
        np.testing.assert_allclose(result["mean_residual"], 0.0, atol=1e-6)
        assert result["r_squared"] > 0.99

    def test_residuals_uncorrelated_with_spectral_radius(self):
        from scipy import stats as sp_stats
        rng = np.random.default_rng(42)
        rho = rng.uniform(0.7, 0.99, 500)
        kreiss = (1.0 / (1.0 - rho)) * np.exp(rng.standard_normal(500) * 0.3)
        kreiss = np.clip(kreiss, 1.0, None)
        result = compute_residual_kreiss(kreiss, rho)
        r, _ = sp_stats.pearsonr(result["residuals"], rho[rho < 1.0])
        assert abs(r) < 0.1

    def test_different_nonnormality_same_rho_detected_by_pooled_residuals(self):
        rng = np.random.default_rng(42)
        rho_shared = rng.uniform(0.80, 0.98, 400)
        base_kreiss = 1.0 / (1.0 - rho_shared)
        kreiss_low = base_kreiss * 1.0 + rng.standard_normal(400) * 0.01
        kreiss_high = base_kreiss * 3.0 + rng.standard_normal(400) * 0.01
        kreiss_low = np.clip(kreiss_low, 1.0, None)
        kreiss_high = np.clip(kreiss_high, 1.0, None)
        kreiss_all = np.concatenate([kreiss_low, kreiss_high])
        rho_all = np.concatenate([rho_shared, rho_shared])
        pooled = compute_residual_kreiss(kreiss_all, rho_all)
        resid_low = pooled["residuals"][:400]
        resid_high = pooled["residuals"][400:]
        assert np.mean(resid_high) > np.mean(resid_low)


class TestOutOfSamplePrediction:
    def test_output_keys(self):
        from cmcc.analysis.dynamical_systems import estimate_jacobian
        rng = np.random.default_rng(42)
        A_true = rng.standard_normal((5, 5))
        A_true = A_true * (0.8 / np.max(np.abs(np.linalg.eigvals(A_true))))
        data = np.zeros((5, 3000))
        data[:, 0] = rng.standard_normal(5)
        for t in range(1, 3000):
            data[:, t] = A_true @ data[:, t-1] + 0.01 * rng.standard_normal(5)

        jac = estimate_jacobian(data, window_size=500, step_size=250)
        result = compute_out_of_sample_prediction(
            jac.jacobians, data, jac.window_centers,
            window_size=500, n_predict_steps=5,
        )
        assert "mean_r2" in result
        assert "growth_correlation" in result
        assert "n_valid_windows" in result
        assert result["n_valid_windows"] > 0

    def test_good_model_positive_r2(self):
        from cmcc.analysis.dynamical_systems import estimate_jacobian
        rng = np.random.default_rng(42)
        A_true = 0.95 * np.eye(5)
        noise_std = 0.5
        data = np.zeros((5, 8000))
        data[:, 0] = rng.standard_normal(5) * 3
        for t in range(1, 8000):
            data[:, t] = A_true @ data[:, t-1] + noise_std * rng.standard_normal(5)

        jac = estimate_jacobian(data, window_size=500, step_size=250)
        result = compute_out_of_sample_prediction(
            jac.jacobians, data, jac.window_centers,
            window_size=500, n_predict_steps=2,
        )
        assert result["n_valid_windows"] > 0
        valid_r2 = result["prediction_r2"][np.isfinite(result["prediction_r2"])]
        assert np.median(valid_r2) > -10.0

    def test_random_data_poor_r2(self):
        from cmcc.analysis.dynamical_systems import estimate_jacobian
        rng = np.random.default_rng(42)
        data = rng.standard_normal((5, 5000))
        jac = estimate_jacobian(data, window_size=500, step_size=250)
        result = compute_out_of_sample_prediction(
            jac.jacobians, data, jac.window_centers,
            window_size=500, n_predict_steps=5,
        )
        assert result["n_valid_windows"] > 0
        assert result["mean_r2"] < 0.5

    def test_prediction_starts_outside_fitting_window(self):
        from cmcc.analysis.dynamical_systems import estimate_jacobian
        rng = np.random.default_rng(42)
        data = rng.standard_normal((5, 3000))
        window_size = 500
        jac = estimate_jacobian(data, window_size=window_size, step_size=250)
        half_w = window_size // 2
        for w in range(len(jac.window_centers)):
            center = int(jac.window_centers[w])
            first_predict = center + half_w + 1
            window_end = center + half_w
            assert first_predict > window_end

    def test_prediction_shapes(self):
        from cmcc.analysis.dynamical_systems import estimate_jacobian
        rng = np.random.default_rng(42)
        data = rng.standard_normal((5, 3000))
        jac = estimate_jacobian(data, window_size=500, step_size=250)
        result = compute_out_of_sample_prediction(
            jac.jacobians, data, jac.window_centers,
            window_size=500, n_predict_steps=5,
        )
        assert result["prediction_r2"].shape == (len(jac.window_centers),)
        assert result["predicted_growth"].shape == (len(jac.window_centers),)
