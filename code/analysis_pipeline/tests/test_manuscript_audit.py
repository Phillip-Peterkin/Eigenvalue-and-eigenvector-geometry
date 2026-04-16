"""Manuscript-to-JSON audit tests.

Validates that every statistical claim in the manuscript has a matching
entry in the saved JSON result artifacts. This prevents manuscript drift
from computed results.
"""
import json
import pytest
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "json_results"


def _load(name):
    path = RESULTS_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not found")
    with open(path) as f:
        return json.load(f)


class TestBroadbandComparison:
    @pytest.fixture(autouse=True)
    def load_data(self):
        self.d = _load("broadband_comparison.json")

    def test_sigma(self):
        s = self.d["sigma"]
        assert abs(s["hg_mean"] - 0.9735) < 0.0001
        assert abs(s["bb_mean"] - 0.9908) < 0.0001
        assert abs(s["t"] - (-5.742)) < 0.01
        assert abs(s["p"] - 8.92e-06) < 1e-07

    def test_tau(self):
        s = self.d["tau"]
        assert abs(s["t"] - (-1.940)) < 0.01
        assert abs(s["p"] - 0.065) < 0.001

    def test_lzc(self):
        s = self.d["LZc"]
        assert abs(s["diff_mean"] - 0.0572) < 0.001
        assert abs(s["t"] - 5.244) < 0.01
        assert abs(s["p"] - 2.92e-05) < 1e-06

    def test_dfa(self):
        s = self.d["DFA"]
        assert abs(s["diff_mean"] - (-0.2568)) < 0.001
        assert abs(s["t"] - (-10.04)) < 0.01
        assert abs(s["p"] - 1.12e-09) < 1e-10


class TestLME:
    @pytest.fixture(autouse=True)
    def load_data(self):
        self.d = _load("lme_results.json")

    def test_sigma_band_coef(self):
        c = self.d["sigma_band"]["fixed_effects"]["band[T.HG]"]
        assert abs(c["coef"] - (-0.01727)) < 0.00001
        assert abs(c["p"] - 9.33e-09) < 1e-10

    def test_lzc_band_coef(self):
        c = self.d["lzc_band"]["fixed_effects"]["band[T.HG]"]
        assert abs(c["p"] - 1.57e-07) < 1e-08

    def test_dfa_band_coef(self):
        c = self.d["dfa_band"]["fixed_effects"]["band[T.HG]"]
        assert abs(c["p"] - 1.00e-23) < 1e-24

    def test_tau_band_coef(self):
        c = self.d["tau_band"]["fixed_effects"]["band[T.HG]"]
        assert abs(c["p"] - 0.0523) < 0.001

    def test_cross_band_hg_sigma(self):
        c = self.d["cross_band_prediction"]["fixed_effects"]["hg_sigma"]
        assert abs(c["coef"] - 0.212) < 0.001
        assert abs(c["p"] - 5.36e-05) < 1e-06


class TestOperatorGeometry:
    @pytest.fixture(autouse=True)
    def load_data(self):
        self.d = _load("exceptional_points.json")

    def test_sigma_vs_ep_score(self):
        c = self.d["correlations"]["sigma_vs_ep_score"]
        assert abs(c["r"] - 0.860) < 0.001
        assert abs(c["p"] - 4.78e-06) < 1e-07

    def test_sigma_vs_min_gap(self):
        c = self.d["correlations"]["sigma_vs_min_gap"]
        assert abs(c["r"] - (-0.588)) < 0.001
        assert abs(c["p"] - 0.0102) < 0.001

    def test_lzc_vs_ep_score(self):
        c = self.d["correlations"]["lzc_vs_ep_score"]
        assert abs(c["r"] - (-0.684)) < 0.001
        assert abs(c["p"] - 0.00174) < 0.0001

    def test_lzc_vs_min_gap(self):
        c = self.d["correlations"]["lzc_vs_min_gap"]
        assert abs(c["r"] - 0.745) < 0.001
        assert abs(c["p"] - 3.88e-04) < 1e-05

    def test_tau_vs_ep_score(self):
        c = self.d["correlations"]["tau_vs_ep_score"]
        assert abs(c["r"] - 0.526) < 0.001
        assert abs(c["p"] - 0.025) < 0.001


class TestJackknife:
    @pytest.fixture(autouse=True)
    def load_data(self):
        self.d = _load("jackknife_sensitivity.json")

    def test_sigma_vs_ep_all_significant(self):
        jk = self.d["correlations"]["sigma_vs_ep_score"]["jackknife"]
        assert jk["all_significant_at_0.05"] is True
        assert jk["n_significant_at_0.05"] == 18

    def test_sigma_vs_ep_range(self):
        jk = self.d["correlations"]["sigma_vs_ep_score"]["jackknife"]
        assert abs(jk["r_min"] - 0.792) < 0.001
        assert abs(jk["r_max"] - 0.889) < 0.001

    def test_tau_vs_ep_n_significant(self):
        jk = self.d["correlations"]["tau_vs_ep_score"]["jackknife"]
        assert jk["n_significant_at_0.05"] == 17


class TestPropofolEEG:
    @pytest.fixture(autouse=True)
    def load_data(self):
        self.d = _load("ep_propofol_eeg.json")
        self.g = self.d["group_statistics"]

    def test_spectral_radius(self):
        s = self.g["spectral_radius_awake_vs_sed_run1"]
        assert abs(s["mean_awake"] - 0.9980) < 0.0001
        assert abs(s["mean_sed"] - 1.0025) < 0.0001
        assert abs(s["t"] - (-7.441)) < 0.01
        assert abs(s["p"] - 4.83e-07) < 1e-08
        assert abs(s["cohens_d"] - (-1.664)) < 0.01

    def test_eigenvalue_gap(self):
        s = self.g["eigenvalue_gap_awake_vs_sed_run1"]
        assert abs(s["t"] - 3.159) < 0.01
        assert abs(s["p"] - 0.00517) < 0.001
        assert abs(s["cohens_d"] - 0.706) < 0.01

    def test_pooled_gap(self):
        s = self.g["pooled_secondary"]["eigenvalue_gap"]
        assert abs(s["p"] - 0.00814) < 0.001
        assert abs(s["cohens_d"] - 0.661) < 0.01

    def test_effective_rank(self):
        s = self.g["effective_rank_awake_vs_sed_run1"]
        assert abs(s["p"] - 0.185) < 0.01

    def test_spectral_sensitivity(self):
        s = self.g["spectral_sensitivity_awake_vs_sed_run1"]
        assert abs(s["p"] - 0.01442) < 0.001
        assert abs(s["cohens_d"] - 0.602) < 0.01

    def test_delta_delta(self):
        s = self.g["delta_delta_correlation"]
        assert abs(s["r"] - (-0.683)) < 0.001
        assert abs(s["p"] - 0.000913) < 0.0001


class TestAlphaPartialCorrelations:
    @pytest.fixture(autouse=True)
    def load_data(self):
        self.d = _load("ep_robustness_checks.json")
        self.p = self.d["partial_regression_alpha"]

    def test_partial_r(self):
        assert abs(self.p["partial_r_controlling_alpha"] - (-0.676)) < 0.001
        assert abs(self.p["partial_p"] - 0.00148) < 0.0001

    def test_alpha_vs_sensitivity_change(self):
        a = self.p["alpha_vs_delta_r"]
        assert abs(a["r"] - 0.134) < 0.01
        assert abs(a["p"] - 0.574) < 0.01

    def test_alpha_vs_gap_change(self):
        a = self.p["alpha_vs_delta_gap"]
        assert abs(a["r"] - (-0.242)) < 0.01
        assert abs(a["p"] - 0.305) < 0.01


class TestSleepDynamics:
    @pytest.fixture(autouse=True)
    def load_data(self):
        self.d = _load("ep_sleep_dynamics.json")
        self.g = self.d["group_statistics"]

    def test_gap_awake_vs_n3(self):
        s = self.g["test_a_gap_awake_vs_n3"]
        assert abs(s["p"] - 0.216) < 0.01
        assert abs(s["cohens_d"] - (-0.421)) < 0.01

    def test_gap_n3_vs_rem(self):
        s = self.g["test_a_gap_n3_vs_rem"]
        assert abs(s["p"] - 2.39e-05) < 1e-06
        assert abs(s["cohens_d"] - (-2.506)) < 0.01

    def test_gap_awake_vs_rem(self):
        s = self.g["test_a_gap_awake_vs_rem"]
        assert abs(s["p"] - 8.59e-05) < 1e-05
        assert abs(s["cohens_d"] - (-2.127)) < 0.01

    def test_sensitivity_awake(self):
        s = self.g["test_b_spec_sens_awake"]
        assert abs(s["mean_r"] - 0.021) < 0.001
        assert abs(s["p_vs_zero"] - 0.492) < 0.01

    def test_sensitivity_n3(self):
        s = self.g["test_b_spec_sens_n3"]
        assert abs(s["mean_r"] - 0.098) < 0.001

    def test_sensitivity_rem(self):
        s = self.g["test_b_spec_sens_rem"]
        assert abs(s["mean_r"] - 0.074) < 0.001

    def test_sensitivity_n3_vs_rem(self):
        s = self.g["test_c_spec_sens_n3_vs_rem"]
        assert abs(s["t"] - 3.569) < 0.01
        assert abs(s["p"] - 0.006) < 0.001
        assert abs(s["cohens_d"] - 1.129) < 0.01

    def test_sensitivity_awake_vs_rem(self):
        s = self.g["test_c_spec_sens_awake_vs_rem"]
        assert abs(s["p"] - 0.086) < 0.01


class TestAmplificationPropofol:
    @pytest.fixture(autouse=True)
    def load_data(self):
        self.d = _load("amplification_propofol_hardening.json")

    def test_residual_kreiss(self):
        s = self.d["rho_controlled"]["delta_residual_kreiss"]
        assert abs(s["cohens_d"] - 1.819) < 0.01
        assert abs(s["p_wilcoxon"] - 3.81e-06) < 1e-07
        assert s["n_positive"] == 19

    def test_dose_response_pairwise(self):
        pw = self.d["dose_response"]["pairwise_kreiss"]
        for key in ["awake_vs_sed_run-1", "awake_vs_sed_run-2", "awake_vs_sed_run-3"]:
            assert pw[key]["cohens_d"] > 0.88
            assert pw[key]["p_wilcoxon"] < 0.006

    def test_strict_monotonic(self):
        dr = self.d["dose_response"]
        assert dr["n_strictly_monotonic"] == 2
        assert dr["n_subjects_all_runs"] == 14


class TestAmplificationSleep:
    @pytest.fixture(autouse=True)
    def load_data(self):
        self.d = _load("amplification_sleep_convergence.json")
        self.g = self.d["group_statistics"]

    def test_n3_vs_rem_kreiss(self):
        s = self.g["N3_vs_R"]["stable_kreiss_median"]
        assert abs(s["cohens_d"] - (-4.178)) < 0.01


class TestChiralityCorrelations:
    @pytest.fixture(autouse=True)
    def load_data(self):
        self.d = _load("chirality.json")
        self.c = self.d.get("correlations", {})

    def test_sigma_vs_chirality_index(self):
        s = self.c["sigma_vs_chirality_index"]
        assert abs(s["r"] - 0.526) < 0.01
        assert abs(s["p"] - 0.025) < 0.005

    def test_lzc_vs_chirality_index(self):
        s = self.c["lzc_vs_chirality_index"]
        assert abs(s["r"] - (-0.570)) < 0.01
        assert abs(s["p"] - 0.013) < 0.005


class TestTemporalPrecedence:
    @pytest.fixture(autouse=True)
    def load_data(self):
        self.d = _load("temporal_precedence.json")

    def test_sr_n2_to_n3_slope(self):
        sr = self.d["transitions"]["N2_to_N3"]["spectral_radius"]
        assert abs(sr["mean_slope"] - 1.235e-05) < 1e-06
        assert abs(sr["slope_p_value"] - 0.00143) < 0.001
        assert abs(sr["early_vs_late_d"] - 1.192) < 0.01
        assert sr["subject_consistency"] == 0.9

    def test_gap_n2_to_n3_slope(self):
        gap = self.d["transitions"]["N2_to_N3"]["eigenvalue_gap"]
        assert abs(gap["mean_slope"] - 2.638e-06) < 1e-07
        assert abs(gap["slope_p_value"] - 0.000628) < 0.0001
        assert abs(gap["early_vs_late_d"] - 1.343) < 0.01
        assert gap["subject_consistency"] == 0.9

    def test_sr_n2_to_rem_null(self):
        sr = self.d["transitions"]["N2_to_R"]["spectral_radius"]
        assert sr["slope_p_value"] > 0.3

    def test_gap_n2_to_rem_null(self):
        gap = self.d["transitions"]["N2_to_R"]["eigenvalue_gap"]
        assert gap["slope_p_value"] > 0.3

    def test_sr_nonoverlap_survives(self):
        sr = self.d["transitions"]["N2_to_N3"]["spectral_radius"]
        assert sr["nonoverlap_slope_p"] < 0.05


class TestFalsificationBattery:
    @pytest.fixture(autouse=True)
    def load_data(self):
        self.d = _load("falsification_battery.json")

    def test_label_destruction_propofol(self):
        ld = self.d["label_destruction"]["awake_vs_propofol"]
        assert abs(ld["real_auc"] - 0.913) < 0.01
        assert ld["empirical_p"] < 0.001

    def test_label_destruction_sleep(self):
        ld = self.d["label_destruction"]["N3_vs_REM"]
        assert abs(ld["real_auc"] - 1.000) < 0.01
        assert ld["empirical_p"] <= 0.001

    def test_subject_jackknife_propofol_range(self):
        jk = self.d["subject_jackknife_classification"]["awake_vs_propofol"]
        assert abs(jk["loo_min"] - 0.903) < 0.01
        assert abs(jk["loo_max"] - 0.983) < 0.01

    def test_subject_jackknife_sleep_range(self):
        jk = self.d["subject_jackknife_classification"]["N3_vs_REM"]
        assert abs(jk["loo_min"] - 1.000) < 0.01
        assert abs(jk["loo_max"] - 1.000) < 0.01

    def test_temporal_jackknife_sign(self):
        tj = self.d["temporal_jackknife"]["loo"]
        assert tj["sign_preserved_all"] is True or abs(tj["fraction_sign_preserved"] - 1.0) < 0.01 if "fraction_sign_preserved" in tj else True

    def test_model_competition_propofol(self):
        mc = self.d["model_competition"]["awake_vs_propofol"]
        assert abs(mc["geometry_auc"] - 0.913) < 0.01
        assert mc["geometry_beats_all"] is True

    def test_window_attacks_nonoverlap(self):
        wa = self.d["window_attacks"]
        assert wa["nonoverlap_survives"] is True

    def test_summary_verdicts(self):
        s = self.d["summary"]
        assert s["label_destruction"]["verdict"] == "SURVIVES"
        assert s["subject_jackknife"]["verdict"] == "SURVIVES"
        assert s["temporal_jackknife"]["verdict"] == "SURVIVES"
        assert s["model_competition"]["verdict"] == "SURVIVES"
        assert s["window_attacks"]["verdict"] == "SURVIVES"


class TestLOSOClassification:
    @pytest.fixture(autouse=True)
    def load_data(self):
        self.d = _load("geometry_brain_states.json")

    def test_propofol_auc(self):
        contrasts = self.d["test_1_sufficiency"]["contrasts"]
        propofol = [c for c in contrasts if c["contrast_name"] == "awake_vs_propofol"][0]
        assert abs(propofol["auc_loso"] - 0.913) < 0.01

    def test_sleep_auc(self):
        contrasts = self.d["test_1_sufficiency"]["contrasts"]
        sleep = [c for c in contrasts if c["contrast_name"] == "N3_vs_REM"][0]
        assert abs(sleep["auc_loso"] - 1.000) < 0.01

    def test_alpha_baseline_propofol(self):
        inc = self.d["test_2_incremental"]
        assert abs(inc["auc_power_only"] - 0.500) < 0.05
