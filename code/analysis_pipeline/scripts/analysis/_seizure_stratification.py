"""Seizure stratification by eigenvalue spacing behavior.

Reprocesses all eligible CHB-MIT seizures and:
1. Stratifies into narrowing vs widening groups (pre-ictal raw spacing vs baseline)
2. Extracts per-seizure spectral, signal, and model features
3. Tests whether narrowing vs widening groups differ on extracted features
4. Tests subject-specificity of behavior
5. Saves per-seizure feature table for downstream analyses (self-alignment, individual plots)

Scientific rationale:
Population-level averaging showed a weak/null pre-ictal narrowing signal.
Stratification asks: are there distinct dynamical sub-populations, and if so,
what separates them? This is a shift from confirmatory to exploratory analysis
and must be clearly labeled as such in any reporting.
"""
from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
warnings.filterwarnings("ignore")

import yaml
from cmcc.io.loader_chbmit import build_seizure_catalog, load_raw_edf
from cmcc.preprocess.seizure_eeg import (
    fit_baseline_pca, preprocess_chbmit_raw, project_to_pca, reject_artifact_windows,
)
from cmcc.analysis.seizure_dynamics import (
    SeizureTrajectory, compute_seizure_trajectory, compute_preictal_slope,
)
from cmcc.analysis.dynamical_systems import estimate_jacobian
from cmcc.features.dfa import compute_dfa
from cmcc.provenance import log_run, save_summary_json

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = CMCC_ROOT / "configs" / "chbmit.yaml"


def log(msg):
    print(msg, flush=True)


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def compute_spectral_slope(psd_freqs, psd_vals, freq_range=(2.0, 40.0)):
    mask = (psd_freqs >= freq_range[0]) & (psd_freqs <= freq_range[1])
    f_sub = psd_freqs[mask]
    p_sub = psd_vals[mask]
    if len(f_sub) < 5 or np.any(p_sub <= 0) or np.any(f_sub <= 0):
        return float("nan")
    log_f = np.log10(f_sub)
    log_p = np.log10(p_sub)
    valid = np.isfinite(log_f) & np.isfinite(log_p)
    if valid.sum() < 5:
        return float("nan")
    coeffs = np.polyfit(log_f[valid], log_p[valid], 1)
    return float(coeffs[0])


def compute_line_length(data_segment):
    if data_segment.ndim == 2:
        ll = np.mean(np.sum(np.abs(np.diff(data_segment, axis=1)), axis=1))
    else:
        ll = np.sum(np.abs(np.diff(data_segment)))
    return float(ll)


def compute_autocorrelation_lag1(data_segment):
    if data_segment.ndim == 2:
        ac_vals = []
        for ch in range(data_segment.shape[0]):
            x = data_segment[ch]
            x = x - np.mean(x)
            var = np.var(x)
            if var == 0:
                ac_vals.append(0.0)
                continue
            ac = np.correlate(x[:-1], x[1:], mode="valid")[0] / (var * (len(x) - 1))
            ac_vals.append(float(ac))
        return float(np.mean(ac_vals))
    x = data_segment - np.mean(data_segment)
    var = np.var(x)
    if var == 0:
        return 0.0
    return float(np.correlate(x[:-1], x[1:], mode="valid")[0] / (var * (len(x) - 1)))


def extract_seizure_features(
    data_pca, sfreq, traj, jac_result, seizure_event, baseline_start, baseline_end,
):
    n_ch, n_samples = data_pca.shape

    bl_start_samp = max(0, int(baseline_start * sfreq))
    bl_end_samp = min(n_samples, int(baseline_end * sfreq))
    pre_start_samp = max(0, int((seizure_event.onset_sec - 600) * sfreq))
    pre_end_samp = min(n_samples, int(seizure_event.onset_sec * sfreq))

    bl_seg = data_pca[:, bl_start_samp:bl_end_samp]
    pre_seg = data_pca[:, pre_start_samp:pre_end_samp]

    features = {
        "subject_id": seizure_event.subject_id,
        "seizure_onset_sec": seizure_event.onset_sec,
        "seizure_duration_sec": seizure_event.duration_sec,
        "event_type": seizure_event.event_type,
    }

    bl_mask = (traj.time_sec >= (baseline_start - seizure_event.onset_sec)) & \
              (traj.time_sec < (baseline_end - seizure_event.onset_sec))
    pre_mask = (traj.time_sec >= -600) & (traj.time_sec < 0)

    bl_raw = traj.min_spacing_raw[bl_mask] if bl_mask.sum() > 0 else np.array([])
    pre_raw = traj.min_spacing_raw[pre_mask] if pre_mask.sum() > 0 else np.array([])

    bl_raw_mean = float(np.nanmean(bl_raw)) if len(bl_raw) > 0 else float("nan")
    pre_raw_mean = float(np.nanmean(pre_raw)) if len(pre_raw) > 0 else float("nan")

    features["bl_raw_spacing_mean"] = bl_raw_mean
    features["pre_raw_spacing_mean"] = pre_raw_mean
    features["raw_spacing_change"] = pre_raw_mean - bl_raw_mean
    features["group"] = "narrowing" if pre_raw_mean < bl_raw_mean else "widening"
    features["preictal_slope"] = traj.preictal_slope
    features["baseline_mean"] = traj.baseline_mean
    features["baseline_std"] = traj.baseline_std

    pre_z = traj.min_spacing_z[pre_mask] if pre_mask.sum() > 0 else np.array([])
    features["pre_z_mean"] = float(np.nanmean(pre_z)) if len(pre_z) > 0 else float("nan")

    if bl_seg.shape[1] > 256:
        from scipy.signal import welch
        f_bl, psd_bl = welch(bl_seg, fs=sfreq, nperseg=min(256, bl_seg.shape[1] // 2), axis=1)
        psd_bl_mean = np.mean(psd_bl, axis=0)
    else:
        f_bl = np.array([])
        psd_bl_mean = np.array([])

    if pre_seg.shape[1] > 256:
        from scipy.signal import welch
        f_pre, psd_pre = welch(pre_seg, fs=sfreq, nperseg=min(256, pre_seg.shape[1] // 2), axis=1)
        psd_pre_mean = np.mean(psd_pre, axis=0)
    else:
        f_pre = np.array([])
        psd_pre_mean = np.array([])

    def band_power(freqs, psd, lo, hi):
        if len(freqs) == 0:
            return float("nan")
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.sum() == 0:
            return float("nan")
        return float(np.mean(psd[mask]))

    features["bl_alpha_power"] = band_power(f_bl, psd_bl_mean, 8, 12)
    features["pre_alpha_power"] = band_power(f_pre, psd_pre_mean, 8, 12)
    features["bl_delta_power"] = band_power(f_bl, psd_bl_mean, 1, 4)
    features["pre_delta_power"] = band_power(f_pre, psd_pre_mean, 1, 4)
    features["bl_broadband_power"] = band_power(f_bl, psd_bl_mean, 0.5, 45)
    features["pre_broadband_power"] = band_power(f_pre, psd_pre_mean, 0.5, 45)

    features["bl_spectral_slope"] = compute_spectral_slope(f_bl, psd_bl_mean) if len(f_bl) > 0 else float("nan")
    features["pre_spectral_slope"] = compute_spectral_slope(f_pre, psd_pre_mean) if len(f_pre) > 0 else float("nan")
    features["spectral_slope_change"] = features["pre_spectral_slope"] - features["bl_spectral_slope"]

    features["alpha_change"] = features["pre_alpha_power"] - features["bl_alpha_power"]
    features["delta_change"] = features["pre_delta_power"] - features["bl_delta_power"]

    features["bl_variance"] = float(np.var(bl_seg)) if bl_seg.shape[1] > 0 else float("nan")
    features["pre_variance"] = float(np.var(pre_seg)) if pre_seg.shape[1] > 0 else float("nan")
    features["variance_change"] = features["pre_variance"] - features["bl_variance"]

    features["bl_line_length"] = compute_line_length(bl_seg) if bl_seg.shape[1] > 1 else float("nan")
    features["pre_line_length"] = compute_line_length(pre_seg) if pre_seg.shape[1] > 1 else float("nan")

    features["bl_autocorr_lag1"] = compute_autocorrelation_lag1(bl_seg) if bl_seg.shape[1] > 10 else float("nan")
    features["pre_autocorr_lag1"] = compute_autocorrelation_lag1(pre_seg) if pre_seg.shape[1] > 10 else float("nan")

    if bl_seg.shape[1] > 100:
        bl_mean_ts = np.mean(bl_seg, axis=0)
        dfa_bl = compute_dfa(bl_mean_ts)
        features["bl_dfa_alpha"] = dfa_bl.alpha
    else:
        features["bl_dfa_alpha"] = float("nan")

    if pre_seg.shape[1] > 100:
        pre_mean_ts = np.mean(pre_seg, axis=0)
        dfa_pre = compute_dfa(pre_mean_ts)
        features["pre_dfa_alpha"] = dfa_pre.alpha
    else:
        features["pre_dfa_alpha"] = float("nan")
    features["dfa_change"] = features["pre_dfa_alpha"] - features["bl_dfa_alpha"]

    n_jac_windows = len(jac_result.spectral_radius)
    jac_time = jac_result.window_centers / sfreq
    jac_bl_mask = (jac_time >= baseline_start) & (jac_time < baseline_end)
    jac_pre_mask = (jac_time >= (seizure_event.onset_sec - 600)) & (jac_time < seizure_event.onset_sec)

    def safe_mean(arr, mask):
        m = mask[:len(arr)]
        vals = arr[m]
        return float(np.nanmean(vals)) if len(vals) > 0 else float("nan")

    features["bl_spectral_radius"] = safe_mean(jac_result.spectral_radius, jac_bl_mask)
    features["pre_spectral_radius"] = safe_mean(jac_result.spectral_radius, jac_pre_mask)
    features["spectral_radius_change"] = features["pre_spectral_radius"] - features["bl_spectral_radius"]

    features["bl_residual_var"] = safe_mean(jac_result.residual_variance, jac_bl_mask)
    features["pre_residual_var"] = safe_mean(jac_result.residual_variance, jac_pre_mask)
    features["residual_var_change"] = features["pre_residual_var"] - features["bl_residual_var"]

    features["bl_condition_number"] = safe_mean(jac_result.condition_numbers, jac_bl_mask)
    features["pre_condition_number"] = safe_mean(jac_result.condition_numbers, jac_pre_mask)

    pre_ep = traj.ep_score_z[pre_mask] if pre_mask.sum() > 0 else np.array([])
    features["pre_ep_score_z"] = float(np.nanmean(pre_ep)) if len(pre_ep) > 0 else float("nan")

    min_idx_mask = (traj.time_sec >= -600) & (traj.time_sec <= 300)
    if min_idx_mask.sum() > 5:
        s = traj.min_spacing_z[min_idx_mask]
        valid = np.isfinite(s)
        if valid.sum() > 3:
            idx = np.nanargmin(s)
            features["min_spacing_time_sec"] = float(traj.time_sec[min_idx_mask][idx])
        else:
            features["min_spacing_time_sec"] = float("nan")
    else:
        features["min_spacing_time_sec"] = float("nan")

    return features


def process_seizure_with_features(seizure_event, cfg, rng):
    pp = cfg["preprocessing"]
    var = cfg["var"]
    sm = cfg["smoothing"]
    art = cfg["artifact"]
    sz_cfg = cfg["seizure"]

    try:
        raw = load_raw_edf(
            cfg["data"]["root"], seizure_event.subject_id,
            seizure_event.session, seizure_event.run, preload=True,
        )
    except Exception as e:
        log(f"      SKIP: {e}")
        return None, None

    try:
        data, sfreq, _ = preprocess_chbmit_raw(
            raw, line_freq=pp["line_freq"], bandpass=tuple(pp["bandpass"]),
        )
    except Exception as e:
        log(f"      SKIP preprocessing: {e}")
        return None, None
    del raw
    gc.collect()

    bl_window = sz_cfg["baseline_window"]
    baseline_start = seizure_event.onset_sec + bl_window[0]
    baseline_end = seizure_event.onset_sec + bl_window[1]
    baseline_start = max(0.0, baseline_start)

    if baseline_end <= baseline_start + 60:
        log(f"      SKIP: insufficient baseline")
        return None, None

    try:
        pca, _ = fit_baseline_pca(
            data, sfreq, baseline_start, baseline_end,
            n_components=pp["n_components"],
        )
    except Exception as e:
        log(f"      SKIP PCA: {e}")
        return None, None

    data_pca = project_to_pca(data, pca)
    del data
    gc.collect()

    artifact_mask = reject_artifact_windows(
        data_pca, sfreq,
        window_sec=var["window_sec"], step_sec=var["step_sec"],
        variance_threshold_sd=art["variance_threshold_sd"],
        kurtosis_threshold=art["kurtosis_threshold"],
    )

    try:
        traj = compute_seizure_trajectory(
            data_pca, sfreq,
            seizure_onset_sec=seizure_event.onset_sec,
            seizure_offset_sec=seizure_event.offset_sec,
            baseline_start_sec=baseline_start, baseline_end_sec=baseline_end,
            window_sec=var["window_sec"], step_sec=var["step_sec"],
            regularization=var["regularization"],
            smoothing_sec=sm["moving_average_sec"],
            subject_id=seizure_event.subject_id,
            seizure_idx=0,
            seizure_duration=seizure_event.duration_sec,
            event_type=seizure_event.event_type,
            artifact_mask=artifact_mask,
        )
    except Exception as e:
        log(f"      SKIP trajectory: {e}")
        del data_pca
        gc.collect()
        return None, None

    n_ch = data_pca.shape[0]
    ch_mean = data_pca.mean(axis=1, keepdims=True)
    ch_std = data_pca.std(axis=1, keepdims=True)
    ch_std[ch_std == 0] = 1.0
    data_z = (data_pca - ch_mean) / ch_std

    window_samples = max(int(var["window_sec"] * sfreq), n_ch + 10)
    step_samples = max(1, int(var["step_sec"] * sfreq))

    jac_result = estimate_jacobian(
        data_z, window_size=window_samples,
        step_size=step_samples, regularization=var["regularization"],
    )

    features = extract_seizure_features(
        data_pca, sfreq, traj, jac_result, seizure_event, baseline_start, baseline_end,
    )

    del data_pca, data_z, jac_result
    gc.collect()

    return traj, features


def group_comparison(features_list):
    narrowing = [f for f in features_list if f["group"] == "narrowing"]
    widening = [f for f in features_list if f["group"] == "widening"]

    results = {
        "n_narrowing": len(narrowing),
        "n_widening": len(widening),
        "pct_narrowing": float(len(narrowing) / len(features_list) * 100) if features_list else 0,
    }

    if len(features_list) > 0:
        from scipy.stats import binomtest
        bt = binomtest(len(narrowing), len(features_list), 0.5, alternative="two-sided")
        results["binomial_p"] = float(bt.pvalue)

    comparison_features = [
        "pre_alpha_power", "pre_delta_power", "pre_broadband_power",
        "pre_spectral_slope", "spectral_slope_change",
        "alpha_change", "delta_change",
        "pre_variance", "variance_change",
        "pre_line_length", "pre_autocorr_lag1",
        "pre_dfa_alpha", "dfa_change",
        "pre_spectral_radius", "spectral_radius_change",
        "pre_residual_var", "residual_var_change",
        "pre_condition_number", "pre_ep_score_z",
        "seizure_duration_sec", "baseline_std",
    ]

    feature_tests = {}
    for feat in comparison_features:
        vals_n = [f[feat] for f in narrowing if np.isfinite(f.get(feat, float("nan")))]
        vals_w = [f[feat] for f in widening if np.isfinite(f.get(feat, float("nan")))]

        if len(vals_n) < 3 or len(vals_w) < 3:
            feature_tests[feat] = {"error": "insufficient_data", "n_narrow": len(vals_n), "n_wide": len(vals_w)}
            continue

        u_stat, p_val = sp_stats.mannwhitneyu(vals_n, vals_w, alternative="two-sided")
        n_total = len(vals_n) + len(vals_w)
        rank_biserial = 1 - (2 * u_stat) / (len(vals_n) * len(vals_w))

        feature_tests[feat] = {
            "median_narrowing": float(np.median(vals_n)),
            "median_widening": float(np.median(vals_w)),
            "mean_narrowing": float(np.mean(vals_n)),
            "mean_widening": float(np.mean(vals_w)),
            "u_stat": float(u_stat),
            "p_value": float(p_val),
            "rank_biserial_r": float(rank_biserial),
            "n_narrowing": len(vals_n),
            "n_widening": len(vals_w),
        }

    p_vals = [v["p_value"] for v in feature_tests.values() if "p_value" in v]
    if p_vals:
        from statsmodels.stats.multitest import multipletests
        reject, p_corrected, _, _ = multipletests(p_vals, method="fdr_bh")
        i = 0
        for feat in feature_tests:
            if "p_value" in feature_tests[feat]:
                feature_tests[feat]["p_fdr"] = float(p_corrected[i])
                feature_tests[feat]["significant_fdr"] = bool(reject[i])
                i += 1

    results["feature_tests"] = feature_tests
    return results


def subject_specificity(features_list):
    from collections import defaultdict
    subject_groups = defaultdict(list)
    for f in features_list:
        subject_groups[f["subject_id"]].append(f["group"])

    results = {}
    for sid, groups in sorted(subject_groups.items()):
        n_narrow = sum(1 for g in groups if g == "narrowing")
        n_wide = sum(1 for g in groups if g == "widening")
        total = len(groups)
        consistency = max(n_narrow, n_wide) / total if total > 0 else 0
        dominant = "narrowing" if n_narrow >= n_wide else "widening"
        results[sid] = {
            "n_seizures": total,
            "n_narrowing": n_narrow,
            "n_widening": n_wide,
            "consistency": float(consistency),
            "dominant_group": dominant,
        }

    consistencies = [v["consistency"] for v in results.values() if v["n_seizures"] >= 2]
    subjects_with_multi = [v for v in results.values() if v["n_seizures"] >= 2]

    summary = {
        "n_subjects": len(results),
        "n_subjects_multi_seizure": len(subjects_with_multi),
        "mean_consistency": float(np.mean(consistencies)) if consistencies else float("nan"),
        "n_always_narrowing": sum(1 for v in subjects_with_multi if v["consistency"] == 1.0 and v["dominant_group"] == "narrowing"),
        "n_always_widening": sum(1 for v in subjects_with_multi if v["consistency"] == 1.0 and v["dominant_group"] == "widening"),
        "n_mixed": sum(1 for v in subjects_with_multi if v["consistency"] < 1.0),
        "per_subject": results,
    }

    return summary


def plot_stratification_figures(features_list, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)

    narrowing = [f for f in features_list if f["group"] == "narrowing"]
    widening = [f for f in features_list if f["group"] == "widening"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    slopes_n = [f["preictal_slope"] for f in narrowing if np.isfinite(f["preictal_slope"])]
    slopes_w = [f["preictal_slope"] for f in widening if np.isfinite(f["preictal_slope"])]
    ax.hist(slopes_n, bins=20, alpha=0.6, color="blue", label=f"Narrowing (n={len(narrowing)})", edgecolor="black")
    ax.hist(slopes_w, bins=20, alpha=0.6, color="red", label=f"Widening (n={len(widening)})", edgecolor="black")
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_xlabel("Pre-ictal slope (z/s)")
    ax.set_ylabel("Count")
    ax.set_title("Pre-ictal Slope by Group")
    ax.legend()

    ax = axes[1]
    changes_n = [f["raw_spacing_change"] for f in narrowing if np.isfinite(f["raw_spacing_change"])]
    changes_w = [f["raw_spacing_change"] for f in widening if np.isfinite(f["raw_spacing_change"])]
    ax.hist(changes_n, bins=20, alpha=0.6, color="blue", label="Narrowing", edgecolor="black")
    ax.hist(changes_w, bins=20, alpha=0.6, color="red", label="Widening", edgecolor="black")
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_xlabel("Raw spacing change (pre - baseline)")
    ax.set_ylabel("Count")
    ax.set_title("Raw Spacing Change Distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "stratification_slopes.png", dpi=200)
    plt.close(fig)

    sig_features = []
    for f in features_list:
        pass

    key_features = [
        ("pre_spectral_slope", "Pre-ictal Spectral Slope"),
        ("pre_dfa_alpha", "Pre-ictal DFA Alpha"),
        ("pre_spectral_radius", "Pre-ictal Spectral Radius"),
        ("pre_residual_var", "Pre-ictal Residual Variance"),
        ("pre_autocorr_lag1", "Pre-ictal Autocorrelation Lag-1"),
        ("pre_variance", "Pre-ictal Variance"),
    ]

    n_feat = len(key_features)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (feat_key, feat_name) in enumerate(key_features):
        ax = axes[i]
        vals_n = [f[feat_key] for f in narrowing if np.isfinite(f.get(feat_key, float("nan")))]
        vals_w = [f[feat_key] for f in widening if np.isfinite(f.get(feat_key, float("nan")))]

        if vals_n and vals_w:
            parts = ax.violinplot([vals_n, vals_w], positions=[0, 1], showmeans=True, showmedians=True)
            if "bodies" in parts:
                parts["bodies"][0].set_facecolor("blue")
                parts["bodies"][0].set_alpha(0.4)
                if len(parts["bodies"]) > 1:
                    parts["bodies"][1].set_facecolor("red")
                    parts["bodies"][1].set_alpha(0.4)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Narrowing", "Widening"])
            if len(vals_n) >= 3 and len(vals_w) >= 3:
                _, p = sp_stats.mannwhitneyu(vals_n, vals_w, alternative="two-sided")
                ax.set_title(f"{feat_name}\np={p:.4f}")
            else:
                ax.set_title(feat_name)
        else:
            ax.set_title(f"{feat_name}\n(insufficient data)")

    fig.tight_layout()
    fig.savefig(fig_dir / "feature_comparison_violin.png", dpi=200)
    plt.close(fig)

    from collections import defaultdict
    subject_groups = defaultdict(lambda: {"narrowing": 0, "widening": 0})
    for f in features_list:
        subject_groups[f["subject_id"]][f["group"]] += 1

    subjects = sorted(subject_groups.keys())
    narrow_counts = [subject_groups[s]["narrowing"] for s in subjects]
    wide_counts = [subject_groups[s]["widening"] for s in subjects]

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    x = np.arange(len(subjects))
    ax.bar(x - 0.2, narrow_counts, 0.4, label="Narrowing", color="blue", alpha=0.7)
    ax.bar(x + 0.2, wide_counts, 0.4, label="Widening", color="red", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Seizure count")
    ax.set_title("Narrowing vs Widening by Subject")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "subject_specificity.png", dpi=200)
    plt.close(fig)


def main():
    t0 = time.time()
    log("=" * 70)
    log("SEIZURE STRATIFICATION & FEATURE EXTRACTION")
    log("=" * 70)

    cfg = load_config()
    seed = cfg["random_seed"]
    rng = np.random.default_rng(seed)

    results_dir = Path(cfg["output"]["results_dir"]) / "analysis"
    fig_dir = Path(cfg["output"]["results_dir"]) / "figures" / "stratification"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log("\n1. Building seizure catalog...")
    sz_cfg = cfg["seizure"]
    catalogs = build_seizure_catalog(
        cfg["data"]["root"],
        min_preictal_sec=sz_cfg["min_preictal_sec"],
        min_inter_seizure_sec=sz_cfg["min_inter_seizure_sec"],
    )

    log("\n2. Processing seizures with full feature extraction...")
    all_features = []
    all_trajectories = []
    seizure_idx_global = 0

    for sid, cat in sorted(catalogs.items()):
        if cat.n_eligible == 0:
            continue

        log(f"\n  {sid}: {cat.n_eligible} eligible seizures")

        for i, sz in enumerate(cat.eligible_seizures):
            log(f"    Seizure {i+1}/{cat.n_eligible}: "
                f"onset={sz.onset_sec:.0f}s, dur={sz.duration_sec:.0f}s")

            traj, features = process_seizure_with_features(sz, cfg, rng)
            if traj is not None and features is not None:
                traj.seizure_idx = seizure_idx_global
                features["seizure_idx_global"] = seizure_idx_global
                all_trajectories.append(traj)
                all_features.append(features)
                log(f"      OK: group={features['group']}, slope={features['preictal_slope']:.6f}, "
                    f"raw_change={features['raw_spacing_change']:.6f}")
                seizure_idx_global += 1

    log(f"\n   Total seizures processed: {len(all_features)}")

    if len(all_features) == 0:
        log("\nERROR: No seizures processed. Exiting.")
        return

    log("\n3. Stratification results...")
    n_narrowing = sum(1 for f in all_features if f["group"] == "narrowing")
    n_widening = sum(1 for f in all_features if f["group"] == "widening")
    log(f"   Narrowing: {n_narrowing} ({n_narrowing/len(all_features)*100:.1f}%)")
    log(f"   Widening:  {n_widening} ({n_widening/len(all_features)*100:.1f}%)")

    log("\n4. Group comparison (narrowing vs widening)...")
    comparison = group_comparison(all_features)
    log(f"   Binomial test p={comparison.get('binomial_p', float('nan')):.4f}")

    sig_feats = []
    for feat, res in comparison.get("feature_tests", {}).items():
        if "p_fdr" in res:
            if res["significant_fdr"]:
                sig_feats.append(feat)
            log(f"   {feat}: p={res['p_value']:.4f}, p_fdr={res['p_fdr']:.4f}, "
                f"r={res['rank_biserial_r']:.3f}"
                f"{' ***' if res['significant_fdr'] else ''}")
        elif "p_value" in res:
            log(f"   {feat}: p={res['p_value']:.4f}, r={res['rank_biserial_r']:.3f}")

    if sig_feats:
        log(f"\n   Significant features (FDR < 0.05): {sig_feats}")
    else:
        log(f"\n   No features significant after FDR correction")

    log("\n5. Subject specificity...")
    subj_spec = subject_specificity(all_features)
    log(f"   Mean consistency: {subj_spec['mean_consistency']:.2f}")
    log(f"   Always narrowing: {subj_spec['n_always_narrowing']}")
    log(f"   Always widening: {subj_spec['n_always_widening']}")
    log(f"   Mixed: {subj_spec['n_mixed']}")

    log("\n6. Generating stratification figures...")
    plot_stratification_figures(all_features, fig_dir)

    log("\n7. Saving results...")
    def sanitize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    final_results = sanitize({
        "n_seizures": len(all_features),
        "n_narrowing": n_narrowing,
        "n_widening": n_widening,
        "group_comparison": comparison,
        "subject_specificity": subj_spec,
        "significant_features": sig_feats,
        "config": cfg,
    })

    save_summary_json(final_results, results_dir, "seizure_stratification.json")

    import csv
    csv_path = results_dir / "per_seizure_features.csv"
    if all_features:
        fieldnames = list(all_features[0].keys())
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for f in all_features:
                writer.writerow(f)
        log(f"   CSV saved: {csv_path}")

    traj_data = {}
    for traj in all_trajectories:
        key = f"{traj.subject_id}_sz{traj.seizure_idx}"
        traj_data[key] = {
            "time_sec": traj.time_sec.tolist(),
            "min_spacing_z": traj.min_spacing_z.tolist(),
            "min_spacing_raw": traj.min_spacing_raw.tolist(),
            "spectral_radius_z": traj.spectral_radius_z.tolist(),
            "ep_score_z": traj.ep_score_z.tolist(),
            "alpha_power_z": traj.alpha_power_z.tolist(),
            "delta_power_z": traj.delta_power_z.tolist(),
            "median_nns_z": traj.median_nns_z.tolist(),
            "p10_nns_z": traj.p10_nns_z.tolist(),
            "subject_id": traj.subject_id,
            "seizure_duration": traj.seizure_duration,
            "preictal_slope": traj.preictal_slope,
            "baseline_mean": traj.baseline_mean,
            "baseline_std": traj.baseline_std,
        }

    np.savez_compressed(
        results_dir / "trajectory_cache.npz",
        **{k: json.dumps(v) for k, v in traj_data.items()},
    )
    log(f"   Trajectory cache saved: {results_dir / 'trajectory_cache.npz'}")

    log_run(cfg, results_dir)

    elapsed = time.time() - t0
    log(f"\n{'=' * 70}")
    log(f"STRATIFICATION COMPLETE in {elapsed/60:.1f} min")
    log(f"Results: {results_dir}")
    log(f"Figures: {fig_dir}")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
