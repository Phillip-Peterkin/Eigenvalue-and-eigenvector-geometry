import os
"""Sleep Dynamics EP Analysis — N3 (Deep Sleep) vs REM vs Awake.

Validates the 'Pathological Locking' hypothesis in natural unconsciousness.
Extends the propofol EP analysis to full-night polysomnography.

Cross-modal context
-------------------
Propofol (ds005620, scalp EEG) showed:
  - Gap TIGHTENING under sedation (opposite of widening prediction)
  - Spectral sensitivity collapse (r: 0.072 → 0.041, p=0.014)
  - Delta-delta correlation r=-0.68 (p=0.0009)
  - Gap is independent of alpha band (topological signature)

This script tests whether natural sleep stages replicate these findings:
  - Test A: Gap narrowing in N3 vs Awake
  - Test B: Spectral sensitivity collapse in N3
  - Test C: REM "recovery" (dreaming exception)
  - Test D: Gap-delta independence in N3

Dataset: ANPHY-Sleep — 10 subjects, 93-channel EEG (10-20/10-05),
1000 Hz, ~7 hrs per recording. Sleep-staged in 30s epochs.

Preprocessing: Remove non-EEG channels → bad channel detection →
notch 50 Hz → bandpass 0.5-45 Hz → downsample 1000→500 Hz →
CSD → PCA (15 components).

Uses longest contiguous run of each sleep state (≥2 min minimum)
to avoid concatenation artifacts from non-contiguous epochs.
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

import mne
from cmcc.preprocess.scalp_eeg import apply_csd, pca_reduce
from cmcc.preprocess.qc import detect_bad_channels
from cmcc.analysis.dynamical_systems import compute_ep_proximity_timecourse
from cmcc.analysis.ep_advanced import (
    compute_spectral_radius_sensitivity,
    compute_alpha_power_per_window,
    _effective_n,
    _adjusted_correlation_p,
)
from cmcc.analysis.contrasts import fdr_correction

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("SLEEP_DATA_ROOT", "./data/ANPHY-Sleep"))
RESULTS_DIR = CMCC_ROOT / "results" / "analysis"
FIG_DIR = CMCC_ROOT / "results" / "figures" / "ep_sleep"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SEC = 0.5
STEP_SEC = 0.1
N_COMPONENTS = 15
DOWNSAMPLE_TO = 500.0
LINE_FREQ = 50.0
SEED = 42

NON_EEG_CHANNELS = [
    "SO1", "SO2", "ZY1", "ZY2",
    "ChEMG1", "ChEMG2",
    "RLEG-", "RLEG+", "LLEG-", "LLEG+",
    "EOG2", "EOG1", "ECG2", "ECG1",
]

OLD_TO_NEW_NAMES = {
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
}

SUBJECTS = {
    "EPCTL01": ("EPCTL01 - fixed.edf", "test1.txt"),
    "EPCTL03": ("EPCTL03.edf", "EPCTL03.txt"),
    "EPCTL05": ("EPCTL05.edf", "EPCTL05.txt"),
    "EPCTL07": ("EPCTL07.edf", "EPCTL07.txt"),
    "EPCTL10": ("EPCTL10.edf", "EPCTL10.txt"),
    "EPCTL14": ("EPCTL14.edf", "EPCTL14.txt"),
    "EPCTL17": ("EPCTL17.edf", "EPCTL17.txt"),
    "EPCTL20": ("EPCTL20.edf", "EPCTL20.txt"),
    "EPCTL24": ("EPCTL24.edf", "EPCTL24.txt"),
    "EPCTL28": ("EPCTL28.edf", "EPCTL28.txt"),
}

MIN_CONTIGUOUS_EPOCHS = 4
TARGET_STATES = ["W", "N3", "R"]
DELTA_BAND = (0.5, 4.0)


def log(msg):
    print(msg, flush=True)


def parse_sleep_staging(staging_path):
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


def find_longest_contiguous_run(epochs, target_stage, min_epochs=4):
    runs = []
    current_start = None
    current_count = 0

    for i, (stage, onset, dur) in enumerate(epochs):
        if stage == target_stage:
            if current_start is None:
                current_start = i
                current_count = 1
            else:
                current_count += 1
        else:
            if current_start is not None and current_count >= min_epochs:
                start_sec = epochs[current_start][1]
                end_sec = epochs[current_start + current_count - 1][1] + epochs[current_start + current_count - 1][2]
                runs.append((start_sec, end_sec, current_count))
            current_start = None
            current_count = 0

    if current_start is not None and current_count >= min_epochs:
        start_sec = epochs[current_start][1]
        end_sec = epochs[current_start + current_count - 1][1] + epochs[current_start + current_count - 1][2]
        runs.append((start_sec, end_sec, current_count))

    if not runs:
        return None

    return max(runs, key=lambda r: r[2])


def load_and_preprocess_segment(edf_path, start_sec, end_sec, sfreq_target=500.0):
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)

    drop_chs = [ch for ch in NON_EEG_CHANNELS if ch in raw.ch_names]
    if drop_chs:
        raw.drop_channels(drop_chs)

    rename_map = {old: new for old, new in OLD_TO_NEW_NAMES.items() if old in raw.ch_names}
    if rename_map:
        raw.rename_channels(rename_map)

    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

    montage = mne.channels.make_standard_montage("standard_1005")
    montage_names = set(montage.ch_names)
    missing = [ch for ch in raw.ch_names if ch not in montage_names]
    if missing:
        raw.drop_channels(missing)

    raw.set_montage(montage, on_missing="warn")

    raw.load_data()
    raw.crop(tmin=start_sec, tmax=end_sec)

    bad = detect_bad_channels(raw)
    raw.info["bads"] = bad
    if bad:
        raw.interpolate_bads(reset_bads=True)

    freqs = [LINE_FREQ * i for i in range(1, 4)]
    raw.notch_filter(freqs, verbose=False)

    raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)

    if raw.info["sfreq"] > sfreq_target:
        raw.resample(sfreq_target, verbose=False)

    raw = apply_csd(raw)

    data = raw.get_data()
    data_pca, pca_obj = pca_reduce(data, n_components=N_COMPONENTS, return_pca=True)

    info = {
        "n_channels_original": data.shape[0],
        "bad_channels": bad,
        "n_channels_after_qc": data.shape[0],
        "sfreq_final": raw.info["sfreq"],
        "duration_sec": float(data_pca.shape[1] / raw.info["sfreq"]),
        "n_components": data_pca.shape[0],
        "cumulative_variance": float(np.sum(pca_obj.explained_variance_ratio_)),
        "channels_dropped_no_montage": missing,
    }

    return data_pca, raw.info["sfreq"], info


def compute_delta_power_per_window(data, sfreq, window_centers, window_samples):
    from scipy.signal import butter, sosfiltfilt

    nyq = sfreq / 2.0
    lo = DELTA_BAND[0] / nyq
    hi = DELTA_BAND[1] / nyq
    if hi >= 1.0:
        hi = 0.99
    sos = butter(4, [lo, hi], btype="band", output="sos")
    data_delta = sosfiltfilt(sos, data, axis=1)

    n_ch, n_total = data.shape
    n_windows = len(window_centers)
    half_w = window_samples // 2
    delta_power = np.zeros(n_windows)
    for i, c in enumerate(window_centers):
        c = int(c)
        start = max(0, c - half_w)
        end = min(n_total, c + half_w)
        if end > start:
            delta_power[i] = np.mean(data_delta[:, start:end] ** 2)

    return delta_power


def analyze_state(subject_id, edf_path, staging_path, target_state):
    epochs = parse_sleep_staging(staging_path)
    run = find_longest_contiguous_run(epochs, target_state, MIN_CONTIGUOUS_EPOCHS)

    if run is None:
        return None

    start_sec, end_sec, n_epochs = run
    duration = end_sec - start_sec

    log(f"    {target_state}: {n_epochs} epochs ({duration:.0f}s) [{start_sec:.0f}-{end_sec:.0f}s]")

    try:
        data_pca, sfreq, preprocess_info = load_and_preprocess_segment(
            edf_path, start_sec, end_sec, sfreq_target=DOWNSAMPLE_TO,
        )
    except Exception as e:
        log(f"    ERROR preprocessing: {e}")
        return None

    ep_tc = compute_ep_proximity_timecourse(
        data_pca, sfreq=sfreq,
        window_sec=WINDOW_SEC, step_sec=STEP_SEC,
        max_channels=N_COMPONENTS, seed=SEED,
    )

    jac = ep_tc["jac_result"]
    ep = ep_tc["ep_result"]
    window_samples = int(WINDOW_SEC * sfreq)

    spec_sens = compute_spectral_radius_sensitivity(jac, ep)

    result = {
        "subject": subject_id,
        "state": target_state,
        "n_epochs": n_epochs,
        "segment_start_sec": start_sec,
        "segment_end_sec": end_sec,
        "duration_sec": preprocess_info["duration_sec"],
        "n_windows": len(jac.window_centers),
        "preprocess": preprocess_info,
        "spectral_sensitivity": spec_sens,
        "mean_spectral_radius": float(np.mean(jac.spectral_radius)),
        "mean_eigenvalue_gap": float(np.mean(ep.min_eigenvalue_gaps)),
        "std_eigenvalue_gap": float(np.std(ep.min_eigenvalue_gaps)),
        "mean_ep_score": float(np.mean(ep.ep_scores)),
    }

    if target_state == "N3":
        delta_power = compute_delta_power_per_window(
            data_pca, sfreq, jac.window_centers, window_samples,
        )
        gaps = ep.min_eigenvalue_gaps
        valid = np.isfinite(gaps) & np.isfinite(delta_power) & (delta_power > 0)
        n_valid = int(valid.sum())

        if n_valid >= 20:
            g = gaps[valid]
            d = delta_power[valid]
            r_gd, p_gd = sp_stats.pearsonr(g, d)
            rho_gd, p_sp_gd = sp_stats.spearmanr(g, d)
            n_eff = _effective_n(g, d)
            p_adj = _adjusted_correlation_p(float(r_gd), n_eff)
            result["gap_vs_delta"] = {
                "pearson_r": float(r_gd),
                "p_nominal": float(p_gd),
                "p_adjusted": float(p_adj),
                "spearman_rho": float(rho_gd),
                "p_spearman": float(p_sp_gd),
                "n_windows": n_valid,
                "n_eff": n_eff,
                "mean_delta_power": float(np.mean(d)),
            }
        else:
            result["gap_vs_delta"] = {"error": "insufficient valid windows", "n_valid": n_valid}

    return result


def _cohens_d_paired(a, b):
    diff = np.array(a) - np.array(b)
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def compute_group_statistics(all_results):
    subjects_with_all = {}
    for r in all_results:
        sid = r["subject"]
        if sid not in subjects_with_all:
            subjects_with_all[sid] = {}
        subjects_with_all[sid][r["state"]] = r

    complete = {sid: states for sid, states in subjects_with_all.items()
                if all(s in states for s in TARGET_STATES)}

    if len(complete) < 3:
        log(f"  Only {len(complete)} subjects with all 3 states — insufficient for group stats")
        return {}

    group = {"n_subjects_complete": len(complete)}
    all_p = []
    all_labels = []

    sids = sorted(complete.keys())

    awake_gap = [complete[s]["W"]["mean_eigenvalue_gap"] for s in sids]
    n3_gap = [complete[s]["N3"]["mean_eigenvalue_gap"] for s in sids]
    rem_gap = [complete[s]["R"]["mean_eigenvalue_gap"] for s in sids]

    t_an, p_an = sp_stats.ttest_rel(n3_gap, awake_gap)
    d_an = _cohens_d_paired(n3_gap, awake_gap)
    group["test_a_gap_awake_vs_n3"] = {
        "mean_awake": float(np.mean(awake_gap)),
        "mean_n3": float(np.mean(n3_gap)),
        "mean_diff_n3_minus_awake": float(np.mean(np.array(n3_gap) - np.array(awake_gap))),
        "t": float(t_an),
        "p": float(p_an),
        "cohens_d": d_an,
        "n": len(sids),
        "hypothesis": "N3 gap differs from Awake (propofol showed tightening)",
    }
    all_p.append(float(p_an))
    all_labels.append("gap_awake_vs_n3")

    t_nr, p_nr = sp_stats.ttest_rel(rem_gap, n3_gap)
    d_nr = _cohens_d_paired(rem_gap, n3_gap)
    group["test_a_gap_n3_vs_rem"] = {
        "mean_n3": float(np.mean(n3_gap)),
        "mean_rem": float(np.mean(rem_gap)),
        "mean_diff_rem_minus_n3": float(np.mean(np.array(rem_gap) - np.array(n3_gap))),
        "t": float(t_nr),
        "p": float(p_nr),
        "cohens_d": d_nr,
        "n": len(sids),
    }
    all_p.append(float(p_nr))
    all_labels.append("gap_n3_vs_rem")

    t_ar, p_ar = sp_stats.ttest_rel(rem_gap, awake_gap)
    d_ar = _cohens_d_paired(rem_gap, awake_gap)
    group["test_a_gap_awake_vs_rem"] = {
        "mean_awake": float(np.mean(awake_gap)),
        "mean_rem": float(np.mean(rem_gap)),
        "mean_diff_rem_minus_awake": float(np.mean(np.array(rem_gap) - np.array(awake_gap))),
        "t": float(t_ar),
        "p": float(p_ar),
        "cohens_d": d_ar,
        "n": len(sids),
    }
    all_p.append(float(p_ar))
    all_labels.append("gap_awake_vs_rem")

    awake_r = [complete[s]["W"]["spectral_sensitivity"]["r"] for s in sids]
    n3_r = [complete[s]["N3"]["spectral_sensitivity"]["r"] for s in sids]
    rem_r = [complete[s]["R"]["spectral_sensitivity"]["r"] for s in sids]

    for label, vals in [("awake", awake_r), ("n3", n3_r), ("rem", rem_r)]:
        t_v, p_v = sp_stats.ttest_1samp(vals, 0.0)
        group[f"test_b_spec_sens_{label}"] = {
            "mean_r": float(np.mean(vals)),
            "std_r": float(np.std(vals)),
            "t_vs_zero": float(t_v),
            "p_vs_zero": float(p_v),
            "n": len(vals),
        }
        all_p.append(float(p_v))
        all_labels.append(f"spec_sens_{label}_vs_zero")

    t_wr, p_wr = sp_stats.ttest_rel(awake_r, n3_r)
    d_wr = _cohens_d_paired(awake_r, n3_r)
    group["test_b_spec_sens_awake_vs_n3"] = {
        "mean_diff_awake_minus_n3": float(np.mean(np.array(awake_r) - np.array(n3_r))),
        "t": float(t_wr),
        "p": float(p_wr),
        "cohens_d": d_wr,
        "n": len(sids),
    }
    all_p.append(float(p_wr))
    all_labels.append("spec_sens_awake_vs_n3")

    t_cr, p_cr = sp_stats.ttest_rel(n3_r, rem_r)
    d_cr = _cohens_d_paired(n3_r, rem_r)
    group["test_c_spec_sens_n3_vs_rem"] = {
        "mean_n3": float(np.mean(n3_r)),
        "mean_rem": float(np.mean(rem_r)),
        "mean_diff_n3_minus_rem": float(np.mean(np.array(n3_r) - np.array(rem_r))),
        "t": float(t_cr),
        "p": float(p_cr),
        "cohens_d": d_cr,
        "n": len(sids),
        "interpretation": (
            "If REM r ~ Awake r and > N3 r: spectral sensitivity marks "
            "subjective experience, not arousal. If REM r ~ N3 r: marks arousal."
        ),
    }
    all_p.append(float(p_cr))
    all_labels.append("spec_sens_n3_vs_rem")

    t_rem_aw, p_rem_aw = sp_stats.ttest_rel(awake_r, rem_r)
    d_rem_aw = _cohens_d_paired(awake_r, rem_r)
    group["test_c_spec_sens_awake_vs_rem"] = {
        "mean_awake": float(np.mean(awake_r)),
        "mean_rem": float(np.mean(rem_r)),
        "mean_diff_awake_minus_rem": float(np.mean(np.array(awake_r) - np.array(rem_r))),
        "t": float(t_rem_aw),
        "p": float(p_rem_aw),
        "cohens_d": d_rem_aw,
        "n": len(sids),
    }
    all_p.append(float(p_rem_aw))
    all_labels.append("spec_sens_awake_vs_rem")

    gap_delta_rs = []
    for s in sids:
        gvd = complete[s]["N3"].get("gap_vs_delta", {})
        if "pearson_r" in gvd:
            gap_delta_rs.append(gvd["pearson_r"])

    if len(gap_delta_rs) >= 3:
        t_gd, p_gd = sp_stats.ttest_1samp(gap_delta_rs, 0.0)
        group["test_d_gap_delta_independence"] = {
            "mean_r": float(np.mean(gap_delta_rs)),
            "std_r": float(np.std(gap_delta_rs)),
            "t_vs_zero": float(t_gd),
            "p_vs_zero": float(p_gd),
            "n": len(gap_delta_rs),
            "n_above_0.3": int(sum(1 for r in gap_delta_rs if abs(r) > 0.3)),
            "n_above_0.5": int(sum(1 for r in gap_delta_rs if abs(r) > 0.5)),
            "n_above_0.8": int(sum(1 for r in gap_delta_rs if abs(r) > 0.8)),
            "interpretation": (
                "If mean |r| < 0.3 and few subjects above 0.3: "
                "gap is independent of delta, confirming topological signature. "
                "If r > 0.8 consistently: gap narrowing is just delta power."
            ),
        }
        all_p.append(float(p_gd))
        all_labels.append("gap_delta_independence")

    if len(all_p) >= 2:
        fdr_sig = fdr_correction(all_p, alpha=0.05)
        group["fdr_correction"] = {
            label: {"p": float(p), "fdr_significant": bool(sig)}
            for label, p, sig in zip(all_labels, all_p, fdr_sig)
        }

    return group


def plot_sleep_summary(all_results, group_stats, output_dir):
    subjects_with_all = {}
    for r in all_results:
        sid = r["subject"]
        if sid not in subjects_with_all:
            subjects_with_all[sid] = {}
        subjects_with_all[sid][r["state"]] = r

    complete = {sid: states for sid, states in subjects_with_all.items()
                if all(s in states for s in TARGET_STATES)}

    if len(complete) < 3:
        return

    sids = sorted(complete.keys())

    awake_gap = [complete[s]["W"]["mean_eigenvalue_gap"] for s in sids]
    n3_gap = [complete[s]["N3"]["mean_eigenvalue_gap"] for s in sids]
    rem_gap = [complete[s]["R"]["mean_eigenvalue_gap"] for s in sids]

    awake_r = [complete[s]["W"]["spectral_sensitivity"]["r"] for s in sids]
    n3_r = [complete[s]["N3"]["spectral_sensitivity"]["r"] for s in sids]
    rem_r = [complete[s]["R"]["spectral_sensitivity"]["r"] for s in sids]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    x = np.arange(len(sids))
    w = 0.25
    ax.bar(x - w, awake_gap, w, label="Awake", color="steelblue", alpha=0.8)
    ax.bar(x, n3_gap, w, label="N3", color="coral", alpha=0.8)
    ax.bar(x + w, rem_gap, w, label="REM", color="seagreen", alpha=0.8)
    ax.set_ylabel("Mean Min Eigenvalue Gap")
    ax.set_xticks(x)
    ax.set_xticklabels([s[-2:] for s in sids], fontsize=8)
    ax.legend(fontsize=8)
    p_an = group_stats.get("test_a_gap_awake_vs_n3", {}).get("p", float("nan"))
    d_an = group_stats.get("test_a_gap_awake_vs_n3", {}).get("cohens_d", float("nan"))
    ax.set_title(f"Test A: Gap by State\nAwake vs N3: p={p_an:.4f}, d={d_an:.2f}")

    ax = axes[0, 1]
    ax.bar(x - w, awake_r, w, label="Awake", color="steelblue", alpha=0.8)
    ax.bar(x, n3_r, w, label="N3", color="coral", alpha=0.8)
    ax.bar(x + w, rem_r, w, label="REM", color="seagreen", alpha=0.8)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("Spectral Sensitivity r")
    ax.set_xticks(x)
    ax.set_xticklabels([s[-2:] for s in sids], fontsize=8)
    ax.legend(fontsize=8)
    p_wr = group_stats.get("test_b_spec_sens_awake_vs_n3", {}).get("p", float("nan"))
    d_wr = group_stats.get("test_b_spec_sens_awake_vs_n3", {}).get("cohens_d", float("nan"))
    ax.set_title(f"Test B: Spectral Sensitivity\nAwake vs N3: p={p_wr:.4f}, d={d_wr:.2f}")

    ax = axes[1, 0]
    states_data = [awake_r, n3_r, rem_r]
    state_labels = ["Awake", "N3", "REM"]
    colors = ["steelblue", "coral", "seagreen"]
    bp = ax.boxplot(states_data, labels=state_labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for i, (data, color) in enumerate(zip(states_data, colors)):
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(data))
        ax.scatter(np.ones(len(data)) * (i + 1) + jitter, data,
                   c=color, s=30, zorder=5, edgecolors="white", linewidth=0.5)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("Spectral Sensitivity r")
    p_cr = group_stats.get("test_c_spec_sens_n3_vs_rem", {}).get("p", float("nan"))
    d_cr = group_stats.get("test_c_spec_sens_n3_vs_rem", {}).get("cohens_d", float("nan"))
    ax.set_title(f"Test C: REM Recovery\nN3 vs REM: p={p_cr:.4f}, d={d_cr:.2f}")

    ax = axes[1, 1]
    gap_delta_rs = []
    gap_delta_sids = []
    for s in sids:
        gvd = complete[s]["N3"].get("gap_vs_delta", {})
        if "pearson_r" in gvd:
            gap_delta_rs.append(gvd["pearson_r"])
            gap_delta_sids.append(s)
    if gap_delta_rs:
        ax.bar(range(len(gap_delta_rs)), gap_delta_rs, color="mediumpurple", alpha=0.7)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(0.3, color="red", linestyle="--", alpha=0.4, label="|r|=0.3")
        ax.axhline(-0.3, color="red", linestyle="--", alpha=0.4)
        ax.set_xticks(range(len(gap_delta_sids)))
        ax.set_xticklabels([s[-2:] for s in gap_delta_sids], fontsize=8)
        ax.set_ylabel("Gap-Delta Pearson r")
        ax.legend(fontsize=8)
        mean_gd = float(np.mean(gap_delta_rs))
        ax.set_title(f"Test D: Gap-Delta Independence (N3)\nmean r={mean_gd:.4f}")
    else:
        ax.text(0.5, 0.5, "No gap-delta data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Test D: Gap-Delta Independence")

    plt.tight_layout()
    fig_path = output_dir / "sleep_dynamics_summary.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Figure: {fig_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (state, gap_vals, color, label) in enumerate([
        ("W", awake_gap, "steelblue", "Awake"),
        ("N3", n3_gap, "coral", "N3"),
        ("R", rem_gap, "seagreen", "REM"),
    ]):
        ax = axes[i]
        ax.hist(gap_vals, bins=10, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(np.mean(gap_vals), color="black", linestyle="--", linewidth=2)
        ax.set_xlabel("Mean Min Eigenvalue Gap")
        ax.set_ylabel("Count")
        ax.set_title(f"{label}\nmean={np.mean(gap_vals):.6f}")

    plt.tight_layout()
    fig_path = output_dir / "sleep_gap_histograms.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Figure: {fig_path}")


def main():
    log("=" * 70)
    log("SLEEP DYNAMICS EP ANALYSIS")
    log(f"States: {TARGET_STATES}")
    log(f"Min contiguous epochs: {MIN_CONTIGUOUS_EPOCHS} ({MIN_CONTIGUOUS_EPOCHS * 30}s)")
    log(f"Parameters: window={WINDOW_SEC}s, step={STEP_SEC}s, PCA={N_COMPONENTS}")
    log("=" * 70)

    all_results = []
    t_start = time.time()

    for subject_id, (edf_name, staging_name) in SUBJECTS.items():
        edf_path = DATA_ROOT / subject_id / edf_name
        staging_path = DATA_ROOT / subject_id / staging_name

        if not edf_path.exists():
            log(f"\n  {subject_id}: EDF not found — SKIP")
            continue
        if not staging_path.exists():
            log(f"\n  {subject_id}: staging file not found — SKIP")
            continue

        log(f"\n{'='*50}")
        log(f"  {subject_id}")

        for state in TARGET_STATES:
            t0 = time.time()
            try:
                result = analyze_state(subject_id, edf_path, staging_path, state)
                if result:
                    r_val = result["spectral_sensitivity"]["r"]
                    gap_val = result["mean_eigenvalue_gap"]
                    log(f"      spec_sens r={r_val:.4f}, gap={gap_val:.6f} "
                        f"[{result['n_windows']} windows, {time.time()-t0:.0f}s]")
                    if "gap_vs_delta" in result and "pearson_r" in result["gap_vs_delta"]:
                        log(f"      gap-delta r={result['gap_vs_delta']['pearson_r']:.4f}")
                    all_results.append(result)
                else:
                    log(f"      SKIP: no contiguous run >= {MIN_CONTIGUOUS_EPOCHS} epochs")
            except Exception as e:
                log(f"      ERROR: {e}")
                import traceback
                traceback.print_exc()

            gc.collect()

    elapsed = time.time() - t_start
    log(f"\n{'='*70}")
    log(f"PROCESSING COMPLETE: {len(all_results)} state-segments in {elapsed:.0f}s")
    log(f"{'='*70}")

    group_stats = compute_group_statistics(all_results)

    if group_stats:
        log(f"\n{'='*70}")
        log("GROUP STATISTICS")
        log(f"{'='*70}")

        ta = group_stats.get("test_a_gap_awake_vs_n3", {})
        if ta:
            log(f"\n  Test A — Gap: Awake vs N3")
            log(f"    Awake gap: {ta.get('mean_awake', 0):.6f}")
            log(f"    N3 gap:    {ta.get('mean_n3', 0):.6f}")
            log(f"    Diff:      {ta.get('mean_diff_n3_minus_awake', 0):.6f}")
            log(f"    t={ta.get('t', 0):.4f}, p={ta.get('p', 0):.6f}, d={ta.get('cohens_d', 0):.3f}")

        for state in ["awake", "n3", "rem"]:
            tb = group_stats.get(f"test_b_spec_sens_{state}", {})
            if tb:
                log(f"\n  Test B — Spectral Sensitivity ({state})")
                log(f"    mean r={tb.get('mean_r', 0):.4f} +/- {tb.get('std_r', 0):.4f}")
                log(f"    t vs 0: t={tb.get('t_vs_zero', 0):.3f}, p={tb.get('p_vs_zero', 0):.6f}")

        tc = group_stats.get("test_c_spec_sens_n3_vs_rem", {})
        if tc:
            log(f"\n  Test C — REM Recovery (N3 vs REM)")
            log(f"    N3 r:  {tc.get('mean_n3', 0):.4f}")
            log(f"    REM r: {tc.get('mean_rem', 0):.4f}")
            log(f"    t={tc.get('t', 0):.4f}, p={tc.get('p', 0):.6f}, d={tc.get('cohens_d', 0):.3f}")

        td = group_stats.get("test_d_gap_delta_independence", {})
        if td:
            log(f"\n  Test D — Gap-Delta Independence (N3)")
            log(f"    mean r={td.get('mean_r', 0):.4f} +/- {td.get('std_r', 0):.4f}")
            log(f"    |r|>0.3: {td.get('n_above_0.3', 0)}, |r|>0.8: {td.get('n_above_0.8', 0)}")

        fdr = group_stats.get("fdr_correction", {})
        if fdr:
            log(f"\n  FDR Correction:")
            for label, info in fdr.items():
                log(f"    {label}: p={info['p']:.6f} sig={info['fdr_significant']}")

    plot_sleep_summary(all_results, group_stats, FIG_DIR)

    output = {
        "analysis": "sleep_dynamics_ep",
        "dataset": "ANPHY-Sleep",
        "parameters": {
            "window_sec": WINDOW_SEC,
            "step_sec": STEP_SEC,
            "n_components": N_COMPONENTS,
            "downsample_to": DOWNSAMPLE_TO,
            "line_freq": LINE_FREQ,
            "min_contiguous_epochs": MIN_CONTIGUOUS_EPOCHS,
            "delta_band_hz": list(DELTA_BAND),
        },
        "per_state_results": all_results,
        "group_statistics": group_stats,
        "n_subjects_attempted": len(SUBJECTS),
        "n_state_segments": len(all_results),
        "elapsed_seconds": elapsed,
    }

    json_path = RESULTS_DIR / "ep_sleep_dynamics.json"

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    log(f"\n  Results: {json_path}")


if __name__ == "__main__":
    main()
